#!/usr/bin/env python3
"""Small distributed save/restore test using the real VERL checkpoint manager.

Run with torchrun on the target runtime before converting production checkpoints.
Only writes into the new --out directory. No downloads or experiment data access.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--verl-root", type=Path, default=Path(__file__).resolve().parents[2] / "SDPO")
args = parser.parse_args()
sys.path.insert(0, str(args.verl_root))

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PretrainedConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.checkpoint.shared_model import compact_model, is_shared, namespace, verify_model


class TinyModel(torch.nn.Module):
    def __init__(self, full_parameter=False):
        super().__init__()
        self.config = PretrainedConfig(hidden_size=1024, model_type="shared-checkpoint-smoke")
        self.base = torch.nn.Parameter(torch.randn(1024, 1024) * 0.01, requires_grad=full_parameter)
        self.adapter_a = torch.nn.Parameter(torch.randn(1024, 8) * 0.01)
        self.adapter_b = torch.nn.Parameter(torch.randn(8, 1024) * 0.01)

    def forward(self, x):
        return x @ self.base + (x @ self.adapter_a) @ self.adapter_b

    def can_generate(self):
        return False


def run_case(full_parameter, out, device, rank, world):
    torch.manual_seed(42)
    mesh = init_device_mesh(device.type, (world,), mesh_dim_names=("fsdp",))
    original = TinyModel(full_parameter)
    config = original.config.to_dict()
    student = FSDP(original.to(device), device_id=device, use_orig_params=True, device_mesh=mesh)
    torch.manual_seed(42)
    teacher = FSDP(TinyModel(full_parameter).to(device), device_id=device, use_orig_params=True, device_mesh=mesh)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
    actor_manager = FSDPCheckpointManager(student, optimizer, scheduler)
    teacher_manager = FSDPCheckpointManager(teacher, checkpoint_config={"save_contents": ["model"], "load_contents": ["model"]})

    def update():
        optimizer.zero_grad()
        student(torch.randn(2, 1024, device=device)).square().mean().backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            for target, current in zip(teacher.parameters(), student.parameters()):
                target.mul_(0.95).add_(current, alpha=0.05)

    def snapshot():
        return [p.detach().clone() for m in (student, teacher) for p in m.parameters()]

    def merge_weights(directory):
        # Exercise the production shard loader without creating an HF model.
        merger = FSDPModelMerger.__new__(FSDPModelMerger)
        merger.config = SimpleNamespace(local_dir=str(directory))
        count = merger._get_world_size()
        first = merger._load_rank_zero_state_dict(count)
        layout, names = merger._extract_device_mesh_info(first, count)
        shards, shape = merger._calculate_shard_configuration(layout, names)
        return merger._load_and_merge_state_dicts(count, shards, shape, names)

    update()
    # First exercise conversion of already-saved ordinary checkpoints.
    os.environ.pop("SDPO_SHARED_CHECKPOINT_STORE", None)
    actor_path = out / "checkpoint-1/actor"
    teacher_path = actor_path / "ema_teacher"
    actor_manager.save_checkpoint(str(actor_path), global_step=1)
    teacher_manager.save_checkpoint(str(teacher_path), global_step=1)
    if rank == 0:
        expected_merged = merge_weights(actor_path)
    dist.barrier()
    for directory in (actor_path, teacher_path):
        shard = directory / f"model_world_size_{world}_rank_{rank}.pt"
        before = verify_model(shard)
        compact_model(shard, out.parent / "shared", namespace(config, world, rank))
        assert verify_model(shard)["logical_sha256"] == before["logical_sha256"]
        assert is_shared(shard)
    dist.barrier()
    if rank == 0:
        actual_merged = merge_weights(actor_path)
        assert actual_merged.keys() == expected_merged.keys()
        for key in expected_merged:
            torch.testing.assert_close(actual_merged[key], expected_merged[key], rtol=0, atol=0)
    dist.barrier()
    update()
    expected = snapshot()
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_lr = scheduler.get_last_lr()

    actor_manager.load_checkpoint(str(actor_path))
    teacher_manager.load_checkpoint(str(teacher_path))
    update()
    for left, right in zip(expected, snapshot()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for key, value in expected_optimizer["state"].items():
        for field, expected_value in value.items():
            actual = optimizer.state_dict()["state"][key][field]
            torch.testing.assert_close(actual, expected_value, rtol=0, atol=0)
    assert expected_lr == scheduler.get_last_lr()

    # Then exercise automatic compaction on newly saved model/teacher shards.
    os.environ["SDPO_SHARED_CHECKPOINT_STORE"] = str(out.parent / "shared")
    next_actor = out / "checkpoint-2/actor"
    actor_manager.save_checkpoint(str(next_actor), global_step=2)
    teacher_manager.save_checkpoint(str(next_actor / "ema_teacher"), global_step=2)
    for directory in (next_actor, next_actor / "ema_teacher"):
        shard = directory / f"model_world_size_{world}_rank_{rank}.pt"
        assert is_shared(shard)
        verify_model(shard)
    dist.barrier()
    if rank == 0:
        print(f"FSDP {'full_parameter' if full_parameter else 'adapter'}: SAVE/RESTORE/NEXT_UPDATE/MERGE PASS", flush=True)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        if rank == 0:
            args.out.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        for full in (False, True):
            run_case(full, args.out / ("full" if full else "adapter"), device, rank, world)
        if rank == 0:
            source = args.verl_root / "verl/utils/checkpoint/shared_model.py"
            receipt = {"status": "PASS", "world_size": world, "device": str(device),
                       "torch": torch.__version__, "codec_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            (args.out / "PASS.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(f"SHARED FSDP SMOKE: PASS; receipt={args.out / 'PASS.json'}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
