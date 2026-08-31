import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

CODEC_PATH = Path(__file__).resolve().parents[3] / "SDPO/verl/utils/checkpoint/shared_model.py"
spec = importlib.util.spec_from_file_location("shared_codec", CODEC_PATH)
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)
try:
    import torch
except ImportError:
    torch = None

CONFIG = {"model_type": "fixture", "hidden_size": 1024}


@unittest.skipIf(torch is None, "CPU torch needed for real serialization tests")
class SharedCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = self.root / "shared"
        self.group = codec.namespace(CONFIG, 8, 0)
        torch.manual_seed(17)
        base = torch.randn(1024, 1024)
        self.state = {"base.weight": base, "base.view": base[:, ::2],
                      "lora_A": torch.randn(32, 16), "counter": 15,
                      "bf16": torch.randn(512, 512).bfloat16()}

    def save(self, name="model.pt", state=None):
        path = self.root / name
        torch.save(self.state if state is None else state, path)
        return path

    def convert(self, path):
        return codec.compact_model(path, self.store, self.group)

    def assert_state(self, actual, expected=None):
        expected = expected or self.state
        for key, value in expected.items():
            if torch.is_tensor(value):
                self.assertTrue(torch.equal(actual[key], value), key)
                self.assertEqual(actual[key].dtype, value.dtype)
                self.assertEqual(actual[key].stride(), value.stride())
            else:
                self.assertEqual(actual[key], value)
        self.assertEqual(actual["base.weight"].untyped_storage().data_ptr(),
                         actual["base.view"].untyped_storage().data_ptr())

    def test_exact_original_bytes_and_torch_state(self):
        path = self.save()
        original = path.read_bytes()
        result = self.convert(path)
        self.assertEqual(result["status"], "shared")
        self.assertLess(path.stat().st_size, len(original) // 20)
        with codec.SharedReader(path) as reader:
            self.assertEqual(reader.read(), original)
            reader.seek(-31, 2)
            self.assertEqual(reader.read(40), original[-31:])
            reader.seek(137)
            data = bytearray(12345)
            self.assertEqual(reader.readinto(data), len(data))
            self.assertEqual(bytes(data), original[137:137 + len(data)])
        self.assert_state(codec.load_model(path, weights_only=False, map_location="cpu"))

    def test_second_student_has_private_adapter_and_shared_base(self):
        first = self.save()
        self.convert(first)
        other = copy.deepcopy(self.state)
        other["lora_A"].add_(1)
        path = self.save("second_different_header_name.pt", other)
        raw = path.read_bytes()
        self.convert(path)
        self.assertEqual(len(list(self.store.glob("*/base.pt"))), 1)
        with codec.SharedReader(path) as reader:
            self.assertEqual(reader.read(), raw)
        self.assert_state(codec.load_model(path, weights_only=False), other)

    def test_teacher_rounding_delta_is_not_discarded(self):
        self.convert(self.save())
        teacher = copy.deepcopy(self.state)
        teacher["base.weight"].mul_(0.95).add_(self.state["base.weight"], alpha=0.05)
        teacher["lora_A"].mul_(0.8)
        self.assertFalse(torch.equal(teacher["base.weight"], self.state["base.weight"]))
        path = self.save("teacher.pt", teacher)
        result = self.convert(path)
        self.assertGreater(result["delta_bytes"], 0)
        self.assert_state(codec.load_model(path, weights_only=False), teacher)

    def test_optimizer_scheduler_rng_and_next_update_equal(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Parameter(torch.randn(256, 256), requires_grad=False)
                self.adapter = torch.nn.Parameter(torch.randn(256))

            def forward(self, x):
                return x @ self.base + self.adapter

        model = Model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        x = torch.randn(1, 256)
        for _ in range(3):
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
            scheduler.step()
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "rng": torch.get_rng_state(), "data_step": 3}
        path = self.save("train.pt", state)
        baseline = torch.load(path, weights_only=False)
        self.convert(path)
        recovered = codec.load_model(path, weights_only=False)
        outputs = []
        for saved in (baseline, recovered):
            m = Model()
            m.load_state_dict(saved["model"])
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            opt.load_state_dict(saved["optimizer"])
            sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.99)
            sched.load_state_dict(saved["scheduler"])
            torch.set_rng_state(saved["rng"])
            m(torch.randn(1, 256)).square().mean().backward()
            opt.step()
            sched.step()
            outputs.append((m.adapter.detach().clone(), sched.state_dict()))
        self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
        self.assertEqual(outputs[0][1], outputs[1][1])

    def test_existing_ordinary_checkpoint_still_loads(self):
        self.assert_state(codec.load_model(self.save(), weights_only=False))

    def test_idempotent(self):
        path = self.save()
        self.convert(path)
        raw = path.read_bytes()
        self.assertEqual(self.convert(path)["status"], "already_shared")
        self.assertEqual(raw, path.read_bytes())

    def test_missing_base_is_error(self):
        path = self.save()
        result = self.convert(path)
        Path(result["base_path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            codec.load_model(path, weights_only=False)

    def test_corrupt_baseline_detected(self):
        path = self.save()
        self.convert(path)
        with codec.SharedReader(path) as reader:
            record = next(r for r in reader.records if r[1] == "base")
            base = reader.base_path
        base.chmod(0o644)
        with base.open("r+b") as out:
            out.seek(record[5])
            value = out.read(1)
            out.seek(record[5])
            out.write(bytes([value[0] ^ 1]))
        with codec.SharedReader(path) as reader, self.assertRaisesRegex(ValueError, "corruption"):
            reader.read()

    def test_corrupt_private_data_detected(self):
        path = self.save()
        self.convert(path)
        with codec.SharedReader(path) as reader:
            offset = reader.payload_offset
        with path.open("r+b") as out:
            out.seek(offset)
            value = out.read(1)
            out.seek(offset)
            out.write(bytes([value[0] ^ 1]))
        with codec.SharedReader(path) as reader, self.assertRaisesRegex(ValueError, "checksum"):
            reader.read()

    def test_replacement_failure_keeps_original(self):
        path = self.save()
        raw = path.read_bytes()
        original_replace = os.replace

        def replace(src, dst):
            if Path(dst) == path:
                raise OSError("simulated interruption")
            return original_replace(src, dst)

        with patch.object(codec.os, "replace", side_effect=replace), self.assertRaises(OSError):
            self.convert(path)
        self.assertEqual(path.read_bytes(), raw)
        self.assertFalse(list(self.root.glob("*.part")))

    def test_missing_index_recovers_without_replacing_base(self):
        path = self.save()
        result = self.convert(path)
        base = Path(result["base_path"])
        before = base.read_bytes()
        (base.parent / "index.json").unlink()
        self.convert(self.save("second.pt"))
        self.assertEqual(base.read_bytes(), before)

    def test_invalid_file_or_symlink_never_replaced(self):
        path = self.root / "bad.pt"
        path.write_bytes(b"not a torch checkpoint")
        with self.assertRaises(Exception):
            self.convert(path)
        self.assertEqual(path.read_bytes(), b"not a torch checkpoint")
        linked = self.root / "link.pt"
        linked.symlink_to(self.save())
        with self.assertRaises(ValueError):
            self.convert(linked)

    def test_namespace_ignores_paths_but_distinguishes_model_and_rank(self):
        self.assertEqual(codec.namespace(CONFIG, 8, 0),
                         codec.namespace(dict(CONFIG, _name_or_path="another mount"), 8, 0))
        self.assertNotEqual(codec.namespace(CONFIG, 8, 0), codec.namespace(CONFIG, 8, 1))
        self.assertNotEqual(codec.namespace(CONFIG, 8, 0), codec.namespace(dict(CONFIG, hidden_size=512), 8, 0))

    def test_saving_disabled_without_env(self):
        path = self.save()
        raw = path.read_bytes()
        with patch.dict(os.environ, {}, clear=True):
            codec.compact_saved_model(path, CONFIG, 8, 0)
        self.assertEqual(raw, path.read_bytes())

    def test_distributed_dtensor_roundtrip(self):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--distributed-smoke", str(self.root)],
                       check=True, timeout=90, capture_output=True, text=True)

    def test_expand_is_byte_identical_and_plain_loadable(self):
        path = self.save()
        raw = path.read_bytes()
        self.convert(path)
        self.assertEqual(codec.verify_model(path)["logical_sha256"], codec.digest(raw))
        self.assertEqual(codec.expand_model(path)["status"], "expanded")
        self.assertEqual(raw, path.read_bytes())
        self.assert_state(torch.load(path, weights_only=False))

    def test_full_parameter_changes_preserved(self):
        self.convert(self.save())
        changed = copy.deepcopy(self.state)
        changed["base.weight"].add_(torch.randn_like(changed["base.weight"]) * 1e-4)
        path = self.save("full_parameter.pt", changed)
        raw = path.read_bytes()
        self.convert(path)
        with codec.checkpoint_stream(path) as reader:
            self.assertEqual(raw, reader.read())
        self.assert_state(codec.load_model(path, weights_only=False), changed)

    def test_mmap_rejected_without_altering_checkpoint(self):
        path = self.save()
        self.convert(path)
        with self.assertRaisesRegex(ValueError, "mmap"):
            codec.load_model(path, weights_only=False, mmap=True)

    def test_sharded_tensor_roundtrip(self):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--sharded-smoke", str(self.root)],
                       check=True, timeout=90, capture_output=True, text=True)


def distributed_worker(rank, root, sharded=False):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard

    root = Path(root)
    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        source = torch.arange(512 * 512, dtype=torch.float32).reshape(512, 512)
        if sharded:
            from torch.distributed._shard.sharded_tensor import init_from_local_shards, Shard as LocalShard
            from torch.distributed._shard.metadata import ShardMetadata
            local = source.chunk(2)[rank].clone()
            metadata = ShardMetadata([rank * 256, 0], [256, 512], f"rank:{rank}/cpu")
            tensor = init_from_local_shards([LocalShard(local, metadata)], *source.shape)
        else:
            tensor = distribute_tensor(source, mesh, [Shard(0)])
        path = root / f"model_world_size_2_rank_{rank}.pt"
        torch.save({"weight": tensor}, path)
        original = path.read_bytes()
        codec.compact_model(path, root / "dtensor-shared", codec.namespace(CONFIG, 2, rank))
        restored = codec.load_model(path, weights_only=False)["weight"]
        if sharded:
            assert torch.equal(restored.local_shards()[0].tensor, tensor.local_shards()[0].tensor)
            assert restored.metadata() == tensor.metadata()
        else:
            assert torch.equal(restored.to_local(), tensor.to_local())
            assert restored.placements == tensor.placements
        with codec.SharedReader(path) as reader:
            assert reader.read() == original
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--distributed-smoke", "--sharded-smoke"):
        torch.multiprocessing.spawn(distributed_worker, args=(sys.argv[2], sys.argv[1] == "--sharded-smoke"), nprocs=2, join=True)
    else:
        unittest.main()
