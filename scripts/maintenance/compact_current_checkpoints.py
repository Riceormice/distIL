#!/usr/bin/env python3
"""Losslessly share native model shards of the current OPSD experiment registry.

Default: read-only inventory. Run --apply --jobs-paused only with all affected
training/evaluation jobs stopped. No checkpoint is deleted. Optimizer, RNG, data,
metrics and unrelated projects are never rewritten. Uses stdlib Python only.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

import audit_opsd_checkpoint_retention as audit
import cleanup_reviewed_math_weights as guarded
from cleanup_verified_physics_weights import journal, UnsafeCleanup


REPO = Path(__file__).resolve().parents[2]
CODEC_REL = Path("verl/utils/checkpoint/shared_model.py")
spec = importlib.util.spec_from_file_location("shared_model_codec", REPO / "SDPO" / CODEC_REL)
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)
STEP = re.compile(r"global_step_(\d+)$")


def current_runs(root: Path, selected: set[str]):
    known = {job.job_id for job, _ in audit.current_jobs(root)}
    if selected - known:
        raise ValueError(f"unknown jobs: {sorted(selected - known)}")
    for job, parent in audit.current_jobs(root):
        if selected and job.job_id not in selected:
            continue
        guarded.safe_path(root, parent)
        matches = sorted(parent.glob(job.pattern)) if parent.exists() else []
        if len(matches) > 1:
            raise UnsafeCleanup(f"ambiguous current run for {job.job_id}: {matches}")
        if matches:
            guarded.safe_path(root, matches[0])
            if not matches[0].is_dir():
                raise UnsafeCleanup(f"run is not a directory: {matches[0]}")
        yield job, matches[0] if matches else None


def checkpoints(root: Path, run: Path, kind: str):
    parent = run / "model_states" if kind == "p0" else run / "native/checkpoints" / run.name
    guarded.safe_path(root, parent)
    found = list(parent.glob("global_step_*"))
    for path in found:
        guarded.safe_path(root, path)
        if STEP.fullmatch(path.name) is None or not path.is_dir():
            raise UnsafeCleanup(f"unexpected native checkpoint: {path}")
    return sorted(found, key=lambda p: int(STEP.fullmatch(p.name)[1]))


def shards(root: Path, checkpoint: Path, kind: str):
    """Validate all ranks before converting the first shard of this checkpoint."""
    actor = checkpoint / "actor"
    guarded.safe_path(root, actor)
    guarded.safe_path(root, actor / "fsdp_config.json")
    fsdp = json.loads((actor / "fsdp_config.json").read_text())
    world = fsdp["world_size"]
    if type(world) is not int or not 1 <= world <= 64:
        raise ValueError(f"invalid world size: {actor}")
    for file in [checkpoint / "data.pt", *[
        actor / f"{prefix}_world_size_{world}_rank_{rank}.pt"
        for prefix in ("optim", "extra_state") for rank in range(world)
    ]]:
        guarded.safe_path(root, file)
        if codec.stamp(file)[2] <= 0:
            raise ValueError(f"empty resume state: {file}")
    teacher = actor / "ema_teacher"
    guarded.safe_path(root, teacher)
    if not teacher.exists():
        if kind != "p0":
            raise ValueError(f"SR-OPSD EMA teacher is missing: {teacher}")
        print(f"WARNING LEGACY_TEACHER_MISSING: {checkpoint}; compaction cannot restore missing history", flush=True)
    result = []
    for group in [actor] + ([teacher] if teacher.exists() else []):
        guarded.safe_path(root, group / "fsdp_config.json")
        if json.loads((group / "fsdp_config.json").read_text())["world_size"] != world:
            raise ValueError(f"teacher/actor world size mismatch: {group}")
        config_path = group / "huggingface/config.json"
        guarded.safe_path(root, config_path)
        config = json.loads(config_path.read_text())
        expected = {f"model_world_size_{world}_rank_{rank}.pt" for rank in range(world)}
        if {p.name for p in group.glob("model*.pt")} != expected:
            raise ValueError(f"missing or unexpected model ranks: {group}")
        for rank in range(world):
            path = group / f"model_world_size_{world}_rank_{rank}.pt"
            guarded.safe_path(root, path)
            if codec.stamp(path)[2] <= 0:
                raise ValueError(f"empty model shard: {path}")
            result.append((path, codec.namespace(config, world, rank)))
    return result


def assert_quiet(root: Path, run: Path, native: list[Path], job):
    # Lock-free legacy jobs can still exist remotely; also require a quiet window.
    paths = [p for p in audit.scheduled.candidate_logs(run, job, root / "nightly_experiment_state")
             if p.is_relative_to(root)]
    paths.extend((run / "logs").glob("*.log"))
    for checkpoint in native:
        for parent, dirs, files in os.walk(checkpoint, followlinks=False):
            for name in dirs + files:
                path = Path(parent) / name
                guarded.safe_path(root, path)
                if path.is_file():
                    paths.append(path)
    for path in paths:
        guarded.safe_path(root, path)
        info = path.lstat()
        if time.time() - max(info.st_mtime, info.st_ctime) < guarded.QUIET_SECONDS:
            raise UnsafeCleanup(f"changed within {guarded.QUIET_SECONDS}s: {path}; stop jobs and wait")


def allocated(path: Path) -> int:
    seen = set()
    total = 0
    if not path.exists():
        return 0
    for parent, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(parent) / name).is_symlink()]
        for name in files:
            info = (Path(parent) / name).lstat()
            key = info.st_dev, info.st_ino
            if stat.S_ISREG(info.st_mode) and key not in seen:
                total += info.st_blocks * 512
                seen.add(key)
    return total


def process_run(root, store, run, job, args, stream):
    native = checkpoints(root, run, job.kind)
    if not native:
        trainer = run / "checkpoints"
        guarded.safe_path(root, trainer)
        adapters = list(trainer.glob("*/checkpoint-*/adapter_model.safetensors"))
        status = "ALREADY_ADAPTER_BASED" if adapters else "NO_NATIVE_CHECKPOINT"
        record = {"job": job.job_id, "status": status, "run": str(run), "adapter_checkpoints": len(adapters)}
        journal(stream, record)
        print(f"{job.job_id}: {status}", flush=True)
        return
    if args.apply and job.kind == "p0":
        target = args.p0_repo / CODEC_REL
        if not target.is_file() or target.read_bytes() != (REPO / "SDPO" / CODEC_REL).read_bytes():
            raise ValueError(f"update P0 shared-checkpoint reader before converting: {target}")
    with ExitStack() as stack:
        if args.apply or args.mode == "verify":
            for directory, filename in (
                (root / "nightly_experiment_state" / job.job_id, "nightly.lock"),
                (run / "state", "pipeline.lock"),
                (run, ".extreme_cleanup.lock"),
            ):
                guarded.safe_path(root, directory / filename)
                directory.mkdir(parents=True, exist_ok=True)
                guarded.take_lock(root, directory / filename, stack, create=True)
            assert_quiet(root, run, native, job)
        planned = [(ckpt, shards(root, ckpt, job.kind)) for ckpt in native]
        before_alloc = sum(allocated(ckpt) for ckpt in native)
        for ckpt, files in planned:
            for path, group in files:
                before = codec.stamp(path)
                is_shared = codec.is_shared(path)
                header = {"job": job.job_id, "path": str(path), "before_bytes": before[2],
                          "was_shared": is_shared, "mode": args.mode, "source_stamp": before}
                journal(stream, {**header, "status": "STARTED" if args.apply else "INVENTORY"})
                if args.mode == "verify":
                    result = {"status": "verified", **codec.verify_model(path)}
                elif not args.apply:
                    result = {"status": "would_" + args.mode, "after_bytes": None}
                elif args.mode == "expand":
                    result = codec.expand_model(path)
                else:
                    result = codec.compact_model(path, store, group)
                journal(stream, {**header, **result})
                after = result.get("after_bytes")
                size = f'{before[2] / 2**30:.3f} -> {after / 2**30:.3f} GiB' if after is not None else f'{before[2] / 2**30:.3f} GiB'
                print(f'{job.job_id} {ckpt.name} {path.parent.name}/{path.name}: {result["status"]} {size}', flush=True)
        after_alloc = sum(allocated(ckpt) for ckpt in native)
        journal(stream, {"status": "RUN_FINISHED", "job": job.job_id,
                         "checkpoint_allocated_before": before_alloc, "checkpoint_allocated_after": after_alloc})
        print(f"RUN_TOTAL {job.job_id}: {before_alloc / 2**30:.3f} -> {after_alloc / 2**30:.3f} GiB (shared store separate)", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=audit.scheduled.BASE)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--job", action="append", default=[])
    parser.add_argument("--mode", choices=("compact", "verify", "expand"), default="compact")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--jobs-paused", action="store_true")
    parser.add_argument("--p0-repo", type=Path, default=Path("/media/damoxing/che-liu-fileset/ylong/sdpo/code/SDPO-p0-mechanism"))
    args = parser.parse_args(argv)
    if args.apply and not args.jobs_paused:
        parser.error("--apply requires --jobs-paused; stop affected training/evaluation and scheduled launches first")
    root = args.root.absolute()
    guarded.safe_path(root, root)
    store = (args.store or root / "sdpo/shared_checkpoint_bases/v1").absolute()
    # Keep shared dependencies inside the protected asset subtree, never a run.
    if not store.is_relative_to(root / "sdpo/shared_checkpoint_bases"):
        parser.error("--store must be inside ROOT/sdpo/shared_checkpoint_bases")
    guarded.safe_path(root, store)
    jobs = list(current_runs(root, set(args.job)))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    with (args.report_dir / "conversion.jsonl").open("x") as stream, ExitStack() as stack:
        if args.apply:
            guarded.take_lock(root, root / ".opsd_extreme_cleanup.lock", stack, create=True)
            guarded.take_lock(root, root / ".shared_checkpoint_conversion.lock", stack, create=True)
        initial_store = allocated(store)
        for job, run in jobs:
            try:
                if run is None:
                    print(f"{job.job_id}: NOT_FOUND", flush=True)
                    journal(stream, {"job": job.job_id, "status": "NOT_FOUND"})
                else:
                    process_run(root, store, run, job, args, stream)
            except (OSError, ValueError, KeyError, UnsafeCleanup) as exc:
                failures += 1
                journal(stream, {"job": job.job_id, "status": "FAILED_OR_SKIPPED", "error": str(exc)})
                print(f"FAILED_OR_SKIPPED {job.job_id}: {exc}", flush=True)
        final_store = allocated(store)
        summary = {"status": "FINISHED", "failures": failures, "jobs": len(jobs),
                   "shared_store": str(store), "store_allocated_before": initial_store,
                   "store_allocated_after": final_store}
        journal(stream, summary)
        print(f"SHARED_STORE {initial_store / 2**30:.3f} -> {final_store / 2**30:.3f} GiB", flush=True)
        print(f"FINISHED failures={failures}; receipts={args.report_dir}; do not delete the shared store", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
