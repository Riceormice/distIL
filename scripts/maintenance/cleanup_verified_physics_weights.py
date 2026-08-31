#!/usr/bin/env python3
"""Delete only five reviewed, completed Physics model_states trees.

Default: dry run. Math resume checkpoints and all raw analysis data are excluded.
Run on the shared-filesystem development machine, without force-relaunching these
completed Physics jobs concurrently. Existing launcher locks are respected, but
their absence is not a global process-liveness check.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import time


ALLOWED_RUNS = (
    "physics_p0_mechanism_20260825_v2/Qwen3-8B/opsd_fkl/seed0",
    "physics_p0_mechanism_20260825_v2/Qwen3-8B/sdpo_rkl/seed0",
    "physics_p0_mechanism_20260825_v2/Qwen3-8B/sr_opsd/seed0",
    "physics_p0_sdpo_fkl_jsd_20260827/Qwen3-8B/sdpo_fkl/seed0",
    "physics_p0_sdpo_fkl_jsd_20260827/Qwen3-8B/sdpo_jsd/seed0",
)
EXPECTED_STEPS = list(range(0, 421, 5))
PROTECTED_NAMES = {
    "raw_logits_audit", "topk_probe", "token_stats", "train_token_stats",
    "generation", "generation_raw", "evaluation", "evaluations", "aggregate",
    "figures", "protocol", "logs",
}


class UnsafeCleanup(RuntimeError):
    pass


def checked_path(root: Path, path: Path) -> Path:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise UnsafeCleanup(f"symlink in path: {current}")
    return path


def required_file(root: Path, path: Path) -> Path:
    checked_path(root, path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise UnsafeCleanup(f"missing/empty evidence file: {path}")
    return path


def read_json(root: Path, path: Path) -> dict:
    required_file(root, path)
    if path.stat().st_size > 4 * 1024**2:
        raise UnsafeCleanup(f"unexpectedly large metadata: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise UnsafeCleanup(f"expected metadata object: {path}")
    return data


def completion_evidence(root: Path, relative: str) -> dict:
    if relative not in ALLOWED_RUNS:
        raise UnsafeCleanup("run is not in the reviewed allowlist")
    run = checked_path(root, root / relative)
    experiment = root / relative.split("/")[0]
    manifest = read_json(root, experiment / "protocol/probe_manifest.json")
    training = read_json(root, run / "state/training.complete")
    complete = read_json(root, run / "state/run.complete")
    launch = read_json(root, run / "launch_config.json")
    probe = manifest.get("probe_id")
    schedule = manifest.get("protocol", {}).get("schedule", {})
    capture = manifest.get("protocol", {}).get("capture", {})
    if not probe or schedule.get("total_steps") != 420 or capture.get("capture_freq") != 5:
        raise UnsafeCleanup("expected the reviewed 420-step, eval5 protocol")
    try:
        world_size = int(launch.get("n_gpus_per_node", 0)) * int(launch.get("nnodes", 0))
    except (TypeError, ValueError) as exc:
        raise UnsafeCleanup("invalid launch world size") from exc
    if launch.get("total_steps") != 420 or not 1 <= world_size <= 64:
        raise UnsafeCleanup("missing/invalid launch steps or world size")
    if (training.get("probe_id") != probe or complete.get("probe_id") != probe
            or training.get("total_training_steps") != 420
            or training.get("expected_capture_steps") != EXPECTED_STEPS
            or training.get("missing_capture_steps") != []
            or complete.get("expected_steps") != EXPECTED_STEPS):
        raise UnsafeCleanup("completion metadata is incomplete or does not match the probe")

    retained = []
    for step in EXPECTED_STEPS:
        tag = f"step_{step:04d}"
        for group in ("capture", "generation"):
            marker = read_json(root, run / f"state/{group}/{tag}.complete")
            if marker.get("probe_id") != probe or marker.get("step") != step:
                raise UnsafeCleanup(f"invalid {group} marker for {tag}")
        for name in (f"generation/{tag}.jsonl", f"evaluation/{tag}.json",
                     f"token_stats/{tag}.parquet"):
            retained.append(required_file(root, run / name))
        for name in ("topk_probe", "raw_logits_audit"):
            if name == "raw_logits_audit" and step % 20:
                continue
            directory = checked_path(root, run / name / tag)
            shards = sorted(directory.glob("rank_*.npz"))
            if [p.name for p in shards] != [f"rank_{rank:04d}.npz" for rank in range(world_size)]:
                raise UnsafeCleanup(f"missing/unexpected saved logits shards: {directory}")
            retained.extend(required_file(root, path) for path in shards)
    for name in ("audit_token_summary.parquet", "topk_ratio_stats.parquet"):
        retained.append(required_file(root, experiment / "aggregate" / name))
    retained_metadata = {
        str(path.relative_to(root)): [path.stat().st_size, path.stat().st_mtime_ns]
        for path in retained
    }
    return {"training": training, "complete": complete, "launch": launch,
            "retained_files": retained_metadata}


def weight_inventory(root: Path, target: Path, quiet_seconds: float = 300) -> dict:
    checked_path(root, target)
    if not target.is_dir() or target.name != "model_states" or os.path.ismount(target):
        raise UnsafeCleanup(f"not an ordinary model_states directory: {target}")
    device = target.stat().st_dev
    inventory = {}

    def walk_error(exc):
        raise exc

    for parent, dirs, files in os.walk(target, followlinks=False, onerror=walk_error):
        for path in [Path(parent), *(Path(parent) / n for n in dirs + files)]:
            info = path.lstat()
            if (stat.S_ISLNK(info.st_mode) or info.st_dev != device
                    or os.path.ismount(path)
                    or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode))):
                raise UnsafeCleanup(f"symlink, mount, or special file in weights: {path}")
            if path.name in PROTECTED_NAMES:
                raise UnsafeCleanup(f"analysis data found inside weights: {path}")
            if time.time() - max(info.st_mtime, info.st_ctime) < quiet_seconds:
                raise UnsafeCleanup(f"weights changed in the last {quiet_seconds:g}s: {path}")
            inventory[str(path.relative_to(target))] = [
                info.st_dev, info.st_ino, info.st_mode, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns,
            ]
    return inventory


def acquire_locks(root: Path, relative: str, stack: ExitStack) -> None:
    run = root / relative
    paths = [run / "state/cleanup_weights.lock", run / "state/pipeline.lock"]
    method = run.parent.name
    if method in ("sdpo_fkl", "sdpo_jsd"):
        paths.append(root / f"nightly_experiment_state/physics_logits_{method}/nightly.lock")
    for index, path in enumerate(paths):
        checked_path(root, path)
        if index and not path.exists():
            continue
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        if index == 0:
            flags |= os.O_CREAT
        fd = os.open(path, flags, 0o600)
        stream = stack.enter_context(os.fdopen(fd, "r+"))
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise UnsafeCleanup(f"not an ordinary lock file: {path}")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise UnsafeCleanup(f"launcher/cleanup lock is held: {path}") from exc


def journal(stream, event: dict) -> None:
    stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **event}) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def clean_one(root: Path, relative: str, report: Path, stream, apply: bool) -> str:
    if relative not in ALLOWED_RUNS:
        raise UnsafeCleanup("run is not in the reviewed allowlist")
    target = checked_path(root, root / relative / "model_states")
    if not target.exists():
        journal(stream, {"status": "ABSENT", "target": str(target)})
        print(f"ABSENT {target}", flush=True)
        return "ABSENT"
    with ExitStack() as stack:
        acquire_locks(root, relative, stack)
        evidence = completion_evidence(root, relative)
        inventory = weight_inventory(root, target)
        usage = subprocess.run(["du", "-sk", str(target)], check=True,
                               text=True, capture_output=True, timeout=180)
        allocated = int(usage.stdout.split()[0]) * 1024
        receipt = report / (hashlib.sha256(relative.encode()).hexdigest()[:16] + ".json")
        with receipt.open("x") as output:
            json.dump({"target": str(target), "allocated_bytes_before": allocated,
                       "evidence": evidence, "weight_inventory": inventory}, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        print(f"{'DELETE' if apply else 'WOULD_DELETE'} {allocated / 1024**3:.2f} GiB {target}", flush=True)
        if not apply:
            journal(stream, {"status": "DRY_RUN", "target": str(target), "receipt": str(receipt)})
            return "DRY_RUN"
        if not shutil.rmtree.avoids_symlink_attacks:
            raise UnsafeCleanup("this Python does not provide fd-safe rmtree")
        if inventory != weight_inventory(root, target) or evidence != completion_evidence(root, relative):
            raise UnsafeCleanup("weights or completion evidence changed during checks")
        journal(stream, {"status": "DELETE_STARTED", "target": str(target), "receipt": str(receipt)})
        shutil.rmtree(target)
        if target.exists() or target.is_symlink():
            raise UnsafeCleanup(f"target still exists after deletion: {target}")
        if evidence != completion_evidence(root, relative):
            raise UnsafeCleanup("retained evidence changed; inspect the receipt")
        journal(stream, {"status": "DELETED", "target": str(target), "receipt": str(receipt)})
        print(f"DELETED {target}", flush=True)
        return "DELETED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/media/vlm-ckp-fileset/ylong"))
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root, report = args.root.resolve(), args.report_dir.resolve()
    if not root.is_dir():
        parser.error(f"root does not exist: {root}")
    if report == root or root in report.parents or report in root.parents:
        parser.error("report-dir must be outside the experimental data root")
    report.mkdir(parents=True, exist_ok=False)
    failures = 0
    print(f"mode={'APPLY' if args.apply else 'DRY_RUN'} report={report}", flush=True)
    print("Scope: five completed Physics model_states only. No Math or other projects.", flush=True)
    with (report / "deletions.jsonl").open("x") as stream:
        for relative in ALLOWED_RUNS:
            try:
                clean_one(root, relative, report, stream, args.apply)
            except (OSError, ValueError, UnsafeCleanup, subprocess.SubprocessError) as exc:
                failures += 1
                journal(stream, {"status": "FAILED_OR_SKIPPED", "run": relative,
                                 "error": str(exc)})
                print(f"FAILED_OR_SKIPPED {relative}: {exc}", flush=True)
    print(f"finished failures_or_skips={failures}; receipts={report}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
