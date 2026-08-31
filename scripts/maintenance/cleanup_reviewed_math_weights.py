#!/usr/bin/env python3
"""Retire only the 24 historical weight paths reviewed on 2026-08-31.

This is NOT completion-based garbage collection. Applying explicitly abandons
these old training trajectories, including unfinished historical evaluations.
Current/nightly jobs, two ahead historical sweeps, and all evidence are excluded.
Stop custom jobs using the retired runs before applying: not all old launchers
take a pipeline lock, and no local process check proves remote inactivity.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

from audit_opsd_checkpoint_retention import PROTECTED, STEP, scopes
from cleanup_verified_physics_weights import UnsafeCleanup, journal


BASE = Path("/media/vlm-ckp-fileset/ylong")
OLD_SWEEP = "sr_opsd_math_alpha_rho_sweep_20260819"
NEW_SWEEP = "sr_opsd_math_alpha_rho_sweep_legacy_allprompts_20260825"
QUIET_SECONDS = 300


@dataclass(frozen=True)
class Candidate:
    path: str
    owner: str
    reason: str
    current_run: str = ""
    minimum_step: int = 0


def sweep_name(alpha: str, rho: str) -> str:
    return (
        f"sr-opsd-8b-seed0-native-verl-forward-renyi-rho{rho}-refw{alpha}"
        "-sync0-ema0.05-lr5e-6-trainbs8-mbs8-rolloutn8-topk100-tailFalse"
        "-clip0.05-temp0.7-tok16384-steps100-sched420-eval5-n16-h200"
    )


def reviewed_candidates() -> tuple[Candidate, ...]:
    result = []

    def native(root, method, name, step, reason, *, current_run="", minimum_step=0, merged=True):
        owner = f"{root}/{method}/{name}"
        paths = [f"{owner}/native/checkpoints/{name}/global_step_{step}"]
        if merged:
            paths.append(f"{owner}/merged/checkpoint-{step}")
        result.extend(Candidate(p, owner, reason, current_run, minimum_step) for p in paths)

    for alpha, rho, old_step, current_step, merged in (
        ("0.7", "0.7", 50, 55, True),
        ("0.9", "0.9", 45, 65, False),
        ("0.7", "0.95", 60, 65, True),
    ):
        name = sweep_name(alpha, rho)
        native(OLD_SWEEP, "sr_opsd", name, old_step, "retire old chunked-evaluation trajectory",
               current_run=f"{NEW_SWEEP}/sr_opsd/{name}", minimum_step=current_step, merged=merged)

    root = "sr_opsd_math_protocol_compare_20260811"
    name = "sr-opsd-8b-seed0-github-original-rho0.95-refw0.9-sync0-lr5e-6-tok8192-steps30-eval5"
    for step in range(5, 31, 5):
        result.append(Candidate(f"{root}/checkpoints/{name}/global_step_{step}", root,
                                "retire old protocol-comparison training"))
    result.append(Candidate(f"{root}/merged/{name}/checkpoint-5", root,
                            "retire old protocol-comparison export"))

    root = "math_train_eval5_n16_h200_20260812"
    name = "sr-opsd-8b-seed0-forward-renyi-rho0.95-refw0.9-sync0-lr5e-6-tok16384-steps100-sched420-eval5-n16-h200"
    native(root, "sr_opsd", name, 5, "retire obsolete non-native-named SR-OPSD run")
    for size, hardware, original_root in (
        ("8b", "h200", root),
        ("4b", "a800", "math_4b_train_eval5_n16_a800_20260812"),
    ):
        name = (f"grpo-{size}-seed0-native-verl-lr5e-6-trainbs8-mbs8-rolloutn8"
                f"-eps0.2-temp0.7-tok16384-steps100-sched420-eval5-n16-{hardware}")
        native(original_root, "grpo", name, 10, "retire superseded native VERL GRPO")
        native(f"math_grpo_{size}_native_verl_eval5_n16_{hardware}_20260827", "grpo", name, 5,
               "retire failed native VERL GRPO; current GRPO uses OPSD/TRL")

    for root, name, step in (
        ("math_train_eval5_n16_h200_20260812",
         "opsd-8b-seed0-lr5e-6-bs1-ga2-steps100-sched420-beta0-clip0.06-topk100-temp0.7-tok16384-eval5-n16-h200", 5),
        ("math_opsd_grouped8x8_eval5_n16_h200_20260820",
         "opsd-8b-seed0-grouped-q8-r8-lr5e-6-steps100-sched420-beta0-clip0.06-topk100-temp0.7-tok16384-eval5-n16-h200", 10),
    ):
        owner = f"{root}/opsd/{name}"
        result.append(Candidate(f"{owner}/checkpoints/{name}/checkpoint-{step}", owner,
                                "retire old OPSD run; current grouped runs are excluded"))
    return tuple(result)


CANDIDATES = reviewed_candidates()
HELD_RUNS = tuple(f"{OLD_SWEEP}/sr_opsd/{sweep_name(a, r)}"
                  for a, r in (("0.9", "0.7"), ("0.7", "0.9")))


def safe_path(root: Path, path: Path) -> Path:
    if not root.is_absolute() or root != Path(os.path.abspath(root)) or root.is_symlink():
        raise UnsafeCleanup("root must be an absolute, normalized, non-symlink directory")
    relative = path.relative_to(root)
    if ".." in relative.parts:
        raise UnsafeCleanup("parent traversal in path")
    device = root.stat().st_dev
    cursor = root
    for part in relative.parts:
        cursor /= part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or info.st_dev != device or os.path.ismount(cursor):
            raise UnsafeCleanup(f"symlink or nested mount: {cursor}")
    return path


def inventory(root: Path, target: Path, *, weights: bool) -> dict:
    safe_path(root, target)
    if not target.is_dir():
        raise UnsafeCleanup(f"not an ordinary directory: {target}")
    result = {}

    def onerror(exc):
        raise exc

    for parent, dirs, files in os.walk(target, followlinks=False, onerror=onerror):
        for path in [Path(parent), *(Path(parent) / n for n in dirs + files)]:
            safe_path(root, path)
            info = path.lstat()
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise UnsafeCleanup(f"special file: {path}")
            if weights and (path.name in PROTECTED or path.name.startswith("wandb")):
                raise UnsafeCleanup(f"evidence/runtime found inside weights: {path}")
            if time.time() - max(info.st_mtime, info.st_ctime) < QUIET_SECONDS:
                raise UnsafeCleanup(f"recently changed (last {QUIET_SECONDS}s): {path}")
            result[str(path.relative_to(target))] = [
                info.st_dev, info.st_ino, info.st_mode, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns, info.st_blocks,
            ]
    if weights and not any(Path(p).suffix in (".pt", ".bin", ".safetensors") for p in result):
        raise UnsafeCleanup(f"no recognized tensor files: {target}")
    return result


def ensure_current_resume(root: Path, candidate: Candidate) -> dict:
    if not candidate.current_run:
        return {}
    run = safe_path(root, root / candidate.current_run)
    parent = safe_path(root, run / "native/checkpoints" / run.name)
    matches = [p for p in parent.glob("global_step_*")
               if STEP.fullmatch(p.name) and int(STEP.fullmatch(p.name)[1]) >= candidate.minimum_step]
    if not matches:
        raise UnsafeCleanup(f"current checkpoint >= {candidate.minimum_step} not found: {run}")
    checkpoint = max(matches, key=lambda p: int(STEP.fullmatch(p.name)[1]))
    required = [checkpoint / "data.pt"]
    for group in ("model", "optim", "extra_state"):
        required.extend(checkpoint / f"actor/{group}_world_size_8_rank_{rank}.pt" for rank in range(8))
    for path in required:
        safe_path(root, path)
        if not path.is_file() or path.stat().st_size == 0:
            raise UnsafeCleanup(f"current resume structure is incomplete: {path}")
    return {"checkpoint": str(checkpoint), "required_files": len(required),
            "note": "structural check only; tensor restore not tested; trajectories are different"}


def ensure_reviewed_scope(root: Path, candidate: Candidate) -> None:
    if candidate not in CANDIDATES:
        raise UnsafeCleanup("not one of the 24 reviewed paths")
    if scopes(root).get(Path(candidate.path).parts[0]) != "historical":
        raise UnsafeCleanup("root is not historical in the current launcher registry")
    allowed = {root / c.path for c in CANDIDATES if c.owner == candidate.owner}
    for parent in {p.parent for p in allowed}:
        safe_path(root, parent)
        if not parent.exists():
            continue
        for child in parent.iterdir():
            if STEP.fullmatch(child.name) and child not in allowed:
                raise UnsafeCleanup(f"unreviewed checkpoint appeared in old run: {child}")


def take_lock(root: Path, path: Path, stack: ExitStack, *, create: bool) -> None:
    safe_path(root, path)
    if not create and not path.exists():
        return
    flags = os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0)
    fd = os.open(path, flags, 0o600)
    stream = stack.enter_context(os.fdopen(fd, "r+"))
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        raise UnsafeCleanup(f"not an ordinary lock: {path}")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise UnsafeCleanup(f"run/cleanup lock is held: {path}") from exc


def evidence_snapshot(root: Path, owner: Path) -> dict:
    result = {}
    for relative in ("evaluations", "logs", "native/logs"):
        path = safe_path(root, owner / relative)
        if path.exists():
            result[relative] = inventory(root, path, weights=False)
    return result


def save_receipt(report: Path, candidate: Candidate, target: Path, items: dict, evidence: dict,
                 resume: dict) -> Path:
    receipt = report / hashlib.sha256(candidate.path.encode()).hexdigest()[:16]
    receipt.mkdir()
    # Retain small checkpoint configs/state before removing the weight directory.
    for relative, info in items.items():
        if (stat.S_ISREG(info[2]) and Path(relative).suffix in (".json", ".yaml", ".yml", ".txt")
                and info[3] <= 4 * 1024**2):
            destination = receipt / "checkpoint_metadata" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(target / relative, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as source, destination.open("xb") as output:
                metadata = source.read(4 * 1024**2 + 1)
                if len(metadata) != info[3]:
                    raise UnsafeCleanup(f"checkpoint metadata changed: {target / relative}")
                output.write(metadata)
                output.flush()
                os.fsync(output.fileno())
    with (receipt / "inventory.json").open("x") as output:
        json.dump({"target": str(target), "reason": candidate.reason, "weights": items,
                   "retained_evidence": evidence, "current_resume": resume}, output, indent=2)
        output.flush()
        os.fsync(output.fileno())
    return receipt


def clean_one(root: Path, candidate: Candidate, report: Path, stream, apply: bool) -> tuple[str, int]:
    ensure_reviewed_scope(root, candidate)
    target = safe_path(root, root / candidate.path)
    if not target.exists():
        journal(stream, {"status": "ABSENT", "target": str(target)})
        print(f"ABSENT {target}", flush=True)
        return "ABSENT", 0
    with ExitStack() as stack:
        owner = root / candidate.owner
        state = safe_path(root, owner / "state")
        if state.is_dir():
            take_lock(root, state / "pipeline.lock", stack, create=True)
        take_lock(root, owner / ".reviewed_weights_cleanup.lock", stack, create=True)
        resume = ensure_current_resume(root, candidate)
        evidence = evidence_snapshot(root, owner)
        items = inventory(root, target, weights=True)
        unique = {(v[0], v[1]): v[6] for v in items.values()}
        allocated = sum(unique.values()) * 512
        receipt = save_receipt(report, candidate, target, items, evidence, resume)
        if not apply:
            status = "WOULD_DELETE"
        else:
            if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
                raise UnsafeCleanup("Python lacks fd-based safe rmtree")
            ensure_reviewed_scope(root, candidate)
            ensure_current_resume(root, candidate)
            if (inventory(root, target, weights=True) != items
                    or evidence_snapshot(root, owner) != evidence):
                raise UnsafeCleanup("weights or retained evidence changed during verification")
            journal(stream, {"status": "DELETE_STARTED", "target": str(target), "receipt": str(receipt)})
            shutil.rmtree(target)
            if evidence_snapshot(root, owner) != evidence:
                raise UnsafeCleanup("retained evidence changed; inspect concurrent old jobs")
            status = "DELETED"
        journal(stream, {"status": status, "target": str(target), "allocated_bytes_before": allocated,
                         "receipt": str(receipt)})
        print(f"{status} {allocated / 1024**3:.2f} GiB {target}", flush=True)
        return status, allocated


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BASE)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--retire-listed-historical-runs", action="store_true",
                        help="confirm these old trajectories are abandoned and no custom job uses them")
    args = parser.parse_args(argv)
    if args.apply and not args.retire_listed_historical_runs:
        parser.error("--apply requires --retire-listed-historical-runs; this is not completion-based cleanup")
    root = args.root.absolute()
    safe_path(root, root)
    report = args.report_dir.absolute()
    if report.resolve() == root.resolve() or root.resolve() in report.resolve().parents:
        parser.error("report must be outside the experiment storage root")
    report.mkdir(parents=True, exist_ok=False)
    failures = deleted = 0
    selected_bytes = 0
    print(f"mode={'APPLY' if args.apply else 'DRY_RUN'} candidates={len(CANDIDATES)} report={report}", flush=True)
    print("Only reviewed old weight paths. No current runs, Physics data, or other projects.", flush=True)
    for run in HELD_RUNS:
        print(f"KEEP_AHEAD_HISTORICAL {root / run}", flush=True)
    with ExitStack() as stack, (report / "events.jsonl").open("x") as stream:
        take_lock(root, root / ".reviewed_math_cleanup.lock", stack, create=True)
        for candidate in CANDIDATES:
            try:
                status, size = clean_one(root, candidate, report, stream, args.apply)
                selected_bytes += size
                deleted += status == "DELETED"
            except (OSError, ValueError, UnsafeCleanup) as exc:
                failures += 1
                journal(stream, {"status": "FAILED_OR_SKIPPED", "target": str(root / candidate.path),
                                 "error": str(exc)})
                print(f"FAILED_OR_SKIPPED {candidate.path}: {exc}", flush=True)
    print(f"finished deleted={deleted} failures_or_skips={failures} "
          f"selected_allocated_GiB={selected_bytes / 1024**3:.2f}", flush=True)
    print("Allocated sizes are not guaranteed reclaimed space (hardlinks/shared storage).", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
