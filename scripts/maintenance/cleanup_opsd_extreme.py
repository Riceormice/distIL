#!/usr/bin/env python3
"""Project-only slimming: retire historical weights and compact Math originals.

Current incomplete runs retain their newest complete native/trainer resume state.
Physics raw logits/generations, other projects, assets and runtimes are excluded.
Run while custom jobs using historical paths are stopped; standard pipeline and
nightly locks are respected. No local PID test can prove remote inactivity.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time

import audit_opsd_checkpoint_retention as audit
import cleanup_reviewed_math_weights as guarded
from cleanup_verified_physics_weights import UnsafeCleanup, journal
import compact_math_evaluations as math_json


BASE = audit.scheduled.BASE
MAX_JSON_BYTES = 512 * 1024**2
NEW_SWEEP = "sr_opsd_math_alpha_rho_sweep_legacy_allprompts_20260825"


def read_stable(root: Path, path: Path) -> tuple[bytes, list]:
    guarded.safe_path(root, path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JSON_BYTES:
        raise UnsafeCleanup(f"not a regular JSON file <=512 MiB: {path}")
    if time.time() - max(info.st_mtime, info.st_ctime) < guarded.QUIET_SECONDS:
        raise UnsafeCleanup(f"recently written: {path}")
    before = file_stamp(info)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as source:
        raw = source.read(MAX_JSON_BYTES + 1)
        if file_stamp(os.fstat(source.fileno())) != before:
            raise UnsafeCleanup(f"file changed while reading: {path}")
    if file_stamp(path.lstat()) != before or len(raw) != info.st_size:
        raise UnsafeCleanup(f"file changed while reading: {path}")
    return raw, before


def file_stamp(info) -> list:
    return [info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def nonempty(root: Path, path: Path) -> bool:
    guarded.safe_path(root, path)
    return path.is_file() and path.stat().st_size > 0


def native_ready(root: Path, checkpoint: Path, *, resume=True) -> bool:
    try:
        config, _ = read_stable(root, checkpoint / "actor/fsdp_config.json")
        world = json.loads(config).get("world_size")
        if type(world) is not int or not 1 <= world <= 64:
            return False
        hf = checkpoint / "actor/huggingface"
        required = [hf / "config.json", hf / "tokenizer_config.json"]
        if not (nonempty(root, hf / "tokenizer.json") or
                all(nonempty(root, hf / name) for name in ("vocab.json", "merges.txt"))):
            return False
        for kind in (("model", "optim", "extra_state") if resume else ("model",)):
            required.extend(checkpoint / f"actor/{kind}_world_size_{world}_rank_{r}.pt"
                            for r in range(world))
        if resume:
            required.append(checkpoint / "data.pt")
            tracker, _ = read_stable(root, checkpoint.parent / "latest_checkpointed_iteration.txt")
            step = audit.STEP.fullmatch(checkpoint.name)
            if step is None or int(tracker.strip()) != int(step[1]):
                return False
            if checkpoint.parent.name.startswith("sr-opsd-"):
                teacher = checkpoint / "actor/ema_teacher"
                teacher_config, _ = read_stable(root, teacher / "fsdp_config.json")
                if json.loads(teacher_config).get("world_size") != world:
                    return False
                required.extend(teacher / f"model_world_size_{world}_rank_{r}.pt" for r in range(world))
        return all(nonempty(root, p) for p in required)
    except (OSError, ValueError, UnsafeCleanup):
        return False


def trainer_ready(root: Path, checkpoint: Path, step: int) -> bool:
    try:
        raw, _ = read_stable(root, checkpoint / "trainer_state.json")
        if json.loads(raw).get("global_step") != step:
            return False
        weights = any(nonempty(root, checkpoint / name)
                      for name in ("adapter_model.safetensors", "model.safetensors", "pytorch_model.bin"))
        # The registered TRL jobs use eight ranks and DeepSpeed ZeRO-2. Do not
        # accept a partially saved checkpoint merely because one shard exists.
        tag, _ = read_stable(root, checkpoint / "latest")
        if tag.decode().strip() != f"global_step{step}":
            return False
        ds = checkpoint / f"global_step{step}"
        guarded.safe_path(root, ds)
        shards = list(ds.glob("*optim_states.pt"))
        ranks = []
        for shard in shards:
            match = re.fullmatch(r"(?:bf16_)?zero_pp_rank_(\d+)_mp_rank_00_optim_states\.pt", shard.name)
            if match is None or not nonempty(root, shard):
                return False
            ranks.append(int(match[1]))
        rng = all(nonempty(root, checkpoint / f"rng_state_{rank}.pth") for rank in range(8))
        return weights and sorted(ranks) == list(range(8)) and rng and nonempty(root, ds / "mp_rank_00_model_states.pt")
    except (OSError, ValueError, UnsafeCleanup):
        return False


def checkpoint_evaluated(root: Path, run: Path, step: int, samples=16) -> bool:
    try:
        for dataset in audit.DATASETS:
            raw, _ = read_stable(root, run / "evaluations" / f"checkpoint-{step}" / f"{dataset}.json")
            data = math_json.load(raw)
            math_json.validate(data, dataset)
            if data["val_n"] != samples:
                return False
        return True
    except (OSError, ValueError, UnsafeCleanup):
        return False


def run_complete(root: Path, run: Path, job) -> bool:
    return (job is not None and nonempty_marker(root, run / "state/complete")
            and all(checkpoint_evaluated(root, run, step) for step in range(5, job.total_steps + 1, 5)))


def nonempty_marker(root: Path, path: Path) -> bool:
    guarded.safe_path(root, path)
    return path.is_file()


def mapped_job(root: Path, run: Path):
    matches = [job for job, parent in audit.current_jobs(root)
               if job.kind == "math" and run.parent == parent
               and fnmatch.fnmatchcase(run.name, job.pattern)]
    return matches[0] if len(matches) == 1 else None


def weight_policy(root: Path, run: Path, rows: list[dict], scope: str, job) -> list[tuple[dict, str]]:
    if scope == "historical":
        # The old chunked sweep is a separate trajectory, not a fallback of the
        # new run. Retire it only once a current counterpart has usable state.
        if run.relative_to(root).parts[0] == guarded.OLD_SWEEP:
            current = root / NEW_SWEEP / "sr_opsd" / run.name
            current_job = mapped_job(root, current)
            candidates = list((current / "native/checkpoints" / current.name).glob("global_step_*"))
            candidates = [p for p in candidates if audit.STEP.fullmatch(p.name)]
            newest = max(candidates, key=lambda p: int(audit.STEP.fullmatch(p.name)[1]), default=None)
            if not run_complete(root, current, current_job) and not (
                newest is not None and native_ready(root, newest)
            ):
                return [(row, "KEEP_OLD_NO_CURRENT_RESUME") for row in rows]
        return [(row, "DELETE_RETIRED_HISTORICAL") for row in rows]
    if job is None:
        return [(row, "KEEP_UNMAPPED") for row in rows]
    if run_complete(root, run, job):
        return [(row, "DELETE_COMPLETED_WEIGHTS") for row in rows]
    native = [row for row in rows if row["kind"] in {"native", "trainer"}]
    latest = max(native, key=lambda row: row["step"], default=None)
    latest_ready = latest is not None and (
        native_ready(root, Path(latest["path"])) if latest["kind"] == "native"
        else trainer_ready(root, Path(latest["path"]), latest["step"])
    )
    results = []
    for row in rows:
        policy = "KEEP_UNCLASSIFIED"
        if row["kind"] in {"native", "trainer"}:
            if latest and row["step"] == latest["step"]:
                policy = "KEEP_LATEST_RESUME"
            elif latest_ready and checkpoint_evaluated(root, run, row["step"]):
                policy = "DELETE_OLDER_EVALUATED_CHECKPOINT"
            else:
                policy = "KEEP_UNVERIFIED_RESUME_OR_PENDING_EVAL"
        elif row["kind"] == "merged":
            native_step = next((r for r in native if r["kind"] == "native" and r["step"] == row["step"]), None)
            if (native_step and native_ready(root, Path(native_step["path"]), resume=False)):
                policy = "DELETE_REBUILDABLE_EXPORT"
            elif checkpoint_evaluated(root, run, row["step"]):
                policy = "DELETE_EVALUATED_EXPORT"
            else:
                policy = "KEEP_ONLY_PENDING_EVAL_COPY"
        results.append((row, policy))
    return results


def delete_tree(root, target, run, report, stream, apply, reason):
    before = guarded.inventory(root, target, weights=True)
    candidate = guarded.Candidate(str(target.relative_to(root)), str(run.relative_to(root)), reason)
    receipt = guarded.save_receipt(report, candidate, target, before, {}, {})
    allocated = sum({(v[0], v[1]): v[6] for v in before.values()}.values()) * 512
    if apply:
        if guarded.inventory(root, target, weights=True) != before:
            raise UnsafeCleanup(f"target changed: {target}")
        if not shutil.rmtree.avoids_symlink_attacks:
            raise UnsafeCleanup("fd-based rmtree unavailable")
        journal(stream, {"status": "DELETE_STARTED", "path": str(target), "receipt": str(receipt)})
        shutil.rmtree(target)
    status = "DELETED" if apply else "WOULD_DELETE"
    journal(stream, {"status": status, "path": str(target), "reason": reason, "allocated_bytes": allocated})
    print(f"{status} {allocated / 2**30:.2f} GiB {target}", flush=True)
    return allocated


def compact_file(root, path, report, stream, apply):
    raw, before = read_stable(root, path)
    digest = hashlib.sha256(raw).hexdigest()
    original = math_json.load(raw)
    compacted, count = math_json.compact(original, path.stem, digest)
    if not count:
        return 0
    key = hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest()[:20]
    temp = None
    try:
        if apply:
            fd, name = tempfile.mkstemp(prefix=path.name + ".slim.", suffix=".part", dir=path.parent)
            temp = Path(name)
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                os.fchmod(out.fileno(), stat.S_IMODE(before[2]))
                json.dump(compacted, out, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                out.write("\n")
                out.flush()
                os.fsync(out.fileno())
            after_raw = temp.read_bytes()
            if math_json.load(after_raw) != compacted:
                raise UnsafeCleanup(f"compacted JSON verification failed: {path}")
            after_size = len(after_raw)
            if after_size >= len(raw):
                return 0
            receipt = {"path": str(path), "before_sha256": digest,
                       "after_sha256": hashlib.sha256(after_raw).hexdigest(),
                       "before_bytes": len(raw), "after_bytes": after_size,
                       "removed_text_fields": count, "raw_text_backed_up": False}
            with (report / f"compact_{key}.json").open("x") as out:
                json.dump(receipt, out, indent=2)
                out.flush()
                os.fsync(out.fileno())
            if file_stamp(path.lstat()) != before:
                raise UnsafeCleanup(f"JSON changed before replacement: {path}")
            os.replace(temp, path)
            temp = None
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            status = "COMPACTED"
        else:
            after_size = len(json.dumps(compacted, ensure_ascii=False, separators=(",", ":")).encode()) + 1
            status = "WOULD_COMPACT"
        saved = max(0, len(raw) - after_size)
        journal(stream, {"status": status, "path": str(path), "logical_bytes_reduced": saved,
                         "removed_text_fields": count, "source_sha256": digest})
        print(f"{status} {saved / 2**20:.1f} MiB {path}", flush=True)
        return saved
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def evaluation_files(experiment: Path):
    # Known flat, per-run and protocol-comparison layouts; do not search weights.
    patterns = ("*/evaluations/checkpoint-*/*.json", "*/*/evaluations/checkpoint-*/*.json",
                "evaluations/*/checkpoint-*/*.json", "evaluations/checkpoint-*/*.json")
    return sorted({p for pattern in patterns for p in experiment.glob(pattern) if p.stem in audit.DATASETS})


def execute(root: Path, report: Path, apply=False, retire_historical=False) -> dict:
    root, report = root.absolute(), report.absolute()
    guarded.safe_path(root, root)
    if report.resolve() == root.resolve() or root.resolve() in report.resolve().parents:
        raise UnsafeCleanup("report directory must be outside experiment storage")
    report.mkdir(parents=True, exist_ok=False)
    totals = {"deleted_trees": 0, "compacted_files": 0, "selected_weight_bytes": 0,
              "text_bytes_reduced": 0, "failures_or_skips": 0}
    with ExitStack() as stack, (report / "events.jsonl").open("x") as stream:
        guarded.take_lock(root, root / ".opsd_extreme_cleanup.lock", stack, create=True)
        for experiment_name, scope in audit.scopes(root).items():
            if scope == "physics":
                print(f"KEEP_PHYSICS_RAW {root / experiment_name}", flush=True)
                continue
            experiment = guarded.safe_path(root, root / experiment_name)
            if not experiment.is_dir():
                continue
            rows, _, errors = audit.discover(experiment, scope)
            if errors:
                raise UnsafeCleanup("; ".join(errors))
            grouped = {}
            for row in rows:
                owner = Path(row["run"]) if row["run"] else experiment
                grouped.setdefault(owner, {"rows": [], "files": []})["rows"].append(row)
            for path in evaluation_files(experiment):
                guarded.safe_path(root, path)
                owner = audit.run_ancestor(path, experiment) or experiment
                grouped.setdefault(owner, {"rows": [], "files": []})["files"].append(path)
            for run, group in grouped.items():
                try:
                    with ExitStack() as run_stack:
                        job = mapped_job(root, run)
                        if scope == "current" and job is None:
                            print(f"KEEP_UNMAPPED_RUN {run}", flush=True)
                            continue
                        if job:
                            state = guarded.safe_path(root, root / "nightly_experiment_state" / job.job_id)
                            state.mkdir(parents=True, exist_ok=True)
                            guarded.take_lock(root, state / "nightly.lock", run_stack, create=True)
                        if (run / "state").is_dir():
                            guarded.take_lock(root, run / "state/pipeline.lock", run_stack, create=True)
                        guarded.take_lock(root, run / ".extreme_cleanup.lock", run_stack, create=True)
                        # Detect active old launchers that predate pipeline locks.
                        guarded.evidence_snapshot(root, run)
                        decisions = weight_policy(root, run, group["rows"], scope, job)
                        for row, policy in decisions:
                            path = Path(row["path"])
                            if policy.startswith("DELETE_") and (scope != "historical" or retire_historical):
                                size = delete_tree(root, path, run, report, stream, apply, policy)
                                totals["selected_weight_bytes"] += size
                                totals["deleted_trees"] += bool(apply)
                            else:
                                journal(stream, {"status": "KEPT", "path": str(path), "reason": policy})
                                print(f"{policy} {path}", flush=True)
                        for path in group["files"]:
                            try:
                                size = compact_file(root, path, report, stream, apply)
                                totals["text_bytes_reduced"] += size
                                totals["compacted_files"] += bool(apply and size)
                            except (OSError, ValueError, UnsafeCleanup) as exc:
                                totals["failures_or_skips"] += 1
                                journal(stream, {"status": "SKIPPED_JSON", "path": str(path), "error": str(exc)})
                                print(f"SKIPPED_JSON {path}: {exc}", flush=True)
                except (OSError, ValueError, UnsafeCleanup) as exc:
                    totals["failures_or_skips"] += 1
                    journal(stream, {"status": "SKIPPED_RUN", "path": str(run), "error": str(exc)})
                    print(f"SKIPPED_RUN {run}: {exc}", flush=True)
        with (report / "summary.json").open("x") as out:
            json.dump(totals, out, indent=2)
        print("FINISHED " + json.dumps(totals), flush=True)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BASE)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--retire-historical", action="store_true",
                        help="abandon all listed historical weight trajectories, including ahead old sweeps")
    args = parser.parse_args()
    try:
        totals = execute(args.root, args.report_dir, args.apply, args.retire_historical)
    except (OSError, ValueError, UnsafeCleanup) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    return int(bool(totals["failures_or_skips"]))


if __name__ == "__main__":
    raise SystemExit(main())
