#!/usr/bin/env python3
"""Read-only, allowlisted OPSD/SDPO storage and checkpoint retention report.

REVIEW is a candidate for human review, never permission to delete. Current
nightly jobs are taken from their registry, not inferred from a local PID.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import fnmatch
import json
import os
import re
import stat
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nightly"))
import show_current_progress as scheduled  # noqa: E402


HISTORICAL_ROOTS = (
    "sr_opsd_math_alpha_rho_sweep_20260819",
    "sr_opsd_math_protocol_compare_20260811",
    "math_train_eval5_n16_h200_20260812",
    "math_4b_train_eval5_n16_a800_20260812",
    "math_grpo_8b_native_verl_eval5_n16_h200_20260827",
    "math_grpo_4b_native_verl_eval5_n16_a800_20260827",
    "math_opsd_grouped8x8_eval5_n16_h200_20260820",
    "sr_opsd_math_refw_sweep30",
)
PHYSICS_ROOTS = (
    "physics_p0_mechanism_20260825_v2",
    "physics_p0_sdpo_fkl_jsd_20260827",
    "sdpo_physics_rho_selfref_grid_eval5_nockpt",
)
WEIGHT_CONTAINERS = {"checkpoints", "merged", "model_states"}
PROTECTED = {
    "raw_logits_audit", "topk_probe", "token_stats", "train_token_stats",
    "generation", "generation_raw", "evaluation", "evaluations", "validation",
    "aggregate", "figures", "protocol", "logs", "state", "training",
    "data", "datasets", "models", "envs", "runtime", "runtime_assets",
    "runtime_overlays", ".git", "__pycache__",
}
STEP = re.compile(r"(?:global_step_|checkpoint-)(\d+)\Z")
DATASETS = ("aime24", "aime25", "hmmt25", "amc23", "minerva")


def current_jobs(root: Path) -> list[tuple[object, Path]]:
    return [(job, root / job.parent.relative_to(scheduled.BASE)) for job in scheduled.jobs()]


def scopes(root: Path) -> dict[str, str]:
    result = dict.fromkeys(HISTORICAL_ROOTS, "historical")
    result.update(dict.fromkeys(PHYSICS_ROOTS, "physics"))
    for job, parent in current_jobs(root):
        if job.kind == "math":
            result[parent.relative_to(root).parts[0]] = "current"
    return result


def disk_bytes(path: Path, timeout: float) -> tuple[int | None, str]:
    try:
        process = subprocess.run(
            ["du", "-skx", str(path)], capture_output=True, text=True, timeout=timeout,
        )
        if process.returncode:
            return None, process.stderr.strip() or f"du exit={process.returncode}"
        return int(process.stdout.split()[0]) * 1024, ""
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        return None, str(exc)


def lock_state(path: Path) -> str:
    if path.is_symlink():
        return "unknown"
    try:
        with path.open("rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(stream, fcntl.LOCK_UN)
        return "available"
    except FileNotFoundError:
        return "missing"
    except BlockingIOError:
        return "held"
    except OSError:
        return "unknown"


def run_ancestor(path: Path, experiment: Path) -> Path | None:
    for parent in path.parents:
        if parent == experiment:
            break
        if (parent / "state").is_dir() or (parent / "evaluations").is_dir():
            return parent
    return None


def small_json(path: Path) -> dict:
    try:
        if path.is_symlink() or path.stat().st_size > 4 * 1024**2:
            return {}
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, ValueError):
        return {}


def nonempty(path: Path) -> bool:
    try:
        return not path.is_symlink() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def resume_hint(row: dict) -> str:
    """Structural hints only: never load pickles or claim restore is tested."""
    path = Path(row["path"])
    if row["kind"] == "native":
        actor = path / "actor"
        if nonempty(path / "data.pt") and not actor.is_symlink() and actor.is_dir():
            return "actor+data.pt; shard/optimizer restore NOT verified"
    if row["kind"] == "trainer":
        state = small_json(path / "trainer_state.json")
        if state.get("global_step") == row["step"] and any(
            nonempty(path / name) for name in ("adapter_model.safetensors", "model.safetensors")
        ):
            return "trainer_state+weights; optimizer/RNG restore NOT verified"
    return "unknown"


def discover(experiment: Path, scope: str) -> tuple[list[dict], list[dict], list[str]]:
    weights, retained, errors = [], [], []

    def visit(parent: Path, container: str = "", depth: int = 0) -> None:
        try:
            entries = sorted(parent.iterdir())
        except OSError as exc:
            errors.append(f"{parent}: {exc}")
            return
        for path in entries:
            if path.is_symlink():
                retained.append({"path": str(path), "policy": "KEEP_SYMLINK_NOT_FOLLOWED"})
                continue
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                errors.append(f"{path}: {exc}")
                continue
            if not stat.S_ISDIR(mode):
                continue
            if os.path.ismount(path):
                retained.append({"path": str(path), "policy": "KEEP_NESTED_MOUNT"})
                continue
            if path.name in PROTECTED or path.name.startswith("wandb"):
                retained.append({"path": str(path), "policy": "KEEP_EVIDENCE_OR_RUNTIME"})
                continue
            match = STEP.fullmatch(path.name)
            if path.name == "model_states" or (container and match):
                kind = (
                    "physics_model_states" if path.name == "model_states" else
                    "merged" if container == "merged" else
                    "native" if path.name.startswith("global_step_") else "trainer"
                )
                run = run_ancestor(path, experiment)
                weights.append({
                    "experiment": experiment.name, "scope": scope,
                    "path": str(path), "run": str(run) if run else "",
                    "kind": kind, "step": int(match[1]) if match else None,
                })
                continue
            if depth >= 10:
                errors.append(f"depth limit, not inspected: {path}")
                continue
            visit(path, path.name if path.name in WEIGHT_CONTAINERS else container, depth + 1)

    visit(experiment)
    return weights, retained, errors


def eval_file_count(run: Path | None, step: int | None) -> int:
    if run is None or step is None:
        return 0
    directory = run / "evaluations" / f"checkpoint-{step}"
    return sum(nonempty(directory / f"{name}.json") for name in DATASETS)


def classify_rows(root: Path, rows: list[dict]) -> None:
    jobs = current_jobs(root)
    groups = defaultdict(list)
    for row in rows:
        if row["run"]:
            groups[row["run"]].append(row)
    for row in rows:
        run = Path(row["run"]) if row["run"] else None
        matches = [job for job, parent in jobs if run is not None
                   and run.parent == parent and fnmatch.fnmatchcase(run.name, job.pattern)]
        job_id = matches[0].job_id if len(matches) == 1 else ""
        locks = [lock_state(run / "state/pipeline.lock")] if run else []
        if job_id:
            locks.append(lock_state(root / "nightly_experiment_state" / job_id / "nightly.lock"))
        row.update({
            "job": job_id, "locks": ",".join(locks) or "unknown",
            "resume_hint": resume_hint(row),
            "eval_files_present_not_validated": eval_file_count(run, row["step"]),
        })
        native_rows = [item for item in groups[row["run"]]
                       if item["kind"] in {"native", "trainer"}]
        newest = max((item["step"] for item in native_rows), default=None)
        row["latest_train_checkpoint"] = newest
        if "held" in locks:
            policy, reason = "KEEP_BUSY", "Launcher lock held; no checkpoint cleanup while this job is running."
        elif "unknown" in locks:
            policy, reason = "KEEP_LOCK_UNKNOWN", "Cannot inspect launcher lock safely."
        elif row["kind"] == "physics_model_states":
            policy, reason = "REVIEW_PHYSICS_WEIGHTS", "Use cleanup_verified_physics_weights.py; it validates all retained evidence first."
        elif row["scope"] == "historical":
            policy, reason = "REVIEW_HISTORICAL_WEIGHTS", "Not in current default launchers; confirm no custom-output job or resume reference still uses it."
        elif not job_id or run is None:
            policy, reason = "KEEP_UNMAPPED", "No exact current job match; do not infer retirement."
        elif (run / "state/complete").is_file():
            policy, reason = "REVIEW_COMPLETED_RUN", "Completion marker found; verify all scheduled evaluation JSONs before deleting final weights."
        elif row["kind"] != "merged" and row["step"] == newest:
            policy, reason = "KEEP_CURRENT_LATEST", "Current/nightly run needs this checkpoint, including optimizer, RNG and data-loader state."
        elif row["kind"] != "merged" and not any(
            item["step"] == newest and resume_hint(item) != "unknown" for item in native_rows
        ):
            policy, reason = "KEEP_NEWER_UNVERIFIED", "Newest checkpoint lacks basic resume files; preserve older checkpoints until restore is verified."
        elif row["kind"] != "merged":
            policy, reason = "REVIEW_CURRENT_OLDER", "Delete only after newer checkpoint restore and this step's required evaluation are verified."
        elif row["eval_files_present_not_validated"] < 5:
            policy, reason = "KEEP_EVALUATION_PENDING", "Merged export may be needed by an incomplete five-dataset evaluation."
        else:
            policy, reason = "REVIEW_EVALUATED_EXPORT", "Five JSON files exist; validate them and confirm no evaluator still uses this export."
        row.update({"policy": policy, "reason": reason})


def write_tsv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["path", "policy"]
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=scheduled.BASE)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--du-timeout", type=float, default=120)
    args = parser.parse_args()
    root, report = args.root.resolve(), args.report_dir.resolve()
    if not root.is_dir() or args.du_timeout <= 0:
        parser.error("root must exist and du-timeout must be positive")
    if report == root or root in report.parents or report in root.parents:
        parser.error("report-dir must be outside the experiment data root")
    if report.exists():
        parser.error("report-dir must be new")
    report.mkdir(parents=True)
    weights, retained, errors, roots = [], [], [], []
    print("READ ONLY: OPSD/SDPO allowlist; no deletion or experiment-state changes.", flush=True)
    for name, scope in scopes(root).items():
        experiment = root / name
        if experiment.is_symlink():
            retained.append({"path": str(experiment), "policy": "KEEP_SYMLINK_NOT_FOLLOWED"})
            continue
        if not experiment.is_dir():
            roots.append({"path": str(experiment), "scope": scope, "status": "MISSING"})
            continue
        print(f"SCAN {scope}: {name}", flush=True)
        amount, error = disk_bytes(experiment, args.du_timeout)
        roots.append({"path": str(experiment), "scope": scope, "allocated_bytes": amount,
                      "status": "ERROR" if error else "SCANNED"})
        if error:
            errors.append(f"{experiment}: {error}")
        found, keep, failed = discover(experiment, scope)
        weights.extend(found)
        retained.extend(keep)
        errors.extend(failed)
    classify_rows(root, weights)
    for row in weights + retained:
        print(f"SIZE {row['policy']} {row['path']}", flush=True)
        amount, error = disk_bytes(Path(row["path"]), args.du_timeout)
        row["allocated_bytes"] = amount
        if error:
            errors.append(f"{row['path']}: {error}")
    weights.sort(key=lambda row: row["allocated_bytes"] or 0, reverse=True)
    write_tsv(report / "checkpoints.tsv", weights)
    write_tsv(report / "retained_data.tsv", retained)
    with (report / "audit.json").open("x") as stream:
        json.dump({"root": str(root), "roots": roots, "checkpoints": weights,
                   "retained_data": retained, "errors": errors}, stream, indent=2)
    lines = [
        "READ ONLY: REVIEW means inspect before deletion, not safe-to-delete.",
        "Paused/nightly jobs retain their latest checkpoint. Local PID absence is not evidence.",
        "Resume hints and evaluation file counts are NOT restore/metric validation.",
        "Sizes are allocated bytes, not guaranteed reclaimed space (hardlinks/shared storage).",
        "", "GiB       POLICY                         STEP   CHECKPOINT",
    ]
    for row in weights:
        amount = row["allocated_bytes"]
        gib = "unknown" if amount is None else f"{amount / 1024**3:.2f}"
        lines.append(f"{gib:>9} {row['policy']:<30} {str(row['step']):>5} {row['path']}")
        lines.append(f"          {row['reason']}")
    lines.extend(["", "LARGEST RETAINED DATA:"])
    for row in sorted(retained, key=lambda item: item["allocated_bytes"] or 0, reverse=True)[:15]:
        amount = row["allocated_bytes"]
        gib = "unknown" if amount is None else f"{amount / 1024**3:.2f}"
        lines.append(f"{gib:>9} GiB {row['policy']} {row['path']}")
    lines.append(f"\nscan_errors={len(errors)}; checkpoint_entries={len(weights)}; report={report}")
    lines.extend(errors)
    summary = "\n".join(lines) + "\n"
    (report / "summary.txt").write_text(summary)
    print(summary, flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
