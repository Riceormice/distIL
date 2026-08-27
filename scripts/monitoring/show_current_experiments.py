#!/usr/bin/env python3
"""Show progress and liveness signals for the current Math and Physics runs.

This is a read-only shared-filesystem monitor. A held pipeline lock is treated as
the strongest remote liveness signal. The legacy Physics launcher has no lock, so
its ACTIVE? state is inferred from a recently updated log and is intentionally
marked with a question mark.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    label: str
    parent: Path
    pattern: str
    total_steps: int
    kind: str = "math"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h200-root",
        type=Path,
        default=Path("/media/vlm-ckp-fileset/ylong/math_train_eval5_n16_h200_20260812"),
    )
    parser.add_argument(
        "--a800-root",
        type=Path,
        default=Path("/media/vlm-ckp-fileset/ylong/math_4b_train_eval5_n16_a800_20260812"),
    )
    parser.add_argument(
        "--grpo-8b-root",
        type=Path,
        default=Path(
            "/media/vlm-ckp-fileset/ylong/"
            "math_grpo_8b_opsd_trl_aligned_eval5_n16_h200_20260827"
        ),
    )
    parser.add_argument(
        "--grpo-4b-root",
        type=Path,
        default=Path(
            "/media/vlm-ckp-fileset/ylong/"
            "math_grpo_4b_opsd_trl_aligned_eval5_n16_a800_20260827"
        ),
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("/media/vlm-ckp-fileset/ylong/sr_opsd_math_alpha_rho_sweep_20260819"),
    )
    parser.add_argument(
        "--grouped-root",
        type=Path,
        default=Path("/media/vlm-ckp-fileset/ylong/math_opsd_grouped8x8_eval5_n16_h200_20260820"),
    )
    parser.add_argument(
        "--physics-root",
        type=Path,
        default=Path("/media/vlm-ckp-fileset/ylong/sdpo_physics_rho_selfref_grid_eval5_nockpt"),
    )
    parser.add_argument(
        "--recent-minutes",
        type=int,
        default=30,
        help="Log-age threshold used only for the legacy Physics ACTIVE? state.",
    )
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    hroot, aroot = args.h200_root, args.a800_root
    specs = [
        RunSpec(
            "Math-8B GRPO OPSD",
            args.grpo_8b_root / "grpo",
            "grpo-8b-seed0-opsd-trl-q8-r8-lr5e-6-eps0.2-lora64a128-"
            "temp0.7-tok16384-steps100-sched420-eval5-n16-h200",
            100,
        ),
        RunSpec(
            "Math-8B SDPO",
            hroot / "sdpo",
            "sdpo-8b-seed0-native-verl-rkl-ema0.05-lr5e-6-trainbs8-mbs8-"
            "rolloutn8-topk100-tailFalse-clip0.05-temp0.7-tok16384-steps100-"
            "sched420-eval5-n16-h200",
            100,
        ),
        RunSpec(
            "Math-8B OPSD",
            hroot / "opsd",
            "opsd-8b-seed0-lr5e-6-bs1-ga8-steps100-sched420-beta0-clip0.06-"
            "topk100-temp0.7-tok16384-eval5-n16-h200",
            100,
        ),
        RunSpec(
            "Math-8B SR-OPSD",
            hroot / "sr_opsd",
            "sr-opsd-8b-seed0-native-verl-forward-renyi-rho0.95-refw0.9-sync0-"
            "ema0.05-lr5e-6-trainbs8-mbs8-rolloutn8-topk100-tailFalse-clip0.05-"
            "temp0.7-tok16384-steps100-sched420-eval5-n16-h200",
            100,
        ),
        RunSpec(
            "Math-4B GRPO OPSD",
            args.grpo_4b_root / "grpo",
            "grpo-4b-seed0-opsd-trl-q8-r8-lr5e-6-eps0.2-lora64a128-"
            "temp0.7-tok16384-steps100-sched420-eval5-n16-a800",
            100,
        ),
        RunSpec(
            "Math-4B SDPO",
            aroot / "sdpo",
            "sdpo-4b-seed0-native-verl-rkl-ema0.05-lr5e-6-trainbs8-mbs8-"
            "rolloutn8-topk100-tailFalse-clip0.05-temp0.7-tok16384-steps100-"
            "sched420-eval5-n16-a800",
            100,
        ),
        RunSpec(
            "Math-4B OPSD",
            aroot / "opsd",
            "opsd-4b-seed0-lr5e-6-bs1-ga8-steps100-sched420-beta0-clip0.05-"
            "topk100-temp0.7-tok16384-eval5-n16-a800",
            100,
        ),
        RunSpec(
            "Math-4B SR-OPSD",
            aroot / "sr_opsd",
            "sr-opsd-4b-seed0-native-verl-forward-renyi-rho0.95-refw0.9-sync0-"
            "ema0.05-lr5e-6-trainbs8-mbs8-rolloutn8-topk100-tailFalse-clip0.05-"
            "temp0.7-tok16384-steps100-sched420-eval5-n16-a800",
            100,
        ),
        RunSpec(
            "Math-8B OPSD 8x8",
            args.grouped_root / "opsd",
            "opsd-8b-seed0-grouped-q8-r8-*-eval5-n16-h200",
            100,
        ),
    ]
    for alpha, rho in (
        ("0.9", "0.7"),
        ("0.9", "0.9"),
        ("0.7", "0.7"),
        ("0.7", "0.9"),
        ("0.7", "0.95"),
    ):
        specs.append(
            RunSpec(
                f"Math sweep a={alpha} r={rho}",
                args.sweep_root / "sr_opsd",
                f"*rho{rho}-refw{alpha}-*",
                100,
            )
        )
    specs.append(
        RunSpec(
            "Physics a=0.9 r=0.9",
            args.physics_root / "runs",
            "*rho0.9-selfref0.9-*",
            420,
            kind="physics",
        )
    )
    return specs


def resolve_run(spec: RunSpec) -> Path | None:
    if not spec.parent.is_dir():
        return None
    matches = [path for path in spec.parent.glob(spec.pattern) if path.is_dir()]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def lock_is_held(path: Path) -> bool:
    if not path.is_file():
        return False
    result = subprocess.run(
        ["flock", "-n", str(path), "-c", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode != 0


def log_files(run: Path, spec: RunSpec, physics_root: Path) -> list[Path]:
    paths: list[Path] = []
    if spec.kind == "physics":
        log_dir = physics_root / "logs" / run.name
        paths.extend(log_dir.glob("*.log"))
    else:
        paths.extend((run / "logs").glob("pipeline_*.log"))
        paths.extend((run / "logs").glob("*.log"))
    return list(dict.fromkeys(paths))


def latest_log(run: Path, spec: RunSpec, physics_root: Path) -> Path | None:
    paths = log_files(run, spec, physics_root)
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def latest_activity(run: Path, spec: RunSpec, physics_root: Path) -> Path | None:
    paths = log_files(run, spec, physics_root)
    paths.extend(metric_paths(run, spec, physics_root))
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def metric_paths(run: Path, spec: RunSpec, physics_root: Path) -> list[Path]:
    if spec.kind == "physics":
        path = physics_root / "logs" / run.name / "metrics.jsonl"
        return [path] if path.is_file() else []
    paths = list((run / "native" / "logs").glob("*/metrics.jsonl"))
    direct = run / "training_metrics.jsonl"
    if direct.is_file():
        paths.append(direct)
    return paths


def max_training_step(run: Path, spec: RunSpec, physics_root: Path) -> int:
    best = 0
    step_pattern = re.compile(r'"step"\s*:\s*(\d+)')
    for path in metric_paths(run, spec, physics_root):
        try:
            with path.open(errors="ignore") as handle:
                for line in handle:
                    for value in step_pattern.findall(line):
                        best = max(best, int(value))
        except OSError:
            continue
    for pattern in (
        "native/checkpoints/*/global_step_*",
        "checkpoints/*/checkpoint-*",
    ):
        for path in run.glob(pattern):
            match = re.search(r"(?:global_step_|checkpoint-)(\d+)$", path.name)
            if match:
                best = max(best, int(match.group(1)))
    return best


def evaluation_progress(
    run: Path, spec: RunSpec, physics_root: Path
) -> tuple[int, int, int]:
    expected = spec.total_steps // 5
    if spec.kind == "physics":
        validation_dir = physics_root / "logs" / run.name / "validation"
        steps = [
            int(path.stem)
            for path in validation_dir.glob("*.jsonl")
            if path.stem.isdigit() and path.stat().st_size > 0
        ] if validation_dir.is_dir() else []
        return len(steps), expected, max(steps, default=0)

    complete_steps: list[int] = []
    evaluation_dir = run / "evaluations"
    if evaluation_dir.is_dir():
        for checkpoint_dir in evaluation_dir.glob("checkpoint-*"):
            match = re.search(r"checkpoint-(\d+)$", checkpoint_dir.name)
            if match and len(list(checkpoint_dir.glob("*.json"))) >= 5:
                complete_steps.append(int(match.group(1)))
    return len(complete_steps), expected, max(complete_steps, default=0)


def format_age(log: Path | None) -> tuple[str, int]:
    if log is None:
        return "--", 10**9
    minutes = max(0, int((time.time() - log.stat().st_mtime) / 60))
    if minutes < 120:
        return f"{minutes}m", minutes
    if minutes < 2880:
        return f"{minutes // 60}h", minutes
    return f"{minutes // 1440}d", minutes


def last_fatal(log: Path | None) -> str:
    if log is None:
        return ""
    try:
        output = subprocess.run(
            ["tail", "-n", "500", str(log)],
            capture_output=True,
            text=True,
            errors="ignore",
            check=False,
        ).stdout
    except OSError:
        return ""
    pattern = re.compile(
        r"ERROR:|Traceback|RuntimeError|ChildFailedError|CUDA out of memory|killed by signal",
        re.IGNORECASE,
    )
    hits = [line.strip() for line in output.splitlines() if pattern.search(line)]
    return hits[-1][:105] if hits else ""


def main() -> None:
    args = parse_args()
    rows = []
    for spec in build_specs(args):
        run = resolve_run(spec)
        if run is None:
            rows.append(
                (spec.label, "NOT_STARTED", f"0/{spec.total_steps}",
                 f"0/{spec.total_steps // 5}", "--", "", None)
            )
            continue

        log = latest_log(run, spec, args.physics_root)
        activity = latest_activity(run, spec, args.physics_root)
        age, age_minutes = format_age(activity)
        complete = (
            (run / "TRAINING_COMPLETE").is_file()
            if spec.kind == "physics"
            else (run / "state" / "complete").is_file()
        )
        held = (
            lock_is_held(run / "state" / "pipeline.lock")
            if spec.kind != "physics"
            else False
        )
        if complete:
            status = "COMPLETE"
        elif held:
            status = "RUNNING"
        elif spec.kind == "physics" and age_minutes <= args.recent_minutes:
            status = "ACTIVE?"
        else:
            status = "STOPPED"

        step = max_training_step(run, spec, args.physics_root)
        done, expected, latest_eval = evaluation_progress(run, spec, args.physics_root)
        rows.append(
            (
                spec.label,
                status,
                f"{step}/{spec.total_steps}",
                f"{done}/{expected}@{latest_eval}",
                age,
                last_fatal(log),
                log,
            )
        )

    print(
        f"{'EXPERIMENT':<29} {'STATUS':<11} {'TRAIN':<9} "
        f"{'EVAL':<12} {'LOG_AGE':<8} LAST_FATAL"
    )
    print("-" * 125)
    for label, status, train, evaluation, age, error, _ in rows:
        print(
            f"{label:<29} {status:<11} {train:<9} "
            f"{evaluation:<12} {age:<8} {error or '--'}"
        )

    print("\nLatest logs for unfinished runs:")
    for label, status, _, _, _, _, log in rows:
        if status not in {"COMPLETE", "NOT_STARTED"}:
            print(f"[{status}] {label}: {log or 'NO LOG'}")

    print("\nStatus semantics:")
    print("  RUNNING  = the remote pipeline still holds its shared lock")
    print("  ACTIVE?  = legacy Physics has no lock; its log changed recently")
    print("  STOPPED  = output exists, but no completion marker or held lock")
    print("  COMPLETE = the launcher's final completion marker exists")


if __name__ == "__main__":
    main()
