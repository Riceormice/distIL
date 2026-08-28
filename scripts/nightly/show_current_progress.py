#!/usr/bin/env python3
"""Show progress for the currently scheduled resumable experiments."""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


BASE = Path("/media/vlm-ckp-fileset/ylong")
DATASETS = ("aime24", "aime25", "hmmt25", "amc23", "minerva")


@dataclass(frozen=True)
class Job:
    job_id: str
    label: str
    parent: Path
    pattern: str
    total_steps: int
    kind: str = "math"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nightly-state-root",
        type=Path,
        default=BASE / "nightly_experiment_state",
    )
    parser.add_argument("--recent-minutes", type=int, default=30)
    return parser.parse_args()


def jobs() -> list[Job]:
    sweep = BASE / "sr_opsd_math_alpha_rho_sweep_legacy_allprompts_20260825/sr_opsd"
    specs = [
        Job(
            "math_alpha070_rho070",
            "Math a=.7 r=.7",
            sweep,
            "*rho0.7-refw0.7-*",
            100,
        ),
        Job(
            "math_alpha070_rho090",
            "Math a=.7 r=.9",
            sweep,
            "*rho0.9-refw0.7-*",
            100,
        ),
        Job(
            "math_alpha070_rho095",
            "Math a=.7 r=.95",
            sweep,
            "*rho0.95-refw0.7-*",
            100,
        ),
        Job(
            "math_alpha090_rho070",
            "Math a=.9 r=.7",
            sweep,
            "*rho0.7-refw0.9-*",
            100,
        ),
        Job(
            "math_alpha090_rho090",
            "Math a=.9 r=.9",
            sweep,
            "*rho0.9-refw0.9-*",
            100,
        ),
        Job(
            "math_grpo_4b",
            "Math GRPO 4B",
            BASE / "math_grpo_4b_opsd_trl_aligned_eval5_n16_a800_20260827/grpo",
            "grpo-4b-*",
            100,
        ),
        Job(
            "math_grpo_8b",
            "Math GRPO 8B",
            BASE / "math_grpo_8b_opsd_trl_aligned_eval5_n16_h200_20260827/grpo",
            "grpo-8b-*",
            100,
        ),
        Job(
            "math_opsd8x8_4b",
            "Math OPSD 8x8 4B",
            BASE / "math_4b_opsd_grouped8x8_eval5_n16_a800_20260827/opsd",
            "opsd-4b-*",
            100,
        ),
        Job(
            "math_opsd8x8_8b",
            "Math OPSD 8x8 8B",
            BASE / "math_opsd_grouped8x8_eval5_n16_h200_legacy_allprompts_20260825/opsd",
            "opsd-8b-*",
            100,
        ),
    ]
    p0 = BASE / "physics_p0_sdpo_fkl_jsd_20260827/Qwen3-8B"
    specs.extend(
        [
            Job(
                "physics_logits_sdpo_fkl",
                "Physics logits FKL",
                p0 / "sdpo_fkl",
                "seed0",
                420,
                kind="p0",
            ),
            Job(
                "physics_logits_sdpo_jsd",
                "Physics logits JSD",
                p0 / "sdpo_jsd",
                "seed0",
                420,
                kind="p0",
            ),
        ]
    )
    return specs


def resolve_run(job: Job) -> Path | None:
    if not job.parent.is_dir():
        return None
    matches = [path for path in job.parent.glob(job.pattern) if path.is_dir()]
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


def candidate_logs(run: Path | None, job: Job, nightly_root: Path) -> list[Path]:
    paths: list[Path] = []
    if run is not None:
        if job.kind == "p0":
            launcher = job.parent.parents[1] / "launcher_logs"
            method = job.parent.name
            paths.extend(launcher.glob(f"{method}_seed0_*.log"))
        else:
            paths.extend((run / "logs").glob("*.log"))
    pointer = nightly_root / job.job_id / "latest_log"
    if pointer.is_file():
        try:
            path = Path(pointer.read_text(encoding="utf-8").strip())
            if path.is_file():
                paths.append(path)
        except OSError:
            pass
    return list(dict.fromkeys(path for path in paths if path.is_file()))


def latest_log(run: Path | None, job: Job, nightly_root: Path) -> Path | None:
    paths = candidate_logs(run, job, nightly_root)
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def age_minutes(path: Path | None) -> int:
    if path is None:
        return 10**9
    return max(0, int((time.time() - path.stat().st_mtime) / 60))


def format_age(minutes: int) -> str:
    if minutes >= 10**9:
        return "--"
    if minutes < 120:
        return f"{minutes}m"
    if minutes < 2880:
        return f"{minutes // 60}h"
    return f"{minutes // 1440}d"


def last_error(log: Path | None) -> str:
    if log is None:
        return "--"
    try:
        text = subprocess.run(
            ["tail", "-n", "800", str(log)],
            capture_output=True,
            text=True,
            errors="ignore",
            check=False,
        ).stdout
    except OSError:
        return "--"
    pattern = re.compile(
        r"ERROR:|Traceback|RuntimeError|ChildFailedError|OutOfMemory|killed by signal",
        re.IGNORECASE,
    )
    hits = [line.strip() for line in text.splitlines() if pattern.search(line)]
    return hits[-1][:100] if hits else "--"


def steps_from_paths(paths: list[Path], pattern: str) -> set[int]:
    regex = re.compile(pattern)
    found: set[int] = set()
    for path in paths:
        match = regex.search(path.name)
        if match:
            found.add(int(match.group(1)))
    return found


def metric_step(run: Path) -> int:
    best = 0
    paths = list((run / "native/logs").glob("*/metrics.jsonl"))
    direct = run / "training_metrics.jsonl"
    if direct.is_file():
        paths.append(direct)
    pattern = re.compile(r'"(?:step|training/global_step)"\s*:\s*(\d+)')
    for path in paths:
        try:
            with path.open(errors="ignore") as stream:
                for line in stream:
                    for value in pattern.findall(line):
                        best = max(best, int(value))
        except OSError:
            continue
    return best


def math_progress(run: Path, total_steps: int) -> tuple[int, str]:
    steps = {metric_step(run)}
    steps.update(
        steps_from_paths(
            list(run.glob("native/checkpoints/*/global_step_*")),
            r"global_step_(\d+)$",
        )
    )
    steps.update(
        steps_from_paths(
            list(run.glob("checkpoints/*/checkpoint-*")),
            r"checkpoint-(\d+)$",
        )
    )

    complete: list[int] = []
    partial: list[tuple[int, int]] = []
    for directory in (run / "evaluations").glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", directory.name)
        if not match:
            continue
        step = int(match.group(1))
        count = sum(
            (directory / f"{dataset}.json").is_file()
            and (directory / f"{dataset}.json").stat().st_size > 0
            for dataset in DATASETS
        )
        steps.add(step)
        if count == len(DATASETS):
            complete.append(step)
        elif count:
            partial.append((step, count))

    train_step = max(steps, default=0)
    expected = total_steps // 5
    latest = max(complete, default=0)
    evidence = f"{len(complete)}/{expected}@{latest}"
    if partial:
        step, count = max(partial)
        evidence += f" partial={step}:{count}/5"
    return min(train_step, total_steps), evidence


def marker_steps(directory: Path) -> set[int]:
    return steps_from_paths(list(directory.glob("step_*.complete")), r"step_(\d+)\.complete$")


def p0_progress(run: Path, total_steps: int) -> tuple[int, str]:
    capture = marker_steps(run / "state/capture")
    generation = marker_steps(run / "state/generation")
    generation.update(
        steps_from_paths(list((run / "generation_raw").glob("*.jsonl")), r"(\d+)\.jsonl$")
    )
    checkpoints = steps_from_paths(
        list((run / "model_states").glob("**/global_step_*")),
        r"global_step_(\d+)$",
    )
    all_steps = capture | generation | checkpoints
    train_step = max(all_steps, default=0)
    expected = total_steps // 5 + 1
    evidence = (
        f"cap={len(capture)}/{expected}@{max(capture, default=0)} "
        f"gen={len(generation)}/{expected}@{max(generation, default=0)}"
    )
    return min(train_step, total_steps), evidence


def status_for(
    run: Path | None,
    job: Job,
    nightly_root: Path,
    log_age: int,
    recent_minutes: int,
) -> str:
    nightly_held = lock_is_held(nightly_root / job.job_id / "nightly.lock")
    pipeline_held = bool(
        run is not None and lock_is_held(run / "state/pipeline.lock")
    )
    if run is None:
        return "RUNNING?" if nightly_held else "NOT_STARTED"
    if job.kind == "p0":
        if (run / "state/run.complete").is_file():
            return "COMPLETE"
        if (run / "state/training.complete").is_file():
            return "TRAIN_DONE"
    elif (run / "state/complete").is_file():
        return "COMPLETE"
    if nightly_held or pipeline_held:
        return "RUNNING"
    if log_age <= recent_minutes:
        return "ACTIVE?"
    return "STOPPED"


def main() -> None:
    args = parse_args()
    rows = []
    for job in jobs():
        run = resolve_run(job)
        log = latest_log(run, job, args.nightly_state_root)
        minutes = age_minutes(log)
        status = status_for(
            run, job, args.nightly_state_root, minutes, args.recent_minutes
        )
        if run is None:
            step, evidence = 0, "--"
        elif job.kind == "p0":
            step, evidence = p0_progress(run, job.total_steps)
        else:
            step, evidence = math_progress(run, job.total_steps)
        rows.append(
            (
                job.label,
                status,
                f"{step}/{job.total_steps}",
                evidence,
                format_age(minutes),
                last_error(log),
                log,
            )
        )

    print(
        f"{'EXPERIMENT':<22} {'STATUS':<11} {'TRAIN':<9} "
        f"{'EVAL/CAPTURE':<31} {'AGE':<7} LAST_ERROR"
    )
    print("-" * 125)
    for label, status, train, evidence, age, error, _ in rows:
        print(
            f"{label:<22} {status:<11} {train:<9} "
            f"{evidence:<31} {age:<7} {error}"
        )

    print("\nLatest logs for unfinished jobs:")
    for label, status, _, _, _, _, log in rows:
        if status not in {"COMPLETE", "NOT_STARTED"}:
            print(f"[{status}] {label}: {log or 'NO LOG'}")

    print("\nRUNNING means a nightly or pipeline lock is held.")
    print("ACTIVE? means no lock is visible, but the latest log changed recently.")


if __name__ == "__main__":
    main()
