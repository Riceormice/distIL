#!/usr/bin/env python3
"""Measure Math pipeline wall time from persisted logs and result mtimes.

The projection is deliberately mechanical: it uses only completed five-step
train/evaluate cycles from the selected run. It does not apply a model- or
hardware-based speed assumption.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DATASETS = ("aime24", "aime25", "hmmt25", "amc23", "minerva")
DEFAULT_OPSD_RUN = Path(
    "/media/vlm-ckp-fileset/ylong/math_opsd_grouped8x8_eval5_n16_h200_20260820/opsd/"
    "opsd-8b-seed0-grouped-q8-r8-lr5e-6-steps100-sched420-beta0-clip0.06-"
    "topk100-temp0.7-tok16384-eval5-n16-h200"
)
DEFAULT_SWEEP_ROOT = Path(
    "/media/vlm-ckp-fileset/ylong/sr_opsd_math_alpha_rho_sweep_20260819/sr_opsd"
)
PIPELINE_STAMP = re.compile(r"pipeline_(\d{8}_\d{6})\.log$")
STEP_PATTERN = re.compile(r'"step"\s*:\s*(\d+)')


@dataclass(frozen=True)
class Cycle:
    step: int
    start: float
    end: float
    phases: dict[str, float]
    restart_logs: int

    @property
    def seconds(self) -> float:
        return self.end - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opsd-run", type=Path, default=DEFAULT_OPSD_RUN)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--alpha", help="Select one sweep alpha instead of auto-selection")
    parser.add_argument("--rho", help="Select one sweep rho instead of auto-selection")
    parser.add_argument("--total-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5)
    return parser.parse_args()


def complete_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 2:
        return False
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 4096))
        return handle.read().rstrip().endswith(b"}")


def result_times(run: Path) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    for checkpoint in (run / "evaluations").glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
        if not match:
            continue
        step = int(match.group(1))
        for dataset in DATASETS:
            path = checkpoint / f"{dataset}.json"
            if complete_file(path):
                output.setdefault(step, {})[dataset] = path.stat().st_mtime
    return output


def pipeline_logs(run: Path) -> list[tuple[float, Path]]:
    logs = []
    for path in (run / "logs").glob("pipeline_*.log"):
        match = PIPELINE_STAMP.search(path.name)
        if match:
            stamp = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").timestamp()
        else:
            stamp = path.stat().st_mtime
        logs.append((stamp, path))
    return sorted(logs)


def training_step(run: Path) -> int:
    best = 0
    candidates = [run / "training_metrics.jsonl"]
    candidates.extend((run / "native" / "logs").glob("*/metrics.jsonl"))
    for path in candidates:
        if not path.is_file():
            continue
        with path.open(errors="ignore") as handle:
            for line in handle:
                for value in STEP_PATTERN.findall(line):
                    best = max(best, int(value))
    for pattern in (
        "checkpoints/*/checkpoint-*",
        "native/checkpoints/*/global_step_*",
    ):
        for path in run.glob(pattern):
            match = re.search(r"(?:checkpoint-|global_step_)(\d+)$", path.name)
            if match:
                best = max(best, int(match.group(1)))
    return best


def build_cycles(run: Path, results: dict[int, dict[str, float]]) -> list[Cycle]:
    launches = pipeline_logs(run)
    if not launches:
        return []
    launch_times = [stamp for stamp, _ in launches]
    previous_end: float | None = None
    cycles = []
    for step in sorted(results):
        times = results[step]
        if any(dataset not in times for dataset in DATASETS):
            continue
        start = previous_end if previous_end is not None else launch_times[0]
        phase_start = start
        phases: dict[str, float] = {}
        ordered = True
        for dataset in DATASETS:
            end = times[dataset]
            duration = end - phase_start
            if duration < 0:
                ordered = False
                break
            phases[dataset] = duration
            phase_start = end
        if not ordered:
            previous_end = max(times.values())
            continue
        end = max(times.values())
        restarts = sum(start < stamp < end for stamp in launch_times)
        cycles.append(Cycle(step, start, end, phases, restarts))
        previous_end = end
    return cycles


def format_seconds(value: float) -> str:
    value = max(0.0, value)
    days, remainder = divmod(int(round(value)), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def format_time(value: float) -> str:
    return datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds")


def run_score(run: Path) -> tuple[int, int, int, int, float]:
    results = result_times(run)
    full = [step for step, times in results.items() if len(times) == len(DATASETS)]
    files = sum(len(times) for times in results.values())
    latest = max(
        [stamp for times in results.values() for stamp in times.values()],
        default=run.stat().st_mtime,
    )
    return len(full), max(full, default=0), files, training_step(run), latest


def select_sweep_run(root: Path, alpha: str | None, rho: str | None) -> Path:
    candidates = [path for path in root.glob("*") if path.is_dir()]
    if alpha is not None or rho is not None:
        if alpha is None or rho is None:
            raise SystemExit("--alpha and --rho must be supplied together")
        candidates = [
            path
            for path in candidates
            if f"rho{rho}-refw{alpha}-" in path.name
        ]
    if not candidates:
        raise SystemExit(f"no matching alpha/rho run under {root}")
    return max(candidates, key=run_score)


def phase_reference(cycles: list[Cycle]) -> tuple[list[Cycle], dict[str, float]]:
    uninterrupted = [cycle for cycle in cycles if cycle.restart_logs == 0]
    reference = uninterrupted or cycles
    phases = {
        dataset: statistics.median(cycle.phases[dataset] for cycle in reference)
        for dataset in DATASETS
    }
    return reference, phases


def analyze(label: str, run: Path, total_steps: int, eval_every: int) -> None:
    if not run.is_dir():
        print(f"\n===== {label} =====\nMISSING: {run}")
        return
    now = time.time()
    results = result_times(run)
    cycles = build_cycles(run, results)
    logs = pipeline_logs(run)
    train_step = training_step(run)
    full_steps = [step for step in sorted(results) if len(results[step]) == len(DATASETS)]
    partial_steps = [step for step in sorted(results) if 0 < len(results[step]) < len(DATASETS)]

    print(f"\n===== {label} =====")
    print(f"run={run}")
    print(f"training_step={train_step}/{total_steps}")
    print(f"complete_eval_steps={full_steps or 'none'}")
    if partial_steps:
        step = partial_steps[-1]
        done = [dataset for dataset in DATASETS if dataset in results[step]]
        missing = [dataset for dataset in DATASETS if dataset not in results[step]]
        print(f"partial_eval_step={step} done={done} missing={missing}")
    if logs:
        latest_log = max((path for _, path in logs), key=lambda path: path.stat().st_mtime)
        print(
            f"latest_log={latest_log} mtime={format_time(latest_log.stat().st_mtime)} "
            f"age={format_seconds(now - latest_log.stat().st_mtime)}"
        )

    if not cycles:
        print("measured_cycles=none; cannot calculate runtime from completed checkpoints")
        return

    print("measured_cycles:")
    for cycle in cycles:
        phase_text = ", ".join(
            f"{dataset}={format_seconds(cycle.phases[dataset])}" for dataset in DATASETS
        )
        print(
            f"  step={cycle.step:3d} start={format_time(cycle.start)} "
            f"end={format_time(cycle.end)} wall={format_seconds(cycle.seconds)} "
            f"restart_logs={cycle.restart_logs}"
        )
        print(f"    phases: {phase_text}")

    reference, phase_medians = phase_reference(cycles)
    cycle_median = statistics.median(cycle.seconds for cycle in reference)
    cycle_mean = statistics.mean(cycle.seconds for cycle in reference)
    source = "uninterrupted cycles" if any(c.restart_logs == 0 for c in cycles) else "all cycles"
    print(
        f"reference={source}, n={len(reference)}, "
        f"cycle_median={format_seconds(cycle_median)}, "
        f"cycle_mean={format_seconds(cycle_mean)}"
    )

    total_cycles = total_steps // eval_every
    completed_cycles = len(full_steps)
    current_step = partial_steps[-1] if partial_steps else None
    remaining_current = 0.0
    if current_step is not None:
        done = [dataset for dataset in DATASETS if dataset in results[current_step]]
        first_missing = next(dataset for dataset in DATASETS if dataset not in done)
        missing_index = DATASETS.index(first_missing)
        if missing_index == 0:
            phase_start = cycles[-1].end
        else:
            phase_start = results[current_step][DATASETS[missing_index - 1]]
        elapsed = now - phase_start
        remaining_current = max(0.0, phase_medians[first_missing] - elapsed)
        remaining_current += sum(
            phase_medians[dataset] for dataset in DATASETS[missing_index + 1 :]
        )
        later_cycles = max(0, total_cycles - completed_cycles - 1)
    else:
        later_cycles = max(0, total_cycles - completed_cycles)

    remaining = remaining_current + later_cycles * cycle_median
    print(
        "continuous_projection="
        f"remaining_current_cycle={format_seconds(remaining_current)}, "
        f"later_full_cycles={later_cycles}, remaining={format_seconds(remaining)}, "
        f"eta={format_time(now + remaining)}"
    )
    print(
        "projection_basis=only the measured file/log timestamps printed above; "
        "future interruption time is excluded"
    )


def main() -> None:
    args = parse_args()
    sweep = select_sweep_run(args.sweep_root, args.alpha, args.rho)
    match = re.search(r"rho([0-9.]+)-refw([0-9.]+)-", sweep.name)
    sweep_label = (
        f"SR-OPSD alpha={match.group(2)} rho={match.group(1)}"
        if match
        else f"SR-OPSD selected run {sweep.name}"
    )
    analyze("OPSD grouped 8x8", args.opsd_run, args.total_steps, args.eval_every)
    analyze(sweep_label, sweep, args.total_steps, args.eval_every)


if __name__ == "__main__":
    main()
