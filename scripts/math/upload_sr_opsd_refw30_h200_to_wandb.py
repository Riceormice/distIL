#!/usr/bin/env python3
"""Upload the four completed H200 Math self-reference-weight runs to W&B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path(
    "/media/vlm-ckp-fileset/ylong/sr_opsd_math_refw_sweep30"
)
CHECKPOINT_STEPS = (5, 10, 15, 20, 25, 30)
DATASETS = ("aime24", "aime25", "hmmt25", "amc23", "minerva")
EXPECTED_PROBLEMS = {
    "aime24": 30,
    "aime25": 30,
    "hmmt25": 30,
    "amc23": 40,
    "minerva": 272,
}
SELF_REFERENCE_WEIGHTS = ("0.95", "0.9", "0.85", "0.8")
RUN_ID_NAMESPACE = "sr-opsd-math-refw30-h200-table2-v1"


@dataclass(frozen=True)
class RunSpec:
    output_root: Path
    self_reference_weight: str

    @property
    def run_name(self) -> str:
        return (
            "sr-opsd-8b-seed0-rho0.95-"
            f"refw{self.self_reference_weight}-sync0-lr5e-6-"
            "tok16384-steps30-eval5"
        )

    @property
    def display_name(self) -> str:
        return (
            "Qwen3-8B-Math-SR-OPSD-rho0.95-"
            f"refw{self.self_reference_weight}-seed0-H200-steps30"
        )

    @property
    def metrics_path(self) -> Path:
        return self.output_root / "logs" / self.run_name / "metrics.jsonl"

    @property
    def eval_dir(self) -> Path:
        return self.output_root / "evaluations" / self.run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--entity", default="wenxuan-yuan-imperial-college-london"
    )
    parser.add_argument("--project", default="SDPO_table2")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    return parser.parse_args()


def read_training_events(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    content = path.read_bytes()
    lines = content.splitlines()
    events: dict[int, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
            step = int(event["step"])
            metrics = event["metrics"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            if line_number == len(lines) and not content.endswith(b"\n"):
                break
            raise RuntimeError(f"{path}:{line_number}: invalid event") from exc
        if not isinstance(metrics, dict):
            raise RuntimeError(f"{path}:{line_number}: metrics is not an object")
        events.setdefault(step, {}).update(metrics)
        events[step].setdefault("training/global_step", step)
    return {step: events[step] for step in sorted(events) if step <= 30}


def file_looks_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 2:
        return False
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 4096))
        return handle.read().rstrip().endswith(b"}")


def read_eval_header(path: Path, expected_dataset: str) -> dict[str, Any]:
    if not file_looks_complete(path):
        raise RuntimeError(f"incomplete evaluation JSON: {path}")
    with path.open("rb") as handle:
        prefix = handle.read(2_097_152)
    marker = prefix.find(b'"results"')
    if marker < 0:
        raise RuntimeError(f"evaluation header is missing results: {path}")
    text = prefix[:marker].decode("utf-8", errors="strict")

    def string_value(key: str) -> str:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', text)
        if match is None:
            raise RuntimeError(f"missing {key} in {path}")
        return match.group(1)

    def number_value(key: str) -> float:
        match = re.search(
            rf'"{re.escape(key)}"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))',
            text,
        )
        if match is None:
            raise RuntimeError(f"missing {key} in {path}")
        return float(match.group(1))

    dataset = string_value("dataset").lower()
    num_problems = int(number_value("num_problems"))
    val_n = int(number_value("val_n"))
    total_solutions = int(number_value("total_solutions"))
    if dataset != expected_dataset:
        raise RuntimeError(f"dataset={dataset!r}, expected {expected_dataset!r}: {path}")
    if num_problems != EXPECTED_PROBLEMS[dataset]:
        raise RuntimeError(
            f"num_problems={num_problems}, expected "
            f"{EXPECTED_PROBLEMS[dataset]}: {path}"
        )
    if val_n != 64 or total_solutions != num_problems * val_n:
        raise RuntimeError(
            f"invalid N=64 dimensions: problems={num_problems}, "
            f"val_n={val_n}, total={total_solutions}: {path}"
        )

    values = {
        "dataset": dataset,
        "num_problems": num_problems,
        "val_n": val_n,
        "total_solutions": total_solutions,
        "avg_pct": number_value("average_at_n_pct"),
        "pass_pct": number_value("pass_at_n_pct"),
        "majority_pct": number_value("majority_vote_at_n_pct"),
        "format_pct": number_value("format_rate"),
    }
    for key in ("avg_pct", "pass_pct", "majority_pct", "format_pct"):
        if not math.isfinite(values[key]) or not 0 <= values[key] <= 100:
            raise RuntimeError(f"invalid {key}={values[key]}: {path}")
    return values


def read_checkpoint(
    spec: RunSpec, step: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checkpoint_dir = spec.eval_dir / f"checkpoint-{step}"
    headers = [
        read_eval_header(checkpoint_dir / f"{dataset}.json", dataset)
        for dataset in DATASETS
    ]
    metrics: dict[str, Any] = {
        "validation/checkpoint": step,
        "validation/complete": 1,
    }
    for header in headers:
        prefix = f"eval/{header['dataset']}"
        metrics[f"{prefix}/avg@64"] = header["avg_pct"] / 100.0
        metrics[f"{prefix}/pass@64"] = header["pass_pct"] / 100.0
        metrics[f"{prefix}/majority@64"] = header["majority_pct"] / 100.0
        metrics[f"{prefix}/format"] = header["format_pct"] / 100.0
        metrics[f"{prefix}/num_problems"] = header["num_problems"]
        metrics[f"{prefix}/total_solutions"] = header["total_solutions"]

    total_problems = sum(row["num_problems"] for row in headers)
    total_solutions = sum(row["total_solutions"] for row in headers)
    weighted_avg = sum(
        row["avg_pct"] * row["total_solutions"] for row in headers
    ) / total_solutions
    weighted_pass = sum(
        row["pass_pct"] * row["num_problems"] for row in headers
    ) / total_problems
    weighted_majority = sum(
        row["majority_pct"] * row["num_problems"] for row in headers
    ) / total_problems
    weighted_format = sum(
        row["format_pct"] * row["total_solutions"] for row in headers
    ) / total_solutions
    macro_avg = sum(row["avg_pct"] for row in headers) / len(headers)

    metrics.update(
        {
            "validation/avg@64": weighted_avg / 100.0,
            "validation/pass@64": weighted_pass / 100.0,
            "validation/majority@64": weighted_majority / 100.0,
            "validation/format": weighted_format / 100.0,
            "validation/macro_avg@64": macro_avg / 100.0,
            "validation/num_problems": total_problems,
            "validation/total_solutions": total_solutions,
        }
    )
    return metrics, headers


def build_events(
    spec: RunSpec,
) -> tuple[
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
    tuple[int, ...],
    str,
]:
    events = read_training_events(spec.metrics_path)
    rows: list[dict[str, Any]] = []
    completed_steps: list[int] = []
    waiting_for = "none"
    for step in CHECKPOINT_STEPS:
        try:
            checkpoint_metrics, headers = read_checkpoint(spec, step)
        except RuntimeError as exc:
            waiting_for = f"checkpoint-{step}: {exc}"
            break
        events.setdefault(step, {}).update(checkpoint_metrics)
        completed_steps.append(step)
        for header in headers:
            rows.append(
                {
                    "step": step,
                    "dataset": header["dataset"],
                    "num_problems": header["num_problems"],
                    "samples_per_problem": header["val_n"],
                    "avg@64_pct": header["avg_pct"],
                    "pass@64_pct": header["pass_pct"],
                    "majority@64_pct": header["majority_pct"],
                    "format_pct": header["format_pct"],
                }
            )
    if not completed_steps:
        raise RuntimeError(f"no complete evaluation checkpoint: {spec.eval_dir}")
    return events, rows, tuple(completed_steps), waiting_for


def run_id_for(spec: RunSpec) -> str:
    value = f"{RUN_ID_NAMESPACE}:{spec.run_name}"
    return "mr" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:14]


def state_path_for(state_dir: Path, spec: RunSpec) -> Path:
    digest = hashlib.sha256(spec.run_name.encode("utf-8")).hexdigest()[:16]
    return state_dir / f"{digest}.json"


def load_state(path: Path, expected_run_id: str) -> dict[str, Any]:
    if not path.is_file():
        return {"last_step": 0, "artifact_logged": False}
    with path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("run_id") != expected_run_id:
        return {"last_step": 0, "artifact_logged": False}
    return state


def save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_summary_csv(
    state_dir: Path, spec: RunSpec, rows: list[dict[str, Any]]
) -> Path:
    summary_dir = state_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"{spec.run_name}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def best_summary(events: dict[int, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset in DATASETS:
        key = f"eval/{dataset}/avg@64"
        candidates = [(step, metrics[key]) for step, metrics in events.items() if key in metrics]
        best_step, best_value = max(candidates, key=lambda item: item[1])
        prefix = f"eval/{dataset}"
        summary[f"best/{dataset}/step"] = best_step
        summary[f"best/{dataset}/avg@64"] = best_value
        summary[f"best/{dataset}/pass@64"] = events[best_step][f"{prefix}/pass@64"]
        summary[f"best/{dataset}/majority@64"] = events[best_step][f"{prefix}/majority@64"]
        summary[f"best/{dataset}/format"] = events[best_step][f"{prefix}/format"]
    aggregate_key = "validation/avg@64"
    aggregate = [
        (step, metrics[aggregate_key])
        for step, metrics in events.items()
        if aggregate_key in metrics
    ]
    best_step, best_value = max(aggregate, key=lambda item: item[1])
    summary["best/weighted/step"] = best_step
    summary["best/weighted/avg@64"] = best_value
    summary["best/weighted/pass@64"] = events[best_step]["validation/pass@64"]
    return summary


def config_for(spec: RunSpec) -> dict[str, Any]:
    return {
        "model": "Qwen3-8B",
        "method": "SR-OPSD",
        "benchmark": "math",
        "datasets": list(DATASETS),
        "hardware": "8xH200",
        "seed": 0,
        "total_training_steps": 30,
        "checkpoint_and_eval_frequency": 5,
        "training_batch_size": 16,
        "ppo_mini_batch_size": 16,
        "ppo_micro_batch_size_per_gpu": 1,
        "training_rollouts_per_question": 8,
        "max_prompt_length": 2048,
        "max_training_response_length": 16384,
        "training_temperature": 0.7,
        "training_top_p": 0.95,
        "training_top_k": 20,
        "learning_rate": 5e-6,
        "learning_rate_schedule": "linear",
        "warmup_steps": 0,
        "weight_decay": 0,
        "gradient_clip_norm": 0.1,
        "projection_objective": "forward_renyi",
        "renyi_order_rho": 0.95,
        "self_reference_weight": float(spec.self_reference_weight),
        "renyi_regularization_level": float(spec.self_reference_weight),
        "renyi_regularization": True,
        "frozen_reference_anchoring": True,
        "renyi_ref_sync_steps": 0,
        "self_teacher": "EMA",
        "teacher_ema_update_rate": 0.01,
        "implementation": "native-sdpo-math-train",
        "full_logit_distillation": True,
        "distillation_top_k": 100,
        "distillation_tail_bucket": True,
        "rollout_is_correction": "token",
        "rollout_is_threshold": 2.0,
        "evaluation_thinking_mode": True,
        "evaluation_samples_per_question": 64,
        "evaluation_temperature": 1.0,
        "evaluation_top_p": 1.0,
        "evaluation_top_k": -1,
        "evaluation_min_p": 0,
        "evaluation_presence_penalty": 0,
        "evaluation_max_new_tokens": 38912,
        "evaluation_tensor_parallel_size": 8,
        "training_metrics_available": spec.metrics_path.is_file(),
        "source_metrics": str(spec.metrics_path) if spec.metrics_path.is_file() else None,
        "source_evaluations": str(spec.eval_dir),
    }


def upload_spec(
    spec: RunSpec,
    *,
    entity: str,
    project: str,
    state_dir: Path,
    dry_run: bool,
) -> bool:
    print(f"VALIDATING {spec.display_name}", flush=True)
    events, rows, completed_steps, waiting_for = build_events(spec)
    summary_csv = write_summary_csv(state_dir, spec, rows)
    summary = best_summary(events)
    latest_eval_step = max(completed_steps)
    complete = completed_steps == CHECKPOINT_STEPS
    run_id = run_id_for(spec)
    state_path = state_path_for(state_dir, spec)
    state = load_state(state_path, run_id)
    last_step = int(state.get("last_step", 0))
    artifact_step = int(state.get("artifact_step", 0))
    pending = [step for step in sorted(events) if step > last_step]
    print(
        f"{'COMPLETE' if complete else 'PARTIAL'} {spec.display_name}: "
        f"latest_event_step={max(events)}, "
        f"eval_checkpoints={len(completed_steps)}/{len(CHECKPOINT_STEPS)}, "
        f"eval_json={len(rows)}, pending_events={len(pending)}, "
        f"waiting_for={waiting_for}",
        flush=True,
    )
    if dry_run or (not pending and artifact_step >= latest_eval_step):
        return complete

    import wandb

    print(f"CONNECTING {spec.display_name}", flush=True)
    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        name=spec.display_name,
        resume="allow",
        group="Qwen3-8B-Math-SR-OPSD-refw30-H200-seed0",
        job_type="ablation",
        tags=[
            "table2",
            "math",
            "Qwen3-8B",
            "SR-OPSD",
            "H200",
            "rho-0.95",
            f"refw-{spec.self_reference_weight}",
            "seed-0",
            "steps-30",
        ],
        config=config_for(spec),
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None:
        raise RuntimeError(f"wandb.init returned no run for {spec.run_name}")

    uploaded_step = last_step
    uploaded_artifact_step = artifact_step
    try:
        run.define_metric("training/global_step")
        run.define_metric("*", step_metric="training/global_step")
        for step in pending:
            events[step]["training/global_step"] = step
            run.log(events[step], step=step)
            uploaded_step = step
            print(f"UPLOADING {spec.display_name}: step={step}/30", flush=True)
        for key, value in summary.items():
            run.summary[key] = value

        if artifact_step < latest_eval_step:
            artifact_name = (
                f"math-sr-opsd-refw{spec.self_reference_weight}-steps30-results"
                .replace(".", "p")
            )
            artifact = wandb.Artifact(
                name=artifact_name,
                type="evaluation-results",
                metadata={
                    "source_run_name": spec.run_name,
                    "rho": 0.95,
                    "self_reference_weight": float(spec.self_reference_weight),
                    "eval_frequency": 5,
                    "evaluation_samples": 64,
                    "latest_complete_evaluation_step": latest_eval_step,
                    "all_evaluations_complete": complete,
                },
            )
            if spec.metrics_path.is_file():
                artifact.add_file(str(spec.metrics_path), name="metrics.jsonl")
            artifact.add_file(str(summary_csv), name="eval5_math_metrics.csv")
            run.log_artifact(artifact)
            uploaded_artifact_step = latest_eval_step
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise

    save_state(
        state_path,
        {
            "run_id": run_id,
            "run_name": spec.run_name,
            "display_name": spec.display_name,
            "last_step": uploaded_step,
            "artifact_step": uploaded_artifact_step,
            "complete": complete,
            "entity": entity,
            "project": project,
        },
    )
    print(f"UPLOADED {spec.display_name}", flush=True)
    return complete


def main() -> None:
    args = parse_args()
    state_dir = args.state_dir or args.output_root / "wandb_upload_state"
    if args.reset_state and state_dir.exists():
        for path in state_dir.glob("*.json"):
            path.unlink()

    specs = [RunSpec(args.output_root, weight) for weight in SELF_REFERENCE_WEIGHTS]
    failures = 0
    complete_runs = 0
    for spec in specs:
        try:
            complete_runs += int(upload_spec(
                spec,
                entity=args.entity,
                project=args.project,
                state_dir=state_dir,
                dry_run=args.dry_run,
            ))
        except Exception as exc:
            failures += 1
            print(
                f"ERROR {spec.display_name}: {type(exc).__name__}: {exc}",
                flush=True,
            )
    mode = "validation" if args.dry_run else "upload"
    partial_runs = len(specs) - complete_runs - failures
    print(
        f"{mode}: runs={len(specs)}, complete={complete_runs}, "
        f"partial={partial_runs}, failures={failures}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)
    if partial_runs:
        print(
            "SUCCESS: all currently complete evaluation checkpoints were "
            + ("validated" if args.dry_run else "uploaded"),
            flush=True,
        )
    else:
        print(
            "SUCCESS: all four H200 Math self-reference-weight runs are complete"
            + ("" if args.dry_run else " and uploaded"),
            flush=True,
        )


if __name__ == "__main__":
    main()
