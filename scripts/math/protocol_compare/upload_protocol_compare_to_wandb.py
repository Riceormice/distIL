#!/usr/bin/env python3
"""Incrementally upload the two SR-OPSD Math protocol-comparison runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path(
    "/media/vlm-ckp-fileset/ylong/sr_opsd_math_protocol_compare_20260811"
)
STEPS = (5, 10, 15, 20, 25, 30)
DATASETS = ("aime24", "aime25", "hmmt25", "amc23", "minerva")
EXPECTED_PROBLEMS = {
    "aime24": 30,
    "aime25": 30,
    "hmmt25": 30,
    "amc23": 40,
    "minerva": 272,
}
RUN_ID_NAMESPACE = "sr-opsd-math-protocol-compare-n64-v1"
VALIDATOR = Path(__file__).resolve().parents[1] / "validate_math_eval.py"


@dataclass(frozen=True)
class RunSpec:
    profile: str
    run_name: str
    display_name: str
    config: dict[str, Any]


RUN_SPECS = (
    RunSpec(
        profile="table_aligned",
        run_name=(
            "sr-opsd-8b-seed0-table-aligned-rho0.95-refw0.9-sync0-"
            "lr5e-6-tok16384-steps30-eval5"
        ),
        display_name="Qwen3-8B-Math-SR-OPSD-table-aligned-N64",
        config={
            "train_batch_size": 8,
            "ppo_mini_batch_size": 8,
            "training_rollouts": 1,
            "max_response_length": 16384,
            "training_temperature": 0.7,
            "training_top_p": 0.95,
            "training_top_k": 20,
            "learning_rate_schedule": "linear",
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "gradient_clip_norm": 0.1,
            "teacher_update_rate": 0.05,
            "distillation_tail_bucket": False,
            "distillation_is_clip": None,
            "token_loss_clip": 0.05,
            "lora_rank": 0,
        },
    ),
    RunSpec(
        profile="github_original",
        run_name=(
            "sr-opsd-8b-seed0-github-original-rho0.95-refw0.9-sync0-"
            "lr5e-6-tok8192-steps30-eval5"
        ),
        display_name="Qwen3-8B-Math-SR-OPSD-github-original-N64",
        config={
            "train_batch_size": 32,
            "ppo_mini_batch_size": 32,
            "training_rollouts": 8,
            "max_response_length": 8192,
            "training_temperature": 0.8,
            "training_top_p": 0.95,
            "training_top_k": -1,
            "learning_rate_schedule": "constant",
            "warmup_steps": 10,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "teacher_update_rate": 0.01,
            "distillation_tail_bucket": True,
            "distillation_is_clip": 2.0,
            "token_loss_clip": None,
            "lora_rank": 64,
        },
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--entity", default="wenxuan-yuan-imperial-college-london"
    )
    parser.add_argument("--project", default="test")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    return parser.parse_args()


def metrics_path(output_root: Path, spec: RunSpec) -> Path:
    return output_root / "logs" / spec.run_name / "metrics.jsonl"


def eval_root(output_root: Path, spec: RunSpec) -> Path:
    return output_root / "evaluations" / spec.run_name


def read_training_events(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    content = path.read_bytes()
    events: dict[int, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
            step = int(event["step"])
            metrics = event["metrics"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            if line_number == len(content.splitlines()) and not content.endswith(b"\n"):
                break
            raise RuntimeError(f"{path}:{line_number}: invalid event") from exc
        if not isinstance(metrics, dict):
            raise RuntimeError(f"{path}:{line_number}: metrics is not an object")
        if 1 <= step <= 30:
            events.setdefault(step, {}).update(metrics)
    return events


def file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def read_eval_header(path: Path, expected_dataset: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size < 2:
        raise RuntimeError("missing")
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 4096))
        if not handle.read().rstrip().endswith(b"}"):
            raise RuntimeError("incomplete JSON")
    validation = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(path),
            "--dataset",
            expected_dataset,
            "--samples",
            "64",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if validation.returncode != 0:
        detail = validation.stderr.strip() or validation.stdout.strip()
        raise RuntimeError(f"strict validation failed: {detail}")
    with path.open("rb") as handle:
        prefix = handle.read(2_097_152)
    marker = prefix.find(b'"results"')
    if marker < 0:
        raise RuntimeError("missing results")
    text = prefix[:marker].decode("utf-8")

    def string_value(key: str) -> str:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', text)
        if match is None:
            raise RuntimeError(f"missing {key}")
        return match.group(1)

    def number_value(key: str) -> float:
        match = re.search(
            rf'"{re.escape(key)}"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))',
            text,
        )
        if match is None:
            raise RuntimeError(f"missing {key}")
        return float(match.group(1))

    dataset = string_value("dataset").lower()
    problems = int(number_value("num_problems"))
    samples = int(number_value("val_n"))
    solutions = int(number_value("total_solutions"))
    if dataset != expected_dataset:
        raise RuntimeError(f"dataset={dataset}, expected={expected_dataset}")
    if problems != EXPECTED_PROBLEMS[dataset]:
        raise RuntimeError(f"problems={problems}")
    if samples != 64 or solutions != problems * samples:
        raise RuntimeError(
            f"invalid dimensions: problems={problems}, N={samples}, total={solutions}"
        )
    values = {
        "dataset": dataset,
        "num_problems": problems,
        "val_n": samples,
        "total_solutions": solutions,
        "average_at_n": int(number_value("average_at_n")),
        "pass_at_n": int(number_value("pass_at_n")),
        "majority_vote_at_n": int(number_value("majority_vote_at_n")),
        "formatted_count": int(number_value("formatted_count")),
        "avg_pct": number_value("average_at_n_pct"),
        "pass_pct": number_value("pass_at_n_pct"),
        "majority_pct": number_value("majority_vote_at_n_pct"),
        "format_pct": number_value("format_rate"),
        "path": path,
        "digest": file_fingerprint(path),
    }
    for key in ("avg_pct", "pass_pct", "majority_pct", "format_pct"):
        if not math.isfinite(values[key]) or not 0 <= values[key] <= 100:
            raise RuntimeError(f"invalid {key}={values[key]}")
    return values


def read_available_evaluations(
    output_root: Path, spec: RunSpec
) -> dict[tuple[int, str], dict[str, Any]]:
    records: dict[tuple[int, str], dict[str, Any]] = {}
    root = eval_root(output_root, spec)
    for step in STEPS:
        for dataset in DATASETS:
            path = root / f"checkpoint-{step}" / f"{dataset}.json"
            if not path.is_file():
                continue
            print(
                f"VALIDATING {spec.profile}: checkpoint-{step} {dataset}",
                flush=True,
            )
            try:
                records[(step, dataset)] = read_eval_header(path, dataset)
                row = records[(step, dataset)]
                print(
                    f"VALID {spec.profile}: checkpoint-{step} {dataset} "
                    f"Avg@64={row['avg_pct']:.2f} Pass@64={row['pass_pct']:.2f}",
                    flush=True,
                )
            except (OSError, RuntimeError) as exc:
                print(
                    f"SKIP INVALID {spec.profile}: checkpoint-{step} "
                    f"{dataset}: {exc}",
                    flush=True,
                )
                continue
    return records


def pooled_metrics(
    records: dict[tuple[int, str], dict[str, Any]], step: int
) -> dict[str, Any] | None:
    rows = [records.get((step, dataset)) for dataset in DATASETS]
    if any(row is None for row in rows):
        return None
    complete_rows = [row for row in rows if row is not None]
    problems = sum(row["num_problems"] for row in complete_rows)
    solutions = sum(row["total_solutions"] for row in complete_rows)
    return {
        "eval/pooled/avg@64": sum(
            row["average_at_n"] for row in complete_rows
        ) / solutions,
        "eval/pooled/pass@64": sum(
            row["pass_at_n"] for row in complete_rows
        ) / problems,
        "eval/pooled/majority@64": sum(
            row["majority_vote_at_n"] for row in complete_rows
        ) / problems,
        "eval/pooled/format": sum(
            row["formatted_count"] for row in complete_rows
        ) / solutions,
        "eval/pooled/num_problems": problems,
        "eval/pooled/total_solutions": solutions,
        "eval/pooled/complete": 1,
    }


def run_id_for(spec: RunSpec) -> str:
    value = f"{RUN_ID_NAMESPACE}:{spec.run_name}"
    return "pc" + hashlib.sha256(value.encode()).hexdigest()[:14]


def load_state(path: Path, run_id: str) -> dict[str, Any]:
    if not path.is_file():
        return {"run_id": run_id, "training_steps": [], "evaluations": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("run_id") != run_id:
        return {"run_id": run_id, "training_steps": [], "evaluations": {}}
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_summary_csv(
    state_dir: Path,
    spec: RunSpec,
    records: dict[tuple[int, str], dict[str, Any]],
) -> Path:
    path = state_dir / "summaries" / f"{spec.profile}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "profile", "step", "dataset", "num_problems", "samples_per_problem",
        "avg_at_64_pct", "pass_at_64_pct", "majority_at_64_pct", "format_pct",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (step, dataset), row in sorted(records.items()):
            writer.writerow({
                "profile": spec.profile,
                "step": step,
                "dataset": dataset,
                "num_problems": row["num_problems"],
                "samples_per_problem": row["val_n"],
                "avg_at_64_pct": row["avg_pct"],
                "pass_at_64_pct": row["pass_pct"],
                "majority_at_64_pct": row["majority_pct"],
                "format_pct": row["format_pct"],
            })
    return path


def full_config(
    output_root: Path, spec: RunSpec, training_path: Path
) -> dict[str, Any]:
    common = {
        "profile": spec.profile,
        "model": "Qwen3-8B",
        "method": "SR-OPSD",
        "seed": 0,
        "hardware": "8xH200/H20Z",
        "training_steps": 30,
        "checkpoint_eval_frequency": 5,
        "learning_rate": 5e-6,
        "projection_objective": "forward_renyi",
        "divergence_alpha": 0.25,
        "renyi_order_rho": 0.95,
        "self_reference_weight": 0.9,
        "reference_sync_steps": 0,
        "entropy_coefficient": 1e-5,
        "max_prompt_length": 2048,
        "evaluation_datasets": list(DATASETS),
        "evaluation_thinking": True,
        "evaluation_samples_per_question": 64,
        "evaluation_temperature": 1.0,
        "evaluation_top_p": 1.0,
        "evaluation_top_k": -1,
        "evaluation_min_p": 0.0,
        "evaluation_presence_penalty": 0.0,
        "evaluation_max_new_tokens": 38912,
        "evaluation_tensor_parallel_size": 8,
        "source_training_metrics": str(training_path),
        "source_evaluations": str(eval_root(output_root, spec)),
    }
    common.update(spec.config)
    return common


def upload_run(
    *,
    output_root: Path,
    spec: RunSpec,
    entity: str,
    project: str,
    state_dir: Path,
    dry_run: bool,
) -> None:
    training_path = metrics_path(output_root, spec)
    training = read_training_events(training_path)
    evaluations = read_available_evaluations(output_root, spec)
    if not training and not evaluations:
        raise RuntimeError(
            f"no training metrics or complete evaluation JSON found for {spec.run_name}"
        )

    run_id = run_id_for(spec)
    state_path = state_dir / f"{spec.profile}.json"
    state = load_state(state_path, run_id)
    uploaded_training = {int(step) for step in state.get("training_steps", [])}
    uploaded_evals = dict(state.get("evaluations", {}))
    pending_training = [step for step in training if step not in uploaded_training]
    pending_evals = [
        (step, dataset)
        for step, dataset in sorted(evaluations)
        if uploaded_evals.get(f"{step}:{dataset}")
        != evaluations[(step, dataset)]["digest"]
    ]
    summary_csv = write_summary_csv(state_dir, spec, evaluations)

    counts = {
        step: sum((step, dataset) in evaluations for dataset in DATASETS)
        for step in STEPS
    }
    print(
        f"{spec.profile}: training_steps={len(training)}, "
        f"eval_datasets={len(evaluations)}/30, "
        f"by_step={','.join(f'{step}:{counts[step]}/5' for step in STEPS)}, "
        f"pending_training={len(pending_training)}, pending_eval={len(pending_evals)}",
        flush=True,
    )
    if dry_run or (not pending_training and not pending_evals):
        return

    import wandb

    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        name=spec.display_name,
        resume="allow",
        group="Qwen3-8B-Math-SR-OPSD-protocol-compare-N64",
        job_type="protocol-comparison",
        tags=["math", "Qwen3-8B", "SR-OPSD", "N64", spec.profile],
        config=full_config(output_root, spec, training_path),
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None:
        raise RuntimeError(f"wandb.init returned no run for {spec.run_name}")

    try:
        run.define_metric("optimizer_step")
        run.define_metric("train/*", step_metric="optimizer_step")
        run.define_metric("eval/*", step_metric="optimizer_step")

        for step in sorted(pending_training):
            event = {"optimizer_step": step}
            for key, value in training[step].items():
                if key == "training/global_step":
                    continue
                normalized = key[len("training/"):] if key.startswith("training/") else key
                event[f"train/{normalized}"] = value
            run.log(event)
            uploaded_training.add(step)
            print(f"UPLOADED {spec.profile}: training step {step}", flush=True)

        touched_steps: set[int] = set()
        for step, dataset in pending_evals:
            row = evaluations[(step, dataset)]
            prefix = f"eval/{dataset}"
            run.log({
                "optimizer_step": step,
                f"{prefix}/avg@64": row["avg_pct"] / 100.0,
                f"{prefix}/pass@64": row["pass_pct"] / 100.0,
                f"{prefix}/majority@64": row["majority_pct"] / 100.0,
                f"{prefix}/format": row["format_pct"] / 100.0,
                f"{prefix}/num_problems": row["num_problems"],
                f"{prefix}/total_solutions": row["total_solutions"],
            })
            uploaded_evals[f"{step}:{dataset}"] = row["digest"]
            touched_steps.add(step)
            print(f"UPLOADED {spec.profile}: checkpoint-{step} {dataset}", flush=True)

        for step in sorted(touched_steps):
            pooled = pooled_metrics(evaluations, step)
            if pooled is not None:
                run.log({"optimizer_step": step, **pooled})

        run.summary["available/training_steps"] = len(training)
        run.summary["available/evaluation_json"] = len(evaluations)
        run.summary["available/complete_checkpoints"] = sum(
            count == len(DATASETS) for count in counts.values()
        )
        run.summary["available/latest_complete_checkpoint"] = max(
            (step for step, count in counts.items() if count == len(DATASETS)),
            default=0,
        )
        run.summary["source/run_name"] = spec.run_name

        artifact = wandb.Artifact(
            name=f"math-protocol-{spec.profile}-available-results",
            type="evaluation-summary",
            metadata={
                "profile": spec.profile,
                "evaluation_json": len(evaluations),
                "samples_per_question": 64,
            },
        )
        artifact.add_file(str(summary_csv), name="available_eval_metrics.csv")
        if training_path.is_file():
            artifact.add_file(str(training_path), name="metrics.jsonl")
        run.log_artifact(artifact)
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise

    save_state(state_path, {
        "run_id": run_id,
        "run_name": spec.run_name,
        "training_steps": sorted(uploaded_training),
        "evaluations": uploaded_evals,
        "entity": entity,
        "project": project,
    })


def main() -> None:
    args = parse_args()
    state_dir = args.state_dir or args.output_root / "wandb_upload_state_protocol"
    if args.reset_state and state_dir.exists():
        for path in state_dir.glob("*.json"):
            path.unlink()

    failures = 0
    for spec in RUN_SPECS:
        try:
            upload_run(
                output_root=args.output_root,
                spec=spec,
                entity=args.entity,
                project=args.project,
                state_dir=state_dir,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            failures += 1
            print(
                f"ERROR {spec.profile}: {type(exc).__name__}: {exc}",
                flush=True,
            )
    if failures:
        raise SystemExit(1)
    print(
        "SUCCESS: all currently available protocol-comparison data "
        + ("validated" if args.dry_run else "uploaded"),
        flush=True,
    )


if __name__ == "__main__":
    main()
