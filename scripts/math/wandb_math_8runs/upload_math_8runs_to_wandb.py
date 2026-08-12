#!/usr/bin/env python3
"""Incrementally upload the current four 4B and four 8B Math runs to W&B."""

from __future__ import annotations

import argparse
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


DATASETS = ("aime24", "aime25", "hmmt25", "amc23", "minerva")
EXPECTED_PROBLEMS = {
    "aime24": 30,
    "aime25": 30,
    "hmmt25": 30,
    "amc23": 40,
    "minerva": 272,
}
MAX_STEP = 100
VAL_N = 16
RUN_NAMESPACE = "math-train-eval5-n16-eight-runs-20260812-v1"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
VALIDATOR = REPO / "scripts/math/validate_math_eval.py"


@dataclass(frozen=True)
class RunSpec:
    model_size: str
    hardware: str
    method_dir: str
    method_label: str
    run_name: str
    output_root: Path

    @property
    def display_name(self) -> str:
        return (
            f"Qwen3-{self.model_size}-Math-{self.method_label}-seed0-"
            f"eval5-N16-{self.hardware}"
        )

    @property
    def profile(self) -> str:
        return f"{self.model_size.lower()}-{self.method_dir}-{self.hardware.lower()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h200-root",
        type=Path,
        default=Path(
            "/media/vlm-ckp-fileset/ylong/math_train_eval5_n16_h200_20260812"
        ),
    )
    parser.add_argument(
        "--a800-root",
        type=Path,
        default=Path(
            "/media/vlm-ckp-fileset/ylong/math_4b_train_eval5_n16_a800_20260812"
        ),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(
            "/media/vlm-ckp-fileset/ylong/"
            "math_train_eval5_n16_wandb_upload_state"
        ),
    )
    parser.add_argument(
        "--entity", default="wenxuan-yuan-imperial-college-london"
    )
    parser.add_argument("--project", default="test")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-strict-validation",
        action="store_true",
        help="Only for local fixture tests; never use for real uploads.",
    )
    return parser.parse_args()


def run_specs(h200_root: Path, a800_root: Path) -> tuple[RunSpec, ...]:
    def names(model: str, hardware: str) -> tuple[tuple[str, str, str], ...]:
        return (
            (
                "grpo",
                "GRPO",
                f"grpo-{model}-seed0-native-verl-lr5e-6-trainbs8-mbs8-"
                f"rolloutn8-eps0.2-temp0.7-tok16384-steps100-sched420-"
                f"eval5-n16-{hardware}",
            ),
            (
                "sdpo",
                "SDPO",
                f"sdpo-{model}-seed0-native-verl-rkl-ema0.05-lr5e-6-"
                f"trainbs8-mbs8-rolloutn8-topk100-tailFalse-clip0.05-"
                f"temp0.7-tok16384-steps100-sched420-eval5-n16-{hardware}",
            ),
            (
                "opsd",
                "OPSD",
                f"opsd-{model}-seed0-lr5e-6-bs1-ga8-steps100-sched420-"
                f"beta0-clip{'0.06' if model == '8b' else '0.05'}-topk100-"
                f"temp0.7-tok16384-eval5-n16-{hardware}",
            ),
            (
                "sr_opsd",
                "SR-OPSD",
                f"sr-opsd-{model}-seed0-native-verl-forward-renyi-rho0.95-"
                f"refw0.9-sync0-ema0.05-lr5e-6-trainbs8-mbs8-rolloutn8-"
                f"topk100-tailFalse-clip0.05-temp0.7-tok16384-steps100-"
                f"sched420-eval5-n16-{hardware}",
            ),
        )

    specs = [
        RunSpec("8B", "H200", method_dir, label, run_name, h200_root)
        for method_dir, label, run_name in names("8b", "h200")
    ]
    specs.extend(
        RunSpec("4B", "A800", method_dir, label, run_name, a800_root)
        for method_dir, label, run_name in names("4b", "a800")
    )
    return tuple(specs)


def find_run_root(spec: RunSpec) -> Path | None:
    run_root = spec.output_root / spec.method_dir / spec.run_name
    return run_root if run_root.is_dir() else None


def parse_parameters(path: Path) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    if not path.is_file():
        return parameters
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        else:
            try:
                parsed = float(value) if any(c in value.lower() for c in ".e") else int(value)
            except ValueError:
                parsed = value
        parameters[key.strip()] = parsed
    return parameters


def training_metrics_path(run_root: Path, spec: RunSpec) -> Path | None:
    direct = run_root / "training_metrics.jsonl"
    if direct.is_file():
        return direct
    expected = run_root / "native/logs" / run_root.name / "metrics.jsonl"
    if expected.is_file():
        return expected
    candidates = sorted((run_root / "native/logs").glob("*/metrics.jsonl"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            f"ambiguous training metrics for {spec.profile}: "
            + ", ".join(str(path) for path in candidates)
        )
    return None


def read_training_events(path: Path | None) -> dict[int, dict[str, float]]:
    if path is None or not path.is_file():
        return {}
    content = path.read_bytes()
    lines = content.splitlines()
    events: dict[int, dict[str, float]] = {}
    for index, raw_line in enumerate(lines):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not content.endswith(b"\n"):
                break
            raise RuntimeError(f"invalid JSONL record in {path} at line {index + 1}")
        if not isinstance(payload, dict) or "step" not in payload:
            continue
        step = int(payload["step"])
        if not 1 <= step <= MAX_STEP:
            continue
        raw_metrics = payload.get("data", payload.get("metrics"))
        if raw_metrics is None:
            raw_metrics = {
                key: value
                for key, value in payload.items()
                if key not in {"step", "timestamp"}
            }
        if not isinstance(raw_metrics, dict):
            continue
        metrics: dict[str, float] = {}
        for key, value in raw_metrics.items():
            if isinstance(value, bool):
                metrics[str(key)] = float(value)
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[str(key)] = float(value)
        events.setdefault(step, {}).update(metrics)
    return events


def fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def complete_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 2:
        return False
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 4096))
        return handle.read().rstrip().endswith(b"}")


def strict_validate(path: Path, dataset: str, skip: bool) -> None:
    if skip:
        return
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(path),
            "--dataset",
            dataset,
            "--samples",
            str(VAL_N),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(detail or "strict validation failed")


def read_eval_header(path: Path, dataset: str, skip_validation: bool) -> dict[str, Any]:
    if not complete_file(path):
        raise RuntimeError("file is still being written")
    strict_validate(path, dataset, skip_validation)
    with path.open("rb") as handle:
        prefix = handle.read(2_097_152)
    marker = prefix.find(b'"results"')
    if marker < 0:
        raise RuntimeError("missing results field")
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

    found_dataset = string_value("dataset").lower()
    problems = int(number_value("num_problems"))
    samples = int(number_value("val_n"))
    total = int(number_value("total_solutions"))
    if found_dataset != dataset:
        raise RuntimeError(f"dataset={found_dataset}, expected={dataset}")
    if problems != EXPECTED_PROBLEMS[dataset]:
        raise RuntimeError(f"num_problems={problems}")
    if samples != VAL_N or total != problems * VAL_N:
        raise RuntimeError(f"invalid dimensions: problems={problems}, N={samples}, total={total}")
    row = {
        "dataset": dataset,
        "num_problems": problems,
        "total_solutions": total,
        "average_at_n": int(number_value("average_at_n")),
        "pass_at_n": int(number_value("pass_at_n")),
        "majority_vote_at_n": int(number_value("majority_vote_at_n")),
        "formatted_count": int(number_value("formatted_count")),
        "avg_pct": number_value("average_at_n_pct"),
        "pass_pct": number_value("pass_at_n_pct"),
        "majority_pct": number_value("majority_vote_at_n_pct"),
        "format_pct": number_value("format_rate"),
        "fingerprint": fingerprint(path),
    }
    for key in ("avg_pct", "pass_pct", "majority_pct", "format_pct"):
        if not math.isfinite(row[key]) or not 0 <= row[key] <= 100:
            raise RuntimeError(f"invalid {key}={row[key]}")
    return row


def available_evaluations(
    run_root: Path,
    state: dict[str, Any],
    skip_validation: bool,
) -> dict[tuple[int, str], dict[str, Any]]:
    records: dict[tuple[int, str], dict[str, Any]] = {}
    eval_root = run_root / "evaluations"
    uploaded = state.get("evaluations", {})
    for checkpoint in sorted(eval_root.glob("checkpoint-*")):
        match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
        if match is None:
            continue
        step = int(match.group(1))
        if not 1 <= step <= MAX_STEP:
            continue
        for dataset in DATASETS:
            path = checkpoint / f"{dataset}.json"
            if not complete_file(path):
                continue
            state_key = f"{step}:{dataset}"
            current_fingerprint = fingerprint(path)
            if uploaded.get(state_key) == current_fingerprint:
                continue
            try:
                records[(step, dataset)] = read_eval_header(
                    path, dataset, skip_validation
                )
            except (OSError, RuntimeError) as exc:
                print(
                    f"SKIP {run_root.name} checkpoint-{step} {dataset}: {exc}",
                    flush=True,
                )
    return records


def pooled_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    problems = sum(row["num_problems"] for row in rows)
    solutions = sum(row["total_solutions"] for row in rows)
    return {
        "eval/pooled/avg@16_pct": 100
        * sum(row["average_at_n"] for row in rows)
        / solutions,
        "eval/pooled/pass@16_pct": 100
        * sum(row["pass_at_n"] for row in rows)
        / problems,
        "eval/pooled/majority@16_pct": 100
        * sum(row["majority_vote_at_n"] for row in rows)
        / problems,
        "eval/pooled/format_pct": 100
        * sum(row["formatted_count"] for row in rows)
        / solutions,
        "eval/pooled/complete_datasets": float(len(rows)),
    }


def run_id_for(spec: RunSpec) -> str:
    value = f"{RUN_NAMESPACE}:{spec.profile}"
    return "m8" + hashlib.sha256(value.encode()).hexdigest()[:14]


def state_path_for(state_dir: Path, spec: RunSpec) -> Path:
    return state_dir / "runs" / f"{spec.profile}.json"


def load_state(path: Path, run_id: str) -> dict[str, Any]:
    empty = {"run_id": run_id, "training_steps": [], "evaluations": {}, "pooled": {}}
    if not path.is_file():
        return empty
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    return state if state.get("run_id") == run_id else empty


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def config_for(spec: RunSpec, run_root: Path, parameters: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": f"Qwen3-{spec.model_size}",
        "method": spec.method_label,
        "hardware": f"8x{spec.hardware}",
        "benchmark": "Math",
        "seed": 0,
        "physical_stop_step": MAX_STEP,
        "scheduler_horizon_steps": 420,
        "evaluation_frequency": 5,
        "evaluation_samples_per_question": VAL_N,
        "evaluation_datasets": list(DATASETS),
        "source": "local_file_and_evaluation_json",
        "source_root": str(run_root),
    }
    config.update(parameters)
    return config


def train_record(metrics: dict[str, float], step: int) -> dict[str, float]:
    record: dict[str, float] = {"train/step": float(step)}
    for key, value in metrics.items():
        if key in {"step", "global_step", "training/global_step"}:
            continue
        record[f"train/{key}"] = value
    return record


def eval_record(row: dict[str, Any], step: int) -> dict[str, float]:
    prefix = f"eval/{row['dataset']}"
    return {
        "eval/step": float(step),
        f"{prefix}/avg@16_pct": row["avg_pct"],
        f"{prefix}/pass@16_pct": row["pass_pct"],
        f"{prefix}/majority@16_pct": row["majority_pct"],
        f"{prefix}/format_pct": row["format_pct"],
        f"{prefix}/num_problems": float(row["num_problems"]),
        f"{prefix}/total_solutions": float(row["total_solutions"]),
    }


def upload_spec(
    spec: RunSpec,
    *,
    entity: str,
    project: str,
    state_dir: Path,
    dry_run: bool,
    skip_validation: bool,
) -> tuple[int, int]:
    run_root = find_run_root(spec)
    if run_root is None:
        print(f"{spec.display_name}: WAITING, output directory not created")
        return 0, 0
    run_id = run_id_for(spec)
    state_path = state_path_for(state_dir, spec)
    state = load_state(state_path, run_id)
    metrics_path = training_metrics_path(run_root, spec)
    training = read_training_events(metrics_path)
    uploaded_steps = {int(step) for step in state.get("training_steps", [])}
    pending_training = {
        step: metrics for step, metrics in training.items() if step not in uploaded_steps
    }
    pending_eval = available_evaluations(run_root, state, skip_validation)

    pooled_pending: dict[int, tuple[list[dict[str, Any]], str]] = {}
    candidate_steps = sorted(
        {step for step, _ in pending_eval}
        | {
            int(path.name.rsplit("-", 1)[-1])
            for path in (run_root / "evaluations").glob("checkpoint-*")
            if path.name.rsplit("-", 1)[-1].isdigit()
        }
    )
    for step in candidate_steps:
        paths = [
            run_root
            / "evaluations"
            / f"checkpoint-{step}"
            / f"{dataset}.json"
            for dataset in DATASETS
        ]
        if any(not complete_file(path) for path in paths):
            continue
        combined_fingerprint = "|".join(fingerprint(path) for path in paths)
        if state.get("pooled", {}).get(str(step)) == combined_fingerprint:
            continue
        rows: list[dict[str, Any]] = []
        for dataset, path in zip(DATASETS, paths, strict=True):
            row = pending_eval.get((step, dataset))
            if row is None:
                try:
                    row = read_eval_header(path, dataset, skip_validation)
                except (OSError, RuntimeError):
                    rows = []
                    break
            rows.append(row)
        if len(rows) == len(DATASETS):
            pooled_pending[step] = (rows, combined_fingerprint)

    latest_step = max(training, default=0)
    completed_eval_files = len(
        set(state.get("evaluations", {}))
        | {f"{step}:{dataset}" for step, dataset in pending_eval}
    )
    print(
        f"{spec.display_name}: local_train_step={latest_step}, "
        f"new_train={len(pending_training)}, new_eval={len(pending_eval)}, "
        f"new_pooled={len(pooled_pending)}, eval_files_seen={completed_eval_files}/100"
    )
    pending_count = len(pending_training) + len(pending_eval) + len(pooled_pending)
    if dry_run or pending_count == 0:
        return pending_count, 0

    import wandb

    parameters = parse_parameters(run_root / "state/parameters.env")
    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        name=spec.display_name,
        resume="allow",
        config=config_for(spec, run_root, parameters),
        group="Math-eval5-N16-seed0-20260812",
        job_type="train-eval",
        tags=[
            "math",
            "eval5",
            "N16",
            "seed0",
            f"Qwen3-{spec.model_size}",
            spec.method_label.lower().replace("-", "_"),
            spec.hardware.lower(),
        ],
    )
    if run is None:
        raise RuntimeError("wandb.init returned None")
    try:
        run.define_metric("train/step")
        run.define_metric("eval/step")
        for step, metrics in sorted(pending_training.items()):
            record = train_record(metrics, step)
            for key in record:
                if key != "train/step":
                    run.define_metric(key, step_metric="train/step")
            run.log(record)
            state.setdefault("training_steps", []).append(step)
            state["training_steps"] = sorted(set(state["training_steps"]))
            save_state(state_path, state)

        for (step, dataset), row in sorted(pending_eval.items()):
            record = eval_record(row, step)
            for key in record:
                if key != "eval/step":
                    run.define_metric(key, step_metric="eval/step")
            run.log(record)
            state.setdefault("evaluations", {})[f"{step}:{dataset}"] = row[
                "fingerprint"
            ]
            save_state(state_path, state)

        for step, (rows, combined_fingerprint) in sorted(pooled_pending.items()):
            record = {"eval/step": float(step), **pooled_metrics(rows)}
            for key in record:
                if key != "eval/step":
                    run.define_metric(key, step_metric="eval/step")
            run.log(record)
            state.setdefault("pooled", {})[str(step)] = combined_fingerprint
            save_state(state_path, state)

        run.summary["progress/latest_train_step"] = latest_step
        run.summary["progress/uploaded_eval_files"] = len(
            state.get("evaluations", {})
        )
        run.summary["progress/complete_eval_checkpoints"] = len(
            state.get("pooled", {})
        )
        run.summary["progress/pipeline_complete"] = int(
            (run_root / "state/complete").is_file()
        )
    finally:
        run.finish()
    return pending_count, 0


def main() -> None:
    args = parse_args()
    failures = 0
    selected = 0
    specs = run_specs(args.h200_root, args.a800_root)
    for spec in specs:
        try:
            pending, _ = upload_spec(
                spec,
                entity=args.entity,
                project=args.project,
                state_dir=args.state_dir,
                dry_run=args.dry_run,
                skip_validation=args.skip_strict_validation,
            )
            selected += pending
        except Exception as exc:
            failures += 1
            print(
                f"ERROR {spec.display_name}: {type(exc).__name__}: {exc}",
                flush=True,
            )
    print(f"runs={len(specs)}, events_selected={selected}, failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
