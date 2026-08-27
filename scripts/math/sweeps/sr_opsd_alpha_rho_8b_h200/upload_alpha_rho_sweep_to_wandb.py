#!/usr/bin/env python3
"""Incrementally upload the five Qwen3-8B SR-OPSD alpha/rho runs to W&B."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MATH_SCRIPTS = SCRIPT_DIR.parents[1]
COMMON_UPLOADER_DIR = MATH_SCRIPTS / "wandb_math_8runs"
sys.path.insert(0, str(COMMON_UPLOADER_DIR))

import upload_math_8runs_to_wandb as common  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(
    "/media/vlm-ckp-fileset/ylong/sr_opsd_math_alpha_rho_sweep_20260819"
)
GRID = (
    ("0.9", "0.7"),
    ("0.9", "0.9"),
    ("0.7", "0.7"),
    ("0.7", "0.9"),
    ("0.7", "0.95"),
)
RUN_NAMESPACE = "sr-opsd-math-alpha-rho-n16-h200-20260819-v1"


@dataclass(frozen=True)
class SweepSpec:
    output_root: Path
    alpha: str
    rho: str
    variant: str = "original"
    display_suffix: str = ""

    @property
    def method_dir(self) -> str:
        return "sr_opsd"

    @property
    def run_name(self) -> str:
        return (
            "sr-opsd-8b-seed0-native-verl-forward-renyi-"
            f"rho{self.rho}-refw{self.alpha}-sync0-ema0.05-lr5e-6-"
            "trainbs8-mbs8-rolloutn8-topk100-tailFalse-clip0.05-"
            "temp0.7-tok16384-steps100-sched420-eval5-n16-h200"
        )

    @property
    def run_root(self) -> Path:
        return self.output_root / self.method_dir / self.run_name

    @property
    def profile(self) -> str:
        base = f"8b-sr-opsd-alpha{self.alpha}-rho{self.rho}-h200"
        return base if self.variant == "original" else f"{base}-{self.variant}"

    @property
    def display_name(self) -> str:
        base = (
            "Qwen3-8B-Math-SR-OPSD-"
            f"alpha{self.alpha}-rho{self.rho}-seed0-eval5-N16-H200"
        )
        return base if not self.display_suffix else f"{base}-{self.display_suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--entity", default="wenxuan-yuan-imperial-college-london"
    )
    parser.add_argument("--project", default="SDPO_math_test")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument(
        "--variant",
        default="original",
        help="Stable run-ID namespace for a distinct evaluation protocol.",
    )
    parser.add_argument(
        "--display-suffix",
        default="",
        help="Suffix appended to W&B names for a distinct protocol.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-strict-validation",
        action="store_true",
        help="Only for local fixtures; never use for real uploads.",
    )
    return parser.parse_args()


def run_id_for(spec: SweepSpec) -> str:
    if spec.variant == "original":
        payload = f"{RUN_NAMESPACE}:{spec.profile}"
    else:
        payload = f"{RUN_NAMESPACE}:{spec.variant}:{spec.profile}"
    return "ar" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:14]


def config_for(
    spec: SweepSpec,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": "Qwen3-8B",
        "method": "SR-OPSD",
        "benchmark": "Math",
        "hardware": "8xH200",
        "seed": 0,
        "self_reference_alpha": float(spec.alpha),
        "self_reference_weight": float(spec.alpha),
        "renyi_order_rho": float(spec.rho),
        "physical_stop_step": common.MAX_STEP,
        "scheduler_horizon_steps": 420,
        "evaluation_frequency": 5,
        "evaluation_samples_per_question": common.VAL_N,
        "evaluation_datasets": list(common.DATASETS),
        "source": "local_file_and_evaluation_json",
        "source_root": str(spec.run_root),
        "evaluation_variant": spec.variant,
    }
    config.update(parameters)
    return config


def completed_eval_count(run_root: Path) -> int:
    count = 0
    for checkpoint in (run_root / "evaluations").glob("checkpoint-*"):
        for dataset in common.DATASETS:
            if common.complete_file(checkpoint / f"{dataset}.json"):
                count += 1
    return count


def build_pooled_pending(
    run_root: Path,
    state: dict[str, Any],
    pending_eval: dict[tuple[int, str], dict[str, Any]],
    skip_validation: bool,
) -> dict[int, tuple[list[dict[str, Any]], str]]:
    pooled: dict[int, tuple[list[dict[str, Any]], str]] = {}
    for checkpoint in sorted((run_root / "evaluations").glob("checkpoint-*")):
        suffix = checkpoint.name.rsplit("-", 1)[-1]
        if not suffix.isdigit():
            continue
        step = int(suffix)
        if not 1 <= step <= common.MAX_STEP:
            continue
        paths = [checkpoint / f"{dataset}.json" for dataset in common.DATASETS]
        if any(not common.complete_file(path) for path in paths):
            continue
        fingerprint = "|".join(common.fingerprint(path) for path in paths)
        if state.get("pooled", {}).get(str(step)) == fingerprint:
            continue
        rows: list[dict[str, Any]] = []
        for dataset, path in zip(common.DATASETS, paths, strict=True):
            row = pending_eval.get((step, dataset))
            if row is None:
                row = common.read_eval_header(path, dataset, skip_validation)
            rows.append(row)
        pooled[step] = (rows, fingerprint)
    return pooled


def best_available_summaries(run_root: Path, skip_validation: bool) -> dict[str, float]:
    by_dataset: dict[str, list[tuple[int, dict[str, Any]]]] = {
        dataset: [] for dataset in common.DATASETS
    }
    pooled: list[tuple[int, dict[str, float]]] = []
    for checkpoint in sorted((run_root / "evaluations").glob("checkpoint-*")):
        suffix = checkpoint.name.rsplit("-", 1)[-1]
        if not suffix.isdigit():
            continue
        step = int(suffix)
        rows: list[dict[str, Any]] = []
        for dataset in common.DATASETS:
            path = checkpoint / f"{dataset}.json"
            if not common.complete_file(path):
                continue
            try:
                row = common.read_eval_header(path, dataset, skip_validation)
            except (OSError, RuntimeError):
                continue
            by_dataset[dataset].append((step, row))
            rows.append(row)
        if len(rows) == len(common.DATASETS):
            pooled.append((step, common.pooled_metrics(rows)))

    summary: dict[str, float] = {}
    for dataset, candidates in by_dataset.items():
        if not candidates:
            continue
        step, row = max(candidates, key=lambda item: item[1]["avg_pct"])
        summary[f"best/{dataset}/step"] = float(step)
        summary[f"best/{dataset}/avg@16_pct"] = float(row["avg_pct"])
        summary[f"best/{dataset}/pass@16_pct"] = float(row["pass_pct"])
        summary[f"best/{dataset}/majority@16_pct"] = float(row["majority_pct"])
    if pooled:
        step, metrics = max(
            pooled, key=lambda item: item[1]["eval/pooled/avg@16_pct"]
        )
        summary["best/pooled/step"] = float(step)
        for key, value in metrics.items():
            if key == "eval/pooled/complete_datasets":
                continue
            summary[f"best/pooled/{key.rsplit('/', 1)[-1]}"] = float(value)
    return summary


def upload_spec(
    spec: SweepSpec,
    *,
    entity: str,
    project: str,
    state_dir: Path,
    dry_run: bool,
    skip_validation: bool,
) -> tuple[int, int]:
    run_root = spec.run_root
    if not run_root.is_dir():
        print(f"{spec.display_name}: NOT_STARTED, missing={run_root}")
        return 0, 0

    run_id = run_id_for(spec)
    state_path = common.state_path_for(state_dir, spec)
    state = common.load_state(state_path, run_id)
    metrics_path = common.training_metrics_path(run_root, spec)
    training = common.read_training_events(metrics_path)
    uploaded_training = {int(step) for step in state.get("training_steps", [])}
    pending_training = {
        step: metrics
        for step, metrics in training.items()
        if step not in uploaded_training
    }
    pending_eval = common.available_evaluations(
        run_root, state, skip_validation
    )
    pooled_pending = build_pooled_pending(
        run_root, state, pending_eval, skip_validation
    )

    latest_train_step = max(training, default=0)
    eval_files = completed_eval_count(run_root)
    pending_count = (
        len(pending_training) + len(pending_eval) + len(pooled_pending)
    )
    print(
        f"{spec.display_name}: local_train_step={latest_train_step}, "
        f"new_train={len(pending_training)}, new_eval={len(pending_eval)}, "
        f"new_pooled={len(pooled_pending)}, eval_files={eval_files}/100"
    )
    if dry_run or pending_count == 0:
        return pending_count, 0

    import wandb

    parameters = common.parse_parameters(run_root / "state/parameters.env")
    group = "Qwen3-8B-Math-SR-OPSD-alpha-rho-N16-seed0"
    tags = [
        "math",
        "alpha-rho-sweep",
        "SR-OPSD",
        "Qwen3-8B",
        "H200",
        "eval5",
        "N16",
        "seed0",
        f"alpha-{spec.alpha}",
        f"rho-{spec.rho}",
    ]
    if spec.variant != "original":
        group = f"{group}-{spec.variant}"
        tags.extend(["evaluation-variant", spec.variant])
    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        name=spec.display_name,
        resume="allow",
        group=group,
        job_type="hyperparameter-sweep",
        tags=tags,
        config=config_for(spec, parameters),
        settings=wandb.Settings(init_timeout=180),
    )
    if run is None:
        raise RuntimeError("wandb.init returned None")

    try:
        run.define_metric("train/step")
        run.define_metric("eval/step")
        for step, metrics in sorted(pending_training.items()):
            record = common.train_record(metrics, step)
            for key in record:
                if key != "train/step":
                    run.define_metric(key, step_metric="train/step")
            run.log(record)
            state.setdefault("training_steps", []).append(step)
            state["training_steps"] = sorted(set(state["training_steps"]))
            common.save_state(state_path, state)

        for (step, dataset), row in sorted(pending_eval.items()):
            record = common.eval_record(row, step)
            for key in record:
                if key != "eval/step":
                    run.define_metric(key, step_metric="eval/step")
            run.log(record)
            state.setdefault("evaluations", {})[f"{step}:{dataset}"] = row[
                "fingerprint"
            ]
            common.save_state(state_path, state)

        for step, (rows, fingerprint) in sorted(pooled_pending.items()):
            record = {"eval/step": float(step), **common.pooled_metrics(rows)}
            for key in record:
                if key != "eval/step":
                    run.define_metric(key, step_metric="eval/step")
            run.log(record)
            state.setdefault("pooled", {})[str(step)] = fingerprint
            common.save_state(state_path, state)

        for key, value in best_available_summaries(
            run_root, skip_validation
        ).items():
            if math.isfinite(value):
                run.summary[key] = value
        run.summary["progress/latest_train_step"] = latest_train_step
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
    state_dir = args.state_dir or (
        args.output_root / "wandb_upload_state_sdpo_math_test"
    )
    failures = 0
    selected = 0
    for alpha, rho in GRID:
        spec = SweepSpec(
            args.output_root,
            alpha,
            rho,
            variant=args.variant,
            display_suffix=args.display_suffix,
        )
        try:
            pending, _ = upload_spec(
                spec,
                entity=args.entity,
                project=args.project,
                state_dir=state_dir,
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
    mode = "dry-run" if args.dry_run else "upload"
    print(
        f"{mode}: runs={len(GRID)}, events_selected={selected}, "
        f"failures={failures}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
