#!/usr/bin/env python3
"""Upload the completed Physics rho x self-reference grid to W&B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path(
    "/media/vlm-ckp-fileset/ylong/sdpo_physics_rho_selfref_grid_eval5_nockpt"
)
EXPECTED_GRID = {
    (self_reference, rho)
    for self_reference in ("0.5", "0.7")
    for rho in ("0.5", "0.7", "0.9", "0.95")
}
REQUIRED_EVAL_KEYS = (
    "actor/entropy",
    "val-core/sciknoweval/acc/mean@16",
    "val-core/sciknoweval/acc/best@16/mean",
    "val-core/sciknoweval/acc/maj@16/mean",
)
PREFERRED_CSV_KEYS = (
    "training/global_step",
    "training/cumulative_step_time_s",
    "actor/entropy",
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/grad_norm",
    "actor/lr",
    "val-core/sciknoweval/acc/mean@16",
    "val-core/sciknoweval/acc/best@16/mean",
    "val-core/sciknoweval/acc/maj@16/mean",
    "val-aux/sciknoweval/incorrect_format/mean@16",
    "response_length/mean",
    "response_length/clip_ratio",
    "perf/time_per_step",
    "perf/throughput",
)


@dataclass(frozen=True)
class RunSource:
    run_name: str
    log_dir: Path
    metrics_path: Path
    validation_dir: Path
    self_reference: str
    rho: str
    selector_alpha: str
    seed: int

    @property
    def display_name(self) -> str:
        return (
            "Qwen3-8B-Physics-SR-OPSD-"
            f"seed{self.seed}-selfref{self.self_reference}-rho{self.rho}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--entity", default="wenxuan-yuan-imperial-college-london"
    )
    parser.add_argument("--project", default="sdpo_ablation_physics")
    parser.add_argument("--total-steps", type=int, default=420)
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument("--validation-lines", type=int, default=1280)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Forget local upload state. Existing W&B runs retain deterministic IDs.",
    )
    return parser.parse_args()


def token(run_name: str, label: str) -> str:
    match = re.search(rf"(?:^|-){re.escape(label)}([^-]+)(?:-|$)", run_name)
    if not match:
        raise ValueError(f"{run_name}: missing {label} token")
    return match.group(1)


def parse_source(metrics_path: Path) -> RunSource:
    run_name = metrics_path.parent.name
    if "physics-sr_opsd_forward_renyi-refTrue" not in run_name:
        raise ValueError(f"unexpected run name: {run_name}")
    return RunSource(
        run_name=run_name,
        log_dir=metrics_path.parent,
        metrics_path=metrics_path,
        validation_dir=metrics_path.parent / "validation",
        self_reference=token(run_name, "selfref"),
        rho=token(run_name, "rho"),
        selector_alpha=token(run_name, "selectorAlpha"),
        seed=int(token(run_name, "seed")),
    )


def read_events(metrics_path: Path) -> dict[int, dict[str, Any]]:
    content = metrics_path.read_bytes()
    if content and not content.endswith(b"\n"):
        content = content.rsplit(b"\n", 1)[0] + b"\n"

    events: dict[int, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
            step = int(payload["step"])
            data = payload["data"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{metrics_path}:{line_number}: invalid metrics event"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(
                f"{metrics_path}:{line_number}: data is not an object"
            )
        events[step] = data
    return events


def count_nonempty_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def add_readable_aliases(data: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in data.items() if is_scalar(value)}
    aliases = {
        "physics/avg@16": "val-core/sciknoweval/acc/mean@16",
        "physics/pass@16": "val-core/sciknoweval/acc/best@16/mean",
        "physics/majority@16": "val-core/sciknoweval/acc/maj@16/mean",
    }
    for alias, original in aliases.items():
        if original in payload:
            payload[alias] = payload[original]

    incorrect_key = "val-aux/sciknoweval/incorrect_format/mean@16"
    incorrect = payload.get(incorrect_key)
    if isinstance(incorrect, (int, float)):
        payload["physics/format@16"] = 1.0 - float(incorrect)
    return payload


def write_eval_csv(
    source: RunSource,
    events: dict[int, dict[str, Any]],
    eval_steps: list[int],
) -> Path:
    output_path = source.log_dir / "eval5_metrics.csv"
    all_keys = {
        key
        for step in eval_steps
        for key, value in add_readable_aliases(events[step]).items()
        if is_scalar(value)
    }
    ordered = [key for key in PREFERRED_CSV_KEYS if key in all_keys]
    ordered.extend(
        key
        for key in (
            "physics/avg@16",
            "physics/pass@16",
            "physics/majority@16",
            "physics/format@16",
        )
        if key in all_keys
    )
    ordered.extend(sorted(all_keys - set(ordered)))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", *ordered])
        writer.writeheader()
        for step in eval_steps:
            payload = add_readable_aliases(events[step])
            row = {"step": step}
            row.update({key: payload.get(key) for key in ordered})
            writer.writerow(row)
    return output_path


def validate_source(
    source: RunSource,
    *,
    total_steps: int,
    eval_freq: int,
    validation_lines: int,
) -> tuple[dict[int, dict[str, Any]], Path]:
    if source.seed != 0:
        raise RuntimeError(f"{source.run_name}: expected seed 0")
    if source.selector_alpha != "0.25":
        raise RuntimeError(
            f"{source.run_name}: expected selector alpha 0.25"
        )

    events = read_events(source.metrics_path)
    expected_steps = set(range(1, total_steps + 1))
    actual_steps = set(events)
    if actual_steps != expected_steps:
        missing = sorted(expected_steps - actual_steps)
        extra = sorted(actual_steps - expected_steps)
        raise RuntimeError(
            f"{source.run_name}: incomplete training metrics; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    eval_steps = list(range(eval_freq, total_steps + 1, eval_freq))
    for step in eval_steps:
        missing_keys = [key for key in REQUIRED_EVAL_KEYS if key not in events[step]]
        if missing_keys:
            raise RuntimeError(
                f"{source.run_name}: step {step} missing {missing_keys}"
            )
        validation_path = source.validation_dir / f"{step}.jsonl"
        if not validation_path.is_file():
            raise RuntimeError(f"missing validation file: {validation_path}")
        actual_lines = count_nonempty_lines(validation_path)
        if actual_lines != validation_lines:
            raise RuntimeError(
                f"{validation_path}: expected {validation_lines} nonempty lines, "
                f"found {actual_lines}"
            )

    return events, write_eval_csv(source, events, eval_steps)


def discover_candidates(
    output_root: Path,
) -> dict[tuple[str, str], list[RunSource]]:
    logs_root = output_root / "logs"
    if not logs_root.is_dir():
        raise RuntimeError(f"logs directory does not exist: {logs_root}")

    candidates: dict[tuple[str, str], list[RunSource]] = {}
    for metrics_path in sorted(logs_root.glob("*/metrics.jsonl")):
        try:
            source = parse_source(metrics_path)
        except ValueError:
            continue
        key = (source.self_reference, source.rho)
        if key in EXPECTED_GRID:
            candidates.setdefault(key, []).append(source)

    missing = sorted(EXPECTED_GRID - set(candidates))
    if missing:
        raise RuntimeError(f"missing grid points: {missing}")
    return candidates


def run_id_for(project: str, source: RunSource) -> str:
    material = f"{project}:{source.run_name}"
    return "phys" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def state_path_for(state_dir: Path, source: RunSource) -> Path:
    safe_name = (
        f"selfref{source.self_reference}_rho{source.rho}".replace(".", "p")
    )
    return state_dir / f"{safe_name}.json"


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def config_for(source: RunSource) -> dict[str, Any]:
    return {
        "model": "Qwen3-8B",
        "dataset": "SciKnowEval/physics",
        "method": "SR-OPSD",
        "projection_objective": "Forward Renyi",
        "seed": source.seed,
        "rho": float(source.rho),
        "self_reference_coefficient": float(source.self_reference),
        "self_reference_config_key": (
            "actor_rollout_ref.actor.self_distillation."
            "renyi_regularization_level"
        ),
        "objective_selector_alpha": float(source.selector_alpha),
        "reference_policy": True,
        "reference_sync_steps": 0,
        "teacher_update_rate": 0.05,
        "distillation_topk": 100,
        "distillation_tail_bucket": True,
        "distillation_is_clip": 2.0,
        "entropy_coeff": 1e-5,
        "learning_rate": 1e-5,
        "warmup_steps": 10,
        "train_batch_size": 32,
        "ppo_mini_batch_size": 32,
        "training_rollouts": 8,
        "training_temperature": 1.0,
        "training_top_p": 1.0,
        "training_top_k": -1,
        "max_prompt_length": 2048,
        "max_response_length": 8192,
        "validation_frequency": 5,
        "validation_samples": 16,
        "validation_temperature": 0.6,
        "validation_top_p": 0.95,
        "total_training_steps": 420,
        "checkpoint_saving": False,
        "source_run_name": source.run_name,
        "source_metrics_path": str(source.metrics_path),
    }


def best_summary(events: dict[int, dict[str, Any]]) -> dict[str, Any]:
    avg_key = "val-core/sciknoweval/acc/mean@16"
    candidates = [
        (step, add_readable_aliases(data))
        for step, data in sorted(events.items())
        if avg_key in data
    ]
    if not candidates:
        return {}
    best_step, best = max(candidates, key=lambda item: float(item[1][avg_key]))
    summary: dict[str, Any] = {"best/step": best_step}
    for key in (
        "physics/avg@16",
        "physics/pass@16",
        "physics/majority@16",
        "physics/format@16",
        "training/cumulative_step_time_s",
        "actor/entropy",
    ):
        if key in best:
            summary[f"best/{key.split('/', 1)[-1]}"] = best[key]
    return summary


def upload_source(
    *,
    source: RunSource,
    events: dict[int, dict[str, Any]],
    eval_csv: Path,
    entity: str,
    project: str,
    state_dir: Path,
    dry_run: bool,
) -> None:
    state_path = state_path_for(state_dir, source)
    state = load_state(state_path)
    last_step = int(state.get("last_step", 0))
    artifact_logged = bool(state.get("artifact_logged", False))
    pending = [step for step in sorted(events) if step > last_step]
    summary = best_summary(events)

    print(
        f"{source.display_name}: local_step={max(events)}, "
        f"uploaded_step={last_step}, pending={len(pending)}, "
        f"best_step={summary.get('best/step')}, "
        f"best_avg16={summary.get('best/avg@16')}"
    )
    if dry_run or (not pending and artifact_logged):
        return

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "The selected Python environment does not contain wandb. "
            "Install it with: python -m pip install wandb"
        ) from exc

    run_id = run_id_for(project, source)
    print(
        f"CONNECTING {source.display_name} to W&B "
        f"({entity}/{project}, id={run_id})",
        flush=True,
    )
    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        name=source.display_name,
        resume="allow",
        group="Qwen3-8B-Physics-SR-OPSD-rho-selfref-grid-seed0",
        job_type="ablation",
        tags=[
            "physics",
            "Qwen3-8B",
            "SR-OPSD",
            "forward-renyi",
            f"rho-{source.rho}",
            f"selfref-{source.self_reference}",
            "seed-0",
        ],
        config=config_for(source),
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None:
        raise RuntimeError(f"wandb.init returned no run for {source.run_name}")

    uploaded_step = last_step
    uploaded_artifact = artifact_logged
    try:
        print(f"CONNECTED {source.display_name}: {run.url}", flush=True)
        run.define_metric("training/global_step")
        run.define_metric("*", step_metric="training/global_step")
        for index, step in enumerate(pending, start=1):
            payload = add_readable_aliases(events[step])
            payload["training/global_step"] = step
            run.log(payload, step=step)
            uploaded_step = step
            if index == 1 or index % 50 == 0 or index == len(pending):
                print(
                    f"UPLOADING {source.display_name}: "
                    f"event {index}/{len(pending)}, step={step}",
                    flush=True,
                )

        for key, value in summary.items():
            run.summary[key] = value

        if not artifact_logged:
            print(
                f"ARTIFACT {source.display_name}: attaching metrics.jsonl "
                "and eval5_metrics.csv",
                flush=True,
            )
            artifact_name = (
                f"physics-grid-selfref{source.self_reference}-rho{source.rho}"
                .replace(".", "p")
            )
            artifact = wandb.Artifact(
                name=artifact_name,
                type="evaluation-results",
                metadata={
                    "source_run_name": source.run_name,
                    "self_reference_coefficient": source.self_reference,
                    "rho": source.rho,
                    "eval_frequency": 5,
                    "validation_samples": 16,
                },
            )
            artifact.add_file(str(eval_csv), name="eval5_metrics.csv")
            artifact.add_file(str(source.metrics_path), name="metrics.jsonl")
            run.log_artifact(artifact)
            uploaded_artifact = True
        print(f"SYNCING {source.display_name} with W&B", flush=True)
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise

    save_state(
        state_path,
        {
            "artifact_logged": uploaded_artifact,
            "entity": entity,
            "last_step": uploaded_step,
            "project": project,
            "run_id": run_id,
            "run_name": source.display_name,
            "source_run_name": source.run_name,
        },
    )
    print(
        f"UPLOADED {source.display_name}: step={uploaded_step}, "
        f"artifact={uploaded_artifact}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    state_dir = args.output_root / "wandb_upload_state"
    if args.reset_state and state_dir.exists():
        for path in state_dir.glob("*.json"):
            path.unlink()

    candidates = discover_candidates(args.output_root)
    print(f"Discovered all {len(candidates)} expected grid points")

    validated: dict[tuple[str, str], tuple[RunSource, dict[int, dict[str, Any]], Path]] = {}
    for key in sorted(candidates, key=lambda item: (float(item[0]), float(item[1]))):
        ordered = sorted(
            candidates[key],
            key=lambda item: item.metrics_path.stat().st_mtime,
            reverse=True,
        )
        errors: list[str] = []
        for source in ordered:
            print(
                f"VALIDATING selfref={key[0]} rho={key[1]} "
                f"source={source.run_name}",
                flush=True,
            )
            try:
                events, eval_csv = validate_source(
                    source,
                    total_steps=args.total_steps,
                    eval_freq=args.eval_freq,
                    validation_lines=args.validation_lines,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(f"{source.run_name}: {exc}")
                continue
            validated[key] = (source, events, eval_csv)
            print(
                f"COMPLETE selfref={key[0]} rho={key[1]} "
                f"events={len(events)} source={source.run_name}"
            )
            break
        if key not in validated:
            detail = "\n  ".join(errors)
            raise RuntimeError(
                f"no complete run for selfref={key[0]} rho={key[1]}:\n  {detail}"
            )

    for key in sorted(validated, key=lambda item: (float(item[0]), float(item[1]))):
        source, events, eval_csv = validated[key]
        upload_source(
            source=source,
            events=events,
            eval_csv=eval_csv,
            entity=args.entity,
            project=args.project,
            state_dir=state_dir,
            dry_run=args.dry_run,
        )

    mode = "dry-run validation" if args.dry_run else "W&B upload"
    print(f"SUCCESS: {mode} completed for all {len(validated)} Physics runs")


if __name__ == "__main__":
    main()
