#!/usr/bin/env python3
"""Validate an eval-every-5 Physics run and write a compact metrics CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = (
    "actor/entropy",
    "val-core/sciknoweval/acc/mean@16",
    "val-core/sciknoweval/acc/best@16/mean",
    "val-core/sciknoweval/acc/maj@16/mean",
)

PREFERRED_KEYS = (
    "training/global_step",
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
    "training/cumulative_step_time_s",
    "perf/time_per_step",
    "perf/throughput",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=420)
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument("--expected-validation-lines", type=int, default=1280)
    return parser.parse_args()


def scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def main() -> None:
    args = parse_args()
    events: dict[int, dict[str, Any]] = {}

    with args.metrics_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            step = int(payload["step"])
            data = payload["data"]
            if not isinstance(data, dict):
                raise ValueError(f"line {line_number}: data is not an object")
            events[step] = data

    expected_train_steps = set(range(1, args.total_steps + 1))
    if set(events) != expected_train_steps:
        missing = sorted(expected_train_steps - set(events))
        extra = sorted(set(events) - expected_train_steps)
        raise ValueError(f"metrics steps incomplete; missing={missing[:10]}, extra={extra[:10]}")

    eval_steps = list(range(args.eval_freq, args.total_steps + 1, args.eval_freq))
    for step in eval_steps:
        missing_keys = [key for key in REQUIRED_KEYS if key not in events[step]]
        if missing_keys:
            raise ValueError(f"step {step}: missing metrics {missing_keys}")

        validation_path = args.validation_dir / f"{step}.jsonl"
        if not validation_path.is_file():
            raise FileNotFoundError(validation_path)
        with validation_path.open("rb") as handle:
            line_count = sum(1 for line in handle if line.strip())
        if line_count != args.expected_validation_lines:
            raise ValueError(
                f"{validation_path}: expected {args.expected_validation_lines} lines, got {line_count}"
            )

    all_keys = {
        key
        for step in eval_steps
        for key, value in events[step].items()
        if scalar(value)
    }
    ordered_keys = [key for key in PREFERRED_KEYS if key in all_keys]
    ordered_keys.extend(sorted(all_keys - set(ordered_keys)))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", *ordered_keys])
        writer.writeheader()
        for step in eval_steps:
            row = {"step": step}
            row.update({key: events[step].get(key) for key in ordered_keys})
            writer.writerow(row)

    print(f"validated_training_steps={len(events)}")
    print(f"validated_eval_steps={len(eval_steps)}")
    print(f"metrics_csv={args.output_csv}")


if __name__ == "__main__":
    main()
