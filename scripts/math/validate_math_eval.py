"""Validate that a math evaluation JSON contains every problem and sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_PROBLEMS = {"aime24": 30, "aime25": 30, "hmmt25": 30, "amc23": 40, "minerva": 272}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--dataset", required=True, choices=EXPECTED_PROBLEMS)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    expected_problems = EXPECTED_PROBLEMS[args.dataset]
    expected_solutions = expected_problems * args.samples
    errors = []
    if payload.get("dataset") != args.dataset:
        errors.append(f"dataset={payload.get('dataset')!r}")
    if payload.get("num_problems") != expected_problems:
        errors.append(f"num_problems={payload.get('num_problems')!r}")
    if payload.get("val_n") != args.samples:
        errors.append(f"val_n={payload.get('val_n')!r}")
    if payload.get("total_solutions") != expected_solutions:
        errors.append(f"total_solutions={payload.get('total_solutions')!r}")
    results = payload.get("results", [])
    if len(results) != expected_problems:
        errors.append(f"result_rows={len(results)}")
    for index, result in enumerate(results):
        if result.get("val_n") != args.samples:
            errors.append(f"results[{index}].val_n={result.get('val_n')!r}")
        generations = result.get("generations", [])
        if len(generations) != args.samples:
            errors.append(f"results[{index}].generations={len(generations)}")
        if len(errors) >= 10:
            errors.append("additional row errors omitted")
            break
    if errors:
        raise SystemExit("incomplete evaluation: " + ", ".join(errors))
    print(
        f"OK {args.dataset}: problems={expected_problems}, samples={args.samples}, "
        f"avg={payload['average_at_n_pct']:.2f}, pass={payload['pass_at_n_pct']:.2f}, "
        f"majority={payload['majority_vote_at_n_pct']:.2f}, format={payload['format_rate']:.2f}"
    )


if __name__ == "__main__":
    main()
