"""Convert distIL's math JSONL into the parquet schema expected by VERL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset


INSTRUCTION = "Let's think step by step and put the final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-size", type=int, default=32)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            problem = raw["problem"].strip()
            answer = str(raw["answer"]).strip()
            rows.append(
                {
                    "data_source": "math",
                    "prompt": [{"role": "user", "content": f"{problem}\n\n{INSTRUCTION}"}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {
                        "split": "train",
                        "index": index,
                        "reference_solution": raw.get("solution", ""),
                        "source": raw.get("source", ""),
                    },
                }
            )
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_size = min(max(args.validation_size, 1), len(rows))
    validation_rows = []
    for row in rows[:validation_size]:
        copied = dict(row)
        copied["extra_info"] = dict(row["extra_info"], split="validation")
        validation_rows.append(copied)

    Dataset.from_list(rows).to_parquet(args.output_dir / "train.parquet")
    Dataset.from_list(validation_rows).to_parquet(args.output_dir / "test.parquet")
    manifest = {
        "source": str(args.input.resolve()),
        "train_records": len(rows),
        "validation_records": len(validation_rows),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
