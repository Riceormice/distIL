#!/usr/bin/env python3
"""Remove model checkpoints only when matching evaluation evidence exists."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


MATH_DATASETS = {"aime24", "aime25", "hmmt25", "amc23", "minerva"}
STEP_DIR_RE = re.compile(r"^(?:global_step_|checkpoint-)(\d+)$")
STEP_FILE_RE = re.compile(r"^(\d+)(?:\.[^.]+)?$")
STEP_TEXT_RE = re.compile(r"(?:global_step|checkpoint|step)[-_]?(\d+)", re.IGNORECASE)
WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors"}
EVIDENCE_WORDS = {"evaluation", "evaluations", "validation", "results", "result", "metrics"}


@dataclass(frozen=True)
class Candidate:
    path: Path
    run_name: str
    step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete checkpoint weight directories only when the same run and step "
            "has complete Math evaluation JSONs or a valid validation record."
        )
    )
    parser.add_argument("roots", nargs="+", type=Path, help="Experiment roots to inspect")
    parser.add_argument("--apply", action="store_true", help="Actually delete; default is dry-run")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("/media/vlm-ckp-fileset/ylong/cleanup_reports"),
        help="Directory for the TSV audit report",
    )
    return parser.parse_args()


def is_weight_checkpoint(path: Path) -> bool:
    if path.name.startswith("global_step_"):
        return (path / "actor").is_dir() or (path / "critic").is_dir()

    for current, directories, files in os.walk(path):
        relative_depth = len(Path(current).relative_to(path).parts)
        if relative_depth > 2:
            directories[:] = []
            continue
        if {"actor", "critic", "lora_adapter"}.intersection(directories):
            return True
        for name in files:
            file_path = Path(name)
            if file_path.suffix in WEIGHT_SUFFIXES or name.startswith(("model_world_size_", "optim_world_size_")):
                return True
    return False


def discover_candidates(root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    if not root.is_dir():
        return candidates

    for current, directories, _ in os.walk(root):
        current_path = Path(current)
        kept: list[str] = []
        for name in directories:
            match = STEP_DIR_RE.match(name)
            candidate_path = current_path / name
            if match and is_weight_checkpoint(candidate_path):
                candidates.append(
                    Candidate(path=candidate_path, run_name=current_path.name, step=int(match.group(1)))
                )
                continue
            kept.append(name)
        directories[:] = kept
    return candidates


def path_step(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = STEP_DIR_RE.match(part)
        if match:
            return int(match.group(1))
    match = STEP_FILE_RE.match(path.name)
    if match:
        return int(match.group(1))
    match = STEP_TEXT_RE.search(path.name)
    return int(match.group(1)) if match else None


def valid_json(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value not in (None, {}, [])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def valid_jsonl(path: Path) -> bool:
    records = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    json.loads(line)
                    records += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return records > 0


def metric_steps(path: Path) -> set[int]:
    """Read validation-bearing steps from a reasonably sized metrics JSONL."""
    if path.stat().st_size > 512 * 1024 * 1024:
        return set()
    steps: set[int] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                has_validation = any(str(key).lower().startswith(("val", "eval")) for key in record)
                if not has_validation:
                    continue
                for key in ("training/global_step", "global_step", "step"):
                    if key in record:
                        steps.add(int(float(record[key])))
                        break
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return set()
    return steps


def discover_evidence(root: Path, run_names: set[str]) -> dict[tuple[str, int], list[str]]:
    math_files: dict[tuple[str, int, Path], set[str]] = defaultdict(set)
    generic: dict[tuple[str, int], list[str]] = defaultdict(list)

    for current, directories, files in os.walk(root):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if not STEP_DIR_RE.match(name) or not is_weight_checkpoint(current_path / name)
        ]
        lowered_parts = {part.lower() for part in current_path.parts}
        likely_evidence_dir = bool(lowered_parts.intersection(EVIDENCE_WORDS))

        for name in files:
            path = current_path / name
            path_text = str(path)
            matching_runs = [run_name for run_name in run_names if run_name in path_text]
            if not matching_runs:
                continue

            suffix = path.suffix.lower()
            step = path_step(path)
            dataset = path.stem.lower()
            if suffix == ".json" and step is not None and dataset in MATH_DATASETS and valid_json(path):
                for run_name in matching_runs:
                    math_files[(run_name, step, path.parent)].add(dataset)
                continue

            if suffix == ".jsonl" and (
                likely_evidence_dir or name in {"metrics.jsonl", "events.jsonl"}
            ):
                if step is not None and valid_jsonl(path):
                    for run_name in matching_runs:
                        generic[(run_name, step)].append(str(path))
                elif name in {"metrics.jsonl", "events.jsonl"}:
                    for metric_step in metric_steps(path):
                        for run_name in matching_runs:
                            generic[(run_name, metric_step)].append(f"{path}#step={metric_step}")

    for (run_name, step, parent), datasets in math_files.items():
        if datasets == MATH_DATASETS:
            generic[(run_name, step)].append(f"{parent}#five_math_jsons")
    return generic


def active_command_lines() -> str:
    command_lines: list[str] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return ""
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "cleanup_completed_checkpoints.py" in raw:
            continue
        if not any(
            marker in raw
            for marker in (
                "verl.trainer",
                "opsd_train.py",
                "accelerate launch",
                "torchrun",
                "run_local_",
                "run_table_experiment",
                "evaluate_math.py",
            )
        ):
            continue
        command_lines.append(raw)
    return "\n".join(command_lines)


def allocated_bytes(path: Path) -> int:
    total = 0
    for current, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(current) / name).stat().st_blocks * 512
            except FileNotFoundError:
                pass
    return total


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f}{unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    roots = [path.resolve() for path in args.roots]
    missing = [path for path in roots if not path.is_dir()]
    if missing:
        for path in missing:
            print(f"ERROR missing root: {path}", file=sys.stderr)
        return 2

    candidates_by_root: dict[Path, list[Candidate]] = {}
    evidence: dict[tuple[str, int], list[str]] = defaultdict(list)
    for root in roots:
        candidates = discover_candidates(root)
        candidates_by_root[root] = candidates
        run_names = {candidate.run_name for candidate in candidates}
        for key, values in discover_evidence(root, run_names).items():
            evidence[key].extend(values)

    active = active_command_lines()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = args.report_dir / f"checkpoint_cleanup_{timestamp}.tsv"
    rows: list[dict[str, str]] = []
    reclaimed = 0

    for root, candidates in candidates_by_root.items():
        for candidate in sorted(candidates, key=lambda item: (item.run_name, item.step, str(item.path))):
            proof = evidence.get((candidate.run_name, candidate.step), [])
            if candidate.run_name and candidate.run_name in active:
                action = "skip_active"
            elif not proof:
                action = "skip_no_evidence"
            else:
                action = "delete" if args.apply else "would_delete"

            size = 0
            if action in {"delete", "would_delete"}:
                size = allocated_bytes(candidate.path)
            print(
                f"{action:16s} {human_bytes(size):>10s} "
                f"step={candidate.step:<5d} run={candidate.run_name} path={candidate.path}"
            )
            if proof:
                print(f"  evidence={proof[0]}")

            rows.append(
                {
                    "action": action,
                    "allocated_bytes": str(size),
                    "root": str(root),
                    "run": candidate.run_name,
                    "step": str(candidate.step),
                    "checkpoint": str(candidate.path),
                    "evidence": " | ".join(proof),
                }
            )
            if action == "delete":
                shutil.rmtree(candidate.path)
                reclaimed += size

    with report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("action", "allocated_bytes", "root", "run", "step", "checkpoint", "evidence"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    mode = "APPLY" if args.apply else "DRY-RUN"
    selected = sum(row["action"] in {"delete", "would_delete"} for row in rows)
    print(f"{mode}: candidates={len(rows)}, selected={selected}, reclaimed={human_bytes(reclaimed)}")
    print(f"Audit report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
