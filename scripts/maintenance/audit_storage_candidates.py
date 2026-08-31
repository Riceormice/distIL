#!/usr/bin/env python3
"""Read-only storage inventory. Review labels never authorize deletion."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


PROTECTED_DIRS = {
    "models", "envs", "code", "datasets", "data", "runtime",
    "runtime_assets", "runtime_overlays", ".cache", ".git", "__pycache__",
}
EVIDENCE_DIRS = {
    "token_stats", "train_token_stats", "generation", "generation_raw",
    "evaluation", "evaluations", "validation", "aggregate", "figures",
    "protocol", "state", "logs", "training", "reports",
}
RAW_DIRS = {"raw_logits_audit", "topk_probe"}
WEIGHT_DIRS = {
    "model_states", "checkpoints", "checkpoint", "ckpts", "merged",
    "host_checkpoints", "checkpoint_cache", "model_cache", "weights",
}
ARCHIVE_SUFFIXES = (".tar.gz", ".tar.zst", ".tar", ".tgz", ".zip")
WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin"}


def classify(name: str, is_dir: bool, size: int, min_file_bytes: int) -> str | None:
    if is_dir:
        if name in PROTECTED_DIRS:
            return "KEEP_RUNTIME_MODEL_DATA"
        if name in RAW_DIRS:
            return "KEEP_RAW_LOGITS"
        if name in EVIDENCE_DIRS:
            return "KEEP_EVIDENCE"
        if name in WEIGHT_DIRS:
            return "REVIEW_WEIGHT_CONTAINER"
        return None
    if name.endswith(ARCHIVE_SUFFIXES):
        return "REVIEW_ARCHIVE"
    if name.endswith(tuple(suffix + ".part" for suffix in ARCHIVE_SUFFIXES)):
        return "REVIEW_PARTIAL_ARCHIVE"
    if size >= min_file_bytes:
        if Path(name).suffix in WEIGHT_SUFFIXES:
            return "REVIEW_TENSOR_OR_WEIGHT_FILE"
        return "REVIEW_LARGE_UNKNOWN_FILE"
    return None


def marker_hints(path: Path, root: Path) -> list[str]:
    """Markers are recorded with their scope, not interpreted as proof."""
    hints = []
    parent = path.parent
    while parent != root and root in parent.parents:
        for suffix in ("state/run.complete", "state/complete", "state/training.complete"):
            marker = parent / suffix
            if marker.is_file():
                hints.append(str(marker))
        parent = parent.parent
    return hints


def discover(root: Path, max_depth: int, min_file_bytes: int, excluded: Path):
    rows, errors = [], []

    def visit(parent: Path, depth: int) -> None:
        try:
            with os.scandir(parent) as entries:
                entries = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            errors.append(f"{parent}: {exc}")
            return
        for entry in entries:
            path = Path(entry.path)
            if path == excluded:
                continue
            try:
                info = entry.stat(follow_symlinks=False)
                if entry.is_symlink():
                    kind = "KEEP_SYMLINK_NOT_FOLLOWED"
                elif parent == root and entry.name == "sdpo" and entry.is_dir(follow_symlinks=False):
                    # ylong/sdpo holds shared assets; nested sdpo folders are methods.
                    kind = "KEEP_RUNTIME_MODEL_DATA"
                else:
                    is_dir = entry.is_dir(follow_symlinks=False)
                    kind = classify(entry.name, is_dir, info.st_size, min_file_bytes)
                    if kind is None and is_dir:
                        if depth < max_depth:
                            visit(path, depth + 1)
                            continue
                        kind = "REVIEW_DEPTH_LIMIT"
                if kind is None:
                    continue
                rows.append({
                    "kind": kind,
                    "path": str(path),
                    "experiment": path.relative_to(root).parts[0],
                    "allocated_bytes": None,
                    "marker_hints": marker_hints(path, root),
                    "activity": "UNKNOWN_ON_OTHER_MACHINES",
                    "error": "",
                })
            except OSError as exc:
                errors.append(f"{path}: {exc}")

    visit(root, 1)
    return rows, errors


def disk_bytes(path: Path, timeout: float) -> tuple[int | None, str]:
    try:
        result = subprocess.run(
            ["du", "-sk", str(path)], text=True, capture_output=True, timeout=timeout,
        )
        if result.returncode:
            return None, result.stderr.strip() or f"du exit={result.returncode}"
        return int(result.stdout.split()[0]) * 1024, ""
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        return None, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-file-mib", type=int, default=64)
    parser.add_argument("--du-timeout", type=float, default=180)
    args = parser.parse_args()
    root, report_dir = args.root.resolve(), args.report_dir.resolve()
    if not root.is_dir():
        parser.error(f"root does not exist: {root}")
    if report_dir == root or report_dir in root.parents:
        parser.error("report directory must not be the input root or its ancestor")
    if args.max_depth < 1 or args.min_file_mib < 1 or args.du_timeout <= 0:
        parser.error("depth, minimum file size, and timeout must be positive")

    print("READ ONLY: no deletion, no payload contents read, no local-process inference", flush=True)
    print("Completion markers are hints only; paused/nightly runs may still need weights.", flush=True)
    rows, errors = discover(root, args.max_depth, args.min_file_mib * 1024**2, report_dir)
    for index, row in enumerate(rows, 1):
        print(f"SIZE {index}/{len(rows)} {row['kind']} {row['path']}", flush=True)
        row["allocated_bytes"], row["error"] = disk_bytes(Path(row["path"]), args.du_timeout)
        if row["error"]:
            errors.append(f"{row['path']}: {row['error']}")
    rows.sort(key=lambda row: row["allocated_bytes"] if row["allocated_bytes"] is not None else -1, reverse=True)

    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = report_dir / f"storage_candidates_{stamp}"
    with stem.with_suffix(".tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["kind", "path"], delimiter="\t")
        writer.writeheader()
        writer.writerows({**row, "marker_hints": " | ".join(row["marker_hints"])} for row in rows)
    with stem.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump({
            "root": str(root), "rows": rows, "errors": errors,
            "note": "Review candidates, not deletion eligibility. No reclaimable-space total; shared/hardlinked files may overlap. Depth limits and scan errors leave unknowns.",
        }, stream, indent=2)

    print("\nLARGEST REVIEW CANDIDATES (not authorization to delete):")
    for row in [row for row in rows if row["kind"].startswith("REVIEW_")][:60]:
        gib = "unknown" if row["allocated_bytes"] is None else f"{row['allocated_bytes'] / 1024**3:.2f}"
        print(f"{gib:>10} GiB {row['kind']} {row['path']}")
        if row["marker_hints"]:
            print("  markers: " + " | ".join(row["marker_hints"]))
    print(f"\nTSV: {stem.with_suffix('.tsv')}")
    print(f"JSON: {stem.with_suffix('.json')}")
    print(f"scan_errors={len(errors)}; protected data is included in the report as KEEP_*")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
