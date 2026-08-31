#!/usr/bin/env python3
"""Compact known distIL evaluation records without changing scores or dimensions.

This module does not choose filesystem scope. The project cleanup caller owns
allowlists, launcher locks, and replacement receipts.
"""

from __future__ import annotations

import hashlib
import json
import math


PROBLEMS = {"aime24": 30, "aime25": 30, "hmmt25": 30, "amc23": 40, "minerva": 272}
SCORE_KEYS = ("average_at_n_pct", "pass_at_n_pct", "majority_vote_at_n_pct", "format_rate")
RETENTION_KEY = "storage_compaction"


def validate(payload: dict, dataset: str) -> None:
    if not isinstance(payload, dict) or dataset not in PROBLEMS:
        raise ValueError("not a supported Math result")
    samples = payload.get("val_n")
    if type(samples) is not int or samples <= 0:
        raise ValueError("invalid val_n")
    if (payload.get("dataset") != dataset or payload.get("num_problems") != PROBLEMS[dataset]
            or payload.get("total_solutions") != PROBLEMS[dataset] * samples):
        raise ValueError("invalid evaluation dimensions")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != PROBLEMS[dataset]:
        raise ValueError("incomplete problem rows")
    for row in results:
        if not isinstance(row, dict) or row.get("val_n") != samples:
            raise ValueError("invalid per-question dimensions")
        generations = row.get("generations")
        if not isinstance(generations, list) or len(generations) != samples:
            raise ValueError("incomplete per-question generations")
        for item in generations:
            if (not isinstance(item, dict) or type(item.get("correct")) is not bool
                    or type(item.get("formatted")) is not bool
                    or "predicted_answer" not in item):
                raise ValueError("missing per-sample answer/correctness/format evidence")
    for key in SCORE_KEYS:
        value = payload.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"invalid {key}")


def compact(payload: dict, dataset: str, original_sha256: str) -> tuple[dict, int]:
    validate(payload, dataset)
    count = 0

    def strip_record(record):
        nonlocal count
        result = dict(record)
        if "full_generation" not in result:
            return result
        raw = result["full_generation"]
        if not isinstance(raw, str):
            raise ValueError("full_generation is not text")
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        for key, value in (("full_generation_sha256", digest),
                           ("full_generation_num_chars", len(raw))):
            if key in result and result[key] != value:
                raise ValueError(f"existing {key} conflicts with original text")
            result[key] = value
        del result["full_generation"]
        count += 1
        return result

    output = dict(payload)
    output["results"] = []
    for row in payload["results"]:
        reduced = strip_record(row)
        reduced["generations"] = [strip_record(item) for item in row["generations"]]
        output["results"].append(reduced)
    if count:
        if RETENTION_KEY in payload:
            raise ValueError("unexpected raw text in an already compacted file")
        output[RETENTION_KEY] = {
            "schema": 1, "removed_field": "full_generation", "original_sha256": original_sha256,
            "removed_text_fields": count, "raw_text_backed_up": False,
        }
    validate(output, dataset)
    return output, count


def load(raw: bytes) -> dict:
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique)
