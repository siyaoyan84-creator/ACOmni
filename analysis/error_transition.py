#!/usr/bin/env python3
"""Compute Corrected/New Error/Both Right/Both Wrong/Net Gain (Table VI)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluation.metrics import transition_counts


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "correct"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="CSV: sample_id,baseline_correct,acor_correct")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = transition_counts(
        [as_bool(row["baseline_correct"]) for row in rows],
        [as_bool(row["acor_correct"]) for row in rows],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
