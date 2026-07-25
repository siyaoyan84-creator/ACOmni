#!/usr/bin/env python3
"""Group samples by cue-type increase and report accuracy/net gain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "correct"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for name, subset in {
        "cue_types_increased": [row for row in rows if float(row["acor_cue_types"]) > float(row["baseline_cue_types"])],
        "not_increased": [row for row in rows if float(row["acor_cue_types"]) <= float(row["baseline_cue_types"])],
    }.items():
        baseline = [truth(row["baseline_correct"]) for row in subset]
        acor = [truth(row["acor_correct"]) for row in subset]
        result[name] = {
            "N": len(subset),
            "baseline_accuracy": sum(baseline) / len(subset) if subset else 0.0,
            "acor_accuracy": sum(acor) / len(subset) if subset else 0.0,
            "net_gain": sum(acor) - sum(baseline),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
