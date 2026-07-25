#!/usr/bin/env python3
"""List corrected samples and summarize their affective cue change."""

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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    corrected = [
        row for row in rows
        if not truth(row["baseline_correct"]) and truth(row["acor_correct"])
    ]
    result = {
        "corrected_count": len(corrected),
        "sample_ids": [row["sample_id"] for row in corrected],
        "cue_types_increased": sum(
            float(row.get("acor_cue_types", 0)) > float(row.get("baseline_cue_types", 0))
            for row in corrected
        ),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
