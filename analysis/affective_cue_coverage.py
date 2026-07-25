#!/usr/bin/env python3
"""Reproduce context-only cue coverage statistics used for Table V."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from acor.response_parser import extract_context


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=Path(__file__).with_name("cue_taxonomy.json"))
    args = parser.parse_args()
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    rows = read_jsonl(args.predictions)
    counts = Counter()
    word_counts = []
    cue_type_counts = []
    valid = 0
    for row in rows:
        context = extract_context(row.get("output", ""))
        if not context:
            continue
        valid += 1
        word_counts.append(len(re.findall(r"\b\w+\b", context)))
        present = 0
        lowered = context.lower()
        for cue, keywords in taxonomy.items():
            if any(keyword.lower() in lowered for keyword in keywords):
                counts[cue] += 1
                present += 1
        cue_type_counts.append(present)
    result = {
        "N": len(rows),
        "ContextValid": valid,
        "AvgWords": sum(word_counts) / len(word_counts) if word_counts else 0.0,
        "AvgCueTypes": sum(cue_type_counts) / len(cue_type_counts) if cue_type_counts else 0.0,
        **{cue: counts[cue] / valid if valid else 0.0 for cue in taxonomy},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
