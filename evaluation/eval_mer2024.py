#!/usr/bin/env python3
"""Evaluate saved free-generation outputs on the public MER2024 split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acor.rewards import EMOTION_ALIASES
from acor.response_parser import extract_answer, normalize_text
from evaluation.metrics import accuracy, macro_f1, parse_valid

LABELS = ["anger", "joy", "sadness", "neutral", "worried", "surprise"]
MER2024_ALIASES = {
    **EMOTION_ALIASES,
    "sad": "sadness",
    "sadness": "sadness",
    "worried": "worried",
    "worry": "worried",
    "anxious": "worried",
}


def normalize_emotion(text: object) -> str | None:
    value = normalize_text(extract_answer(text) or text)
    for alias, category in MER2024_ALIASES.items():
        if alias in value.split():
            return category
    return None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    gold = [normalize_emotion(row.get("label", row.get("answer"))) for row in rows]
    predictions = [normalize_emotion(row.get("output", row.get("prediction"))) for row in rows]
    valid_pairs = [(g, p) for g, p in zip(gold, predictions) if g is not None and p is not None]
    metrics = {
        "samples": len(rows),
        "accuracy": accuracy([g for g, _ in valid_pairs], [p for _, p in valid_pairs]),
        "macro_f1": macro_f1([g for g, _ in valid_pairs], [p for _, p in valid_pairs], LABELS),
        "parse_valid": parse_valid(predictions),
    }
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
