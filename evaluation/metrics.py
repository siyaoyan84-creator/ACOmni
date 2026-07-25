"""Dependency-light metrics shared by ACOR evaluations."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence


def accuracy(gold: Sequence[str], prediction: Sequence[str]) -> float:
    return sum(g == p for g, p in zip(gold, prediction)) / len(gold) if gold else 0.0


def macro_f1(gold: Sequence[str], prediction: Sequence[str], labels: Iterable[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, prediction))
        fp = sum(g != label and p == label for g, p in zip(gold, prediction))
        fn = sum(g == label and p != label for g, p in zip(gold, prediction))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def parse_valid(prediction: Sequence[str | None]) -> float:
    return sum(item is not None for item in prediction) / len(prediction) if prediction else 0.0


def transition_counts(baseline: Sequence[bool], acor: Sequence[bool]) -> dict[str, int]:
    counts = Counter()
    for before, after in zip(baseline, acor):
        if not before and after:
            counts["corrected"] += 1
        elif before and not after:
            counts["new_error"] += 1
        elif before and after:
            counts["both_right"] += 1
        else:
            counts["both_wrong"] += 1
    counts["net_gain"] = counts["corrected"] - counts["new_error"]
    return dict(counts)
