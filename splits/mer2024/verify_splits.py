#!/usr/bin/env python3
"""Verify MER2024 split size, uniqueness, and non-overlap."""

from __future__ import annotations

import argparse
from pathlib import Path


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced", type=Path, default=Path(__file__).with_name("balanced50_ids.txt"))
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--expected-balanced", type=int, default=300)
    parser.add_argument("--expected-fresh", type=int, default=300)
    args = parser.parse_args()
    balanced, fresh = read_ids(args.balanced), read_ids(args.fresh)
    assert len(balanced) == args.expected_balanced, (len(balanced), args.expected_balanced)
    assert len(fresh) == args.expected_fresh, (len(fresh), args.expected_fresh)
    assert len(set(balanced)) == len(balanced), "balanced split contains duplicates"
    assert len(set(fresh)) == len(fresh), "fresh split contains duplicates"
    overlap = sorted(set(balanced) & set(fresh))
    assert not overlap, f"splits overlap: {overlap[:10]}"
    print("MER2024 split verification passed")


if __name__ == "__main__":
    main()
