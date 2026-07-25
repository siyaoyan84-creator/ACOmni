#!/usr/bin/env python3
"""Build ACOR-Affective or ACOR-A.R. from public adapter checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from acor.model_utils import adapter_soup, interpolate_adapters


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)

    interp = subparsers.add_parser("interpolate")
    interp.add_argument("--source-a", type=Path, required=True)
    interp.add_argument("--source-b", type=Path, required=True)
    interp.add_argument("--alpha", type=float, required=True)
    interp.add_argument("--output", type=Path, required=True)

    soup = subparsers.add_parser("soup")
    soup.add_argument("--source", action="append", nargs=2, metavar=("PATH", "WEIGHT"), required=True)
    soup.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.operation == "interpolate":
        interpolate_adapters(args.source_a, args.source_b, args.output, args.alpha)
    else:
        adapter_soup({Path(path): float(weight) for path, weight in args.source}, args.output)


if __name__ == "__main__":
    main()
