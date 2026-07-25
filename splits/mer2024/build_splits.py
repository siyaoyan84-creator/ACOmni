#!/usr/bin/env python3
"""Prepare MER2024 calibration and fresh test splits.

This script builds two balanced, non-overlapping splits from the MER2024 label CSV:
- calibration360: 60 samples per class
- fresh_test300: 50 samples per class

It excludes the 300 samples already used by the current v2-neutral evaluation,
reuses the exact reference question and choices from that JSONL, and writes two
JSONL files with stable sample identifiers.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("prepare_mer2024_calibration_fresh_split")

LABEL_MAP = {
    "neutral": "neutral",
    "angry": "anger",
    "happy": "joy",
    "sad": "sadness",
    "worried": "worried",
    "surprise": "surprise",
}
LABEL_ORDER = ["neutral", "anger", "joy", "sadness", "worried", "surprise"]
CHOICE_ORDER = [
    ("A", "neutral"),
    ("B", "anger"),
    ("C", "joy"),
    ("D", "sadness"),
    ("E", "worried"),
    ("F", "surprise"),
]
PROMPT_VERSION_FALLBACK = "MER2024"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MER2024 calibration and fresh test splits")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--used-test-jsonl", required=True)
    parser.add_argument("--output-root", default="eval_results/mer2024_acor_eval")
    parser.add_argument("--calibration-per-class", type=int, default=60)
    parser.add_argument("--fresh-test-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--calibration-file-name", default="mer2024_calibration60_v2_neutral.jsonl")
    parser.add_argument("--fresh-test-file-name", default="mer2024_fresh_test50_v2_neutral.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"Empty JSONL file: {path}")
    return rows


def extract_stable_sample_name(record: Dict[str, Any]) -> str:
    for key in ("name", "sample_name"):
        value = record.get(key)
        if value:
            return Path(str(value)).stem
    for key in ("video", "video_path"):
        value = record.get(key)
        if value:
            return Path(str(value)).stem
    raise RuntimeError(f"Cannot extract stable sample name from record: {record}")


def normalize_label(raw_label: Any, sample_name: str) -> str:
    if raw_label is None:
        raise RuntimeError(f"Missing label for sample {sample_name}")
    normalized = str(raw_label).strip().lower()
    if normalized not in LABEL_MAP:
        raise RuntimeError(f"Unsupported label for sample {sample_name}: {raw_label!r}")
    return LABEL_MAP[normalized]


def normalize_choice(choice: Any) -> Tuple[str, str]:
    text = str(choice).strip()
    if not text:
        raise RuntimeError("Encountered empty choice in reference prompt")
    letter = None
    label = None
    for expected_letter, expected_label in CHOICE_ORDER:
        if text.startswith(expected_letter):
            letter = expected_letter
            remainder = text[len(expected_letter):].strip()
            remainder = remainder.lstrip(".):\t ")
            label = remainder.lower()
            break
    if letter is None or label is None:
        raise RuntimeError(f"Unrecognized choice format: {choice!r}")
    if label not in LABEL_ORDER:
        raise RuntimeError(f"Unrecognized choice label: {choice!r}")
    return letter, label


def load_reference_prompt(path: Path) -> Tuple[str, List[Any], str, str]:
    rows = read_jsonl(path)
    first = rows[0]
    if "question" not in first:
        raise RuntimeError(f"Reference JSONL missing question field: {path}")
    if "choices" not in first:
        raise RuntimeError(f"Reference JSONL missing choices field: {path}")
    if "prompt_version" not in first:
        raise RuntimeError(f"Reference JSONL missing prompt_version field: {path}")
    question = str(first["question"]).strip()
    if not question:
        raise RuntimeError(f"Reference question is empty in {path}")
    choices = first["choices"]
    if not isinstance(choices, list) or len(choices) != 6:
        raise RuntimeError(f"Reference choices must be a list of length 6 in {path}")
    prompt_version = str(first["prompt_version"]).strip()
    if not prompt_version:
        raise RuntimeError(f"Reference prompt_version is empty in {path}")
    source = str(first.get("source") or "").strip() or PROMPT_VERSION_FALLBACK
    normalized = [normalize_choice(choice) for choice in choices]
    expected = CHOICE_ORDER
    if normalized != expected:
        raise RuntimeError(f"Reference choices do not match expected v2-neutral order: {normalized!r}")
    return question, choices, prompt_version, source


def build_video_map(video_root: Path) -> Dict[str, Path]:
    if not video_root.exists():
        raise FileNotFoundError(f"Video root does not exist: {video_root}")
    if not video_root.is_dir():
        raise RuntimeError(f"Video root is not a directory: {video_root}")
    video_map: Dict[str, Path] = {}
    videos = list(video_root.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No mp4 files found directly under video root: {video_root}")
    for video_path in videos:
        stem = video_path.stem
        if stem in video_map:
            raise RuntimeError(f"Duplicate video stem detected: {stem}")
        video_map[stem] = video_path.resolve()
    return video_map


def load_csv_samples(labels_csv: Path) -> List[Dict[str, Any]]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    rows: List[Dict[str, Any]] = []
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RuntimeError(f"Labels CSV has no header: {labels_csv}")
        required_fields = {"name", "discrete", "valence"}
        missing = required_fields - set(reader.fieldnames)
        if missing:
            raise RuntimeError(f"Labels CSV missing required fields {sorted(missing)}; actual columns: {reader.fieldnames}")
        for index, row in enumerate(reader):
            sample_name = str(row.get("name") or "").strip()
            if not sample_name:
                raise RuntimeError(f"Missing name in CSV row {index}")
            raw_label = row.get("discrete")
            label = normalize_label(raw_label, sample_name)
            rows.append(
                {
                    "name": sample_name,
                    "raw_label": raw_label,
                    "label": label,
                    "valence": row.get("valence"),
                    "source_row_index": index,
                }
            )
    if not rows:
        raise RuntimeError(f"Empty labels CSV: {labels_csv}")
    return rows


def build_used_test_name_set(used_test_jsonl: Path) -> Tuple[str, List[Any], str, str, set[str]]:
    rows = read_jsonl(used_test_jsonl)
    question, choices, prompt_version, source = load_reference_prompt(used_test_jsonl)
    names: List[str] = []
    for record in rows:
        names.append(extract_stable_sample_name(record))
    if len(names) != 300:
        raise RuntimeError(f"Used test JSONL must contain exactly 300 records, found {len(names)}")
    counter = Counter(names)
    duplicates = [name for name, count in counter.items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Duplicate stable sample names in used test JSONL: {duplicates}")
    return question, choices, prompt_version, source, set(names)


def build_candidates(
    rows: Sequence[Dict[str, Any]],
    video_map: Dict[str, Path],
    used_test_names: set[str],
) -> Dict[str, List[Dict[str, Any]]]:
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_names: Dict[str, int] = {}
    for row in rows:
        name = row["name"]
        if name in seen_names:
            raise RuntimeError(f"Duplicate sample name in CSV: {name}")
        seen_names[name] = 1
        if name in used_test_names:
            continue
        video_path = video_map.get(name)
        if video_path is None:
            continue
        item = dict(row)
        item["video"] = str(video_path)
        by_label[row["label"]].append(item)
    return by_label


def split_samples(
    candidates: Dict[str, List[Dict[str, Any]]],
    calibration_per_class: int,
    fresh_test_per_class: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    calibration: List[Dict[str, Any]] = []
    fresh_test: List[Dict[str, Any]] = []
    required = calibration_per_class + fresh_test_per_class
    for label in LABEL_ORDER:
        pool = sorted(candidates.get(label, []), key=lambda item: item["name"])
        if len(pool) < required:
            raise RuntimeError(
                f"Insufficient samples for label {label}: total={len(pool)}, required={required}"
            )
        shuffled = pool[:]
        rng.shuffle(shuffled)
        calibration.extend(shuffled[:calibration_per_class])
        fresh_test.extend(shuffled[calibration_per_class:calibration_per_class + fresh_test_per_class])
    rng.shuffle(calibration)
    rng.shuffle(fresh_test)
    return calibration, fresh_test


def build_output_record(
    item: Dict[str, Any],
    sample_idx: int,
    question: str,
    choices: List[Any],
    split: str,
    prompt_version: str,
    source: str,
) -> Dict[str, Any]:
    valence = item["valence"]
    try:
        valence_value: Any = float(valence) if valence is not None and str(valence).strip() != "" else valence
    except (TypeError, ValueError):
        valence_value = valence
    answer_letter = {
        "neutral": "A",
        "anger": "B",
        "joy": "C",
        "sadness": "D",
        "worried": "E",
        "surprise": "F",
    }[item["label"]]
    answer = f"{answer_letter}. {item['label']}"
    return {
        "sample_idx": sample_idx,
        "sample_id": item["name"],
        "sample_name": item["name"],
        "name": item["name"],
        "video": item["video"],
        "audio": None,
        "transcription": "",
        "question": question,
        "choices": choices,
        "answer": answer,
        "answer_letter": answer_letter,
        "label": item["label"],
        "raw_label": item["raw_label"],
        "valence": valence_value,
        "source": source,
        "split": split,
        "prompt_version": prompt_version,
        "source_row_index": item["source_row_index"],
    }


def validate_final_splits(
    calibration: Sequence[Dict[str, Any]],
    fresh_test: Sequence[Dict[str, Any]],
    used_test_names: set[str],
    calibration_per_class: int,
    fresh_test_per_class: int,
    question: str,
    choices: List[Any],
    prompt_version: str,
    source: str,
) -> None:
    required_fields = {
        "sample_idx",
        "sample_id",
        "sample_name",
        "name",
        "video",
        "audio",
        "transcription",
        "question",
        "choices",
        "answer",
        "answer_letter",
        "label",
        "raw_label",
        "valence",
        "source",
        "split",
        "prompt_version",
        "source_row_index",
    }

    if len(calibration) != 6 * calibration_per_class:
        raise RuntimeError(f"Calibration sample count mismatch: {len(calibration)}")
    if len(fresh_test) != 6 * fresh_test_per_class:
        raise RuntimeError(f"Fresh test sample count mismatch: {len(fresh_test)}")

    cal_counter = Counter(item["label"] for item in calibration)
    fresh_counter = Counter(item["label"] for item in fresh_test)
    expected_cal = {label: calibration_per_class for label in LABEL_ORDER}
    expected_fresh = {label: fresh_test_per_class for label in LABEL_ORDER}
    if dict(cal_counter) != expected_cal:
        raise RuntimeError(f"Calibration label distribution mismatch: {dict(cal_counter)}")
    if dict(fresh_counter) != expected_fresh:
        raise RuntimeError(f"Fresh test label distribution mismatch: {dict(fresh_counter)}")

    for split_name, records in (("calibration", calibration), ("fresh_test", fresh_test)):
        sample_indices = [item["sample_idx"] for item in records]
        if sample_indices != list(range(len(records))):
            raise RuntimeError(f"{split_name} sample_idx must be a continuous sequence starting from 0")
        if len(sample_indices) != len(set(sample_indices)):
            raise RuntimeError(f"{split_name} sample_idx duplicates detected")
        sample_ids = [item["sample_id"] for item in records]
        if len(sample_ids) != len(set(sample_ids)):
            raise RuntimeError(f"{split_name} sample_id duplicates detected")
        for index, item in enumerate(records):
            missing_fields = required_fields - set(item)
            if missing_fields:
                raise RuntimeError(
                    f"Missing required output fields in {split_name} record index {index}: {sorted(missing_fields)}"
                )
            if item["question"] != question:
                raise RuntimeError(f"Question mismatch in {split_name} record index {index}")
            if item["choices"] != choices:
                raise RuntimeError(f"Choices mismatch in {split_name} record index {index}")
            if item["prompt_version"] != prompt_version:
                raise RuntimeError(f"Prompt version mismatch in {split_name} record index {index}")
            if item["source"] != source:
                raise RuntimeError(f"Source mismatch in {split_name} record index {index}")
            if item["audio"] is not None:
                raise RuntimeError(f"Audio field must be None in {split_name} record index {index}")
            if item["transcription"] != "":
                raise RuntimeError(f"Transcription field must be empty string in {split_name} record index {index}")
            if item["sample_id"] != item["sample_name"] or item["sample_id"] != item["name"]:
                raise RuntimeError(f"Stable sample identifiers mismatch in {split_name} record index {index}")
            if item["answer_letter"] != {
                "neutral": "A",
                "anger": "B",
                "joy": "C",
                "sadness": "D",
                "worried": "E",
                "surprise": "F",
            }[item["label"]]:
                raise RuntimeError(f"Answer letter mismatch in {split_name} record index {index}")
            if item["answer"] != f"{item['answer_letter']}. {item['label']}":
                raise RuntimeError(f"Answer text mismatch in {split_name} record index {index}")
            if item["label"] not in LABEL_ORDER:
                raise RuntimeError(f"Invalid label in final splits: {item['label']}")

    cal_names = [item["name"] for item in calibration]
    fresh_names = [item["name"] for item in fresh_test]
    cal_set = set(cal_names)
    fresh_set = set(fresh_names)
    if cal_set & fresh_set:
        raise RuntimeError(f"Calibration and fresh test overlap detected: {sorted(cal_set & fresh_set)}")
    if cal_set & used_test_names:
        raise RuntimeError(f"Calibration overlaps with used test set: {sorted(cal_set & used_test_names)[:10]}")
    if fresh_set & used_test_names:
        raise RuntimeError(f"Fresh test overlaps with used test set: {sorted(fresh_set & used_test_names)[:10]}")


def write_jsonl_atomic(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        shutil.move(str(temp_path), str(path))
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    labels_csv = Path(args.labels_csv)
    video_root = Path(args.video_root)
    used_test_jsonl = Path(args.used_test_jsonl)
    output_root = Path(args.output_root)
    calibration_path = output_root / args.calibration_file_name
    fresh_test_path = output_root / args.fresh_test_file_name

    if (calibration_path.exists() or fresh_test_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {calibration_path if calibration_path.exists() else fresh_test_path}"
        )

    video_map = build_video_map(video_root)
    question, choices, prompt_version, source, used_test_names = build_used_test_name_set(used_test_jsonl)
    if len(used_test_names) != 300:
        raise RuntimeError(f"Used test set must contain exactly 300 unique sample names, found {len(used_test_names)}")

    csv_rows = load_csv_samples(labels_csv)
    candidates = build_candidates(csv_rows, video_map, used_test_names)

    for label in LABEL_ORDER:
        pool = candidates.get(label, [])
        required = args.calibration_per_class + args.fresh_test_per_class
        if len(pool) < required:
            raise RuntimeError(
                f"Insufficient samples for label {label}: total={len(pool)}, exclude_used={sum(1 for row in csv_rows if row['label'] == label and row['name'] not in used_test_names)}, video_available={len(pool)}, required={required}"
            )

    rng = random.Random(args.seed)
    calibration_items, fresh_test_items = split_samples(
        candidates,
        args.calibration_per_class,
        args.fresh_test_per_class,
        rng,
    )
    calibration_records = [
        build_output_record(item, index, question, list(choices), "calibration", prompt_version, source)
        for index, item in enumerate(calibration_items)
    ]
    fresh_test_records = [
        build_output_record(item, index, question, list(choices), "fresh_test", prompt_version, source)
        for index, item in enumerate(fresh_test_items)
    ]

    validate_final_splits(
        calibration_records,
        fresh_test_records,
        used_test_names,
        args.calibration_per_class,
        args.fresh_test_per_class,
        question,
        choices,
        prompt_version,
        source,
    )

    write_jsonl_atomic(calibration_path, calibration_records)
    write_jsonl_atomic(fresh_test_path, fresh_test_records)

    cal_dist = Counter(item["label"] for item in calibration_items)
    fresh_dist = Counter(item["label"] for item in fresh_test_items)
    cal_set = {item["name"] for item in calibration_items}
    fresh_set = {item["name"] for item in fresh_test_items}
    print(f"seed = {args.seed}")
    print(f"used_test_count = {len(used_test_names)}")
    print(f"calibration_count = {len(calibration_records)}")
    print(f"fresh_test_count = {len(fresh_test_records)}")
    print(f"calibration_distribution = {dict(cal_dist)}")
    print(f"fresh_test_distribution = {dict(fresh_dist)}")
    print(f"calibration_used_overlap = {len(cal_set & used_test_names)}")
    print(f"fresh_test_used_overlap = {len(fresh_set & used_test_names)}")
    print(f"calibration_fresh_overlap = {len(cal_set & fresh_set)}")
    print(f"calibration_output = {calibration_path}")
    print(f"fresh_test_output = {fresh_test_path}")


if __name__ == "__main__":
    main()
