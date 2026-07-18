import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

# Historical model labels are optional; model-generation code is not included.
MODEL_SPECS = [
    {
        "key": "humanomniv2_paper_baseline",
        "label": "ACOR-public paper baseline",
        "prefixes": [],
        "fixed": {
            "ib": 69.33,
            "emotion": 82.41,
            "deception": 64.00,
            "do": 58.47,
            "ws": 47.10,
            "avg": 58.30,
        },
    },
    {
        "key": "grpo1_2100",
        "label": "GRPO1-2100",
        "prefixes": ["ib_stage2B_emotion_preserving_grpo2100_ib_clean_v1"],
        "fixed": {
            "ib": 67.73,
            "emotion": 81.31,
            "deception": 64.50,
            "do": 61.74,
            "ws": 46.22,
            "avg": 58.56,
        },
    },
    {
        "key": "b500_acor_balanced",
        "label": "B500 / ACOR-Balanced",
        "prefixes": ["stage2B_emotion_preserving_grpo500"],
    },
    {
        "key": "alpha20_acor_affective",
        "label": "alpha20 / ACOR-Affective",
        "prefixes": ["stage2B_interp_b500_b700_alpha20"],
    },
    {
        "key": "pair_alpha20_repair20_lam025_acor_affective_repaired",
        "label": "pair_alpha20_repair20_lam025 / ACOR-Affective-Repaired",
        "prefixes": ["acor_step1_pair_alpha20_repair20_lam025"],
    },
]

DAILY_TASKS = [
    "event_sequence",
    "av_event_alignment",
    "temporal_localization",
    "audio_visual_reasoning",
    "counting",
    "other",
]
WORLD_DOMAINS = [
    "Understanding",
    "Reasoning",
    "Recognition",
    "Prediction",
    "Planning",
    "Causal",
    "Spatial",
    "Audio",
]


def resolve_eval_root(args_eval_root):
    eval_root = args_eval_root
    if eval_root is None:
        eval_root_env = os.environ.get("EVAL_ROOT")
        if eval_root_env:
            eval_root = Path(eval_root_env)
        else:
            output_root = os.environ.get("OUTPUT_ROOT")
            if not output_root:
                raise RuntimeError("Set --eval-root, EVAL_ROOT, or OUTPUT_ROOT.")
            eval_root = Path(output_root) / "eval_results"
    return eval_root.expanduser().resolve()


def safe_load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_jsonl(path: Path):
    records = []
    line_count = 0
    malformed = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            line_count += 1
            try:
                records.append(json.loads(line))
            except Exception:
                malformed += 1
    return records, line_count, malformed


def normalize_text(value):
    return str(value).strip() if value is not None else ""


def score_from_record(record):
    for key in ["reward", "accuracy", "acc", "score", "mean_acc"]:
        if key in record:
            try:
                return float(record[key])
            except Exception:
                pass
    pred = normalize_text(record.get("prediction", record.get("pred", "")))
    gt = normalize_text(record.get("gt", record.get("answer", record.get("solution", ""))))
    if pred and gt and pred == gt:
        return 1.0
    return 0.0


# benchmark-to-file mapping.
def metric_key_for_record(record):
    dataset = normalize_text(record.get("dataset", record.get("dataset_name", ""))).lower()
    file_name = normalize_text(record.get("file_name", "")).lower()
    source = record.get("source_sample", {}) if isinstance(record.get("source_sample", {}), dict) else {}
    text = " ".join(
        normalize_text(x).lower()
        for x in [
            dataset,
            file_name,
            record.get("problem_type", ""),
            source.get("Type", ""),
            source.get("content_parent_category", ""),
            source.get("content_fine_category", ""),
            source.get("domain", ""),
            source.get("sub_category", ""),
        ]
    )
    if dataset == "ib" or "intent" in text:
        return "ib"
    if dataset == "daily" or "daily" in text:
        return "daily"
    if dataset == "world" or "world" in text:
        return "world"
    return "unknown"


def daily_task_key(record):
    source = record.get("source_sample", {}) if isinstance(record.get("source_sample", {}), dict) else {}
    raw = " ".join(
        normalize_text(x).lower()
        for x in [
            record.get("problem_type", ""),
            source.get("Type", ""),
            source.get("content_parent_category", ""),
            source.get("content_fine_category", ""),
            source.get("video_category", ""),
            source.get("question", ""),
        ]
    )
    if "event sequence" in raw:
        return "event_sequence"
    if "av event alignment" in raw:
        return "av_event_alignment"
    if "temporal localization" in raw:
        return "temporal_localization"
    if "audio visual" in raw or "audio-visual" in raw or "a/v" in raw:
        return "audio_visual_reasoning"
    if "count" in raw:
        return "counting"
    return "other"


def daily_subset_key(record):
    source = record.get("source_sample", {}) if isinstance(record.get("source_sample", {}), dict) else {}
    duration = normalize_text(source.get("video_duration", "")).lower()
    if "30s" in duration or "<1min" in duration or duration == "30s":
        return "30s"
    if "60s" in duration or duration == "1min" or duration == "1-2min":
        return "60s"
    return "unknown"


def world_domain_key(record):
    source = record.get("source_sample", {}) if isinstance(record.get("source_sample", {}), dict) else {}
    domain = normalize_text(source.get("domain", record.get("world_domain", ""))).strip()
    return domain if domain else "UNKNOWN"


def build_model_files(model_spec, eval_root: Path):
    prefixes = model_spec.get("prefixes", [])
    matched = []
    for path in eval_root.glob("**/*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        name = path.name
        if any(name.startswith(prefix) for prefix in prefixes):
            matched.append(path)
    return sorted(matched)


def aggregate_model(model_spec, eval_root: Path):
    fixed = model_spec.get("fixed")
    if fixed:
        return {
            "label": model_spec["label"],
            "main": fixed,
            "daily": {},
            "world": {},
            "missing": [],
        }

    matched_files = build_model_files(model_spec, eval_root)
    missing = []
    if not matched_files:
        missing.append(f"missing_files: {model_spec['label']}")
        return {"label": model_spec["label"], "main": {}, "daily": {}, "world": {}, "missing": missing}

    grouped = defaultdict(list)
    daily_tasks = defaultdict(list)
    daily_subset = defaultdict(list)
    world_domains = defaultdict(list)
    line_info = []

    for file_path in matched_files:
        if file_path.suffix.lower() == ".jsonl":
            records, line_count, malformed = read_jsonl(file_path)
            line_info.append((file_path.name, line_count, malformed))
            for record in records:
                group = metric_key_for_record(record)
                if group == "daily":
                    task = daily_task_key(record)
                    subset = daily_subset_key(record)
                    daily_tasks[task].append(score_from_record(record))
                    daily_subset[subset].append(score_from_record(record))
                elif group == "world":
                    domain = world_domain_key(record)
                    world_domains[domain].append(score_from_record(record))
                elif group == "ib":
                    grouped["ib"].append(score_from_record(record))
                else:
                    grouped["unknown"].append(score_from_record(record))
        else:
            data = safe_load_json(file_path)
            if data is None:
                raise FileNotFoundError(f"Unreadable JSON file: {file_path}")
            if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
                records = data["results"]
            elif isinstance(data, list):
                records = data
            else:
                records = [data]
            for record in records:
                if not isinstance(record, dict):
                    continue
                group = metric_key_for_record(record)
                if group == "daily":
                    task = daily_task_key(record)
                    subset = daily_subset_key(record)
                    daily_tasks[task].append(score_from_record(record))
                    daily_subset[subset].append(score_from_record(record))
                elif group == "world":
                    domain = world_domain_key(record)
                    world_domains[domain].append(score_from_record(record))
                elif group == "ib":
                    grouped["ib"].append(score_from_record(record))
                else:
                    grouped["unknown"].append(score_from_record(record))

    main = {
        "ib": round(sum(grouped["ib"]) / len(grouped["ib"]) * 100, 2) if grouped["ib"] else None,
        "emotion": None,
        "deception": None,
        "do": None,
        "ws": None,
        "avg": None,
    }

    daily = {k: (round(sum(v) / len(v) * 100, 2) if v else None) for k, v in daily_tasks.items()}
    daily["subset_30s"] = round(sum(daily_subset.get("30s", [])) / len(daily_subset.get("30s", [])) * 100, 2) if daily_subset.get("30s") else None
    daily["subset_60s"] = round(sum(daily_subset.get("60s", [])) / len(daily_subset.get("60s", [])) * 100, 2) if daily_subset.get("60s") else None

    world = {k: (round(sum(v) / len(v) * 100, 2) if v else None) for k, v in world_domains.items()}

    if any(malformed for _, _, malformed in line_info):
        malformed_entries = [f"{name}={malformed}" for name, _, malformed in line_info if malformed]
        missing.append("malformed_jsonl_lines: " + ", ".join(malformed_entries))

    pair_files = [name for name, _, _ in line_info if "pair_alpha20_repair20" in name]
    if pair_files and any(count != 3172 for name, count, _ in line_info if "pair_alpha20_repair20" in name):
        missing.append("incomplete_lines: pair_alpha20_repair20 ws jsonl not equal to 3172")

    if grouped["unknown"]:
        missing.append(f"UNKNOWN_group: {len(grouped['unknown'])} records")

    return {"label": model_spec["label"], "main": main, "daily": daily, "world": world, "missing": missing}


# final_acc extraction.
def fmt(v):
    return "N/A" if v is None else f"{v:.2f}"


# summary-table construction.
def make_main_table(rows):
    header = "| Model | IB | Emotion | Deception | DO | WS | Avg |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        main = row["main"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                row["label"],
                fmt(main.get("ib")),
                fmt(main.get("emotion")),
                fmt(main.get("deception")),
                fmt(main.get("do")),
                fmt(main.get("ws")),
                fmt(main.get("avg")),
            )
        )
    return "\n".join(lines)


# benchmark averaging.
def make_daily_table(rows):
    header = "| Model | Daily-Omni 6 task | 30s subset | 60s subset |"
    sep = "|---|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        daily = row["daily"]
        task_value = None
        for task in DAILY_TASKS:
            if task in daily:
                task_value = daily.get(task)
                break
        lines.append(
            f"| {row['label']} | {fmt(task_value)} | {fmt(daily.get('subset_30s'))} | {fmt(daily.get('subset_60s'))} |"
        )
    return "\n".join(lines)


# benchmark averaging.
def make_world_table(rows):
    header = "| Model | " + " | ".join(WORLD_DOMAINS) + " |"
    sep = "|---|" + "---:|" * len(WORLD_DOMAINS)
    lines = [header, sep]
    for row in rows:
        world = row["world"]
        lines.append("| {} | {} |".format(row["label"], " | ".join(fmt(world.get(domain)) for domain in WORLD_DOMAINS)))
    return "\n".join(lines)


# model ordering.
def main():
    parser = argparse.ArgumentParser(description="ACOR evaluation summary helper (read-only).")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="Root directory containing evaluation outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file for the generated summary table.",
    )
    args = parser.parse_args()

    eval_root = resolve_eval_root(args.eval_root)
    results = [aggregate_model(spec, eval_root) for spec in MODEL_SPECS]
    print("main_results_markdown")
    print(make_main_table(results))
    print()
    print("daily_task_markdown")
    print(make_daily_table(results))
    print()
    print("worldsense_domain_markdown")
    print(make_world_table(results))
    print()
    print("missing_items")
    missing = []
    for row in results:
        missing.extend(row.get("missing", []))
    if missing:
        for item in missing:
            print(f"- {item}")
    else:
        print("- none")

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.write_text(
            "\n".join([
                "main_results_markdown",
                make_main_table(results),
                "",
                "daily_task_markdown",
                make_daily_table(results),
                "",
                "worldsense_domain_markdown",
                make_world_table(results),
                "",
                "missing_items",
                *([f"- {item}" for item in missing] if missing else ["- none"]),
            ]),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
