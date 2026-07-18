#!/usr/bin/env python3
"""Read-only ACOR paper statistics helper."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Historical result labels are optional; model-generation code is not included.
HISTORICAL_RESULT_LABELS = [
    "baseline",
    "B500",
    "alpha20",
    "repair20",
    "pair_lam025",
]

KEYWORDS = [
    "emotion_full_sft",
    "stage2B_emotion_preserving_grpo500",
    "alpha20",
    "pair_alpha20",
    "acor_ablate",
    "ib_clean",
    "do_clean",
    "ws_fixed",
    "predictions_rank0",
]

CHECKPOINT_PATHS = []


def eprint(*args: Any) -> None:
    print(*args)


def safe_json_load(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def safe_read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int, int]:
    rows: List[Dict[str, Any]] = []
    raw_lines = 0
    malformed = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw_lines += 1
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                    else:
                        malformed += 1
                except Exception:
                    malformed += 1
    except Exception:
        return [], 0, 0
    return rows, raw_lines, malformed


def resolve_eval_root(args_eval_root: Optional[Path]) -> Path:
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


def iter_eval_files(eval_root: Path) -> Iterable[Path]:
    if not eval_root.exists():
        return []
    return sorted(
        [p for p in eval_root.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}],
        key=lambda p: str(p).lower(),
    )


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return " ".join(normalize_text(i) for i in x)
    if isinstance(x, dict):
        return " ".join(normalize_text(v) for v in x.values())
    return str(x)


def pick_first(record: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def get_generation_text(record: Dict[str, Any]) -> str:
    for key in ["output", "raw_output", "full_response", "model_response", "generated_response", "decoded_output", "completion_text", "response", "text", "completion", "model_output", "generated_text"]:
        if key in record and record[key] not in (None, ""):
            return normalize_text(record[key])
    if record:
        print("missing_generation_keys:\t" + ",".join(sorted(record.keys())))
    return ""


def get_answer_text(record: Dict[str, Any]) -> str:
    for key in ["prediction", "chosen_answer_for_eval", "first_answer", "last_answer", "pred", "model_prediction", "answer_pred", "final_answer", "generated_answer"]:
        if key in record and record[key] not in (None, ""):
            return normalize_text(record[key])
    return ""


def get_pred_text(record: Dict[str, Any]) -> str:
    return get_answer_text(record)


def get_gt_text(record: Dict[str, Any]) -> str:
    for key in ["gt", "gt_answer", "ground_truth", "solution", "label", "target", "gold", "reference", "answer"]:
        if key in record and record[key] not in (None, ""):
            return normalize_text(record[key])
    return ""


def extract_context(text: str) -> str:
    m = re.search(r"<context>\s*(.*?)\s*</context>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def maybe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def score_from_record(record: Dict[str, Any]) -> Optional[float]:
    for key in ["mean_acc", "accuracy", "acc", "score", "reward"]:
        if key in record:
            val = maybe_float(record[key])
            if val is not None:
                return val
    pred = get_pred_text(record)
    gt = get_gt_text(record)
    if pred and gt:
        return 1.0 if pred.strip() == gt.strip() else 0.0
    return None


def rough_model_name(path: Path, record: Optional[Dict[str, Any]] = None) -> str:
    candidates = []
    if record:
        for key in ["model_name", "model", "base_model", "checkpoint", "run_name", "output_dir"]:
            v = record.get(key)
            if v:
                candidates.append(normalize_text(v))
    candidates.append(path.name)
    candidates.append(str(path.parent.name))
    joined = " ".join(candidates).lower()
    if "emotion_preserving" in joined:
        return "emotion_preserving"
    if "stage2b" in joined and "b500" in joined:
        return "stage2B_b500"
    if "pair_alpha20" in joined or "repair20" in joined:
        return "pair_alpha20_repair20"
    if "alpha20" in joined:
        return "alpha20"
    if "acor" in joined:
        return "acor"
    if "ib_clean" in joined:
        return "ib_clean"
    if "do_clean" in joined:
        return "do_clean"
    if "ws_fixed" in joined:
        return "ws_fixed"
    return path.stem[:80]


def list_files_cmd(eval_root: Path) -> int:
    found = []
    for p in iter_eval_files(eval_root):
        low = p.name.lower()
        if any(k.lower() in low for k in KEYWORDS):
            found.append(p)
    for p in found:
        print(p.as_posix())
    return 0


def read_result_file(path: Path) -> Tuple[List[Dict[str, Any]], int, int, int]:
    if path.suffix.lower() == ".jsonl":
        rows, raw_lines, malformed = safe_read_jsonl(path)
        duplicate_count = len(rows) - len({json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows})
        return rows, raw_lines, malformed, max(0, duplicate_count)
    data = safe_json_load(path)
    if data is None:
        return [], 0, 0, 0
    if isinstance(data, list):
        rows = [r for r in data if isinstance(r, dict)]
    elif isinstance(data, dict) and isinstance(data.get("results"), list):
        rows = [r for r in data["results"] if isinstance(r, dict)]
    elif isinstance(data, dict):
        rows = [data]
    else:
        rows = []
    duplicate_count = len(rows) - len({json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows})
    return rows, len(rows), 0, max(0, duplicate_count)


def summary_cmd(eval_root: Path) -> int:
    files = list(iter_eval_files(eval_root))
    print("file_name\tmean_acc\traw_lines\tmerged_count\tduplicate_count\tdataset_len\tmodel")
    for path in files:
        rows, raw_lines, malformed, dup = read_result_file(path)
        scores = [s for s in (score_from_record(r) for r in rows) if s is not None]
        mean_acc = sum(scores) / len(scores) if scores else None
        dataset_len = None
        for key in ["dataset_len", "dataset_length", "num_samples", "total", "count"]:
            if rows:
                v = rows[0].get(key)
                if isinstance(v, int):
                    dataset_len = v
                    break
        merged_count = len(rows)
        model = rough_model_name(path, rows[0] if rows else None)
        print(
            f"{path.name}\t"
            f"{('N/A' if mean_acc is None else f'{mean_acc:.4f}')}\t"
            f"{raw_lines}\t{merged_count}\t{dup}\t"
            f"{dataset_len if dataset_len is not None else 'N/A'}\t{model}"
        )
    return 0


def _field_text(rec: Dict[str, Any], keys: Iterable[str]) -> str:
    return " ".join(normalize_text(rec.get(k, "")) for k in keys).lower()


def is_emotion_record(rec: Dict[str, Any]) -> bool:
    keys = ["source", "dataset", "dataset_name", "problem_type", "type", "Type", "question_type", "id", "qid", "sample_id", "sample_index", "uid", "video", "video_id", "path"]
    text = _field_text(rec, keys)
    if "emer" in text:
        return True
    for key in ["source", "dataset", "dataset_name", "problem_type", "type", "Type", "question_type"]:
        if "emotion" in normalize_text(rec.get(key, "")).lower():
            return True
    return False


def is_deception_record(rec: Dict[str, Any]) -> bool:
    keys = ["source", "dataset", "dataset_name", "problem_type", "type", "Type", "question_type", "id", "qid", "sample_id", "sample_index", "uid", "video", "video_id", "path"]
    text = _field_text(rec, keys)
    if "mdpe" in text:
        return True
    for key in ["source", "dataset", "dataset_name", "problem_type", "type", "Type", "question_type"]:
        if "deception" in normalize_text(rec.get(key, "")).lower():
            return True
    return False


def ib_group(rec: Dict[str, Any]) -> str:
    t = normalize_text(rec.get("Type", rec.get("type", rec.get("problem_type", rec.get("question_type", ""))))).strip().lower()
    if is_emotion_record(rec):
        return "Emotion"
    if is_deception_record(rec):
        return "Deception"
    if "why" in t:
        return "Why"
    if "how" in t:
        return "How"
    if "what" in t:
        return "What"
    if "when" in t:
        return "When"
    if "who" in t or "which" in t:
        return "Who/Which"
    return "Other"


def canonical_answer(text: str) -> str:
    t = normalize_text(text).strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" .,!?:;\"'[]{}()")
    return t


def _extract_choice_letter(text: str) -> str:
    raw = normalize_text(text).strip()
    if not raw:
        return ""

    patterns = [
        r"\boption\s*([A-E])\b",
        r"\b(?:final\s+)?answer\s*(?:is|:)?\s*\(?\s*([A-E])\s*\)?\b",
        r"\banswer\s*(?:is|:)?\s*\(?\s*([A-E])\s*\)?\b",
        r"\(\s*([A-E])\s*\)",
        r"\b([A-E])\s*[\).:]",
    ]
    for pat in patterns:
        m = re.search(pat, raw, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    canonical = canonical_answer(raw)
    if canonical in {"a", "b", "c", "d", "e"}:
        return canonical.upper()
    if re.fullmatch(r"\(?\s*([A-E])\s*\)?\.?", raw, flags=re.IGNORECASE):
        return re.fullmatch(r"\(?\s*([A-E])\s*\)?\.?", raw, flags=re.IGNORECASE).group(1).upper()
    return ""


def extract_final_answer(text: str) -> str:
    raw = normalize_text(text)
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        letter = _extract_choice_letter(candidate)
        if letter:
            return letter
        return canonical_answer(candidate)
    for pat in [r"(?:^|\n)\s*(?:final\s+)?answer\s*:\s*(.+)$", r"(?:^|\n)\s*answer\s*:\s*(.+)$"]:
        m = re.search(pat, raw, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            candidate = m.group(1).strip()
            letter = _extract_choice_letter(candidate)
            if letter:
                return letter
            return canonical_answer(candidate)
    letter = _extract_choice_letter(raw)
    if letter:
        return letter
    return canonical_answer(raw)


def _coerce_correct_value(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value > 0
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "correct", "yes", "y"}:
            return True
        if low in {"false", "0", "incorrect", "no", "n"}:
            return False
    return None


def record_correct(rec: Dict[str, Any]) -> bool:
    for key in ["correct", "is_correct", "acc", "accuracy", "score", "reward"]:
        if key in rec:
            coerced = _coerce_correct_value(rec[key])
            if coerced is not None:
                return coerced
    ans_text = get_answer_text(rec)
    gen_text = get_generation_text(rec) if not ans_text else ""
    pred = extract_final_answer(ans_text) if ans_text else extract_final_answer(gen_text)
    gt_raw = get_gt_text(rec)
    gt = extract_final_answer(gt_raw)
    if pred and gt:
        return pred == gt or canonical_answer(pred) == canonical_answer(gt_raw) or canonical_answer(ans_text or gen_text) == canonical_answer(gt_raw)
    return False


def ib_strict_cmd(pred_path: Path) -> int:
    rows, raw_lines, malformed = safe_read_jsonl(pred_path)
    if not rows:
        print(f"No readable rows in {pred_path}")
        return 0

    buckets = defaultdict(lambda: {"count": 0, "correct": 0})
    for rec in rows:
        g = ib_group(rec)
        correct = 1 if record_correct(rec) else 0
        buckets[g]["count"] += 1
        buckets[g]["correct"] += correct

    order = ["Why", "How", "What", "When", "Who/Which", "Other", "Emotion", "Deception"]
    print(f"file\t{pred_path.name}")
    print(f"raw_lines\t{raw_lines}")
    print(f"malformed\t{malformed}")
    group_accs = []
    nonempty_accs = []
    for g in order:
        c = buckets[g]["count"]
        corr = buckets[g]["correct"]
        acc = corr / c if c else 0.0
        group_accs.append(acc)
        if c:
            nonempty_accs.append(acc)
        print(f"{g}\tcount={c}\tcorrect={corr}\tacc={acc:.4f}")
    grouped_avg_all8 = sum(group_accs) / len(group_accs) if group_accs else 0.0
    grouped_avg_nonempty = sum(nonempty_accs) / len(nonempty_accs) if nonempty_accs else 0.0
    print(f"grouped_average_nonempty\t{grouped_avg_nonempty:.4f}")
    print(f"grouped_average_all8\t{grouped_avg_all8:.4f}")
    return 0


CUE_PATTERNS = {
    "face": [r"\bface\b", r"\bfacial\b", r"\bexpression\b", r"\bsmile\b", r"\bfrown\b", r"\bgaze\b", r"\beye\b", r"\beyes\b", r"\bmouth\b", r"\bbrow\b"],
    "voice": [r"\bvoice\b", r"\btone\b", r"\bpitch\b", r"\bvolume\b", r"\bpace\b", r"\bpause\b", r"\bhesitation\b", r"\bsigh\b", r"\blaugh\b", r"\bcry\b", r"\bwhisper\b"],
    "gesture_posture": [r"\bgesture\b", r"\bposture\b", r"\bpose\b", r"\bhand\b", r"\bhands\b", r"\bnod\b", r"\bshrug\b", r"\blean\b", r"\bbody\b", r"\bhead\b"],
    "interaction": [r"\binteraction\b", r"\binteract\b", r"\bconversation\b", r"\brespond\b", r"\breaction\b", r"\bdialogue\b", r"\bturn[- ]taking\b", r"\bengage\b"],
    "inconsistency": [r"\binconsistent\b", r"\binconsistency\b", r"\bcontradict\b", r"\bcontradiction\b", r"\bmismatch\b", r"\bconflict\b", r"\bdeceptive\b", r"\bnervous\b", r"\bavoidance\b"],
}


def detect_emotion_subset(rec: Dict[str, Any]) -> bool:
    return is_emotion_record(rec)


def cue_rates(text: str) -> Dict[str, int]:
    low = text.lower()
    return {k: int(any(re.search(p, low) for p in pats)) for k, pats in CUE_PATTERNS.items()}


def cue_cmd(pred_path: Path) -> int:
    rows, raw_lines, malformed = safe_read_jsonl(pred_path)
    if not rows:
        print(f"No readable rows in {pred_path}")
        return 0

    totals = Counter()
    cue_hits = Counter()
    emotion_totals = Counter()
    emotion_hits = Counter()
    valid_context = 0
    total_context_words = 0

    for rec in rows:
        text = get_generation_text(rec)
        if not text:
            print("missing_text_keys:\t" + ",".join(sorted(rec.keys())))
            continue
        ctx = extract_context(text)
        if re.search(r"<context>\s*(.*?)\s*</context>", text, flags=re.DOTALL | re.IGNORECASE):
            valid_context += 1
        words = len(ctx.split()) if ctx else 0
        total_context_words += words
        rates = cue_rates(ctx)
        for k, v in rates.items():
            cue_hits[k] += v
        totals["n"] += 1

        if detect_emotion_subset(rec):
            emotion_totals["n"] += 1
            for k, v in rates.items():
                emotion_hits[k] += v

    n = totals["n"] or 1
    print(f"file\t{pred_path.name}")
    print(f"raw_lines\t{raw_lines}")
    print(f"malformed\t{malformed}")
    print(f"valid_context_rate\t{valid_context / n:.4f}")
    print(f"avg_context_words\t{total_context_words / n:.2f}")
    for k in CUE_PATTERNS:
        print(f"{k}_rate\t{cue_hits[k] / n:.4f}")
    if emotion_totals["n"]:
        en = emotion_totals["n"]
        print(f"emotion_subset_n\t{en}")
        for k in CUE_PATTERNS:
            print(f"emotion_subset_{k}_rate\t{emotion_hits[k] / en:.4f}")
    return 0


def get_nested_source(record: Dict[str, Any], key: str) -> Any:
    src = record.get("source_sample")
    if isinstance(src, dict):
        return src.get(key)
    return None


def _first_nonempty(*values: Any) -> Any:
    for v in values:
        if v not in (None, ""):
            return v
    return ""


def pair_rows(base_rows: List[Dict[str, Any]], acor_rows: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    index = defaultdict(list)
    match_keys = ["sample_index", "id", "sample_id", "question_id", "qid", "uid", "index", "idx", "video_id"]
    for i, r in enumerate(acor_rows):
        for key in match_keys:
            if key in r and r[key] not in (None, ""):
                index[(key, normalize_text(r[key]))].append((i, r))
    pairs = []
    used = set()
    for i, br in enumerate(base_rows):
        matched = None
        for key in match_keys:
            if key in br and br[key] not in (None, ""):
                cand = index.get((key, normalize_text(br[key])), [])
                for j, ar in cand:
                    if j not in used:
                        matched = (j, ar)
                        break
            if matched:
                break
        if matched is None and i < len(acor_rows) and i not in used:
            matched = (i, acor_rows[i])
        if matched:
            used.add(matched[0])
            pairs.append((i, br, matched[1]))
    return pairs


def answer_ok(rec: Dict[str, Any]) -> bool:
    return record_correct(rec)


def case_cmd(base_path: Path, acor_path: Path, top_k: int) -> int:
    base_rows, _, _ = safe_read_jsonl(base_path)
    acor_rows, _, _ = safe_read_jsonl(acor_path)
    if not base_rows or not acor_rows:
        print("No readable rows in one or both files")
        return 0

    pairs = pair_rows(base_rows, acor_rows)
    candidates = []
    for idx, br, ar in pairs:
        base_correct = answer_ok(br)
        acor_correct = answer_ok(ar)
        if base_correct:
            continue
        if not acor_correct:
            continue
        acor_ctx = extract_context(get_pred_text(ar))
        cue_hit = any(k in acor_ctx.lower() for k in ["face", "voice", "gesture", "interaction", "inconsisten"])
        subset = ib_group(ar)
        if subset in {"Emotion", "Deception"} or cue_hit:
            candidates.append((idx, subset, br, ar))

    print(f"base_file\t{base_path.name}")
    print(f"acor_file\t{acor_path.name}")
    print(f"paired\t{len(pairs)}")
    print(f"candidates\t{len(candidates)}")
    for idx, subset, br, ar in candidates[:top_k]:
        print("---")
        sample_index = pick_first(ar, ["sample_index", "id", "qid", "sample_id", "idx"], pick_first(br, ["sample_index", "id", "qid", "sample_id", "idx"], idx))
        print(f"sample_index\t{sample_index}")
        print(f"subset\t{subset}")
        print(f"video\t{pick_first(ar, ['video', 'video_id', 'path'], pick_first(br, ['video', 'video_id', 'path'], ''))}")
        print(f"question\t{normalize_text(pick_first(ar, ['question', 'prompt', 'problem'], pick_first(br, ['question', 'prompt', 'problem'], '')))[:500]}")
        print(f"gt\t{get_gt_text(ar) or get_gt_text(br)}")
        print(f"base_pred\t{get_pred_text(br)[:500]}")
        print(f"acor_pred\t{get_pred_text(ar)[:500]}")
    return 0


def checkpoints_cmd() -> int:
    for p in CHECKPOINT_PATHS:
        print(f"{p}\t{'YES' if p.exists() else 'NO'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ACOR paper statistics helper (read-only).")
    ap.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="Root directory containing evaluation outputs.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")
    sub.add_parser("summary")

    p = sub.add_parser("ib-strict")
    p.add_argument("--pred", required=True, type=Path)

    p = sub.add_parser("cue")
    p.add_argument("--pred", required=True, type=Path)

    p = sub.add_parser("case")
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--acor", required=True, type=Path)
    p.add_argument("--top-k", type=int, default=10)

    sub.add_parser("checkpoints")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    eval_root = resolve_eval_root(args.eval_root)

    if args.cmd == "list":
        return list_files_cmd(eval_root)
    if args.cmd == "summary":
        return summary_cmd(eval_root)
    if args.cmd == "ib-strict":
        return ib_strict_cmd(args.pred)
    if args.cmd == "cue":
        return cue_cmd(args.pred)
    if args.cmd == "case":
        return case_cmd(args.base, args.acor, args.top_k)
    if args.cmd == "checkpoints":
        return checkpoints_cmd()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
