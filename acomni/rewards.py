"""Dependency-light public interfaces for ACOmni rewards.

The paper term *Emotion Category Consistency Reward* corresponds to
``emotion_consistency_reward``. The training entrypoint calls the original
implementations in ``src/open_r1/vlm_modules/qwenomni_module.py``; this module
exposes the same staged affective scoring rule for reuse in evaluation/tests.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterable, Mapping, Optional, Sequence
from urllib import request

from .response_parser import (
    extract_answer,
    extract_context,
    normalize_text,
    parse_multilabel_answer,
    parse_multiple_choice_answer,
    validate_structured_output,
)

EMOTION_ALIASES = {
    "angry": "anger", "anger": "anger",
    "happy": "joy", "happiness": "joy", "joy": "joy", "mocking": "joy", "proud": "joy",
    "sad": "sad", "sadness": "sad",
    "fear": "fear",
    "disgust": "disgust",
    "neutral": "neutral",
    "surprised": "surprise", "surprise": "surprise",
}

EMOTION_KEYWORDS = (
    "emotion", "emotional", "feeling", "feel", "felt", "happy", "sad", "angry",
    "fear", "surprise", "disgust", "joy", "anxious", "calm", "excited",
    "frustrated", "relaxed", "tense", "pleased", "disappointed", "satisfied",
    "mood", "sentiment", "affective", "affect",
)
MULTIMODAL_EVIDENCE_KEYWORDS = (
    "visual", "see", "saw", "observe", "look", "watch", "facial", "expression",
    "face", "acoustic", "hear", "heard", "voice", "tone", "sound", "audio",
    "speak", "said", "textual", "text", "subtitle", "word", "caption", "read",
    "evidence", "cue", "signal", "indicator", "sign",
)
TEMPORAL_TRAJECTORY_KEYWORDS = (
    "trajectory", "change", "changed", "shift", "shifted", "transition",
    "initially", "first", "then", "later", "finally", "eventually", "from", "to",
    "become", "became", "transform", "evolve", "progression", "development",
    "over time",
)

JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "qwen-plus")
JUDGE_TEMPERATURE = float(os.environ.get("LLM_JUDGE_TEMPERATURE", "0"))
JUDGE_MAX_TOKENS = int(os.environ.get("LLM_JUDGE_MAX_TOKENS", "16"))

JUDGE_PROMPT = """Evaluate how well the 'hypothesis' captures affective/emotional aspects from the 'reference'.

Score 0-5:
5: Accurately identifies key emotional states and affective tones
4: Captures most emotional aspects with minor gaps
3: Shows partial understanding of affective content
2: Limited connection to affective aspects
1: Barely touches on emotional content
0: No affective aspects reflected

reference: {reference}
hypothesis: {hypothesis}

Return only the score number:
"""


def _completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion or "")


def format_reward(completions: Sequence[object], **_: object) -> list[float]:
    return [float(validate_structured_output(_completion_text(item)).valid) for item in completions]


def accuracy_reward(
    completions: Sequence[object],
    solution: Sequence[object],
    problem_type: Optional[Sequence[str]] = None,
    choices: Optional[Sequence[Sequence[str]]] = None,
    **_: object,
) -> list[float]:
    rewards = []
    for index, (completion, gold) in enumerate(zip(completions, solution)):
        task = (problem_type[index] if problem_type and index < len(problem_type) else "multiple choice").lower()
        prediction_text = _completion_text(completion)
        gold_text = str(gold)
        if "multiple" in task or task in {"mc", "emotion", "emer_ov_mc"}:
            option_list = choices[index] if choices and index < len(choices) else None
            pred = parse_multiple_choice_answer(prediction_text, option_list)
            target = parse_multiple_choice_answer(gold_text, option_list)
            rewards.append(float(pred is not None and pred == target))
        elif "multi" in task:
            labels = sorted(set(EMOTION_ALIASES.values()))
            pred = parse_multilabel_answer(prediction_text, labels)
            target = parse_multilabel_answer(gold_text, labels)
            if not pred and not target:
                rewards.append(0.0)
            else:
                tp = len(pred & target)
                precision = tp / len(pred) if pred else 0.0
                recall = tp / len(target) if target else 0.0
                rewards.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        else:
            rewards.append(float(normalize_text(extract_answer(prediction_text)) == normalize_text(extract_answer(gold_text))))
    return rewards


def _keyword_fraction(text: str, keywords: Iterable[str]) -> float:
    lowered = text.lower()
    hits = sum(1 for keyword in keywords if keyword in lowered)
    return min(1.0, hits / 3.0)


def parse_judge_score(response: object) -> float:
    match = re.search(r"(?<!\d)([0-5](?:\.\d+)?)", str(response or ""))
    if not match:
        return 0.0
    return max(0.0, min(1.0, float(match.group(1)) / 5.0))


def call_llm_judge(reference: str, hypothesis: str) -> float:
    api_key = os.environ.get("LLM_JUDGE_API_KEY")
    api_base = os.environ.get("LLM_JUDGE_API_BASE")
    if not api_key or not api_base:
        return 0.0
    payload = {
        "model": JUDGE_MODEL,
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": JUDGE_MAX_TOKENS,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(reference=reference, hypothesis=hypothesis)}],
    }
    req = request.Request(
        api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        return parse_judge_score(body["choices"][0]["message"]["content"])
    except Exception:
        return 0.0


def affective_context_reward(
    completions: Sequence[object],
    solution: Sequence[object],
    use_llm_judge: bool = True,
    **_: object,
) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, solution):
        context = extract_context(_completion_text(completion))
        reference = extract_context(gold)
        if not context or not reference:
            rewards.append(0.0)
            continue
        reward = 0.2
        reward += 0.2 if any(keyword in context.lower() for keyword in EMOTION_KEYWORDS) else 0.0
        reward += 0.2 if any(keyword in context.lower() for keyword in MULTIMODAL_EVIDENCE_KEYWORDS) else 0.0
        reward += 0.2 if any(keyword in context.lower() for keyword in TEMPORAL_TRAJECTORY_KEYWORDS) else 0.0
        judge = call_llm_judge(reference, context) if use_llm_judge and len(context) >= 10 else 0.0
        rewards.append(max(0.0, min(1.0, reward + 0.2 * judge)))
    return rewards


def _normalize_emotion(text: object) -> Optional[str]:
    normalized = normalize_text(extract_answer(text) or text)
    for alias, category in EMOTION_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return category
    return None


def emotion_consistency_reward(
    completions: Sequence[object],
    solution: Sequence[object],
    **_: object,
) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, solution):
        pred = _normalize_emotion(_completion_text(completion))
        target = _normalize_emotion(gold)
        rewards.append(float(pred is not None and target is not None and pred == target))
    return rewards
