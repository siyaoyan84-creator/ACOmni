"""Shared structured-output parsing for training, evaluation, and analysis."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

TAG_PATTERN = r"<{tag}>\s*([\s\S]*?)\s*</{tag}>"


def _extract_tag(text: object, tag: str) -> str:
    match = re.search(TAG_PATTERN.format(tag=re.escape(tag)), str(text or ""), re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_context(text: object) -> str:
    return _extract_tag(text, "context")


def extract_think(text: object) -> str:
    return _extract_tag(text, "think")


def extract_answer(text: object) -> str:
    return _extract_tag(text, "answer")


@dataclass(frozen=True)
class StructuredOutput:
    context: str
    think: str
    answer: str
    valid: bool
    missing_tags: tuple[str, ...]


def validate_structured_output(text: object, require_context: bool = True) -> StructuredOutput:
    fields = {
        "context": extract_context(text),
        "think": extract_think(text),
        "answer": extract_answer(text),
    }
    required = ("context", "think", "answer") if require_context else ("think", "answer")
    missing = tuple(name for name in required if not fields[name])
    return StructuredOutput(**fields, valid=not missing, missing_tags=missing)


def normalize_text(text: object) -> str:
    value = str(text or "").strip().lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", value)


def parse_multiple_choice_answer(
    text: object,
    choices: Optional[Sequence[str]] = None,
    aliases: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return a normalized option letter, or ``None`` for invalid output."""
    answer = extract_answer(text) or str(text or "")
    letter_match = re.search(r"(?<![A-Za-z])([A-H])(?:\s*[\.\):]|(?![A-Za-z]))", answer, re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()

    normalized = normalize_text(answer)
    alias_map = {normalize_text(k): v.upper() for k, v in (aliases or {}).items()}
    if normalized in alias_map:
        return alias_map[normalized]

    if choices:
        for index, choice in enumerate(choices):
            if normalize_text(choice) in normalized or normalized in normalize_text(choice):
                return chr(ord("A") + index)
    return None


def parse_multilabel_answer(text: object, labels: Iterable[str]) -> set[str]:
    answer = normalize_text(extract_answer(text) or text)
    return {label for label in labels if normalize_text(label) in answer}
