"""ACOR affective-context prompts and SFT target construction."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

AFFECTIVE_CONTEXT_INSTRUCTION = (
    "First summarize the global scene and the affective context. Describe relevant "
    "facial expression, gaze, vocal tone, speaking pace, gesture, hesitation, "
    "interaction, and temporal emotion trajectory. Then reason from that evidence."
)

STRUCTURED_OUTPUT_INSTRUCTION = (
    "Return exactly three sections in this order: "
    "<context>...</context><think>...</think><answer>...</answer>."
)


def build_affective_prompt(question: str, options: Optional[Iterable[str]] = None) -> str:
    option_text = ""
    if options:
        option_text = "\nOptions:\n" + "\n".join(str(option) for option in options)
    return f"{question.strip()}{option_text}\n\n{AFFECTIVE_CONTEXT_INSTRUCTION}\n{STRUCTURED_OUTPUT_INSTRUCTION}"


def build_structured_target(context: str, reasoning: str, answer: str) -> str:
    return (
        f"<context>{context.strip()}</context>"
        f"<think>{reasoning.strip()}</think>"
        f"<answer>{answer.strip()}</answer>"
    )


def format_sft_sample(sample: Mapping[str, object]) -> dict:
    """Convert a public sample record into the message format used by ACOR SFT."""
    question = str(sample.get("question", ""))
    options = sample.get("options")
    prompt = build_affective_prompt(question, options if isinstance(options, list) else None)
    target = build_structured_target(
        str(sample.get("context", "")),
        str(sample.get("reasoning", "")),
        str(sample.get("answer", "")),
    )
    content = []
    if sample.get("video"):
        content.append({"type": "video", "video": sample["video"]})
    if sample.get("audio"):
        content.append({"type": "audio", "audio": sample["audio"]})
    content.append({"type": "text", "text": prompt})
    return {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": target}]}
