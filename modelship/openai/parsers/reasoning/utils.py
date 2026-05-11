"""Reasoning parser detection by chat-template inspection."""

from __future__ import annotations

from pathlib import Path

from modelship.openai.parsers.utils import read_chat_template


def detect_reasoning_parser(model_path: str | Path) -> str | None:
    """Return the name of the reasoning parser the model needs, or None."""
    template = read_chat_template(model_path)
    if template is None:
        return None
    return classify_template(template)


def classify_template(template: str) -> str | None:
    """Map a chat-template string to a reasoning parser name based on markers."""
    if "<|channel>thought" in template:
        return "gemma4"
    if "<think>" in template or "</think>" in template:
        return "deepseek_r1"
    return None
