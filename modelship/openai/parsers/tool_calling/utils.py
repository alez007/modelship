"""Tool-call parser detection by chat-template inspection."""

from __future__ import annotations

from pathlib import Path

from modelship.openai.parsers.utils import read_chat_template


def detect_tool_parser(model_path: str | Path) -> str | None:
    """Return the name of the tool-call parser required by the model or None.

    - Returns "hermes" if `<tool_call>` markers are found.
    - Returns "mistral" if `[TOOL_CALLS]` markers are found.
    - Returns "llama3_json" if `<|python_tag|>` markers are found.
    - Returns "unknown" if tool logic is found but format markers are not recognized.
    - Returns ``None`` if no chat template / no tool logic is detected.
    """
    template = read_chat_template(model_path)
    if template is None:
        return None
    return classify_template(template)


def classify_template(template: str) -> str | None:
    """Map a chat-template string to a parser name based on tool-call markers."""
    if "tools" not in template and "tool_calls" not in template:
        return None

    if "<tool_call>" in template:
        return "hermes"

    if "[TOOL_CALLS]" in template:
        return "mistral"

    if "<|python_tag|>" in template or "<|eom_id|>" in template:
        return "llama3_json"

    return "unknown"
