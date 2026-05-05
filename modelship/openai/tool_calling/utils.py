"""Utility functions for tool-calling.

Includes logic for detecting tool-calling support and format from model
configuration by inspecting ``tokenizer_config.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from modelship.logging import get_logger

logger = get_logger("tool_calling.utils")


def detect_tool_parser(model_path: str | Path) -> str | None:
    """Return the name of the tool-call parser required by the model or None.

    Inspects ``tokenizer_config.json`` in ``model_path``.

    - Returns "hermes" if `<tool_call>` markers are found.
    - Returns "mistral" if `[TOOL_CALLS]` markers are found.
    - Returns "llama3_json" if `<|python_tag|>` markers are found.
    - Returns "unknown" if tool logic is found but format markers are not recognized.
    """
    path = Path(model_path)
    # If model_path is a file (e.g. a GGUF), the config is in its parent dir.
    config_dir = path.parent if path.is_file() else path
    config_path = config_dir / "tokenizer_config.json"

    if not config_path.exists():
        return None

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to load %s for tool detection: %s", config_path, e)
        return None

    template = config.get("chat_template")
    if not template or not isinstance(template, str):
        return None

    # 1. Check for tool-calling support at all.
    # Most templates use `tools` or `tool_calls` variables in Jinja2 logic.
    if "tools" not in template and "tool_calls" not in template:
        return None

    # 2. Identify the specific format/parser.
    if "<tool_call>" in template:
        return "hermes"

    if "[TOOL_CALLS]" in template:
        return "mistral"

    if "<|python_tag|>" in template or "<|eom_id|>" in template:
        # Llama 3.1+ specific markers.
        return "llama3_json"

    # Tool logic found but no recognized markers.
    return "unknown"
