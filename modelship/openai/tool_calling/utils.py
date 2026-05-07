"""Utility functions for tool-calling.

Detects the tool-call parser a model needs by inspecting its chat template.
For GGUF files, the template is read from embedded metadata (authoritative —
it ships with the weights). For HF directories (and as a fallback for GGUF
dirs that also ship HF tokenizer files), the template is read from
``tokenizer_config.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from modelship.logging import get_logger

logger = get_logger("tool_calling.utils")


def read_chat_template(model_path: str | Path) -> str | None:
    """Read a model's chat template string from disk.

    For GGUF files, prefers ``tokenizer.chat_template`` in the file's
    embedded metadata (authoritative — it's what shipped with the weights).
    Falls back to ``tokenizer_config.json`` in the same directory if the
    GGUF read fails or the key is absent. For non-GGUF paths, reads
    ``tokenizer_config.json`` directly.
    """
    path = Path(model_path)
    if path.is_file() and path.suffix == ".gguf":
        template = _read_chat_template_from_gguf(path)
        if template is not None:
            return template
        config_dir = path.parent
    else:
        config_dir = path
    return _read_chat_template_from_tokenizer_config(config_dir)


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


def _read_chat_template_from_tokenizer_config(config_dir: Path) -> str | None:
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
    return template


def _read_chat_template_from_gguf(gguf_path: Path) -> str | None:
    """Read ``tokenizer.chat_template`` from a GGUF file's metadata.

    Returns ``None`` if the key is absent, the file isn't readable as GGUF,
    or the gguf package is unavailable.
    """
    try:
        from gguf import GGUFReader
    except ImportError:
        logger.debug("gguf package not available; skipping GGUF metadata read for %s", gguf_path)
        return None
    try:
        reader = GGUFReader(str(gguf_path))
    except Exception as e:
        logger.warning("Failed to open %s as GGUF for tool detection: %s", gguf_path, e)
        return None
    field = reader.get_field("tokenizer.chat_template")
    if field is None:
        return None
    try:
        # GGUF string fields expose `.parts` (numpy arrays of bytes) selected by `.data` indices.
        parts = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
    except Exception as e:
        logger.warning("Failed to decode chat_template from %s: %s", gguf_path, e)
        return None
    template = "".join(parts)
    if not template:
        return None
    return template
