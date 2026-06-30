"""Shared chat-template reading for parser detection (tool-calling, reasoning)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modelship.logging import get_logger

logger = get_logger("openai.parsers.utils")

# Render args we supply ourselves; a user kwarg of the same name would be a
# duplicate-keyword TypeError in render_jinja_template.
_RESERVED_RENDER_KEYS = frozenset(
    {"conversations", "messages", "tools", "chat_template", "add_generation_prompt", "bos_token", "eos_token"}
)


def render_generation_prompt(chat_template: str, chat_template_kwargs: dict[str, Any] | None = None) -> str:
    """Render a minimal generation prompt from a chat-template string.

    Standalone Jinja render (HF's ``render_jinja_template``, with its undefined-safe
    polyfills) of a single user turn with ``add_generation_prompt``. Used to probe
    whether reasoning is primed under the deployment's effective kwargs without
    touching a live engine. ``bos_token``/``eos_token`` are irrelevant to reasoning
    markers, so they are left empty.
    """
    from transformers.utils.chat_template_utils import render_jinja_template

    kwargs = {k: v for k, v in (chat_template_kwargs or {}).items() if k not in _RESERVED_RENDER_KEYS}
    rendered, _ = render_jinja_template(
        conversations=[[{"role": "user", "content": "hi"}]],
        chat_template=chat_template,
        add_generation_prompt=True,
        bos_token="",
        eos_token="",
        **kwargs,
    )
    return rendered[0]


def read_chat_template(model_path: str | Path) -> str | None:
    """Read a model's chat template string from disk.

    For GGUF files, prefers ``tokenizer.chat_template`` in the file's
    embedded metadata. Falls back to ``tokenizer_config.json`` in the
    same directory. For non-GGUF paths, reads ``tokenizer_config.json``
    directly.
    """
    path = Path(model_path)
    if path.is_file() and path.suffix.lower() == ".gguf":
        template = _read_chat_template_from_gguf(path)
        if template is not None:
            return template
        config_dir = path.parent
    else:
        config_dir = path
    return _read_chat_template_from_tokenizer_config(config_dir)


def _read_chat_template_from_tokenizer_config(config_dir: Path) -> str | None:
    config_path = config_dir / "tokenizer_config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load %s for parser detection: %s", config_path, e)
        else:
            template = config.get("chat_template")
            if template and isinstance(template, str):
                return template

    # HF newer convention (transformers 4.43+): chat_template lives in
    # a separate file alongside tokenizer_config.json. AWQ/GPTQ quantized
    # repacks often ship it this way.
    jinja_path = config_dir / "chat_template.jinja"
    if jinja_path.exists():
        try:
            return jinja_path.read_text(encoding="utf-8") or None
        except OSError as e:
            logger.warning("Failed to load %s for parser detection: %s", jinja_path, e)

    return None


def _read_chat_template_from_gguf(gguf_path: Path) -> str | None:
    try:
        from gguf import GGUFReader
    except ImportError:
        logger.debug("gguf package not available; skipping GGUF metadata read for %s", gguf_path)
        return None
    try:
        reader = GGUFReader(str(gguf_path))
    except Exception as e:
        logger.warning("Failed to open %s as GGUF for parser detection: %s", gguf_path, e)
        return None
    field = reader.get_field("tokenizer.chat_template")
    if field is None:
        return None
    try:
        # GGUF string fields expose `.parts` (numpy bytes arrays) selected by `.data` indices.
        parts = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
    except Exception as e:
        logger.warning("Failed to decode chat_template from %s: %s", gguf_path, e)
        return None
    template = "".join(parts)
    if not template:
        return None
    return template
