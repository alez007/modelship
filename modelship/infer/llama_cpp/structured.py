"""OpenAI ``response_format`` → llama.cpp ``LlamaGrammar`` conversion.

llama-cpp-python's ``create_chat_completion`` accepts a ``response_format``
kwarg, but its built-in handler only recognizes the narrower
``{"type": "json_object", "schema": ...}`` shape and silently returns no
grammar for OpenAI's ``{"type": "json_schema", "json_schema": {...}}``.
We convert the OpenAI shape into a ``LlamaGrammar`` here and pass it via
the ``grammar`` kwarg, which both ``create_chat_completion`` and
``create_completion`` accept.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from llama_cpp import LlamaGrammar

from modelship.logging import get_logger

logger = get_logger("infer.llama_cpp.structured")

# Permissive JSON-object grammar: any valid JSON object. Used when the
# caller requested ``{"type": "json_object"}`` without a schema.
_JSON_OBJECT_SCHEMA = {"type": "object"}


@lru_cache(maxsize=1)
def _json_object_grammar() -> LlamaGrammar:
    """Compile the permissive json_object grammar once per process."""
    return LlamaGrammar.from_json_schema(json.dumps(_JSON_OBJECT_SCHEMA), verbose=False)


def build_llama_grammar(response_format: dict[str, Any] | None) -> LlamaGrammar | None:
    """Convert an OpenAI-shaped ``response_format`` to a ``LlamaGrammar``.

    Returns ``None`` when no constraint should apply (missing, ``text``,
    or a malformed payload — logged as a warning).
    """
    if not response_format:
        return None

    fmt_type = response_format.get("type")
    if fmt_type in (None, "text"):
        return None

    if fmt_type == "json_object":
        return _json_object_grammar()

    if fmt_type == "json_schema":
        spec = response_format.get("json_schema") or {}
        schema = spec.get("schema")
        if not isinstance(schema, dict):
            logger.warning(
                "response_format.json_schema is missing a 'schema' object; skipping grammar constraint",
            )
            return None
        try:
            return LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
        except Exception as exc:
            logger.warning("failed to compile json_schema into LlamaGrammar: %s; skipping", exc)
            return None

    logger.warning("unsupported response_format type %r; skipping grammar constraint", fmt_type)
    return None
