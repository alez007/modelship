"""Build a GBNF ``LlamaGrammar`` that constrains tool-call output.

Compiled from a request's ``tools`` and passed to
``llama.create_completion(grammar=...)`` — the same ``grammar`` kwarg
:mod:`structured` uses for ``response_format``. Unlike a ``response_format``
grammar (which forces the whole response to be one JSON value), this grammar's
root permits *either* free text *or* a bounded sequence of enveloped tool
calls, so a non-tool answer is still reachable.

JSON-family parsers (Hermes) wrap each call's ``{"name", "arguments"}`` JSON in
literal marker tags, so the body is built by reusing llama.cpp's own
``json_schema_to_gbnf`` and only the envelope is hand-written. Custom-syntax
families return ``None`` (no emitter yet).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from llama_cpp import LlamaGrammar
from llama_cpp.llama_grammar import json_schema_to_gbnf

from modelship.logging import get_logger

if TYPE_CHECKING:
    from modelship.openai.parsers.tool_calling import ToolCallParser

logger = get_logger("infer.llama_cpp.tool_grammar")

# Parsers whose body is a JSON ``{"name", "arguments"}`` object wrapped in literal
# ``start_marker`` / ``end_marker`` tags — the only shape ``json_schema_to_gbnf``
# plus the envelope below can express. Qwen3-Coder shares Hermes' markers but has
# an XML body, so it's excluded; Mistral / llama3_json are JSON but use a
# different envelope shape, not wired here.
_JSON_FAMILY_PARSERS = frozenset({"hermes"})

# Parser names already logged as unconstrained, to avoid repeating per request.
_logged_unconstrained: set[str] = set()


def build_tool_call_gbnf(parser: ToolCallParser, tools: list[dict[str, Any]]) -> str | None:
    """Assemble the GBNF text for the tool-call grammar, or ``None`` if unsupported.

    Split from compilation so the emitted grammar can be inspected/tested as
    text. Returns ``None`` (logged once per parser) for parsers without an
    emitter and on conversion failure.
    """
    if not tools:
        return None

    if parser.name not in _JSON_FAMILY_PARSERS:
        if parser.name not in _logged_unconstrained:
            _logged_unconstrained.add(parser.name)
            logger.info(
                "constrain_tool_calls: no grammar emitter for tool_call_parser %r; left unconstrained",
                parser.name,
            )
        return None

    # One anyOf branch per tool: ``name`` pinned to the tool's name via ``const``,
    # ``arguments`` constrained to its parameter schema. ``additionalProperties:
    # false`` forbids keys outside the envelope.
    meta_schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"const": (t.get("function") or {}).get("name")},
                    "arguments": (t.get("function") or {}).get("parameters") or {"type": "object"},
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
            for t in tools
        ]
    }

    try:
        inner = json_schema_to_gbnf(json.dumps(meta_schema))
    except Exception as exc:
        logger.warning("constrain_tool_calls: failed to convert tool schemas to GBNF: %s; skipping", exc)
        return None

    # The converter's entry rule is ``root``; rename only its LHS so our own
    # ``root`` can wrap it. Nothing references it, so a line-anchored sub suffices.
    inner = re.sub(r"^root ::=", "tc-json ::=", inner, flags=re.M)

    start = json.dumps(parser.start_marker)
    end = json.dumps(parser.end_marker)
    # `tool-calls` caps at two calls so decoding can't loop indefinitely. The
    # `content` branch ([^<]+) preserves a free-text answer but cannot contain a
    # literal `<`.
    prefix = (
        "root ::= tool-calls | content\n"
        'tool-calls ::= tool-call ( "\\n" tool-call )?\n'
        f'tool-call ::= {start} "\\n" tc-json "\\n" {end}\n'
        "content ::= [^<]+\n"
    )
    return prefix + inner


def build_tool_call_grammar(parser: ToolCallParser, tools: list[dict[str, Any]]) -> LlamaGrammar | None:
    """Compile a tool-call-constraining grammar, or ``None`` if unsupported.

    ``parser`` supplies the envelope markers; ``tools`` is the OpenAI-shaped list
    already resolved for the request. Returns ``None`` for parsers without an
    emitter and on any conversion/compile failure, so the caller falls back to
    unconstrained decoding rather than erroring.
    """
    text = build_tool_call_gbnf(parser, tools)
    if text is None:
        return None
    try:
        return LlamaGrammar.from_string(text, verbose=False)
    except Exception as exc:
        logger.warning("constrain_tool_calls: failed to compile tool-call grammar: %s; skipping", exc)
        return None
