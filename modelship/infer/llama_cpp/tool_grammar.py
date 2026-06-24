"""Build a GBNF ``LlamaGrammar`` that constrains tool-call output.

Compiled from a request's ``tools`` and passed to
``llama.create_completion(grammar=...)`` — the same ``grammar`` kwarg
:mod:`structured` uses for ``response_format``. Unlike a ``response_format``
grammar (which forces the whole response to be one JSON value), this grammar's
root permits a free-text answer *or* a bounded sequence of enveloped tool
calls wrapped in optional conversational text, so a non-tool answer and a
chatty preamble before a tool call are both reachable.

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
    # false`` forbids keys outside the envelope. Entries lacking a function name
    # are skipped — a ``None`` const yields an unsatisfiable branch.
    tool_schemas = []
    for t in tools:
        func = t.get("function")
        if not func or not func.get("name"):
            continue
        tool_schemas.append(
            {
                "type": "object",
                "properties": {
                    "name": {"const": func["name"]},
                    "arguments": func.get("parameters") or {"type": "object"},
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )

    if not tool_schemas:
        return None

    meta_schema = {"anyOf": tool_schemas}

    try:
        inner = json_schema_to_gbnf(json.dumps(meta_schema))
    except Exception as exc:
        logger.warning("constrain_tool_calls: failed to convert tool schemas to GBNF: %s; skipping", exc)
        return None

    # The converter's entry rule is ``root``; rename only its LHS so our own
    # ``root`` can wrap it. Nothing references it, so a line-anchored sub suffices.
    inner = re.sub(r"^\s*root\s*::=", "tc-json ::=", inner, flags=re.M)

    start = json.dumps(parser.start_marker)
    end = json.dumps(parser.end_marker)

    # Exclude the start marker's first char from `content` so free text yields to a
    # tool call when the marker begins. Escape it if it's special inside a GBNF class.
    start_char = parser.start_marker[0] if parser.start_marker else "<"
    escaped_char = f"\\{start_char}" if start_char in "]-^\\" else start_char

    # Allow optional leading/trailing conversational text and optional whitespace
    # around the JSON envelope, so a chatty prefix can't strand the model in the
    # content branch. `tool-calls` caps at two calls so decoding can't loop forever.
    prefix = (
        "root ::= ( content )? tool-calls ( content )? | content\n"
        "tool-calls ::= tool-call ( tc-ws tool-call )?\n"
        f"tool-call ::= {start} tc-ws tc-json tc-ws {end}\n"
        f"content ::= [^{escaped_char}]+\n"
        "tc-ws ::= [ \\t\\n\\r]*\n"
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
