"""Registry of named reasoning parsers, dispatched by configuration string.

Mirrors ``modelship.openai.parsers.tool_calling.registry``: loaders look
up a reasoning parser by the name resolved on the driver
(``_resolved_reasoning_parser``) and hand it to the unified
``ChatOutputStreamer``. vLLM bypasses this registry — it has its own
built-in reasoning parsers and consumes only the resolved name.
"""

from __future__ import annotations

from modelship.openai.parsers.reasoning.parsers import (
    DeepseekR1ReasoningParser,
    Gemma4ReasoningParser,
    ReasoningParser,
)

_PARSERS: dict[str, ReasoningParser] = {
    DeepseekR1ReasoningParser.name: DeepseekR1ReasoningParser(),
    Gemma4ReasoningParser.name: Gemma4ReasoningParser(),
}


def get_parser(name: str) -> ReasoningParser:
    """Return the parser registered under ``name`` or raise ``ValueError``."""
    try:
        return _PARSERS[name]
    except KeyError:
        available = ", ".join(sorted(_PARSERS)) or "(none)"
        raise ValueError(f"unknown reasoning_parser {name!r}; available: {available}") from None


def available_parsers() -> list[str]:
    return sorted(_PARSERS)


def register_parser(parser: ReasoningParser) -> None:
    """Register an additional parser. Intended for tests and plugin code."""
    _PARSERS[parser.name] = parser
