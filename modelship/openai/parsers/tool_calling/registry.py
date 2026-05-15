"""Registry of named tool-call parsers, dispatched by configuration string.

The registry is the single seam between a loader and the per-family parsers.
Loaders that emit raw text (Transformers, plugin-wrapped engines) look up a
parser by the name configured on the deployment and feed it the model's
output. Loaders with native tool-call support (vLLM, llama.cpp via a
function-calling chat handler) bypass the registry entirely.
"""

from __future__ import annotations

from modelship.openai.parsers.tool_calling.parsers import (
    FunctionGemmaToolCallParser,
    Gemma4ToolCallParser,
    HermesToolCallParser,
    Llama3JsonToolCallParser,
    MistralToolCallParser,
    Qwen3CoderToolCallParser,
    ToolCallParser,
)

_PARSERS: dict[str, ToolCallParser] = {
    FunctionGemmaToolCallParser.name: FunctionGemmaToolCallParser(),
    Gemma4ToolCallParser.name: Gemma4ToolCallParser(),
    HermesToolCallParser.name: HermesToolCallParser(),
    Llama3JsonToolCallParser.name: Llama3JsonToolCallParser(),
    MistralToolCallParser.name: MistralToolCallParser(),
    Qwen3CoderToolCallParser.name: Qwen3CoderToolCallParser(),
}


def get_parser(name: str) -> ToolCallParser:
    """Return the parser registered under ``name`` or raise ``ValueError``."""
    try:
        return _PARSERS[name]
    except KeyError:
        available = ", ".join(sorted(_PARSERS)) or "(none)"
        raise ValueError(f"unknown tool_call_parser {name!r}; available: {available}") from None


def available_parsers() -> list[str]:
    return sorted(_PARSERS)


def register_parser(parser: ToolCallParser) -> None:
    """Register an additional parser. Intended for tests and plugin code."""
    _PARSERS[parser.name] = parser
