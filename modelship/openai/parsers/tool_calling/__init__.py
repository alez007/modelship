"""Cross-loader tool-calling parser package.

Per-family tool-call parsers (Hermes today, more later) plus the
registry that maps a configured name to a parser instance, plus tool
input resolution. The unified streamer that drives parsing for
streaming and non-streaming responses lives one level up at
``modelship.openai.parsers.output``; loaders should import the
streaming helpers from ``modelship.openai.parsers.streaming``.
"""

from modelship.openai.parsers.tool_calling.input import request_forces_tool_call, resolve_tools_for_request
from modelship.openai.parsers.tool_calling.parsers import (
    HermesToolCallParser,
    Llama3JsonToolCallParser,
    MistralToolCallParser,
    ToolCallParser,
)
from modelship.openai.parsers.tool_calling.registry import available_parsers, get_parser, register_parser

__all__ = [
    "HermesToolCallParser",
    "Llama3JsonToolCallParser",
    "MistralToolCallParser",
    "ToolCallParser",
    "available_parsers",
    "get_parser",
    "register_parser",
    "request_forces_tool_call",
    "resolve_tools_for_request",
]
