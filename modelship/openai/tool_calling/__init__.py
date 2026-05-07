"""Cross-loader tool-calling toolkit.

Loaders without native tool-call support (Transformers today, plugin-wrapped
raw-text engines tomorrow) use the parsers and helpers in this package to
turn raw model output into OpenAI-shape ``tool_calls``. Loaders whose engines
already emit structured tool calls (vLLM, llama.cpp via a function-calling
chat handler) bypass it.
"""

from modelship.openai.tool_calling.input import resolve_tools_for_request
from modelship.openai.tool_calling.parsers import ParsedToolCalls, ToolCallParser, ToolCallStreamer
from modelship.openai.tool_calling.registry import available_parsers, get_parser, register_parser
from modelship.openai.tool_calling.streaming import (
    build_chat_completion_response,
    finish_reason_for,
    stream_chat_completion,
)

__all__ = [
    "ParsedToolCalls",
    "ToolCallParser",
    "ToolCallStreamer",
    "available_parsers",
    "build_chat_completion_response",
    "finish_reason_for",
    "get_parser",
    "register_parser",
    "resolve_tools_for_request",
    "stream_chat_completion",
]
