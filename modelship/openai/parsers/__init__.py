"""Cross-loader parser toolkit: reasoning + tool calls.

Loaders import streaming helpers and the unified streamer/result
classes from this package; the per-family parsers live under the
``reasoning`` and ``tool_calling`` subpackages.
"""

from modelship.openai.parsers.output import ChatOutputStreamer, ParsedChatOutput
from modelship.openai.parsers.streaming import (
    build_chat_completion_response,
    finish_reason_for,
    stream_chat_completion,
)

__all__ = [
    "ChatOutputStreamer",
    "ParsedChatOutput",
    "build_chat_completion_response",
    "finish_reason_for",
    "stream_chat_completion",
]
