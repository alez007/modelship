"""Base class for model-family-specific tool-call output parsers.

Parsers in this codebase are *marker-based*: each family wraps tool-call
JSON in a fixed pair of literal strings (``<tool_call>`` / ``</tool_call>``
for Hermes, ``[TOOL_CALLS]`` / closing token for Mistral, ...). A subclass
declares the marker pair plus two small extractors that pick a function
name and an arguments substring out of the (possibly partial) JSON between
the markers.

The cumulative-text streaming and non-streaming paths both run through
:class:`~modelship.openai.parsers.output.ChatOutputStreamer`, which
consumes a parser instance plus an optional reasoning parser and walks
the model output once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelship.openai.parsers.output import ParsedChatOutput


class ToolCallParser(ABC):
    """Family-specific knobs the streamer needs to drive its diff loop.

    Subclasses set ``start_marker`` / ``end_marker`` and implement the two
    extractors. They never touch streaming state — that lives on
    :class:`~modelship.openai.parsers.output.ChatOutputStreamer`, which is
    instantiated once per request.
    """

    name: str
    start_marker: str
    end_marker: str

    @abstractmethod
    def extract_partial_name(self, partial_payload: str) -> str | None:
        """Return the function name if a complete quoted name is visible yet, else ``None``.

        Called every delta until it yields a non-``None`` result; once a name
        has been emitted to the client, the streamer stops asking and starts
        forwarding arguments bytes for that tool.
        """

    @abstractmethod
    def extract_partial_args(self, partial_payload: str) -> str | None:
        """Return the arguments substring as the client should see it so far.

        The streamer takes a length-diff of successive returns and forwards
        only the new bytes. Implementations must withhold any trailing bytes
        that could plausibly be the envelope closer landing ahead of the
        family's end marker — otherwise those bytes leak into the args
        stream and the client receives malformed JSON.
        """

    def parse(self, text: str) -> ParsedChatOutput:
        # Local import: ``output`` imports ``ToolCallParser`` for typing,
        # so importing it at module top would create a cycle.
        from modelship.openai.parsers.output import ChatOutputStreamer

        streamer = ChatOutputStreamer(self)
        streamer.extract_streaming(text)
        streamer.finalize()
        return streamer.result
