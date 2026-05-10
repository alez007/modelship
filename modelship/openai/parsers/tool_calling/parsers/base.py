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
    # When False, the streamer locates ``start_marker`` but does NOT skip past
    # it: the marker bytes stay at the head of the payload fed to the
    # extractors. Used by parsers (e.g. ``llama3_json``) whose marker is part
    # of the JSON they need to parse — ``{"name"`` cannot be consumed without
    # destroying the object's structure. Defaults to True to preserve the
    # Hermes/Mistral semantics where the marker is an envelope tag stripped
    # before the body.
    consume_start_marker: bool = True
    # Set True when the parser's marker(s) are registered as *special tokens*
    # in the tokenizers of the model families this parser targets — the way
    # ``[TOOL_CALLS]`` is in Mistral v3+ tokenizers
    # (``added_tokens_decoder`` with ``special=True``). Loaders that decode
    # with ``skip_special_tokens=True`` by default would otherwise strip the
    # marker before this parser ever sees it, leaving the parser permanently
    # idle. The transformers loader reads this flag at startup, flips its
    # streamer's ``skip_special_tokens`` to ``False`` when set, and noise-
    # strips every OTHER registered special from chunks itself. Hermes and
    # llama3_json keep the default — their markers are regular text.
    markers_are_specials: bool = False

    @abstractmethod
    def extract_partial_name(self, partial_payload: str) -> str | None:
        """Return the function name if a complete quoted name is visible yet, else ``None``.

        Called every delta until it yields a non-``None`` result; once a name
        has been emitted to the client, the streamer stops asking and starts
        forwarding arguments bytes for that tool.
        """

    @abstractmethod
    def extract_partial_args(self, partial_payload: str, is_complete: bool = False) -> str | None:
        """Return the arguments substring as the client should see it so far.

        The streamer takes a length-diff of successive returns and forwards
        only the new bytes. Implementations must withhold any trailing bytes
        that could plausibly be the envelope closer landing ahead of the
        family's end marker — otherwise those bytes leak into the args
        stream and the client receives malformed JSON.

        ``is_complete`` is True when the streamer has confirmed the tool call
        is fully terminated (or at the end of generation). This allows parsers
        that construct JSON dynamically (e.g., from custom syntax) to flush
        withheld structural bytes.
        """

    def split_payload(self, payload: str, is_complete: bool) -> list[tuple[str, bool]]:
        """Split a region payload into ``(sub_payload, is_complete)`` per-call entries.

        Default: one call per region (Hermes-style). Families like Mistral
        whose envelope wraps a JSON array of calls override this to yield
        one entry per array element. Each entry is then fed independently to
        :meth:`extract_partial_name` / :meth:`extract_partial_args`, and each
        becomes its own OpenAI ``tool_calls[i]`` slot.
        """
        return [(payload, is_complete)]

    def parse(self, text: str) -> ParsedChatOutput:
        # Local import: ``output`` imports ``ToolCallParser`` for typing,
        # so importing it at module top would create a cycle.
        from modelship.openai.parsers.output import ChatOutputStreamer

        streamer = ChatOutputStreamer(self)
        streamer.extract_streaming(text)
        streamer.finalize()
        return streamer.result
