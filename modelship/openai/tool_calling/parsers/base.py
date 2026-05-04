"""Base class for model-family-specific tool-call output parsers.

Parsers in this codebase are *marker-based*: each family wraps tool-call
JSON in a fixed pair of literal strings (``<tool_call>`` / ``</tool_call>``
for Hermes, ``[TOOL_CALLS]`` / closing token for Mistral, ...). A subclass
declares the marker pair plus two small extractors that pick a function
name and an arguments substring out of the (possibly partial) JSON between
the markers. Both the streaming and non-streaming paths run the same
:class:`ToolCallStreamer` so behavior cannot drift between them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from modelship.openai.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
    random_uuid,
)


@dataclass(frozen=True)
class ParsedToolCalls:
    """Aggregate result of parsing a model's full chat-completion text.

    ``content`` carries the residual non-tool-call text once any tool-call
    markers are stripped. It is ``None`` when tool calls were extracted *and*
    the residual is empty, matching OpenAI's behavior of nulling ``content``
    alongside ``tool_calls``.
    """

    content: str | None
    tool_calls: list[ToolCall]

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ToolCallParser(ABC):
    """Family-specific knobs the streamer needs to drive its diff loop.

    Subclasses set ``start_marker`` / ``end_marker`` and implement the two
    extractors. They never touch streaming state — that lives on
    :class:`ToolCallStreamer`, which can be instantiated once per request.
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

    def parse(self, text: str) -> ParsedToolCalls:
        streamer = ToolCallStreamer(self)
        streamer.extract_streaming(text)
        streamer.finalize()
        return streamer.result


class ToolCallStreamer:
    """Per-request, stateful tool-call extractor.

    Mirrors vLLM's approach: hold a small amount of "what we've already sent"
    state, re-parse the cumulative ``current_text`` on every delta, and diff.
    Returns either a :class:`DeltaMessage` carrying any newly-emittable
    content / tool-call fragments or ``None`` if there is nothing to send yet.

    State held per request:

    - ``_sent_content_idx`` — number of content-stream chars already shipped.
      The "content stream" view is the original text with every (complete or
      open) tool-call region excised.
    - ``_sent_name`` / ``_sent_id`` per tool index — whether the function name
      and id deltas have been emitted.
    - ``_sent_args`` per tool index — number of arguments chars already
      shipped (the suffix-diff cursor).
    - ``_finalized_calls`` — :class:`ToolCall` objects accumulated for blocks
      that have closed, used to populate :attr:`result` for the non-streaming
      path and for the final ``finish_reason``.
    """

    def __init__(self, parser: ToolCallParser):
        self._parser = parser
        self._start = parser.start_marker
        self._end = parser.end_marker
        self._sent_content_idx = 0
        self._sent_name: list[bool] = []
        self._sent_id: list[str] = []
        self._sent_args: list[str] = []
        self._finalized_calls: list[ToolCall] = []
        self._content_parts_len = 0  # bookkeeping for `result.content`
        self._last_text = ""

    def extract_streaming(self, current_text: str) -> DeltaMessage | None:
        """Run one diff pass against ``current_text`` and return any new deltas."""
        self._last_text = current_text
        content_delta = self._emit_new_content(current_text, hold_marker_tail=True)
        tool_call_deltas = self._emit_new_tool_call_fragments(current_text)
        if not content_delta and not tool_call_deltas:
            return None
        return DeltaMessage(content=content_delta, tool_calls=tool_call_deltas)

    def finalize(self) -> DeltaMessage | None:
        """Flush any held-back content tail once no more text is coming."""
        content_delta = self._emit_new_content(self._last_text, hold_marker_tail=False)
        if content_delta is None:
            return None
        return DeltaMessage(content=content_delta)

    @property
    def result(self) -> ParsedToolCalls:
        """Final view, suitable for the non-streaming response shape."""
        # The content-view we accumulated as we streamed; reconstruct from the cursor.
        view = self._build_content_view(self._last_text, hold_marker_tail=False)
        content = (view.strip() or None) if self._finalized_calls else (view or None)
        return ParsedToolCalls(content=content, tool_calls=list(self._finalized_calls))

    # ------------------------------------------------------------------
    # Content stream
    # ------------------------------------------------------------------

    def _emit_new_content(self, current_text: str, *, hold_marker_tail: bool) -> str | None:
        view = self._build_content_view(current_text, hold_marker_tail=hold_marker_tail)
        if len(view) <= self._sent_content_idx:
            return None
        new = view[self._sent_content_idx :]
        self._sent_content_idx = len(view)
        return new or None

    def _build_content_view(self, text: str, *, hold_marker_tail: bool) -> str:
        """Build the content-stream view: the original text with tool-call regions excised.

        When ``hold_marker_tail`` is true (mid-stream), withhold a trailing
        suffix that could be the start of a new ``start_marker``, so the
        client never sees half of an opening tag. At finalize time we know
        no more text is coming, so the held-back tail is safe to flush.
        """
        parts: list[str] = []
        pos = 0
        while pos < len(text):
            start = text.find(self._start, pos)
            if start < 0:
                remainder = text[pos:]
                if hold_marker_tail:
                    safe = _safe_outside_flush_index(remainder, self._start)
                    parts.append(remainder[:safe])
                else:
                    parts.append(remainder)
                break
            parts.append(text[pos:start])
            payload_start = start + len(self._start)
            end = text.find(self._end, payload_start)
            if end < 0:
                # Open block, not yet closed — nothing more to append to content.
                break
            pos = end + len(self._end)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Tool-call fragments
    # ------------------------------------------------------------------

    def _emit_new_tool_call_fragments(self, current_text: str) -> list[DeltaToolCall]:
        deltas: list[DeltaToolCall] = []
        for i, (payload, is_complete) in enumerate(self._iter_tool_call_blocks(current_text)):
            self._ensure_slot(i)

            if not self._sent_name[i]:
                name = self._parser.extract_partial_name(payload)
                if name is None:
                    # Per OpenAI streaming convention the name is sent first;
                    # don't advance to a later block until this one has one.
                    break
                tool_id = f"chatcmpl-tool-{random_uuid()}"
                self._sent_name[i] = True
                self._sent_id[i] = tool_id
                deltas.append(
                    DeltaToolCall(
                        index=i,
                        id=tool_id,
                        type="function",
                        function=DeltaFunctionCall(name=name),
                    )
                )

            args = self._parser.extract_partial_args(payload)
            if args is not None and len(args) > len(self._sent_args[i]):
                diff = args[len(self._sent_args[i]) :]
                self._sent_args[i] = args
                deltas.append(
                    DeltaToolCall(
                        index=i,
                        function=DeltaFunctionCall(arguments=diff),
                    )
                )

            if is_complete and i == len(self._finalized_calls) and self._sent_name[i]:
                self._finalized_calls.append(
                    ToolCall(
                        id=self._sent_id[i],
                        type="function",
                        function=FunctionCall(
                            name=self._extract_committed_name(i),
                            arguments=self._sent_args[i],
                        ),
                    )
                )
        return deltas

    def _ensure_slot(self, i: int) -> None:
        while len(self._sent_name) <= i:
            self._sent_name.append(False)
            self._sent_id.append("")
            self._sent_args.append("")

    def _extract_committed_name(self, i: int) -> str:
        # We only stash the bool that the name was sent, not the value, so
        # re-derive from the current payload (cheap, the regex is small).
        for j, (payload, _) in enumerate(self._iter_tool_call_blocks(self._last_text)):
            if j == i:
                name = self._parser.extract_partial_name(payload)
                return name or ""
        return ""

    def _iter_tool_call_blocks(self, text: str):
        """Yield ``(partial_payload, is_complete)`` for each tool-call region in order.

        For the still-open final block, the partial payload has any tail
        suffix that could be the start of ``end_marker`` withheld, so the
        client never sees a fragment of the closing tag forwarded as
        argument bytes.
        """
        pos = 0
        while True:
            start = text.find(self._start, pos)
            if start < 0:
                return
            payload_start = start + len(self._start)
            end = text.find(self._end, payload_start)
            if end < 0:
                partial = text[payload_start:]
                safe = _safe_outside_flush_index(partial, self._end)
                yield partial[:safe], False
                return
            yield text[payload_start:end], True
            pos = end + len(self._end)


def _safe_outside_flush_index(buf: str, start_marker: str) -> int:
    """Index up to which ``buf`` can be flushed without risking a split marker.

    ``buf`` is known not to contain the full marker. The unsafe tail is the
    longest proper-prefix overlap between ``buf`` and ``start_marker``: if
    the next chunk completes that prefix, we'd have to retract bytes we
    already streamed. Holding them back avoids the retraction.
    """
    max_overlap = min(len(buf), len(start_marker) - 1)
    for k in range(max_overlap, 0, -1):
        if buf.endswith(start_marker[:k]):
            return len(buf) - k
    return len(buf)
