"""Single-pass chat-output streamer that handles reasoning + tool calls.

This is the loader-shared driver consumed by every chat serving path
that emits raw text (transformers, llama.cpp, plugin-wrapped engines).
It walks the cumulative model output once and routes each region to
its appropriate stream:

1. **Reasoning regions** — text inside the optional reasoning parser's
   marker pair. Surfaced via ``DeltaMessage.reasoning``.
2. **Tool-call regions** — text inside the tool-call parser's marker
   pair, with name/arguments extraction. Surfaced via
   ``DeltaMessage.tool_calls``.
3. **Content regions** — everything else. Surfaced via
   ``DeltaMessage.content``.

The single-pass design matters: a ``<tool_call>`` marker emitted inside
``<think>`` is part of the model's reasoning, not a real tool call.
Two chained streamers would parse it as a real call; one streamer that
knows about both marker kinds correctly routes it to the reasoning
view. Both the streaming and non-streaming paths run the same streamer
so behavior cannot drift between them.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelship.openai.parsers.reasoning.parsers import ReasoningParser
from modelship.openai.parsers.tool_calling.parsers import ToolCallParser
from modelship.openai.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
    random_uuid,
)


@dataclass(frozen=True)
class ParsedChatOutput:
    """Aggregate result of parsing a model's full chat-completion text.

    ``content`` carries the residual non-reasoning, non-tool-call text
    once both kinds of markers are stripped. It is ``None`` when tool
    calls were extracted *and* the residual is empty, matching OpenAI's
    behavior of nulling ``content`` alongside ``tool_calls``.
    ``reasoning`` is the concatenation of all ``<think>...`` regions
    (or whichever marker pair the reasoning parser declared); ``None``
    when reasoning was not enabled or no markers were emitted.
    """

    content: str | None
    reasoning: str | None
    tool_calls: list[ToolCall]

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ChatOutputStreamer:
    """Per-request, stateful chat-output extractor.

    State held per request:

    - ``_sent_content_idx`` / ``_sent_reasoning_idx`` — chars already
      shipped from each non-tool-call stream.
    - ``_sent_name`` / ``_sent_id`` per tool index — whether the
      function name and id deltas have been emitted.
    - ``_sent_args`` per tool index — chars of arguments already
      shipped (the suffix-diff cursor).
    - ``_finalized_indices`` / ``_finalized_calls`` — closed tool-call
      blocks accumulated for :attr:`result` and ``finish_reason``.
    """

    def __init__(
        self,
        tool_call_parser: ToolCallParser | None,
        reasoning_parser: ReasoningParser | None = None,
    ):
        if tool_call_parser is None and reasoning_parser is None:
            raise ValueError("ChatOutputStreamer requires at least one parser (tool-call or reasoning)")
        self._tool_parser = tool_call_parser
        self._tool_start = tool_call_parser.start_marker if tool_call_parser else ""
        self._tool_end = tool_call_parser.end_marker if tool_call_parser else ""
        self._reasoning = reasoning_parser
        self._reasoning_start = reasoning_parser.start_marker if reasoning_parser else ""
        self._reasoning_end = reasoning_parser.end_marker if reasoning_parser else ""
        self._sent_content_idx = 0
        self._sent_reasoning_idx = 0
        self._sent_name: list[bool] = []
        self._sent_id: list[str] = []
        self._sent_args: list[str] = []
        self._finalized_indices: set[int] = set()
        self._finalized_calls: list[ToolCall] = []
        self._last_text = ""

    def extract_streaming(self, current_text: str) -> DeltaMessage | None:
        """Run one diff pass against ``current_text`` and return any new deltas."""
        self._last_text = current_text
        regions = self._scan(current_text, hold_marker_tail=True)
        content_delta = self._diff_content(regions)
        reasoning_delta = self._diff_reasoning(regions)
        tool_call_deltas = self._emit_new_tool_call_fragments(regions)
        if not content_delta and not reasoning_delta and not tool_call_deltas:
            return None
        return DeltaMessage(content=content_delta, reasoning=reasoning_delta, tool_calls=tool_call_deltas)

    def finalize(self) -> DeltaMessage | None:
        """Flush any held-back tails once no more text is coming."""
        regions = self._scan(self._last_text, hold_marker_tail=False)
        content_delta = self._diff_content(regions)
        reasoning_delta = self._diff_reasoning(regions)
        if content_delta is None and reasoning_delta is None:
            return None
        return DeltaMessage(content=content_delta, reasoning=reasoning_delta)

    @property
    def result(self) -> ParsedChatOutput:
        """Final view, suitable for the non-streaming response shape."""
        regions = self._scan(self._last_text, hold_marker_tail=False)
        content_view = "".join(payload for kind, payload, _ in regions if kind == "content")
        reasoning_view = "".join(payload for kind, payload, _ in regions if kind == "reasoning")
        content = (content_view if content_view.strip() else None) if self._finalized_calls else (content_view or None)
        return ParsedChatOutput(
            content=content,
            reasoning=reasoning_view or None,
            tool_calls=list(self._finalized_calls),
        )

    # ------------------------------------------------------------------
    # Single-pass scanner
    # ------------------------------------------------------------------

    def _scan(self, text: str, *, hold_marker_tail: bool) -> list[tuple[str, str, bool]]:
        """Walk ``text`` once, returning ordered ``(kind, payload, is_complete)`` regions.

        ``kind`` is one of ``"content"``, ``"reasoning"``, ``"tool_call"``.
        ``is_complete`` is meaningful only for ``"tool_call"``; for the
        other kinds it is always ``True`` (their boundaries are handled
        via the marker-tail holdback in the content/reasoning views).

        ``hold_marker_tail`` controls mid-stream holdback: trailing
        bytes that could be the prefix of any expected opening or
        closing marker are withheld, so the client never sees a
        fragment of a tag forwarded as content/reasoning bytes. At
        finalize time we know no more text is coming, so all held
        tails are flushed.
        """
        regions: list[tuple[str, str, bool]] = []
        pos = 0
        n = len(text)
        while pos < n:
            tool_start = text.find(self._tool_start, pos) if self._tool_start else -1
            reason_start = text.find(self._reasoning_start, pos) if self._reasoning_start else -1
            next_starts = [s for s in (tool_start, reason_start) if s >= 0]
            if not next_starts:
                # No more openings ahead; the rest is content.
                remainder = text[pos:]
                if hold_marker_tail:
                    safe = self._safe_outside_flush_index(remainder)
                    regions.append(("content", remainder[:safe], True))
                else:
                    regions.append(("content", remainder, True))
                break

            next_pos = min(next_starts)
            if next_pos > pos:
                regions.append(("content", text[pos:next_pos], True))

            if next_pos == reason_start:
                payload_start = next_pos + len(self._reasoning_start)
                end = text.find(self._reasoning_end, payload_start) if self._reasoning_end else -1
                if end < 0:
                    inner = text[payload_start:]
                    if hold_marker_tail:
                        safe = _safe_overlap_index(inner, self._reasoning_end)
                        regions.append(("reasoning", inner[:safe], False))
                    else:
                        regions.append(("reasoning", inner, False))
                    break
                regions.append(("reasoning", text[payload_start:end], True))
                pos = end + len(self._reasoning_end)
            else:
                payload_start = next_pos + len(self._tool_start)
                end = text.find(self._tool_end, payload_start) if self._tool_end else -1
                if end < 0:
                    partial = text[payload_start:]
                    if hold_marker_tail:
                        safe = _safe_overlap_index(partial, self._tool_end)
                        regions.append(("tool_call", partial[:safe], False))
                    else:
                        regions.append(("tool_call", partial, False))
                    break
                regions.append(("tool_call", text[payload_start:end], True))
                pos = end + len(self._tool_end)
        return regions

    def _safe_outside_flush_index(self, buf: str) -> int:
        """Index up to which ``buf`` (a content tail) can be flushed.

        ``buf`` does not contain a full opening marker; the unsafe tail
        is the longest proper-prefix overlap with whichever opening
        marker(s) the streamer is watching for.
        """
        cap = len(buf)
        for marker in (self._tool_start, self._reasoning_start):
            if not marker:
                continue
            cap = min(cap, _safe_overlap_index(buf, marker))
        return cap

    # ------------------------------------------------------------------
    # Content + reasoning diff
    # ------------------------------------------------------------------

    def _diff_content(self, regions: list[tuple[str, str, bool]]) -> str | None:
        view = "".join(payload for kind, payload, _ in regions if kind == "content")
        if len(view) <= self._sent_content_idx:
            return None
        new = view[self._sent_content_idx :]
        self._sent_content_idx = len(view)
        return new or None

    def _diff_reasoning(self, regions: list[tuple[str, str, bool]]) -> str | None:
        view = "".join(payload for kind, payload, _ in regions if kind == "reasoning")
        if len(view) <= self._sent_reasoning_idx:
            return None
        new = view[self._sent_reasoning_idx :]
        self._sent_reasoning_idx = len(view)
        return new or None

    # ------------------------------------------------------------------
    # Tool-call fragments
    # ------------------------------------------------------------------

    def _emit_new_tool_call_fragments(self, regions: list[tuple[str, str, bool]]) -> list[DeltaToolCall]:
        if self._tool_parser is None:
            return []
        deltas: list[DeltaToolCall] = []
        tool_blocks = [(payload, complete) for kind, payload, complete in regions if kind == "tool_call"]
        for i, (payload, is_complete) in enumerate(tool_blocks):
            self._ensure_slot(i)

            if not self._sent_name[i]:
                name = self._tool_parser.extract_partial_name(payload)
                if name is None:
                    if is_complete:
                        # Closed but malformed (no extractable name) — skip,
                        # but still process later blocks.
                        continue
                    # Mid-stream and no name yet; per OpenAI convention we don't
                    # advance to later blocks until the current one has a name.
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

            args = self._tool_parser.extract_partial_args(payload)
            if args is not None and len(args) > len(self._sent_args[i]):
                diff = args[len(self._sent_args[i]) :]
                self._sent_args[i] = args
                deltas.append(
                    DeltaToolCall(
                        index=i,
                        function=DeltaFunctionCall(arguments=diff),
                    )
                )

            if is_complete and i not in self._finalized_indices and self._sent_name[i]:
                self._finalized_indices.add(i)
                self._finalized_calls.append(
                    ToolCall(
                        id=self._sent_id[i],
                        type="function",
                        function=FunctionCall(
                            name=self._tool_parser.extract_partial_name(payload) or "",
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


def _safe_overlap_index(buf: str, marker: str) -> int:
    """Index up to which ``buf`` can be flushed without risking a split marker.

    ``buf`` is known not to contain the full marker. The unsafe tail is
    the longest proper-prefix overlap between ``buf`` and ``marker``: if
    the next chunk completes that prefix, we'd have to retract bytes we
    already streamed. Holding them back avoids the retraction.
    """
    if not marker:
        return len(buf)
    max_overlap = min(len(buf), len(marker) - 1)
    for k in range(max_overlap, 0, -1):
        if buf.endswith(marker[:k]):
            return len(buf) - k
    return len(buf)
