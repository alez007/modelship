"""Tests for the cross-loader reasoning parser toolkit.

Covers:
- The reasoning parser registry and the DeepSeek-R1 marker pair.
- ``ChatOutputStreamer`` running with only a reasoning parser.
- Composition: reasoning + tool-call parsers together, including the
  case where a tool-call marker appears *inside* a reasoning region
  (must not be parsed as a real tool call).
"""

from __future__ import annotations

import json

import pytest

from modelship.openai.parsers.output import ChatOutputStreamer
from modelship.openai.parsers.reasoning import (
    DeepseekR1ReasoningParser,
    available_parsers,
    get_parser,
    register_parser,
)
from modelship.openai.parsers.reasoning.parsers import ReasoningParser
from modelship.openai.parsers.tool_calling.parsers import HermesToolCallParser


class TestReasoningRegistry:
    def test_default_registry_includes_deepseek_r1(self):
        assert "deepseek_r1" in available_parsers()

    def test_get_parser_returns_singleton(self):
        a = get_parser("deepseek_r1")
        b = get_parser("deepseek_r1")
        assert a is b

    def test_unknown_parser_raises_with_available_list(self):
        with pytest.raises(ValueError, match="deepseek_r1"):
            get_parser("does-not-exist")

    def test_register_parser_makes_it_findable(self):
        class Stub(ReasoningParser):
            name = "stub-reasoning"
            start_marker = "<<r>>"
            end_marker = "<</r>>"

        register_parser(Stub())
        try:
            assert get_parser("stub-reasoning").name == "stub-reasoning"
        finally:
            from modelship.openai.parsers.reasoning import registry

            registry._PARSERS.pop("stub-reasoning", None)


class TestDeepseekR1Markers:
    def test_marker_pair(self):
        p = DeepseekR1ReasoningParser()
        assert p.start_marker == "<think>"
        assert p.end_marker == "</think>"


class TestReasoningOnlyStreamer:
    """``ChatOutputStreamer`` driven by a reasoning parser alone (no tools)."""

    def _feed(self, chunks: list[str]) -> tuple[ChatOutputStreamer, list]:
        streamer = ChatOutputStreamer(None, DeepseekR1ReasoningParser())
        deltas = []
        cumulative = ""
        for chunk in chunks:
            cumulative += chunk
            d = streamer.extract_streaming(cumulative)
            if d is not None:
                deltas.append(d)
        final = streamer.finalize()
        if final is not None:
            deltas.append(final)
        return streamer, deltas

    def test_at_least_one_parser_required(self):
        with pytest.raises(ValueError, match="at least one parser"):
            ChatOutputStreamer(None, None)

    def test_pure_content_streams_immediately(self):
        _, deltas = self._feed(["Hello", " ", "world"])
        assert "".join(d.content or "" for d in deltas) == "Hello world"
        assert all((d.reasoning or "") == "" for d in deltas)

    def test_reasoning_block_routed_to_reasoning_field(self):
        text = "<think>thinking step</think>final answer"
        _, deltas = self._feed([text])
        reasoning = "".join(d.reasoning or "" for d in deltas)
        content = "".join(d.content or "" for d in deltas)
        assert reasoning == "thinking step"
        assert content == "final answer"

    def test_reasoning_streams_incrementally(self):
        # Drip-feed a `<think>` block one char at a time. We want at least
        # several reasoning deltas, not one big shot at finalize.
        text = "<think>abc</think>"
        _, deltas = self._feed(list(text))
        reasoning_deltas = [d for d in deltas if d.reasoning]
        assert len(reasoning_deltas) >= 2
        assert "".join(d.reasoning or "" for d in deltas) == "abc"

    def test_holds_back_marker_prefix_until_disambiguated(self):
        # `<th` could be the first chars of `<think>`; the streamer must
        # not ship those as content mid-stream.
        streamer = ChatOutputStreamer(None, DeepseekR1ReasoningParser())
        d = streamer.extract_streaming("hello <th")
        assert d is not None and d.content == "hello "
        # Disambiguate by completing the marker → still no content (we
        # entered a reasoning region instead).
        d = streamer.extract_streaming("hello <think>x")
        assert d is None or (d.reasoning or "") == "x"

    def test_held_tail_flushes_when_disambiguated_as_content(self):
        # `<thr` proves it wasn't `<think>`; the held tail should flush.
        streamer = ChatOutputStreamer(None, DeepseekR1ReasoningParser())
        streamer.extract_streaming("hi <th")
        d = streamer.extract_streaming("hi <thr")
        # Content should now contain the previously-held `<th` plus `r`.
        assert d is not None and "thr" in (d.content or "")

    def test_holds_back_close_marker_prefix_inside_reasoning(self):
        # Inside an open reasoning block, `</thi` could be the start of
        # `</think>`; mid-stream we must not ship it as reasoning bytes
        # because the next chunk might prove it was the closer.
        streamer = ChatOutputStreamer(None, DeepseekR1ReasoningParser())
        d = streamer.extract_streaming("<think>thinking</thi")
        # Reasoning so far: "thinking" (the tail "</thi" is held back).
        assert d is not None and (d.reasoning or "") == "thinking"

    def test_multiple_reasoning_blocks_are_concatenated(self):
        text = "<think>a</think>between<think>b</think>after"
        streamer, _ = self._feed([text])
        result = streamer.result
        assert result.reasoning == "ab"
        assert result.content == "betweenafter"

    def test_unterminated_reasoning_block_flushes_at_finalize(self):
        # Model emits `<think>...` but never closes. Finalize should not
        # raise; the open block becomes reasoning content (best-effort
        # rendering of what the model produced).
        text = "<think>incomplete"
        streamer, _ = self._feed([text])
        assert streamer.result.reasoning == "incomplete"
        assert streamer.result.content is None

    def test_no_reasoning_no_content_returns_no_delta(self):
        streamer = ChatOutputStreamer(None, DeepseekR1ReasoningParser())
        # An empty cumulative text produces nothing.
        assert streamer.extract_streaming("") is None

    def test_chunk_boundary_in_open_marker(self):
        # The opening `<think>` is split exactly at `<think` / `>`.
        streamer = ChatOutputStreamer(None, DeepseekR1ReasoningParser())
        d1 = streamer.extract_streaming("hi <think")
        d2 = streamer.extract_streaming("hi <think>r")
        d3 = streamer.extract_streaming("hi <think>r</think>ok")
        # Combine all yielded deltas + finalize.
        deltas = [d for d in (d1, d2, d3) if d is not None]
        final = streamer.finalize()
        if final is not None:
            deltas.append(final)
        assert "".join(d.reasoning or "" for d in deltas) == "r"
        assert "".join(d.content or "" for d in deltas) == "hi ok"


class TestComposition:
    """ChatOutputStreamer with both reasoning + tool-call parsers active.

    The single-pass design must route tool-call markers that appear
    inside a reasoning region to the reasoning view, NOT parse them as
    real tool calls.
    """

    def _feed(self, chunks: list[str]) -> tuple[ChatOutputStreamer, list]:
        streamer = ChatOutputStreamer(HermesToolCallParser(), DeepseekR1ReasoningParser())
        deltas = []
        cumulative = ""
        for chunk in chunks:
            cumulative += chunk
            d = streamer.extract_streaming(cumulative)
            if d is not None:
                deltas.append(d)
        final = streamer.finalize()
        if final is not None:
            deltas.append(final)
        return streamer, deltas

    def test_reasoning_then_tool_call_then_residual(self):
        text = (
            "<think>I should call get_weather</think>"
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
            "ok"
        )
        streamer, deltas = self._feed([text])
        # Reasoning surfaced.
        assert "".join(d.reasoning or "" for d in deltas) == "I should call get_weather"
        # Real tool call extracted.
        assert len(streamer.result.tool_calls) == 1
        assert streamer.result.tool_calls[0].function.name == "get_weather"
        # `ok` ends up in content view.
        assert (streamer.result.content or "") == "ok"

    def test_tool_marker_inside_reasoning_is_not_a_real_tool_call(self):
        # The model is "thinking out loud" about tool calls inside its
        # `<think>` block. Those markers must NOT be parsed as real
        # tool calls — they're part of the reasoning text.
        text = '<think>I might need <tool_call>{"name":"x","arguments":{}}</tool_call> later</think>actual answer'
        streamer, _ = self._feed([text])
        assert streamer.result.tool_calls == []
        # Reasoning includes the literal markers because the streamer
        # surfaces the reasoning region's bytes verbatim.
        assert (streamer.result.reasoning or "").startswith("I might need <tool_call>")
        assert streamer.result.content == "actual answer"

    def test_real_tool_call_after_reasoning_streams_correctly(self):
        # Char-by-char to confirm both streams progress incrementally.
        text = '<think>plan</think><tool_call>{"name": "ping", "arguments": {"x": 1}}</tool_call>'
        _, deltas = self._feed(list(text))

        reasoning = "".join(d.reasoning or "" for d in deltas)
        assert reasoning == "plan"

        tool_deltas = [tc for d in deltas for tc in d.tool_calls]
        names = [tc.function.name for tc in tool_deltas if tc.function and tc.function.name]
        assert names == ["ping"]
        joined_args = "".join(
            tc.function.arguments or "" for tc in tool_deltas if tc.function and tc.function.arguments
        )
        assert json.loads(joined_args) == {"x": 1}

    def test_chunk_boundary_at_marker_seam(self):
        # The chunk lands exactly at `<thi` / `nk>...</think><tool_call>...`
        # — the streamer must hold `<thi` until the next chunk proves
        # which marker (or none) it belonged to.
        chunks = [
            "hello ",
            "<thi",
            'nk>r</think><tool_call>{"name": "p", "arguments": {}}</tool_call>',
        ]
        streamer, deltas = self._feed(chunks)
        # `hello ` lands as content, never `hello <thi`.
        contents = "".join(d.content or "" for d in deltas)
        # Note: residual content may be empty after the tool call (the
        # model emitted nothing post-call), but `hello ` is in there.
        assert contents.startswith("hello ")
        # Real tool call extracted.
        assert [tc.function.name for tc in streamer.result.tool_calls] == ["p"]
        # Reasoning extracted.
        assert streamer.result.reasoning == "r"

    def test_finish_reason_with_reasoning_only_is_stop(self):
        from modelship.openai.parsers.streaming import finish_reason_for

        text = "<think>r</think>ok"
        streamer, _ = self._feed([text])
        assert finish_reason_for(streamer.result, completion_tokens=10, max_tokens=100) == "stop"

    def test_finish_reason_with_tool_calls_overrides(self):
        from modelship.openai.parsers.streaming import finish_reason_for

        text = '<think>r</think><tool_call>{"name": "f", "arguments": {}}</tool_call>'
        streamer, _ = self._feed([text])
        assert finish_reason_for(streamer.result, completion_tokens=10, max_tokens=100) == "tool_calls"


class TestParseFullViaToolCallParser:
    """`ToolCallParser.parse()` returns the unified ParsedChatOutput."""

    def test_parse_returns_parsed_chat_output(self):
        from modelship.openai.parsers.output import ParsedChatOutput

        text = '<tool_call>{"name": "x", "arguments": {}}</tool_call>'
        result = HermesToolCallParser().parse(text)
        assert isinstance(result, ParsedChatOutput)
        assert result.has_tool_calls
        assert result.reasoning is None
