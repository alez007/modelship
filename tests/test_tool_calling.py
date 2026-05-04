"""Tests for the cross-loader tool-calling toolkit."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from modelship.openai.tool_calling import (
    ToolCallStreamer,
    available_parsers,
    get_parser,
    register_parser,
    resolve_tools_for_request,
)
from modelship.openai.tool_calling.parsers import HermesToolCallParser, ToolCallParser


class TestRegistry:
    def test_default_registry_includes_hermes(self):
        assert "hermes" in available_parsers()

    def test_get_parser_returns_singleton(self):
        a = get_parser("hermes")
        b = get_parser("hermes")
        assert a is b

    def test_unknown_parser_raises_with_available_list(self):
        with pytest.raises(ValueError, match="hermes"):
            get_parser("does-not-exist")

    def test_register_parser_makes_it_findable(self):
        class Stub(ToolCallParser):
            name = "stub-test-parser"
            start_marker = "<<"
            end_marker = ">>"

            def extract_partial_name(self, partial_payload: str) -> str | None:
                return None

            def extract_partial_args(self, partial_payload: str) -> str | None:
                return None

        register_parser(Stub())
        try:
            assert get_parser("stub-test-parser").name == "stub-test-parser"
        finally:
            # Clean up so other tests don't see the stub.
            from modelship.openai.tool_calling import registry

            registry._PARSERS.pop("stub-test-parser", None)


class TestHermesParser:
    parser = HermesToolCallParser()

    def test_no_tool_calls_returns_text_unchanged(self):
        result = self.parser.parse("just a regular response")
        assert result.tool_calls == []
        assert result.content == "just a regular response"
        assert result.has_tool_calls is False

    def test_single_tool_call(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Paris"}
        assert result.content is None

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
        )
        result = self.parser.parse(text)
        assert [tc.function.name for tc in result.tool_calls] == ["a", "b"]

    def test_tool_call_with_residual_text(self):
        text = 'Sure, calling that.\n<tool_call>{"name": "ping", "arguments": {}}</tool_call>'
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 1
        assert result.content == "Sure, calling that."

    def test_string_arguments_forwarded_verbatim(self):
        # vLLM-style: the streamer forwards the raw bytes of the arguments
        # value as the model emitted them, including any surrounding quotes
        # if the model wrapped its arguments in a JSON string literal. The
        # OpenAI streaming contract treats `arguments` as an opaque string
        # the client concatenates and parses.
        text = '<tool_call>{"name": "x", "arguments": "{\\"a\\": 1}"}</tool_call>'
        result = self.parser.parse(text)
        assert result.tool_calls[0].function.arguments == '"{\\"a\\": 1}"'

    def test_object_arguments_passed_through(self):
        text = '<tool_call>{"name": "x", "arguments": {"a": 1, "b": [2, 3]}}</tool_call>'
        result = self.parser.parse(text)
        assert json.loads(result.tool_calls[0].function.arguments) == {"a": 1, "b": [2, 3]}

    def test_block_without_extractable_name_is_dropped(self):
        # When the block contains nothing the name regex can hook onto we
        # silently drop it — there is nothing to tell the client about.
        text = "<tool_call>{not valid json}</tool_call>"
        result = self.parser.parse(text)
        assert result.tool_calls == []

    def test_missing_name_drops_call(self):
        text = '<tool_call>{"arguments": {}}</tool_call>'
        result = self.parser.parse(text)
        assert result.tool_calls == []

    def test_empty_name_drops_call(self):
        text = '<tool_call>{"name": "", "arguments": {}}</tool_call>'
        result = self.parser.parse(text)
        assert result.tool_calls == []

    def test_each_tool_call_gets_unique_id(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        result = self.parser.parse(text)
        assert result.tool_calls[0].id != result.tool_calls[1].id


class TestToolCallStreamer:
    """Drive the streamer one chunk at a time and verify the deltas.

    The cumulative-text protocol matches what serving_chat does in production:
    every fed string is the *full* generated text so far, not just the latest
    delta.
    """

    def _feed(self, chunks: list[str]) -> tuple[ToolCallStreamer, list]:
        streamer = ToolCallStreamer(HermesToolCallParser())
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

    def test_pure_content_streams_immediately(self):
        _, deltas = self._feed(["Hello", " ", "world"])
        assert "".join(d.content or "" for d in deltas) == "Hello world"
        assert all(not d.tool_calls for d in deltas)

    def test_holds_back_marker_prefix_in_content_until_finalize(self):
        # `<` could be the first char of `<tool_call>`; the streamer must not
        # ship it mid-stream. Once finalize() runs (no more text coming) the
        # held tail is safe to flush as content.
        streamer = ToolCallStreamer(HermesToolCallParser())
        mid = streamer.extract_streaming("before <")
        assert mid is not None and mid.content == "before "
        final = streamer.finalize()
        assert final is not None and final.content == "<"

    def test_held_tail_flushes_when_disambiguated(self):
        # `<too` arriving means we still hold; the next chunk `g` proves it
        # was not the marker, so the held tail flushes as content.
        streamer, deltas = self._feed(["text <too", "g"])
        assert "".join(d.content or "" for d in deltas) == "text <toog"
        assert streamer.result.tool_calls == []

    def test_emits_name_before_args(self):
        # Stream a tool call in many small chunks; verify the first tool-call
        # delta carries the function name and id, subsequent deltas carry
        # arguments fragments only.
        chunks = list('<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>')
        _, deltas = self._feed(chunks)

        tool_deltas = [tc for d in deltas for tc in d.tool_calls]
        # First tool delta: name + id, no arguments.
        assert tool_deltas[0].function is not None
        assert tool_deltas[0].function.name == "get_weather"
        assert tool_deltas[0].id is not None
        assert tool_deltas[0].function.arguments is None
        # Subsequent deltas carry arguments fragments only (no name).
        arg_deltas = tool_deltas[1:]
        assert all(d.function and d.function.name is None for d in arg_deltas)
        # Concatenated arguments form valid JSON.
        joined_args = "".join(d.function.arguments or "" for d in arg_deltas)
        assert json.loads(joined_args) == {"city": "Paris"}

    def test_arguments_stream_incrementally(self):
        # Feed the args char-by-char; each char-after-name should generate
        # an args delta of length 1 (or close to it).
        prefix = '<tool_call>{"name": "ping", "arguments": '
        suffix = '{"x": 42}}</tool_call>'
        _, deltas = self._feed([prefix, *list(suffix)])

        arg_deltas = [tc for d in deltas for tc in d.tool_calls if tc.function and tc.function.arguments is not None]
        assert len(arg_deltas) >= 3  # incremental, not one big shot
        joined = "".join(d.function.arguments or "" for d in arg_deltas)
        assert json.loads(joined) == {"x": 42}

    def test_multiple_tool_calls_get_distinct_indices(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
        )
        _, deltas = self._feed([text])

        tool_deltas = [tc for d in deltas for tc in d.tool_calls]
        indices = {d.index for d in tool_deltas}
        assert indices == {0, 1}
        names = [d.function.name for d in tool_deltas if d.function and d.function.name]
        assert names == ["a", "b"]

    def test_content_after_tool_call_resumes_streaming(self):
        text = '<tool_call>{"name": "p", "arguments": {}}</tool_call> ok'
        _, deltas = self._feed([text])
        joined_content = "".join(d.content or "" for d in deltas)
        assert "ok" in joined_content

    def test_partial_name_held_until_closing_quote(self):
        # While the model is mid-name (``"name": "get_wea`` so far), the
        # streamer must NOT send a partial name — wait for the closing quote.
        streamer = ToolCallStreamer(HermesToolCallParser())
        partial = '<tool_call>{"name": "get_wea'
        d = streamer.extract_streaming(partial)
        # No name yet → no tool delta.
        assert d is None or all(not tc.function or not tc.function.name for tc in d.tool_calls)

    def test_unterminated_block_is_finalized_without_crash(self):
        # Model stops mid tool-call. The streamer should not raise on finalize
        # and should not claim a finalized ToolCall (the call wasn't closed).
        text = '<tool_call>{"name": "incomplete", "arguments": {"a": 1'
        streamer, _ = self._feed([text])
        assert streamer.result.tool_calls == []


class TestResolveToolsForRequest:
    tools: ClassVar = [
        {"type": "function", "function": {"name": "alpha"}},
        {"type": "function", "function": {"name": "beta"}},
    ]

    def test_no_tools_returns_none(self):
        assert resolve_tools_for_request(None, "auto") is None
        assert resolve_tools_for_request([], "auto") is None

    def test_auto_passes_through(self):
        assert resolve_tools_for_request(self.tools, "auto") == self.tools

    def test_unset_tool_choice_passes_through(self):
        assert resolve_tools_for_request(self.tools, None) == self.tools

    def test_none_suppresses_tools(self):
        assert resolve_tools_for_request(self.tools, "none") is None

    def test_required_passes_through(self):
        # We cannot strictly enforce a tool call without constrained decoding,
        # so "required" downgrades to "auto" semantics (with a logged warning).
        assert resolve_tools_for_request(self.tools, "required") == self.tools

    def test_specific_function_filters_to_that_tool(self):
        result = resolve_tools_for_request(self.tools, {"type": "function", "function": {"name": "beta"}})
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "beta"

    def test_unknown_function_falls_back_to_all(self):
        # If the named function isn't in the tools list, fall back to passing
        # them all through rather than emitting an empty list.
        result = resolve_tools_for_request(self.tools, {"type": "function", "function": {"name": "missing"}})
        assert result == self.tools

    def test_unrecognized_choice_falls_back_to_all(self):
        result = resolve_tools_for_request(self.tools, "weird-mode")
        assert result == self.tools
