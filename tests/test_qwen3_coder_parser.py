"""Qwen3-Coder XML-style tool-call parser tests.

Covers parsing of the ``<tool_call><function=...><parameter=...>…``
envelope, multi-call back-to-back streams, streaming monotonicity at
marker boundaries, and the ``classify_template`` disambiguation that
keeps Qwen3-Coder templates from falling through to the Hermes parser.
"""

from __future__ import annotations

import json

from modelship.openai.parsers.output import ChatOutputStreamer
from modelship.openai.parsers.tool_calling.parsers.qwen3_coder import Qwen3CoderToolCallParser
from modelship.openai.parsers.tool_calling.utils import classify_template


class TestQwen3CoderClassify:
    """``classify_template`` must route Qwen3-Coder templates here, not to Hermes."""

    def test_function_marker_routes_to_qwen3_coder(self):
        # The chat template mentions tools (gating clause) and contains
        # ``<function=`` — must not fall through to Hermes.
        template = "{% if tools %}<tool_call>\n<function={{ name }}>{% endif %}"
        assert classify_template(template) == "qwen3_coder"

    def test_parameter_marker_routes_to_qwen3_coder(self):
        template = "{% if tools %}<parameter={{ key }}>value</parameter>{% endif %}"
        assert classify_template(template) == "qwen3_coder"

    def test_hermes_template_without_function_marker_still_hermes(self):
        # Plain Hermes-style template: ``<tool_call>`` envelope but no
        # ``<function=`` body — must stay as hermes.
        template = '{% if tools %}<tool_call>{"name": "x"}</tool_call>{% endif %}'
        assert classify_template(template) == "hermes"


class TestQwen3CoderParser:
    def test_single_tool_call_one_param(self):
        text = (
            "<tool_call>\n<function=get_weather>\n<parameter=location>\nTokyo\n</parameter>\n</function>\n</tool_call>"
        )
        result = Qwen3CoderToolCallParser().parse(text)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"location": "Tokyo"}

    def test_single_tool_call_multi_params(self):
        text = (
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>\nrust async runtimes\n</parameter>\n"
            "<parameter=top_k>\n5\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = Qwen3CoderToolCallParser().parse(text)
        assert result.tool_calls[0].function.name == "search"
        # Values are kept as strings — see parser module docstring on why.
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "query": "rust async runtimes",
            "top_k": "5",
        }

    def test_inline_values_without_newlines(self):
        text = "<tool_call><function=ping><parameter=host>example.com</parameter></function></tool_call>"
        result = Qwen3CoderToolCallParser().parse(text)
        assert json.loads(result.tool_calls[0].function.arguments) == {"host": "example.com"}

    def test_two_back_to_back_tool_calls(self):
        # Each call is its own ``<tool_call>…</tool_call>`` envelope; the
        # streamer treats them as independent regions.
        text = (
            "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"
            "<tool_call>\n<function=b>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>"
        )
        result = Qwen3CoderToolCallParser().parse(text)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "a"
        assert result.tool_calls[1].function.name == "b"
        assert json.loads(result.tool_calls[0].function.arguments) == {"x": "1"}
        assert json.loads(result.tool_calls[1].function.arguments) == {"y": "2"}

    def test_no_tool_calls_returns_text_unchanged(self):
        result = Qwen3CoderToolCallParser().parse("just a regular response")
        assert result.tool_calls == []
        assert result.content == "just a regular response"
        assert result.has_tool_calls is False

    def test_value_preserves_internal_whitespace(self):
        # Leading/trailing whitespace stripped, internal preserved.
        text = "<tool_call><function=run><parameter=code>\ndef f():\n    return 1\n</parameter></function></tool_call>"
        result = Qwen3CoderToolCallParser().parse(text)
        assert json.loads(result.tool_calls[0].function.arguments) == {
            "code": "def f():\n    return 1",
        }


class TestQwen3CoderStreaming:
    def test_streaming_basic(self):
        parser = Qwen3CoderToolCallParser()
        streamer = ChatOutputStreamer(parser)

        d1 = streamer.extract_streaming("<tool_call>\n<function=get_weather>\n<param")
        assert d1.tool_calls and d1.tool_calls[0].function.name == "get_weather"

        # Partial parameter value visible.
        streamer.extract_streaming("<tool_call>\n<function=get_weather>\n<parameter=location>\nTok")

        # Final close.
        streamer.extract_streaming(
            "<tool_call>\n<function=get_weather>\n<parameter=location>\nTokyo\n</parameter>\n</function>\n</tool_call>"
        )
        streamer.finalize()

        result = streamer.result
        assert len(result.tool_calls) == 1
        assert json.loads(result.tool_calls[0].function.arguments) == {"location": "Tokyo"}

    def test_streaming_monotonic_concat_matches_final(self):
        # The concatenation of all streamed arg deltas must equal the
        # final arguments JSON (the streamer's prefix-extension contract).
        parser = Qwen3CoderToolCallParser()
        streamer = ChatOutputStreamer(parser)
        chunks = [
            "<tool_call>\n<function=run>\n<parameter=cmd>\nls",
            "<tool_call>\n<function=run>\n<parameter=cmd>\nls -la",
            "<tool_call>\n<function=run>\n<parameter=cmd>\nls -la\n</parameter>\n"
            "<parameter=cwd>\n/tmp\n</parameter>\n</function>\n</tool_call>",
        ]
        seen = ""
        for c in chunks:
            d = streamer.extract_streaming(c)
            if d and d.tool_calls:
                for t in d.tool_calls:
                    if t.function and t.function.arguments:
                        seen += t.function.arguments
        streamer.finalize()
        assert json.loads(seen) == {"cmd": "ls -la", "cwd": "/tmp"}

    def test_streaming_chunk_mid_parameter_close(self):
        # Regression: a chunk that ends mid-``</parameter>`` (e.g.
        # ``</param``) must not leak those bytes into the value stream.
        parser = Qwen3CoderToolCallParser()
        streamer = ChatOutputStreamer(parser)
        chunks = [
            "<tool_call>\n<function=f>\n<parameter=x>\nval</param",
            "<tool_call>\n<function=f>\n<parameter=x>\nval</parameter>\n</function>\n</tool_call>",
        ]
        seen = ""
        for c in chunks:
            d = streamer.extract_streaming(c)
            if d and d.tool_calls:
                for t in d.tool_calls:
                    if t.function and t.function.arguments:
                        seen += t.function.arguments
        streamer.finalize()
        assert json.loads(seen) == {"x": "val"}

    def test_streaming_chunk_mid_next_parameter_opener(self):
        # Regression: between parameters, a chunk ending in ``<parameter``
        # must not leak those bytes into the previous value.
        parser = Qwen3CoderToolCallParser()
        streamer = ChatOutputStreamer(parser)
        chunks = [
            "<tool_call>\n<function=f>\n<parameter=a>\nfirst\n</parameter>\n<parameter",
            "<tool_call>\n<function=f>\n<parameter=a>\nfirst\n</parameter>\n"
            "<parameter=b>\nsecond\n</parameter>\n</function>\n</tool_call>",
        ]
        seen = ""
        for c in chunks:
            d = streamer.extract_streaming(c)
            if d and d.tool_calls:
                for t in d.tool_calls:
                    if t.function and t.function.arguments:
                        seen += t.function.arguments
        streamer.finalize()
        assert json.loads(seen) == {"a": "first", "b": "second"}
