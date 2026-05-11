import json

from modelship.openai.parsers.output import ChatOutputStreamer
from modelship.openai.parsers.reasoning.parsers.gemma import Gemma4ReasoningParser
from modelship.openai.parsers.tool_calling.parsers.gemma import (
    FunctionGemmaToolCallParser,
    Gemma4ToolCallParser,
)


class TestGemma4ReasoningParser:
    def test_extracts_reasoning_and_strips_label(self):
        parser = Gemma4ReasoningParser()
        streamer = ChatOutputStreamer(None, parser)

        # Simulate model output: <|channel>thought\nI am reasoning<channel|>Final answer
        text = "<|channel>thought\nI am reasoning<channel|>Final answer"
        streamer.extract_streaming(text)
        streamer.finalize()

        result = streamer.result
        assert result.reasoning == "I am reasoning"
        assert result.content == "Final answer"


class TestGemma4ToolCallParser:
    def test_single_tool_call(self):
        parser = Gemma4ToolCallParser()
        # Simulate: <|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>
        text = '<|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>'
        result = parser.parse(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"location": "Tokyo"}

    def test_multiple_concatenated_tool_calls(self):
        parser = Gemma4ToolCallParser()
        # Simulate: <|tool_call>call:a{x:1}call:b{y:2}<tool_call|>
        text = '<|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}call:get_time{timezone:<|"|>JST<|"|>}<tool_call|>'
        result = parser.parse(text)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"
        assert json.loads(result.tool_calls[0].function.arguments) == {"location": "Tokyo"}
        assert json.loads(result.tool_calls[1].function.arguments) == {"timezone": "JST"}

    def test_streaming_tool_call(self):
        parser = Gemma4ToolCallParser()
        streamer = ChatOutputStreamer(parser)

        # Chunk 1: Start and name
        d1 = streamer.extract_streaming("<|tool_call>call:get_weather{loc")
        assert d1.tool_calls[0].function.name == "get_weather"

        # Chunk 2: Partial arguments
        d2 = streamer.extract_streaming('<|tool_call>call:get_weather{location:<|"|>Tok')
        # "location": "Tok" -> length diff
        assert "location" in d2.tool_calls[0].function.arguments

        # Chunk 3: End of call
        d3 = streamer.extract_streaming('<|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>')
        assert "yo" in d3.tool_calls[0].function.arguments

        streamer.finalize()
        result = streamer.result
        assert len(result.tool_calls) == 1
        assert json.loads(result.tool_calls[0].function.arguments) == {"location": "Tokyo"}


class TestFunctionGemmaToolCallParser:
    def test_single_tool_call(self):
        parser = FunctionGemmaToolCallParser()
        # Simulate: <start_function_call>call:get_weather{location:<escape>Tokyo<escape>}<end_function_call>
        text = "<start_function_call>call:get_weather{location:<escape>Tokyo<escape>}<end_function_call>"
        result = parser.parse(text)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"location": "Tokyo"}

    def test_streaming_tool_call(self):
        parser = FunctionGemmaToolCallParser()
        streamer = ChatOutputStreamer(parser)

        # Chunk 1: Start and name
        d1 = streamer.extract_streaming("<start_function_call>call:get_weather{loc")
        assert d1.tool_calls[0].function.name == "get_weather"

        # Chunk 2: Partial arguments
        d2 = streamer.extract_streaming("<start_function_call>call:get_weather{location:<escape>Tok")
        # "location": "Tok" -> length diff
        assert "location" in d2.tool_calls[0].function.arguments

        # Chunk 3: End of call
        d3 = streamer.extract_streaming(
            "<start_function_call>call:get_weather{location:<escape>Tokyo<escape>}<end_function_call>"
        )
        assert "yo" in d3.tool_calls[0].function.arguments

        streamer.finalize()
        result = streamer.result
        assert len(result.tool_calls) == 1
        assert json.loads(result.tool_calls[0].function.arguments) == {"location": "Tokyo"}

    def test_markers_are_specials_is_true(self):
        # FunctionGemma's envelope and string-delim tokens are registered
        # specials on the tokenizer; the transformers loader keys off this
        # flag to switch ``skip_special_tokens=False`` and keep the markers
        # visible to the parser.
        assert FunctionGemmaToolCallParser.markers_are_specials is True


class TestGemmaNestedStructures:
    """Custom-syntax parser handles nested objects and arrays correctly."""

    def test_array_of_objects(self):
        parser = Gemma4ToolCallParser()
        text = "<|tool_call>call:f{items:[{a:1},{a:2}]}<tool_call|>"
        result = parser.parse(text)
        assert json.loads(result.tool_calls[0].function.arguments) == {"items": [{"a": 1}, {"a": 2}]}

    def test_array_of_mixed_types(self):
        parser = Gemma4ToolCallParser()
        text = '<|tool_call>call:f{items:[<|"|>x<|"|>,{n:1},<|"|>y<|"|>]}<tool_call|>'
        result = parser.parse(text)
        assert json.loads(result.tool_calls[0].function.arguments) == {"items": ["x", {"n": 1}, "y"]}

    def test_nested_arrays(self):
        parser = Gemma4ToolCallParser()
        text = "<|tool_call>call:f{grid:[[1,2],[3,4]]}<tool_call|>"
        result = parser.parse(text)
        assert json.loads(result.tool_calls[0].function.arguments) == {"grid": [[1, 2], [3, 4]]}

    def test_function_gemma_array_of_objects(self):
        parser = FunctionGemmaToolCallParser()
        text = "<start_function_call>call:f{items:[{a:1},{a:2}]}<end_function_call>"
        result = parser.parse(text)
        assert json.loads(result.tool_calls[0].function.arguments) == {"items": [{"a": 1}, {"a": 2}]}


class TestGemmaStreamingMonotonicity:
    """Partial string parsing must yield monotonic prefixes of the final value.

    Regression: when a chunk ends mid-way through the closing string delim
    (e.g. ``<es`` for FunctionGemma's ``<escape>``), the partial parse used
    to include those bytes in the string value, producing a stripped JSON
    longer than the eventual clean output — and the streamer's
    length-greater diff machinery could not retract the transient bytes,
    leaving the client stuck with a malformed value.
    """

    def test_function_gemma_chunk_mid_closing_delim(self):
        parser = FunctionGemmaToolCallParser()
        streamer = ChatOutputStreamer(parser)
        chunks = [
            "<start_function_call>call:f{location:<escape>Tokyo<es",
            "<start_function_call>call:f{location:<escape>Tokyo<escape>}<end_function_call>",
        ]
        seen = ""
        for c in chunks:
            d = streamer.extract_streaming(c)
            if d and d.tool_calls:
                for t in d.tool_calls:
                    if t.function and t.function.arguments:
                        seen += t.function.arguments
        streamer.finalize()
        assert json.loads(seen) == {"location": "Tokyo"}

    def test_gemma4_chunk_mid_closing_delim(self):
        parser = Gemma4ToolCallParser()
        streamer = ChatOutputStreamer(parser)
        chunks = [
            '<|tool_call>call:f{x:<|"|>val<',
            '<|tool_call>call:f{x:<|"|>val<|"|>}<tool_call|>',
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
