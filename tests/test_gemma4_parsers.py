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
