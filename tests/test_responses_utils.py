"""Tests for modelship.openai.utils.responses's parsed-output shaping."""

from modelship.openai.protocol import FunctionCall, ToolCall
from modelship.openai.utils.chat import ParsedChatOutput
from modelship.openai.utils.responses import build_responses_items_from_parsed


def test_build_responses_items_from_parsed_orders_reasoning_message_tools():
    parsed = ParsedChatOutput(
        content="Hello!",
        reasoning="Thinking...",
        tool_calls=[ToolCall(id="call_1", type="function", function=FunctionCall(name="get_weather", arguments="{}"))],
    )

    output = build_responses_items_from_parsed(parsed)

    assert [item.type for item in output] == ["reasoning", "message", "function_call"]
    assert output[0].summary[0].text == "Thinking..."
    assert output[1].content[0].text == "Hello!"
    assert output[2].call_id == "call_1"
    assert output[2].name == "get_weather"
    assert output[2].arguments == "{}"


def test_build_responses_items_from_parsed_skips_empty_fields():
    parsed = ParsedChatOutput(content=None, reasoning=None, tool_calls=[])

    assert build_responses_items_from_parsed(parsed) == []
