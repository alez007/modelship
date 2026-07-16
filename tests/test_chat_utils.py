"""Tests for modelship.openai.utils.chat."""

from modelship.openai.protocol import FunctionCall, ToolCall, UsageInfo
from modelship.openai.utils.chat import ParsedChatOutput, build_from_parsed, normalize_chat_messages


def test_tool_message_name_backfilled_from_assistant_call():
    messages = [
        {"role": "user", "content": "turn off the lights"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "abc", "type": "function", "function": {"name": "HassTurnOff"}}],
        },
        {"role": "tool", "tool_call_id": "abc", "content": '{"success": true}'},
    ]

    out = normalize_chat_messages(messages)

    tool_msg = out[-1]
    assert tool_msg["name"] == "HassTurnOff"
    # caller's dicts are not mutated
    assert "name" not in messages[-1]


def test_existing_tool_message_name_preserved():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "abc", "function": {"name": "HassTurnOff"}}],
        },
        {"role": "tool", "tool_call_id": "abc", "name": "explicit", "content": "ok"},
    ]

    out = normalize_chat_messages(messages)

    assert out[-1]["name"] == "explicit"


def test_orphan_tool_message_left_unchanged():
    messages = [
        {"role": "tool", "tool_call_id": "missing", "content": "ok"},
    ]

    out = normalize_chat_messages(messages)

    assert "name" not in out[-1]


def test_malformed_tool_calls_do_not_raise():
    messages = [
        {
            "role": "assistant",
            # tool_calls / function in unexpected shapes must not blow up
            "tool_calls": [
                "not-a-dict",
                {"id": "x", "function": "not-a-dict"},
                {"id": ["unhashable"], "function": {"name": "Skipped"}},
                {"id": "y", "function": {"name": "Real"}},
            ],
        },
        {"role": "assistant", "tool_calls": "not-a-list"},
        {"role": "tool", "tool_call_id": "y", "content": "ok"},
        {"role": "tool", "tool_call_id": "x", "content": "ok"},
    ]

    out = normalize_chat_messages(messages)

    assert out[-2]["name"] == "Real"
    # the call whose function was malformed yields no name
    assert "name" not in out[-1]


def test_non_string_tool_call_id_does_not_raise():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "abc", "function": {"name": "HassTurnOff"}}],
        },
        # unhashable tool_call_id must not raise on the mapping lookup
        {"role": "tool", "tool_call_id": ["abc"], "content": "ok"},
    ]

    out = normalize_chat_messages(messages)

    assert "name" not in out[-1]


def test_text_part_collapse_still_works():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "input_text", "text": "world"},
            ],
        },
    ]

    out = normalize_chat_messages(messages)

    assert out[0]["content"] == "hello\nworld"


def test_build_from_parsed_multi_choice_and_dto():
    choices = [
        ParsedChatOutput(
            content="Hello from choice 0",
            reasoning="Thinking 0...",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="get_weather", arguments="{}"),
                )
            ],
        ),
        ParsedChatOutput(
            content="Hello from choice 1",
            reasoning="Thinking 1...",
            tool_calls=[],
        ),
    ]
    usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    # Test single string finish_reason
    res1 = build_from_parsed(
        request_id="test_req_1",
        model_name="test_model",
        choices=choices,
        usage=usage,
        finish_reasons="length",
        created=12345,
    )

    assert res1.id == "test_req_1"
    assert res1.model == "test_model"
    assert len(res1.choices) == 2
    assert res1.choices[0].index == 0
    assert res1.choices[0].message.content == "Hello from choice 0"
    assert res1.choices[0].message.reasoning == "Thinking 0..."
    assert len(res1.choices[0].message.tool_calls) == 1
    assert res1.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert res1.choices[0].finish_reason == "length"

    assert res1.choices[1].index == 1
    assert res1.choices[1].message.content == "Hello from choice 1"
    assert res1.choices[1].message.reasoning == "Thinking 1..."
    assert len(res1.choices[1].message.tool_calls) == 0
    assert res1.choices[1].finish_reason == "length"

    # Test automatic finish_reason derivation
    res2 = build_from_parsed(
        request_id="test_req_2",
        model_name="test_model",
        choices=choices,
        usage=usage,
        created=12345,
    )
    assert res2.choices[0].finish_reason == "tool_calls"
    assert res2.choices[1].finish_reason == "stop"

    # Test list finish_reasons
    res3 = build_from_parsed(
        request_id="test_req_3",
        model_name="test_model",
        choices=choices,
        usage=usage,
        finish_reasons=["length", "stop"],
        created=12345,
    )
    assert res3.choices[0].finish_reason == "length"
    assert res3.choices[1].finish_reason == "stop"
