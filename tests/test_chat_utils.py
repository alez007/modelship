"""Tests for modelship.openai.chat_utils.normalize_chat_messages."""

from modelship.openai.chat_utils import normalize_chat_messages


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
