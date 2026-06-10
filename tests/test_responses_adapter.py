"""Tests for the stateless Responses <-> chat-completions adapter (Phase A)."""

import pytest

from modelship.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    FunctionCall,
    ResponsesRequest,
    ToolCall,
    UsageInfo,
)
from modelship.openai.protocol.responses import (
    UnsupportedResponsesFeatureError,
    chat_response_to_responses,
    responses_request_to_chat,
)


def _req(**overrides) -> ResponsesRequest:
    payload = {"model": "m", "input": "hello"}
    payload.update(overrides)
    return ResponsesRequest(**payload)


class TestRequestInputTranslation:
    def test_string_input_becomes_user_message(self):
        chat = responses_request_to_chat(_req(input="hi there"))
        assert chat.messages == [{"role": "user", "content": "hi there"}]
        assert chat.model == "m"
        assert chat.stream is False

    def test_instructions_become_leading_system_message(self):
        chat = responses_request_to_chat(_req(input="hi", instructions="be terse"))
        assert chat.messages[0] == {"role": "system", "content": "be terse"}
        assert chat.messages[1] == {"role": "user", "content": "hi"}

    def test_message_items_with_content_parts_flatten_to_text(self):
        chat = responses_request_to_chat(
            _req(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}],
                    },
                ]
            )
        )
        assert chat.messages == [{"role": "user", "content": "ab"}]

    def test_role_shorthand_without_type_is_a_message(self):
        chat = responses_request_to_chat(_req(input=[{"role": "user", "content": "yo"}]))
        assert chat.messages == [{"role": "user", "content": "yo"}]

    def test_function_call_and_output_round_trip(self):
        chat = responses_request_to_chat(
            _req(
                input=[
                    {"role": "user", "content": "weather?"},
                    {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
                ]
            )
        )
        assert chat.messages[1]["role"] == "assistant"
        assert chat.messages[1]["tool_calls"][0]["id"] == "call_1"
        assert chat.messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert chat.messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}

    def test_reasoning_input_item_is_dropped(self):
        chat = responses_request_to_chat(
            _req(
                input=[
                    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "x"}]},
                    {"role": "user", "content": "go"},
                ]
            )
        )
        assert chat.messages == [{"role": "user", "content": "go"}]

    def test_unknown_input_item_type_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError):
            responses_request_to_chat(_req(input=[{"type": "image_generation_call"}]))


class TestRequestFieldTranslation:
    def test_max_output_tokens_maps_to_max_completion_tokens(self):
        chat = responses_request_to_chat(_req(max_output_tokens=128))
        assert chat.max_completion_tokens == 128

    def test_tools_flattened_to_nested(self):
        chat = responses_request_to_chat(
            _req(tools=[{"type": "function", "name": "f", "description": "d", "parameters": {"type": "object"}}])
        )
        assert chat.tools == [
            {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}
        ]

    def test_tool_choice_object_translated(self):
        chat = responses_request_to_chat(_req(tool_choice={"type": "function", "name": "f"}))
        assert chat.tool_choice == {"type": "function", "function": {"name": "f"}}

    def test_text_format_json_schema_nested(self):
        chat = responses_request_to_chat(
            _req(text={"format": {"type": "json_schema", "name": "p", "schema": {"type": "object"}, "strict": True}})
        )
        assert chat.response_format == {
            "type": "json_schema",
            "json_schema": {"name": "p", "schema": {"type": "object"}, "strict": True},
        }

    def test_reasoning_effort_passes_through(self):
        chat = responses_request_to_chat(_req(reasoning={"effort": "high"}))
        assert chat.reasoning_effort == "high"


class TestRequestRejections:
    def test_previous_response_id_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError, match="previous_response_id"):
            responses_request_to_chat(_req(previous_response_id="resp_1"))

    def test_background_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError, match="background"):
            responses_request_to_chat(_req(background=True))

    def test_hosted_tool_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError, match="hosted tool"):
            responses_request_to_chat(_req(tools=[{"type": "web_search"}]))

    def test_store_true_is_accepted(self):
        # store defaults to true on OpenAI; we accept-but-don't-persist rather than reject.
        chat = responses_request_to_chat(_req(store=True))
        assert chat.messages == [{"role": "user", "content": "hello"}]


def _chat_response(*, content=None, reasoning=None, tool_calls=None, finish_reason="stop") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model="m",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=content,
                    reasoning=reasoning,
                    tool_calls=tool_calls or [],
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=7, total_tokens=12),
    )


class TestResponseTranslation:
    def test_text_becomes_output_message(self):
        resp = chat_response_to_responses(_chat_response(content="hi"), _req())
        assert resp.object == "response"
        assert resp.status == "completed"
        assert len(resp.output) == 1
        msg = resp.output[0]
        assert msg.type == "message"
        assert msg.content[0].type == "output_text"
        assert msg.content[0].text == "hi"

    def test_reasoning_becomes_reasoning_item_first(self):
        resp = chat_response_to_responses(_chat_response(content="answer", reasoning="because"), _req())
        assert resp.output[0].type == "reasoning"
        assert resp.output[0].summary[0].text == "because"
        assert resp.output[1].type == "message"

    def test_tool_calls_become_function_call_items(self):
        tc = ToolCall(id="call_9", function=FunctionCall(name="f", arguments="{}"))
        resp = chat_response_to_responses(_chat_response(tool_calls=[tc]), _req())
        fc = [o for o in resp.output if o.type == "function_call"]
        assert len(fc) == 1
        assert fc[0].call_id == "call_9"
        assert fc[0].name == "f"

    def test_usage_remapped(self):
        resp = chat_response_to_responses(_chat_response(content="x"), _req())
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 7
        assert resp.usage.total_tokens == 12

    def test_length_finish_reason_is_incomplete(self):
        resp = chat_response_to_responses(_chat_response(content="x", finish_reason="length"), _req())
        assert resp.status == "incomplete"
        assert resp.incomplete_details == {"reason": "max_output_tokens"}

    def test_store_always_false_and_settings_echoed(self):
        resp = chat_response_to_responses(
            _chat_response(content="x"),
            _req(store=True, instructions="sys", max_output_tokens=64, metadata={"k": "v"}),
        )
        assert resp.store is False
        assert resp.instructions == "sys"
        assert resp.max_output_tokens == 64
        assert resp.metadata == {"k": "v"}
