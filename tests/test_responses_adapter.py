"""Tests for the Responses -> chat-completions request-side adapter and the shared
response envelope."""

import pytest

from modelship.openai.protocol import ResponsesRequest
from modelship.openai.protocol.responses import (
    UnsupportedResponsesFeatureError,
    responses_request_to_chat,
)
from modelship.openai.protocol.responses.adapter import build_response_object


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

    def test_function_call_missing_name_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError, match="function_call"):
            responses_request_to_chat(_req(input=[{"type": "function_call", "call_id": "call_1", "arguments": "{}"}]))

    def test_function_call_missing_call_id_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError, match="function_call"):
            responses_request_to_chat(_req(input=[{"type": "function_call", "name": "f", "arguments": "{}"}]))

    def test_function_call_output_missing_call_id_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError, match="function_call_output"):
            responses_request_to_chat(_req(input=[{"type": "function_call_output", "output": "sunny"}]))

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


class TestResponseEnvelopeEcho:
    """`store` / `previous_response_id` are echoed from the request. The gateway reads
    the same `store` field to decide whether to persist, so the response can't claim
    one thing while the store did another."""

    def _build(self, request):
        return build_response_object(request, status="completed", output=[], usage=None, incomplete=None)

    def test_store_defaults_to_true_when_unset(self):
        # OpenAI stores by default; a client that never sends the field still expects
        # its previous_response_id to work on the next turn.
        assert self._build(_req()).store is True

    def test_store_true_echoed(self):
        assert self._build(_req(store=True)).store is True

    def test_explicit_store_false_echoed(self):
        assert self._build(_req(store=False)).store is False

    def test_previous_response_id_echoed(self):
        assert self._build(_req(previous_response_id="resp_1")).previous_response_id == "resp_1"

    def test_previous_response_id_absent_is_none(self):
        assert self._build(_req()).previous_response_id is None


class TestRequestRejections:
    def test_previous_response_id_accepted(self):
        # The gateway resolves it into `input` before the Ray hop; the adapter only
        # echoes it, so reaching here with one set is legitimate.
        chat = responses_request_to_chat(_req(previous_response_id="resp_1"))
        assert chat.messages == [{"role": "user", "content": "hello"}]

    def test_background_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError, match="background"):
            responses_request_to_chat(_req(background=True))

    def test_hosted_tool_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError, match="hosted tool"):
            responses_request_to_chat(_req(tools=[{"type": "web_search"}]))

    def test_text_format_as_string_rejected(self):
        # A malformed text.format (string instead of object) must be a clean
        # 400, not an AttributeError -> 500.
        with pytest.raises(UnsupportedResponsesFeatureError, match="must be an object"):
            responses_request_to_chat(_req(text={"format": "json_object"}))

    def test_store_true_is_accepted(self):
        # store defaults to true on OpenAI; we accept-but-don't-persist rather than reject.
        chat = responses_request_to_chat(_req(store=True))
        assert chat.messages == [{"role": "user", "content": "hello"}]
