"""Tests for the Responses -> chat-completions request-side adapter and the shared
response envelope."""

import pytest

from modelship.openai import compaction_crypto
from modelship.openai.protocol import ResponsesRequest
from modelship.openai.protocol.responses import (
    UnsupportedResponsesFeatureError,
    responses_request_to_chat,
)
from modelship.openai.protocol.responses.adapter import _content_to_chat, build_response_object
from modelship.openai.protocol.responses.schemas import ResponseReasoningItem


@pytest.fixture(autouse=True)
def _reset_ephemeral_compaction_key():
    compaction_crypto._ephemeral_key = None
    yield
    compaction_crypto._ephemeral_key = None


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

    def test_message_with_image_part_reaches_chat_as_image_url(self):
        chat = responses_request_to_chat(
            _req(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "what is this?"},
                            {"type": "input_image", "image_url": "https://example.com/cat.png"},
                        ],
                    },
                ]
            )
        )
        content = chat.messages[0]["content"]
        assert content == [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ]

    def test_unknown_content_part_type_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError, match="content part"):
            responses_request_to_chat(
                _req(input=[{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}])
            )

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

    def test_function_call_output_with_non_text_output_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError, match="content shape"):
            responses_request_to_chat(
                _req(input=[{"type": "function_call_output", "call_id": "call_1", "output": {"bad": "shape"}}])
            )

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


class TestCompactionInputDecode:
    """Round-trip half of ``/v1/responses/compact``: a ``compaction`` input item is
    decrypted back into the items it was built from and spliced into the message list,
    exactly as ``websocket-compact-new-chain`` requires."""

    def test_compaction_item_decodes_into_messages(self):
        summary_items = [{"role": "assistant", "content": "previously: the user is named Alex"}]
        blob = compaction_crypto.encrypt_items(summary_items)
        chat = responses_request_to_chat(
            _req(
                input=[
                    {"type": "compaction", "id": "cmp_1", "encrypted_content": blob},
                    {"role": "user", "content": "what's my name?"},
                ]
            )
        )
        assert chat.messages == [
            {"role": "assistant", "content": "previously: the user is named Alex"},
            {"role": "user", "content": "what's my name?"},
        ]

    def test_missing_encrypted_content_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError, match="encrypted_content"):
            responses_request_to_chat(_req(input=[{"type": "compaction", "id": "cmp_1"}]))

    def test_tampered_encrypted_content_is_a_clean_rejection(self):
        blob = compaction_crypto.encrypt_items([{"role": "assistant", "content": "x"}])
        tampered = blob[:-4] + ("AAAA" if blob[-4:] != "AAAA" else "BBBB")
        with pytest.raises(UnsupportedResponsesFeatureError, match="could not be decoded"):
            responses_request_to_chat(
                _req(input=[{"type": "compaction", "id": "cmp_1", "encrypted_content": tampered}])
            )

    def test_wrong_key_is_the_same_clean_rejection_as_tampering(self, monkeypatch):
        # Both a tampered blob and one encrypted under a different key must produce
        # the identical error — the message must never reveal which case it was.
        from cryptography.fernet import Fernet

        blob = compaction_crypto.encrypt_items([{"role": "assistant", "content": "x"}])
        monkeypatch.setenv("MSHIP_COMPACTION_KEY", Fernet.generate_key().decode("ascii"))
        with pytest.raises(UnsupportedResponsesFeatureError, match="could not be decoded"):
            responses_request_to_chat(_req(input=[{"type": "compaction", "id": "cmp_1", "encrypted_content": blob}]))


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

    def _build(self, request, **kwargs):
        return build_response_object(request, status="completed", output=[], usage=None, incomplete=None, **kwargs)

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


class TestResponseResourceRequiredFields:
    """The Open Responses conformance suite validates against `ResponseResource`'s
    required-field list; these were previously missing or sent as `null`."""

    def _build(self, request, **kwargs):
        return build_response_object(request, status="completed", output=[], usage=None, incomplete=None, **kwargs)

    def test_static_defaults_for_previously_missing_fields(self):
        dumped = self._build(_req()).model_dump(mode="json")
        assert dumped["truncation"] == "disabled"
        assert dumped["presence_penalty"] == 0.0
        assert dumped["frequency_penalty"] == 0.0
        assert dumped["top_logprobs"] == 0
        assert dumped["max_tool_calls"] is None
        assert dumped["background"] is False
        assert dumped["service_tier"] == "default"
        assert dumped["safety_identifier"] is None
        assert dumped["prompt_cache_key"] is None

    def test_temperature_and_top_p_resolve_to_openai_defaults_when_unset(self):
        resp = self._build(_req())
        assert resp.temperature == 1.0
        assert resp.top_p == 1.0

    def test_explicit_temperature_and_top_p_are_preserved(self):
        resp = self._build(_req(temperature=0.2, top_p=0.5))
        assert resp.temperature == 0.2
        assert resp.top_p == 0.5

    def test_text_defaults_to_plain_text_format_when_unset(self):
        resp = self._build(_req())
        assert resp.text == {"format": {"type": "text"}}

    def test_explicit_text_is_preserved(self):
        resp = self._build(_req(text={"format": {"type": "json_object"}}))
        assert resp.text == {"format": {"type": "json_object"}}

    def test_completed_at_absent_by_default(self):
        assert self._build(_req()).completed_at is None

    def test_completed_at_set_when_passed(self):
        resp = self._build(_req(), completed_at=1234)
        assert resp.completed_at == 1234


class TestEchoedTools:
    def _tools_on(self, request):
        return build_response_object(request, status="completed", output=[], usage=None, incomplete=None).tools

    def test_partial_tool_backfilled_with_all_five_keys(self):
        tools = self._tools_on(_req(tools=[{"type": "function", "name": "f"}]))
        assert tools == [{"type": "function", "name": "f", "description": None, "parameters": None, "strict": None}]

    def test_fully_specified_tool_unchanged(self):
        tool = {"type": "function", "name": "f", "description": "d", "parameters": {"type": "object"}, "strict": True}
        assert self._tools_on(_req(tools=[tool])) == [tool]

    def test_no_tools_echoes_empty_list(self):
        assert self._tools_on(_req()) == []


class TestReasoningItemSerialization:
    def test_encrypted_content_omitted_when_unset(self):
        dumped = ResponseReasoningItem().model_dump(mode="json")
        assert "encrypted_content" not in dumped

    def test_encrypted_content_present_when_set(self):
        dumped = ResponseReasoningItem(encrypted_content="abc").model_dump(mode="json")
        assert dumped["encrypted_content"] == "abc"


class TestContentToChat:
    def test_plain_string_passes_through(self):
        assert _content_to_chat("hi") == "hi"

    def test_none_passes_through(self):
        assert _content_to_chat(None) is None

    def test_text_only_parts_collapse_to_string(self):
        assert _content_to_chat([{"type": "input_text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"

    def test_image_part_produces_parts_list(self):
        result = _content_to_chat([{"type": "input_image", "image_url": "https://x/y.png"}])
        assert result == [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]

    def test_mixed_text_and_image_produces_parts_list(self):
        result = _content_to_chat(
            [{"type": "input_text", "text": "look"}, {"type": "input_image", "image_url": "https://x/y.png"}]
        )
        assert result == [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]

    def test_unknown_part_type_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError, match="content part type"):
            _content_to_chat([{"type": "input_audio", "input_audio": {}}])

    def test_non_dict_part_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError, match="content part"):
            _content_to_chat(["oops"])

    def test_non_list_non_string_content_rejected(self):
        with pytest.raises(UnsupportedResponsesFeatureError, match="content shape"):
            _content_to_chat({"type": "input_text", "text": "not wrapped in a list"})


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
