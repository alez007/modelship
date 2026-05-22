"""Tests for structured-outputs (`response_format`) handling across loaders."""

import pytest

from modelship.openai.protocol import ChatCompletionRequest


def _base_request(**overrides) -> dict:
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(overrides)
    return payload


_TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}

JSON_SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "person",
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        },
        "strict": True,
    },
}


class TestResponseFormatValidation:
    def test_response_format_propagates_without_tools(self):
        req = ChatCompletionRequest(**_base_request(response_format=JSON_SCHEMA_FORMAT))
        assert req.response_format == JSON_SCHEMA_FORMAT

    def test_json_object_format_propagates(self):
        req = ChatCompletionRequest(**_base_request(response_format={"type": "json_object"}))
        assert req.response_format == {"type": "json_object"}

    def test_text_format_with_tools_allowed(self):
        # response_format={"type":"text"} is a no-op and must not block tools.
        req = ChatCompletionRequest(
            **_base_request(response_format={"type": "text"}, tools=[_TOOL]),
        )
        assert req.response_format == {"type": "text"}
        assert req.tools == [_TOOL]

    def test_response_format_with_tools_allowed_when_tool_choice_none(self):
        req = ChatCompletionRequest(
            **_base_request(
                response_format=JSON_SCHEMA_FORMAT,
                tools=[_TOOL],
                tool_choice="none",
            ),
        )
        assert req.response_format == JSON_SCHEMA_FORMAT
        assert req.tool_choice == "none"

    @pytest.mark.parametrize("tool_choice", [None, "auto", "required"])
    def test_response_format_with_tools_rejected_for_active_tool_choice(self, tool_choice):
        payload = _base_request(response_format=JSON_SCHEMA_FORMAT, tools=[_TOOL])
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        with pytest.raises(ValueError, match="tool_choice='none'"):
            ChatCompletionRequest(**payload)

    def test_response_format_with_tools_rejected_for_named_tool(self):
        payload = _base_request(
            response_format=JSON_SCHEMA_FORMAT,
            tools=[_TOOL],
            tool_choice={"type": "function", "function": {"name": "f"}},
        )
        with pytest.raises(ValueError, match="tool_choice='none'"):
            ChatCompletionRequest(**payload)

    def test_json_object_with_tools_also_rejected(self):
        with pytest.raises(ValueError, match="tool_choice='none'"):
            ChatCompletionRequest(
                **_base_request(response_format={"type": "json_object"}, tools=[_TOOL]),
            )

    def test_no_response_format_no_op(self):
        req = ChatCompletionRequest(**_base_request())
        assert req.response_format is None


class TestVllmRoundTrip:
    """Confirm our request shape survives ``VllmChatCompletionRequest(**model_dump())``."""

    def test_response_format_round_trip(self):
        vllm = pytest.importorskip("vllm.entrypoints.openai.chat_completion.protocol")
        req = ChatCompletionRequest(**_base_request(response_format=JSON_SCHEMA_FORMAT))
        vllm_req = vllm.ChatCompletionRequest(**req.model_dump())
        assert vllm_req.response_format is not None
        assert vllm_req.response_format.type == "json_schema"
        assert vllm_req.response_format.json_schema is not None
        assert vllm_req.response_format.json_schema.name == "person"
        assert vllm_req.response_format.json_schema.json_schema == JSON_SCHEMA_FORMAT["json_schema"]["schema"]
        assert vllm_req.response_format.json_schema.strict is True

    def test_json_object_round_trip(self):
        vllm = pytest.importorskip("vllm.entrypoints.openai.chat_completion.protocol")
        req = ChatCompletionRequest(**_base_request(response_format={"type": "json_object"}))
        vllm_req = vllm.ChatCompletionRequest(**req.model_dump())
        assert vllm_req.response_format is not None
        assert vllm_req.response_format.type == "json_object"

    def test_response_format_with_tools_and_tool_choice_none_round_trip(self):
        """The one combination the validator allows still survives the dump."""
        vllm = pytest.importorskip("vllm.entrypoints.openai.chat_completion.protocol")
        req = ChatCompletionRequest(
            **_base_request(
                response_format=JSON_SCHEMA_FORMAT,
                tools=[_TOOL],
                tool_choice="none",
            ),
        )
        vllm_req = vllm.ChatCompletionRequest(**req.model_dump())
        assert vllm_req.response_format is not None
        assert vllm_req.tools is not None
        assert vllm_req.tool_choice == "none"
