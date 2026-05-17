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


class TestResponseFormatPropagation:
    """response_format passes through our protocol unchanged. The protocol
    layer takes no opinion on tools/response_format coexistence — that's a
    per-loader concern (OpenAI allows both together).
    """

    def test_response_format_propagates_without_tools(self):
        req = ChatCompletionRequest(**_base_request(response_format=JSON_SCHEMA_FORMAT))
        assert req.response_format == JSON_SCHEMA_FORMAT

    def test_json_object_format_propagates(self):
        req = ChatCompletionRequest(**_base_request(response_format={"type": "json_object"}))
        assert req.response_format == {"type": "json_object"}

    def test_response_format_propagates_with_tools(self):
        tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
        req = ChatCompletionRequest(
            **_base_request(response_format=JSON_SCHEMA_FORMAT, tools=[tool]),
        )
        assert req.response_format == JSON_SCHEMA_FORMAT
        assert req.tools == [tool]

    def test_response_format_propagates_with_tool_choice_none(self):
        tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
        req = ChatCompletionRequest(
            **_base_request(response_format=JSON_SCHEMA_FORMAT, tools=[tool], tool_choice="none"),
        )
        assert req.response_format == JSON_SCHEMA_FORMAT
        assert req.tool_choice == "none"

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

    def test_response_format_and_tools_coexist(self):
        """vLLM honors both natively — confirm our model_dump() preserves the pair."""
        vllm = pytest.importorskip("vllm.entrypoints.openai.chat_completion.protocol")
        tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
        req = ChatCompletionRequest(
            **_base_request(response_format=JSON_SCHEMA_FORMAT, tools=[tool]),
        )
        vllm_req = vllm.ChatCompletionRequest(**req.model_dump())
        assert vllm_req.response_format is not None
        assert vllm_req.response_format.type == "json_schema"
        assert vllm_req.tools is not None
        assert len(vllm_req.tools) == 1
