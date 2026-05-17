"""Tests for structured-outputs (`response_format`) handling across loaders."""

import logging

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


class TestResponseFormatValidator:
    def test_response_format_propagates_without_tools(self):
        req = ChatCompletionRequest(**_base_request(response_format=JSON_SCHEMA_FORMAT))
        assert req.response_format == JSON_SCHEMA_FORMAT

    def test_json_object_format_propagates(self):
        req = ChatCompletionRequest(**_base_request(response_format={"type": "json_object"}))
        assert req.response_format == {"type": "json_object"}

    def test_tools_drop_response_format(self, caplog):
        # `modelship.*` loggers have propagate=False after configure_logging(),
        # so attach caplog's handler directly to catch warnings from this test.
        tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
        target = logging.getLogger("modelship.openai.protocol")
        target.addHandler(caplog.handler)
        try:
            caplog.set_level(logging.WARNING)
            req = ChatCompletionRequest(
                **_base_request(response_format=JSON_SCHEMA_FORMAT, tools=[tool]),
            )
        finally:
            target.removeHandler(caplog.handler)
        assert req.response_format is None
        assert req.tools == [tool]
        assert any("ignoring response_format" in r.message for r in caplog.records)

    def test_no_response_format_no_op(self):
        req = ChatCompletionRequest(**_base_request())
        assert req.response_format is None

    def test_empty_tools_list_does_not_drop(self):
        # An explicitly empty `tools` should be falsy and not trigger the drop.
        req = ChatCompletionRequest(**_base_request(response_format=JSON_SCHEMA_FORMAT, tools=[]))
        assert req.response_format == JSON_SCHEMA_FORMAT


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

    def test_tools_present_strips_response_format_before_vllm(self):
        vllm = pytest.importorskip("vllm.entrypoints.openai.chat_completion.protocol")
        tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
        req = ChatCompletionRequest(
            **_base_request(response_format=JSON_SCHEMA_FORMAT, tools=[tool]),
        )
        vllm_req = vllm.ChatCompletionRequest(**req.model_dump())
        assert vllm_req.response_format is None
