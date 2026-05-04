"""Tests for the Transformers chat path's tool-call handling.

These tests bypass the real HF pipeline by injecting a callable that returns
a canned generation, so they run offline and do not touch any model weights.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from modelship.infer.infer_config import RawRequestProxy, TransformersConfig
from modelship.infer.transformers.capabilities import TransformersCapabilities
from modelship.infer.transformers.openai.serving_chat import OpenAIServingChat
from modelship.openai.protocol import ChatCompletionRequest, ChatCompletionResponse


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [0] * len(text.split())

    def apply_chat_template(self, messages: list[dict], **kwargs: Any) -> Any:
        prompt = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)
        if "tools" in kwargs:
            prompt = f"[TOOLS:{len(kwargs['tools'])}]\n" + prompt
        if kwargs.get("tokenize"):
            return [0] * len(prompt.split())
        return prompt


class _FakePipeline:
    """Stand-in for ``transformers.Pipeline`` that records calls and replays canned output."""

    def __init__(self, generated_text: str):
        self.tokenizer = _FakeTokenizer()
        self.task = "text-generation"
        self.generated_text = generated_text
        self.last_input: Any = None
        self.last_kwargs: dict[str, Any] = {}

    def __call__(self, inputs: Any, **kwargs: Any) -> list[dict]:
        self.last_input = inputs
        self.last_kwargs = kwargs
        return [{"generated_text": self.generated_text}]


def _make_serving(generated: str) -> tuple[OpenAIServingChat, _FakePipeline]:
    pipe = _FakePipeline(generated)
    serving = OpenAIServingChat(
        pipeline=pipe,  # type: ignore[arg-type]
        model_name="test-model",
        config=TransformersConfig(),
        capabilities=TransformersCapabilities(supports_image=False, supports_audio=False),
    )
    return serving, pipe


def _raw_request() -> RawRequestProxy:
    return RawRequestProxy(None, {})


@pytest.mark.asyncio
async def test_response_without_tools_carries_content_only():
    serving, _ = _make_serving("hello there")
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], stream=False)
    resp = await serving.create_chat_completion(req, _raw_request())

    assert isinstance(resp, ChatCompletionResponse)
    msg = resp.choices[0].message
    assert msg.content == "hello there"
    assert msg.tool_calls == []
    assert resp.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_tools_only_render_when_requested():
    serving, pipe = _make_serving("hello")
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], stream=False)
    await serving.create_chat_completion(req, _raw_request())

    # Without `tools` in the request, the pipeline receives the message list
    # directly — no pre-rendered prompt.
    assert isinstance(pipe.last_input, list)


@pytest.mark.asyncio
async def test_tools_in_request_pre_renders_prompt_and_parses_tool_call():
    raw = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    serving, pipe = _make_serving(raw)
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "weather in paris?"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="auto",
        stream=False,
    )
    resp = await serving.create_chat_completion(req, _raw_request())

    assert isinstance(resp, ChatCompletionResponse)
    # Pre-rendered prompt is a string carrying the tool marker injected by our fake template.
    assert isinstance(pipe.last_input, str)
    assert pipe.last_input.startswith("[TOOLS:1]")

    msg = resp.choices[0].message
    assert msg.content is None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].function.name == "get_weather"
    assert json.loads(msg.tool_calls[0].function.arguments) == {"city": "Paris"}
    assert resp.choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_tool_choice_none_skips_tool_rendering():
    serving, pipe = _make_serving("regular reply")
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "noop"}}],
        tool_choice="none",
        stream=False,
    )
    resp = await serving.create_chat_completion(req, _raw_request())

    # tool_choice="none" — pipeline should receive messages, not a rendered prompt.
    assert isinstance(pipe.last_input, list)
    assert isinstance(resp, ChatCompletionResponse)
    assert resp.choices[0].message.content == "regular reply"
    assert resp.choices[0].message.tool_calls == []
    assert resp.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_tool_call_with_trailing_text_preserves_content():
    raw = 'Calling now.\n<tool_call>{"name": "ping", "arguments": {}}</tool_call>'
    serving, _ = _make_serving(raw)
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "ping?"}],
        tools=[{"type": "function", "function": {"name": "ping"}}],
        stream=False,
    )
    resp = await serving.create_chat_completion(req, _raw_request())

    assert isinstance(resp, ChatCompletionResponse)
    msg = resp.choices[0].message
    assert msg.content == "Calling now."
    assert len(msg.tool_calls) == 1


@pytest.mark.asyncio
async def test_unknown_parser_at_init_raises():
    pipe = _FakePipeline("anything")
    with pytest.raises(ValueError, match="unknown tool_call_parser"):
        OpenAIServingChat(
            pipeline=pipe,  # type: ignore[arg-type]
            model_name="test-model",
            config=TransformersConfig(tool_call_parser="not-a-real-parser"),
            capabilities=TransformersCapabilities(supports_image=False, supports_audio=False),
        )
