"""Tests for the Transformers chat path's reasoning + tool-call composition.

These tests bypass the real HF pipeline by injecting a callable that returns
a canned generation, so they run offline and do not touch any model weights.
The shared parser/streamer logic is exercised in ``test_reasoning.py`` and
``test_tool_calling.py``; this file focuses on the wiring between
``transformers/openai/serving_chat.py`` and the unified streamer.
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
    def __init__(self, generated_text: str):
        self.tokenizer = _FakeTokenizer()
        self.task = "text-generation"
        self.generated_text = generated_text

    def __call__(self, inputs: Any, **kwargs: Any) -> list[dict]:
        return [{"generated_text": self.generated_text}]


def _make_serving(
    generated: str,
    *,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = "deepseek_r1",
) -> OpenAIServingChat:
    pipe = _FakePipeline(generated)
    return OpenAIServingChat(
        pipeline=pipe,  # type: ignore[arg-type]
        model_name="test-model",
        config=TransformersConfig(),
        capabilities=TransformersCapabilities(supports_image=False, supports_audio=False),
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
    )


def _raw_request() -> RawRequestProxy:
    return RawRequestProxy(None, {})


@pytest.mark.asyncio
async def test_reasoning_only_response_splits_thinking_and_content():
    raw = "<think>let me think</think>The answer is 42."
    serving = _make_serving(raw)
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], stream=False)
    resp = await serving.create_chat_completion(req, _raw_request())

    assert isinstance(resp, ChatCompletionResponse)
    msg = resp.choices[0].message
    assert msg.reasoning == "let me think"
    assert msg.content == "The answer is 42."
    assert msg.tool_calls == []
    assert resp.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_reasoning_with_no_markers_falls_back_to_content():
    # Reasoning parser is configured, but the model emitted no <think>
    # block — content should pass through intact and reasoning is None.
    serving = _make_serving("just a regular reply")
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], stream=False)
    resp = await serving.create_chat_completion(req, _raw_request())

    assert isinstance(resp, ChatCompletionResponse)
    msg = resp.choices[0].message
    assert msg.reasoning is None
    assert msg.content == "just a regular reply"


@pytest.mark.asyncio
async def test_tool_call_marker_inside_thinking_is_not_parsed_as_tool_call():
    # The single-pass streamer routes the tool marker inside <think> to the
    # reasoning view; it must NOT show up as a finalized tool call.
    raw = (
        '<think>I would call <tool_call>{"name": "x", "arguments": {}}</tool_call> here</think>'
        "Sure, I'll call it for real now."
    )
    serving = _make_serving(raw, tool_call_parser="hermes", reasoning_parser="deepseek_r1")
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "ping"}],
        tools=[{"type": "function", "function": {"name": "x"}}],
        stream=False,
    )
    resp = await serving.create_chat_completion(req, _raw_request())

    assert isinstance(resp, ChatCompletionResponse)
    msg = resp.choices[0].message
    assert msg.tool_calls == []
    assert msg.reasoning is not None
    assert "<tool_call>" in msg.reasoning  # the marker text lives inside reasoning
    assert msg.content == "Sure, I'll call it for real now."
    assert resp.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_thinking_then_real_tool_call_after_close_marker():
    # Reasoning closes before a real tool call is emitted; both should land.
    raw = (
        "<think>I should call get_weather for Paris.</think>"
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    )
    serving = _make_serving(raw, tool_call_parser="hermes", reasoning_parser="deepseek_r1")
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "weather paris?"}],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
        stream=False,
    )
    resp = await serving.create_chat_completion(req, _raw_request())

    assert isinstance(resp, ChatCompletionResponse)
    msg = resp.choices[0].message
    assert msg.reasoning == "I should call get_weather for Paris."
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].function.name == "get_weather"
    assert json.loads(msg.tool_calls[0].function.arguments) == {"city": "Paris"}
    assert msg.content is None
    assert resp.choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_unknown_reasoning_parser_at_init_raises():
    pipe = _FakePipeline("anything")
    with pytest.raises(ValueError, match="reasoning"):
        OpenAIServingChat(
            pipeline=pipe,  # type: ignore[arg-type]
            model_name="test-model",
            config=TransformersConfig(),
            capabilities=TransformersCapabilities(supports_image=False, supports_audio=False),
            reasoning_parser="not-a-real-parser",
        )


@pytest.mark.asyncio
async def test_no_parsers_configured_passes_text_through_unchanged():
    # Both parsers None: the chat path must not extract anything, even if the
    # model happens to emit marker-shaped tokens.
    raw = "<think>shouldn't be split</think>plain"
    pipe = _FakePipeline(raw)
    serving = OpenAIServingChat(
        pipeline=pipe,  # type: ignore[arg-type]
        model_name="test-model",
        config=TransformersConfig(),
        capabilities=TransformersCapabilities(supports_image=False, supports_audio=False),
    )
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], stream=False)
    resp = await serving.create_chat_completion(req, _raw_request())

    assert isinstance(resp, ChatCompletionResponse)
    msg = resp.choices[0].message
    assert msg.reasoning is None
    assert msg.content == raw
    assert msg.tool_calls == []
