"""TRACE-level logging of raw model output on the llama.cpp chat path.

The raw completion text (before tool-call/reasoning parsing) is what's needed
to debug whether ``<tool_call>`` tags are present, so it must be logged at
TRACE. Tests mock Ray Serve via ``__new__`` to bypass the deployment wrapper.
"""

import asyncio
import inspect

import pytest

from modelship.infer.llama_cpp.openai.serving_chat import OpenAIServingChat
from modelship.logging import TRACE
from modelship.openai.protocol import ChatCompletionRequest

LOGGER_NAME = "modelship.infer.llama_cpp.chat"


class _FakeRenderer:
    def render(self, messages, tools):
        return "PROMPT"

    def count_tokens(self, text):
        return len(text.split())


class _FakeLlama:
    def __init__(self, text):
        self._text = text

    def create_completion(self, **kwargs):
        return {
            "choices": [{"text": self._text}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }


def _serving(text: str) -> OpenAIServingChat:
    # Bypass __init__ / the @serve.deployment wrapper; wire up only what
    # _handle_with_parsers touches on the non-streaming path.
    from llama_cpp import Llama

    chat = OpenAIServingChat.__new__(OpenAIServingChat)
    chat.model_name = "x"
    chat._lock = asyncio.Lock()
    chat._renderer = _FakeRenderer()
    chat._llama = _FakeLlama(text)
    chat.reasoning_parser = None
    chat._completion_accepted_params = set(inspect.signature(Llama.create_completion).parameters)
    return chat


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="x", messages=[{"role": "user", "content": "hi"}])


class TestTraceResponseLogging:
    @pytest.mark.asyncio
    async def test_non_streaming_logs_raw_response_at_trace(self, caplog):
        text = '<tool_call>{"name": "get_weather"}</tool_call>'
        chat = _serving(text)
        with caplog.at_level(TRACE, logger=LOGGER_NAME):
            await chat._handle_with_parsers(
                _request(),
                "chat-abc",
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                tool_parser_name=None,
            )

        records = [r for r in caplog.records if r.name == LOGGER_NAME and "chat response" in r.message]
        assert records, "expected a TRACE 'chat response' record"
        rendered = records[0].getMessage()
        assert "chat-abc" in rendered
        assert text in rendered
