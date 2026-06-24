"""constrain_tool_calls must yield to the reasoning block on the llama.cpp chat path.

The tool-call grammar's ``content`` rule excludes the start marker's first char
(``<``), which also opens ``<think>``. Applying it on a reasoning-enabled
deployment makes ``<think>`` unreachable, so the model emits a junk token in its
place every turn. The grammar must therefore be skipped when a reasoning parser
is active — tool calls are still extracted from the raw output by the parser.
"""

import asyncio
import inspect

import pytest

from modelship.infer.llama_cpp.openai.serving_chat import OpenAIServingChat
from modelship.openai.protocol import ChatCompletionRequest

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "HassTurnOn",
            "parameters": {"type": "object", "properties": {"area": {"type": "string"}}},
        },
    }
]


class _FakeRenderer:
    def render(self, messages, tools):
        return "PROMPT"

    def count_tokens(self, text):
        return len(text.split())


class _CapturingLlama:
    """Records the kwargs of the last non-streaming create_completion call."""

    def __init__(self):
        self.last_kwargs: dict | None = None

    def create_completion(self, *, stream=False, **kwargs):
        self.last_kwargs = kwargs
        return {"choices": [{"text": "hi"}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


def _serving(*, reasoning_parser: str | None) -> tuple[OpenAIServingChat, _CapturingLlama]:
    from llama_cpp import Llama

    chat = OpenAIServingChat.__new__(OpenAIServingChat)
    chat.model_name = "x"
    chat._lock = asyncio.Lock()
    chat._renderer = _FakeRenderer()
    llama = _CapturingLlama()
    chat._llama = llama
    chat.tool_call_parser = "hermes"
    chat.reasoning_parser = reasoning_parser
    chat._constrain_tool_calls = True
    chat._logged_reasoning_unconstrained = False
    chat._completion_accepted_params = set(inspect.signature(Llama.create_completion).parameters)
    return chat, llama


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="x", messages=[{"role": "user", "content": "hi"}], tools=TOOLS)


async def _run(reasoning_parser: str | None) -> dict:
    chat, llama = _serving(reasoning_parser=reasoning_parser)
    await chat._handle_with_parsers(
        _request(),
        "chat-1",
        messages=[{"role": "user", "content": "hi"}],
        tools=TOOLS,
        tool_parser_name="hermes",
    )
    assert llama.last_kwargs is not None
    return llama.last_kwargs


@pytest.mark.asyncio
async def test_grammar_skipped_when_reasoning_parser_active():
    kwargs = await _run(reasoning_parser="deepseek_r1")
    assert "grammar" not in kwargs, "tool-call grammar must not be applied on a reasoning deployment"


@pytest.mark.asyncio
async def test_grammar_applied_without_reasoning_parser():
    kwargs = await _run(reasoning_parser=None)
    assert kwargs.get("grammar") is not None, "tool-call grammar should constrain a non-reasoning deployment"
