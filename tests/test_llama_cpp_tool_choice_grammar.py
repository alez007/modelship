"""tool_choice drives the llama.cpp tool-call grammar.

Format-constraining is automatic when a request carries tools and the model has a
usable parser. The grammar *root* is chosen per request from ``tool_choice``:

- ``required`` / named-function force a tool-only root, which has no free-text branch
  and so cannot emit ``<think>`` — it is applied even on a reasoning deployment.
- ``auto`` (default) keeps a free-text branch whose ``content ::= [^<]+`` rule
  excludes ``<`` (which opens ``<think>``), so it yields to a reasoning deployment.
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
    chat._logged_reasoning_unconstrained = False
    chat._completion_accepted_params = set(inspect.signature(Llama.create_completion).parameters)
    return chat, llama


def _request(tool_choice=None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="x", messages=[{"role": "user", "content": "hi"}], tools=TOOLS, tool_choice=tool_choice
    )


async def _run(reasoning_parser: str | None, tool_choice=None) -> dict:
    chat, llama = _serving(reasoning_parser=reasoning_parser)
    await chat._handle_with_parsers(
        _request(tool_choice),
        "chat-1",
        messages=[{"role": "user", "content": "hi"}],
        tools=TOOLS,
        tool_parser_name="hermes",
    )
    assert llama.last_kwargs is not None
    return llama.last_kwargs


@pytest.mark.asyncio
async def test_auto_constrains_without_reasoning_parser():
    kwargs = await _run(reasoning_parser=None, tool_choice="auto")
    assert kwargs.get("grammar") is not None, "auto should auto-constrain a non-reasoning deployment"


@pytest.mark.asyncio
async def test_auto_yields_to_reasoning_parser():
    kwargs = await _run(reasoning_parser="deepseek_r1", tool_choice="auto")
    assert "grammar" not in kwargs, "auto must not constrain a reasoning deployment"


@pytest.mark.asyncio
async def test_required_forces_grammar_even_with_reasoning_parser():
    kwargs = await _run(reasoning_parser="deepseek_r1", tool_choice="required")
    assert kwargs.get("grammar") is not None, "required must force the grammar regardless of reasoning"


@pytest.mark.asyncio
async def test_named_function_forces_grammar_even_with_reasoning_parser():
    choice = {"type": "function", "function": {"name": "HassTurnOn"}}
    kwargs = await _run(reasoning_parser="deepseek_r1", tool_choice=choice)
    assert kwargs.get("grammar") is not None, "named-function must force the grammar regardless of reasoning"
