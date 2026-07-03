"""Quarantine for every vLLM-internal touchpoint the vllm loader needs.

Modelship types go in; a parsed 3-tuple (via vllm.parser) or a raw engine
stream comes out. Nothing outside this module should import from
`vllm.entrypoints.*`/`vllm.parser`/`vllm.v1.engine.*` directly — that keeps a
vLLM version bump's blast radius confined to one file.
"""

from collections.abc import AsyncGenerator, Mapping
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as VllmChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse as VllmErrorResponse
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.inputs import EngineInput
from vllm.outputs import RequestOutput
from vllm.parser import Parser
from vllm.renderers.inputs.preprocess import extract_prompt_components, extract_prompt_len
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine.async_llm import AsyncLLM

from modelship.openai.protocol import ChatCompletionRequest


def build_vllm_request(
    request: ChatCompletionRequest,
    chat_template_kwargs: dict[str, Any] | None,
) -> VllmChatCompletionRequest:
    """Shape a modelship chat request into vLLM's own request model.

    Merges the model's default `chat_template_kwargs` under any per-request
    value (request wins) — vLLM renders the chat template internally, so
    unlike llama.cpp-family loaders this can't be patched in after the fact.
    """
    request_data = request.model_dump()
    if chat_template_kwargs:
        request_data["chat_template_kwargs"] = {
            **chat_template_kwargs,
            **(request_data.get("chat_template_kwargs") or {}),
        }
    return VllmChatCompletionRequest(**request_data)


async def render_and_params(
    render: OpenAIServingRender,
    vllm_req: VllmChatCompletionRequest,
) -> tuple[EngineInput, SamplingParams] | VllmErrorResponse:
    """Render the chat template and derive `SamplingParams`, in that order.

    `render_chat` mutates `vllm_req` in place as a side effect of rendering
    (`ToolParser.adjust_request` sets `structured_outputs` /
    `_grammar_from_tool_parser`), and `to_sampling_params` reads that
    mutation. The order is load-bearing — this function exists specifically
    so callers can't split the two apart or run them against a rebuilt copy
    of the request, which would silently drop the mutation.
    """
    result = await render.render_chat(vllm_req)
    if isinstance(result, VllmErrorResponse):
        return result
    _conversation, engine_inputs = result
    if len(engine_inputs) != 1:
        raise RuntimeError(f"expected exactly 1 rendered engine prompt for a chat request, got {len(engine_inputs)}")
    engine_input = engine_inputs[0]

    max_tokens = get_max_tokens(
        render.model_config.max_model_len,
        vllm_req.max_completion_tokens if vllm_req.max_completion_tokens is not None else vllm_req.max_tokens,
        extract_prompt_len(render.model_config, engine_input),
        render.default_sampling_params,
        render.override_max_tokens,
        truncate_prompt_tokens=vllm_req.truncate_prompt_tokens,
    )
    sampling_params = vllm_req.to_sampling_params(max_tokens, render.default_sampling_params)
    return engine_input, sampling_params


def extract_prompt_token_ids(render: OpenAIServingRender, engine_input: EngineInput) -> list[int]:
    """Extract the rendered prompt's token IDs, needed for `derive_reasoning_ended`."""
    return list(extract_prompt_components(render.model_config, engine_input).token_ids or [])


def make_parsers(
    render: OpenAIServingRender,
    tokenizer: TokenizerLike,
    vllm_req: VllmChatCompletionRequest,
    chat_template_kwargs: dict[str, Any] | None,
    n: int,
) -> list[Parser | None]:
    """Instantiate one parser per choice.

    Parsers carry per-choice streaming state (`Parser._stream_state`), so a
    request with `n > 1` needs `n` independent instances — sharing one across
    choices corrupts state on every choice after the first. `render.parser`
    is the same class `render_chat` already resolved internally via
    `ParserManager.get_parser`, so this can't drift out of sync with it.
    """
    if render.parser is None:
        return [None] * n
    parser_cls = render.parser
    return [
        parser_cls(tokenizer, vllm_req.tools, chat_template_kwargs=chat_template_kwargs)  # type: ignore[arg-type]
        for _ in range(n)
    ]


def derive_reasoning_ended(
    vllm_req: VllmChatCompletionRequest,
    parser: Parser | None,
    prompt_token_ids: list[int],
) -> bool | None:
    """Replicates the reasoning_ended precedence in vLLM's own chat completion serving.

    Mistral's grammar (when built) already encodes an optional `think?` rule
    covering both reasoning and non-reasoning outputs, so `reasoning_ended`
    is forced True whenever `_grammar_from_tool_parser` is set. But that flag
    is only set on the grammar-building branch of
    `MistralToolParser.adjust_request` — a request with tools but no
    structured-outputs constraint active takes an early-return branch that
    leaves it False, so this must not assume the flag is reliably True
    whenever a mistral tool parser is in play.
    """
    if not vllm_req.include_reasoning:
        return True
    if vllm_req._grammar_from_tool_parser:
        return True
    if parser is not None and parser.reasoning_parser is not None:
        return parser.is_reasoning_end(prompt_token_ids)
    return None


def generate(
    engine: AsyncLLM,
    engine_input: EngineInput,
    sampling_params: SamplingParams,
    request_id: str,
    *,
    reasoning_ended: bool | None,
    parser: Parser | None,
    chat_template_kwargs: dict[str, Any] | None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    data_parallel_rank: int | None = None,
) -> AsyncGenerator[RequestOutput, None]:
    """Thin wrapper over `AsyncLLM.generate` — the only place this loader touches the engine directly."""
    reasoning_parser_kwargs = None
    if parser is not None and parser.reasoning_parser is not None:
        reasoning_parser_kwargs = {"chat_template_kwargs": chat_template_kwargs}
    return engine.generate(
        engine_input,
        sampling_params,
        request_id,
        trace_headers=trace_headers,
        priority=priority,
        data_parallel_rank=data_parallel_rank,
        reasoning_ended=reasoning_ended,
        reasoning_parser_kwargs=reasoning_parser_kwargs,
    )
