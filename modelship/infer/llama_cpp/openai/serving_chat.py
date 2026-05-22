import asyncio
import inspect
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Any, cast

from llama_cpp import Llama

from modelship.infer.base_serving import OpenAIServing
from modelship.infer.infer_config import RawRequestProxy
from modelship.infer.llama_cpp.capabilities import LlamaCppCapabilities
from modelship.infer.llama_cpp.structured import build_llama_grammar
from modelship.infer.llama_cpp.utils import LlamaCppToolCallRenderer
from modelship.logging import get_logger
from modelship.openai.chat_utils import UnsupportedContentError, normalize_chat_messages
from modelship.openai.parsers.reasoning import get_parser as get_reasoning_parser
from modelship.openai.parsers.streaming import build_chat_completion_response, stream_chat_completion
from modelship.openai.parsers.tool_calling import get_parser, resolve_tools_for_request
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    create_error_response,
)
from modelship.utils import base_request_id

logger = get_logger("infer.llama_cpp.chat")


class OpenAIServingChat(OpenAIServing):
    request_id_prefix = "chat"

    def __init__(
        self,
        llama: Llama,
        model_name: str,
        capabilities: LlamaCppCapabilities,
        tool_call_parser: str | None = None,
        reasoning_parser: str | None = None,
        renderer: LlamaCppToolCallRenderer | None = None,
    ):
        self._llama = llama
        self.model_name = model_name
        self._caps = capabilities
        self._lock = asyncio.Lock()
        self._accepted_params = set(inspect.signature(llama.create_chat_completion).parameters)
        self._completion_accepted_params = set(inspect.signature(llama.create_completion).parameters)
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self._renderer = renderer
        # The renderer's presence is the sole switch between paths:
        #  - renderer set → drive `create_completion` raw, route every
        #    response through `ChatOutputStreamer`. Reasoning + tool-call
        #    extraction happens here. Default for any model that exposes
        #    a chat template via GGUF metadata.
        #  - renderer None → fall back to llama-cpp's
        #    `create_chat_completion`. Used when the user opted into
        #    llama-cpp's own templating with `chat_format` on
        #    LlamaCppConfig — they're presumed to know our parsers won't
        #    fire for that config.
        if tool_call_parser is not None:
            assert renderer is not None, "renderer is required when tool_call_parser is set"
            get_parser(tool_call_parser)
        if reasoning_parser is not None:
            assert renderer is not None, "renderer is required when reasoning_parser is set"
            get_reasoning_parser(reasoning_parser)

    async def warmup(self) -> None:
        logger.info("Warming up llama.cpp chat model: %s", self.model_name)
        request = ChatCompletionRequest(
            model=self.model_name,
            messages=[{"role": "user", "content": "warmup"}],
            max_tokens=1,
        )
        await self.create_chat_completion(request, RawRequestProxy(None, {}))
        logger.info("Warmup chat done for %s", self.model_name)

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info("chat completion request %s: stream=%s", request_id, request.stream)

        try:
            messages = normalize_chat_messages(
                request.messages,
                supports_image=self._caps.supports_image,
                supports_audio=self._caps.supports_audio,
            )
        except UnsupportedContentError as e:
            logger.warning("chat request %s rejected: %s", request_id, e)
            return create_error_response(e)

        tools = resolve_tools_for_request(request.tools, request.tool_choice)
        if tools and self.tool_call_parser is None:
            logger.warning(
                "chat request %s asks for %d tool(s) but model %r has no usable tool-call parser; ignoring tools",
                request_id,
                len(tools),
                self.model_name,
            )
            tools = None

        # Reasoning-enabled deployments emit `<think>...</think>` before content.
        # A JSON grammar from response_format would exclude the `<` token and
        # break reasoning emission. Reject upfront so the user picks a
        # non-reasoning deployment for schema-constrained traffic.
        if self.reasoning_parser and request.response_format:
            fmt_type = request.response_format.get("type")
            if fmt_type not in (None, "text"):
                msg = (
                    f"response_format with type={fmt_type!r} cannot be combined with a "
                    f"reasoning-enabled deployment ({self.model_name!r}). The schema grammar "
                    "prevents the reasoning parser's required tokens from being emitted. "
                    "Use a non-reasoning deployment, or drop response_format."
                )
                logger.warning("chat request %s rejected: %s", request_id, msg)
                return create_error_response(msg)

        tool_parser_name = self.tool_call_parser if tools else None
        # Renderer presence decides the path (see __init__ comment): when
        # the user set `chat_format` on LlamaCppConfig we leave it to
        # llama-cpp; otherwise we render the prompt ourselves so the
        # ChatOutputStreamer sees the model's raw bytes regardless of
        # whether reasoning or tools are active for this request.
        if self._renderer is not None:
            return await self._handle_with_parsers(
                request, request_id, messages, tools, tool_parser_name=tool_parser_name
            )
        return await self._handle_native(request, request_id, messages)

    async def _handle_native(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        messages: list[dict],
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        """Pass-through to ``llama.create_chat_completion`` — no parser involved.

        Used when neither tool calling nor reasoning is active for this
        deployment, or when the user configured ``chat_format`` (in
        which case our parsers are intentionally bypassed and
        llama-cpp-python's own chat handler is responsible).
        """
        kwargs = self._build_kwargs(request, messages)
        loop = asyncio.get_event_loop()
        llama = self._llama

        if request.stream:

            async def stream_generator() -> AsyncGenerator[str, None]:
                async with self._lock:
                    raw = await loop.run_in_executor(
                        None,
                        lambda: llama.create_chat_completion(**kwargs, stream=True),  # type: ignore[arg-type]
                    )
                    iterator = cast(Iterator[dict], raw)
                    while True:
                        chunk = await loop.run_in_executor(None, lambda: next(iterator, None))
                        if chunk is None:
                            break
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(0)
                    yield "data: [DONE]\n\n"

            return stream_generator()

        async with self._lock:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: llama.create_chat_completion(**kwargs, stream=False),  # type: ignore[arg-type]
                )
                return ChatCompletionResponse.model_validate(result)
            except Exception as e:
                logger.warning("llama_cpp chat inference failed: %s", e)
                return create_error_response(e)

    async def _handle_with_parsers(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        messages: list[dict],
        tools: list[dict[str, Any]] | None,
        *,
        tool_parser_name: str | None,
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        """Render prompt ourselves, run raw completion, parse via shared streamer.

        Bypasses ``llama.create_chat_completion`` because that re-runs
        llama-cpp's own chat templating (which also strips tokens our
        parsers need to see). The unified streamer handles reasoning
        and tool calls in a single pass over the model output.
        """
        assert self._renderer is not None  # guarded at __init__
        renderer = self._renderer
        reasoning_parser_name = self.reasoning_parser

        try:
            prompt = renderer.render(messages, tools)
        except Exception as e:
            logger.warning("llama_cpp prompt rendering failed for %s: %s", request_id, e)
            return create_error_response(e)

        max_tokens = request.max_tokens
        if max_tokens is None and request.max_completion_tokens is not None:
            max_tokens = request.max_completion_tokens

        completion_kwargs = self._build_completion_kwargs(request, prompt)
        prompt_tokens = renderer.count_tokens(prompt)
        loop = asyncio.get_event_loop()
        llama = self._llama

        if request.stream:
            include_usage = bool(request.stream_options and request.stream_options.include_usage)
            return self._locked_stream_with_parsers(
                request_id=request_id,
                completion_kwargs=completion_kwargs,
                tool_parser_name=tool_parser_name,
                reasoning_parser_name=reasoning_parser_name,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                include_usage=include_usage,
            )

        async with self._lock:
            try:
                raw = await loop.run_in_executor(
                    None,
                    lambda: llama.create_completion(**completion_kwargs, stream=False),  # type: ignore[arg-type]
                )
                result = cast(dict, raw)
            except Exception as e:
                logger.warning("llama_cpp inference failed for %s: %s", request_id, e)
                return create_error_response(e)

        completion_text = result["choices"][0]["text"] if result.get("choices") else ""
        usage = result.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or renderer.count_tokens(completion_text))

        return build_chat_completion_response(
            request_id=request_id,
            model_name=self.model_name,
            text=completion_text,
            parser_name=tool_parser_name,
            reasoning_parser_name=reasoning_parser_name,
            prompt_tokens=int(usage.get("prompt_tokens") or prompt_tokens),
            completion_tokens=completion_tokens,
            max_tokens=max_tokens,
            created=int(time.time()),
        )

    async def _locked_stream_with_parsers(
        self,
        *,
        request_id: str,
        completion_kwargs: dict,
        tool_parser_name: str | None,
        reasoning_parser_name: str | None,
        prompt_tokens: int,
        max_tokens: int | None,
        include_usage: bool,
    ) -> AsyncGenerator[str, None]:
        assert self._renderer is not None
        renderer = self._renderer
        async with self._lock:
            async for chunk in stream_chat_completion(
                request_id=request_id,
                model_name=self.model_name,
                text_chunks=self._raw_text_chunks(completion_kwargs),
                parser_name=tool_parser_name,
                reasoning_parser_name=reasoning_parser_name,
                count_tokens=renderer.count_tokens,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                include_usage=include_usage,
                created=int(time.time()),
            ):
                yield chunk

    async def _raw_text_chunks(self, completion_kwargs: dict) -> AsyncIterator[str]:
        """Drive ``llama.create_completion(stream=True)`` and yield text pieces."""
        loop = asyncio.get_event_loop()
        llama = self._llama
        raw = await loop.run_in_executor(
            None,
            lambda: llama.create_completion(**completion_kwargs, stream=True),  # type: ignore[arg-type]
        )
        iterator = cast(Iterator[dict], raw)
        while True:
            chunk = await loop.run_in_executor(None, lambda: next(iterator, None))
            if chunk is None:
                return
            text = (chunk.get("choices") or [{}])[0].get("text") or ""
            if text:
                yield text

    def _build_kwargs(self, request: ChatCompletionRequest, messages: list[dict]) -> dict:
        params = request.model_dump(exclude_none=True)
        params["messages"] = messages
        if "max_tokens" not in params and "max_completion_tokens" in params:
            params["max_tokens"] = params["max_completion_tokens"]

        # Convert OpenAI-shaped response_format → LlamaGrammar. llama-cpp-python's
        # own handler only recognizes {"type": "json_object", "schema": ...} and
        # silently drops {"type": "json_schema", ...}, so we own the conversion.
        grammar = build_llama_grammar(params.pop("response_format", None))

        kwargs: dict = {}
        dropped: list[str] = []
        for k, v in params.items():
            if k == "stream":
                continue
            if k in self._accepted_params:
                kwargs[k] = v
            else:
                dropped.append(k)
        if dropped:
            logger.warning(
                "llama_cpp: dropping request params not supported by create_chat_completion: %s",
                dropped,
            )
        if grammar is not None:
            kwargs["grammar"] = grammar
        return kwargs

    def _build_completion_kwargs(self, request: ChatCompletionRequest, prompt: str) -> dict:
        params = request.model_dump(exclude_none=True)
        params["prompt"] = prompt
        if "max_tokens" not in params and "max_completion_tokens" in params:
            params["max_tokens"] = params["max_completion_tokens"]
        # These are consumed by our renderer/parser, not by `create_completion`.
        for k in ("messages", "tools", "tool_choice", "stream", "stream_options"):
            params.pop(k, None)
        for k in ("logprobs", "top_logprobs"):
            params.pop(k, None)

        grammar = build_llama_grammar(params.pop("response_format", None))

        kwargs = {k: v for k, v in params.items() if k in self._completion_accepted_params}
        if grammar is not None:
            kwargs["grammar"] = grammar
        return kwargs
