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
from modelship.infer.llama_cpp.utils import LlamaCppToolCallRenderer
from modelship.logging import get_logger
from modelship.openai.chat_utils import UnsupportedContentError, normalize_chat_messages
from modelship.openai.parsers.tool_calling import (
    build_chat_completion_response,
    get_parser,
    resolve_tools_for_request,
    stream_chat_completion,
)
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
        renderer: LlamaCppToolCallRenderer | None = None,
    ):
        self._llama = llama
        self.model_name = model_name
        self._caps = capabilities
        self._lock = asyncio.Lock()
        self._accepted_params = set(inspect.signature(llama.create_chat_completion).parameters)
        self._completion_accepted_params = set(inspect.signature(llama.create_completion).parameters)
        self.tool_call_parser = tool_call_parser
        self._renderer = renderer
        if tool_call_parser is not None:
            assert renderer is not None, "renderer is required when tool_call_parser is set"
            # Validate at startup so misconfiguration surfaces before the first request.
            get_parser(tool_call_parser)

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

        if tools and self.tool_call_parser is not None:
            return await self._handle_with_tools(request, request_id, messages, tools)
        return await self._handle_without_tools(request, request_id, messages)

    async def _handle_without_tools(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        messages: list[dict],
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        """Pass-through to ``llama.create_chat_completion`` — no parser involved.

        This path is used when no tools are requested, or when the user
        configured ``chat_format`` (in which case our parser is intentionally
        bypassed and llama-cpp-python handles tool calling itself if at all).
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

    async def _handle_with_tools(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        messages: list[dict],
        tools: list[dict[str, Any]],
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        """Render prompt with tools, run raw completion, parse via shared streamer.

        Bypasses ``llama.create_chat_completion`` because that re-runs llama-cpp's
        own chat templating (which produces tokens conditioned on a different
        template than the one our parser expects).
        """
        assert self._renderer is not None  # guarded at __init__
        assert self.tool_call_parser is not None
        parser_name = self.tool_call_parser
        renderer = self._renderer

        try:
            prompt = renderer.render(messages, tools)
        except Exception as e:
            logger.warning("llama_cpp tool-call prompt rendering failed for %s: %s", request_id, e)
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
            return self._locked_stream_with_tools(
                request_id=request_id,
                completion_kwargs=completion_kwargs,
                parser_name=parser_name,
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
                logger.warning("llama_cpp tool-call inference failed for %s: %s", request_id, e)
                return create_error_response(e)

        completion_text = result["choices"][0]["text"] if result.get("choices") else ""
        usage = result.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or renderer.count_tokens(completion_text))

        return build_chat_completion_response(
            request_id=request_id,
            model_name=self.model_name,
            text=completion_text,
            parser_name=parser_name,
            prompt_tokens=int(usage.get("prompt_tokens") or prompt_tokens),
            completion_tokens=completion_tokens,
            max_tokens=max_tokens,
            created=int(time.time()),
        )

    async def _locked_stream_with_tools(
        self,
        *,
        request_id: str,
        completion_kwargs: dict,
        parser_name: str,
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
                parser_name=parser_name,
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

        return {k: v for k, v in params.items() if k in self._completion_accepted_params}
