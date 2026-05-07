import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from threading import Thread
from typing import Any

from transformers import Pipeline, PreTrainedTokenizerBase, TextIteratorStreamer

from modelship.infer.base_serving import OpenAIServing
from modelship.infer.infer_config import RawRequestProxy, TransformersConfig
from modelship.infer.transformers.capabilities import TransformersCapabilities
from modelship.logging import TRACE, get_logger
from modelship.openai.chat_utils import UnsupportedContentError, normalize_chat_messages
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    create_error_response,
)
from modelship.openai.tool_calling import (
    build_chat_completion_response,
    get_parser,
    resolve_tools_for_request,
    stream_chat_completion,
)
from modelship.utils import base_request_id

logger = get_logger("infer.transformers.chat")


class OpenAIServingChat(OpenAIServing):
    request_id_prefix = "chatcmpl"

    def __init__(
        self,
        pipeline: Pipeline,
        model_name: str,
        config: TransformersConfig,
        capabilities: TransformersCapabilities,
        tool_call_parser: str | None,
    ):
        self.pipeline = pipeline
        self.model_name = model_name
        self.config = config
        self.capabilities = capabilities
        self.tool_call_parser = tool_call_parser
        assert pipeline.tokenizer is not None, "text-generation pipeline must have a tokenizer"
        self.tokenizer: PreTrainedTokenizerBase = pipeline.tokenizer
        self._lock = asyncio.Lock()
        # Validate the configured parser at startup so misconfiguration surfaces
        # before the first request rather than mid-generation. None means
        # auto-detection found no usable parser; tool calls in requests will be
        # dropped with a warning at request time.
        if tool_call_parser is not None:
            get_parser(tool_call_parser)

    async def warmup(self) -> None:
        logger.info("Warming up chat model: %s", self.model_name)
        await self.run_in_executor(
            self._run,
            [{"role": "user", "content": "warmup"}],
            None,
            1,
        )
        logger.info("Warmup chat done for %s", self.model_name)

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: RawRequestProxy
    ) -> ChatCompletionResponse | AsyncGenerator[str, None] | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info("chat completion request %s: stream=%s", request_id, request.stream)
        logger.log(
            TRACE, "chat request %s: messages=%s, max_tokens=%s", request_id, request.messages, request.max_tokens
        )

        try:
            messages = normalize_chat_messages(
                request.messages,
                supports_image=self.capabilities.supports_image,
                supports_audio=self.capabilities.supports_audio,
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

        max_tokens = request.max_tokens
        if max_tokens is None and request.max_completion_tokens is not None:
            max_tokens = request.max_completion_tokens

        parser_name = self.tool_call_parser if tools else None

        if request.stream:
            include_usage = bool(request.stream_options and request.stream_options.include_usage)
            return self._locked_stream(
                request_id, messages, tools, max_tokens, parser_name=parser_name, include_usage=include_usage
            )

        async with self._lock:
            try:
                result = await self.run_in_executor(self._run, messages, tools, max_tokens)
            except Exception:
                logger.exception("chat completion inference failed for %s", request_id)
                return create_error_response("chat completion inference failed")

        prompt_tokens = self._count_prompt_tokens(messages, tools)
        completion_text = self._extract_completion_text(result)
        completion_tokens = len(self.tokenizer.encode(completion_text, add_special_tokens=False))

        response = build_chat_completion_response(
            request_id=request_id,
            model_name=self.model_name,
            text=completion_text,
            parser_name=parser_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            max_tokens=max_tokens,
            created=int(time.time()),
        )
        logger.log(
            TRACE,
            "chat response %s: text=%r, tool_calls=%d, prompt_tokens=%d, completion_tokens=%d",
            request_id,
            completion_text,
            len(response.choices[0].message.tool_calls),
            prompt_tokens,
            completion_tokens,
        )
        return response

    @staticmethod
    def _extract_completion_text(result: list) -> str:
        generated = result[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1]["content"]
        return generated

    def _count_prompt_tokens(self, messages: list[dict], tools: list[dict[str, Any]] | None) -> int:
        # apply_chat_template returns a string by default (character count!) — force tokenize=True.
        kwargs: dict[str, Any] = {"tokenize": True}
        if tools:
            kwargs["tools"] = tools
        token_ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        return len(token_ids)

    def _render_prompt(self, messages: list[dict], tools: list[dict[str, Any]]) -> str:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,  # type: ignore[arg-type]
            tokenize=False,
            add_generation_prompt=True,
        )
        assert isinstance(rendered, str), "apply_chat_template(tokenize=False) must return str"
        return rendered

    def _run(self, messages: list[dict], tools: list[dict[str, Any]] | None, max_tokens: int | None) -> list:
        kwargs = {**self.config.pipeline_kwargs}
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens
        if tools:
            # The standard text-generation pipeline does not forward `tools` to
            # `apply_chat_template`, so we render the prompt ourselves and feed
            # it as a plain string.
            prompt = self._render_prompt(messages, tools)
            return self.pipeline(prompt, return_full_text=False, **kwargs)  # type: ignore[return-value]
        return self.pipeline(messages, return_full_text=False, **kwargs)  # type: ignore[return-value]

    async def _locked_stream(
        self,
        request_id: str,
        messages: list[dict],
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        *,
        parser_name: str | None,
        include_usage: bool,
    ) -> AsyncGenerator[str, None]:
        async with self._lock:
            async for chunk in self._stream(
                request_id, messages, tools, max_tokens, parser_name=parser_name, include_usage=include_usage
            ):
                yield chunk

    async def _stream(
        self,
        request_id: str,
        messages: list[dict],
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        *,
        parser_name: str | None,
        include_usage: bool,
    ) -> AsyncGenerator[str, None]:
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore[arg-type]

        kwargs = {**self.config.pipeline_kwargs}
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens

        if tools:
            prompt: Any = self._render_prompt(messages, tools)
        else:
            prompt = messages

        thread = Thread(
            target=self.pipeline,
            args=(prompt,),
            kwargs={"streamer": streamer, "return_full_text": False, **kwargs},
        )
        thread.start()

        try:
            async for chunk in stream_chat_completion(
                request_id=request_id,
                model_name=self.model_name,
                text_chunks=_async_iter(streamer),
                parser_name=parser_name,
                count_tokens=lambda text: len(self.tokenizer.encode(text, add_special_tokens=False)),
                prompt_tokens=self._count_prompt_tokens(messages, tools),
                max_tokens=max_tokens,
                include_usage=include_usage,
                created=int(time.time()),
            ):
                yield chunk
        finally:
            thread.join()


async def _async_iter(streamer: TextIteratorStreamer) -> AsyncIterator[str]:
    """Adapt a synchronous ``TextIteratorStreamer`` to an async iterator.

    HF's streamer is driven by a background thread and exposes a blocking
    ``__next__``. We hop to a thread per pull so the event loop keeps spinning
    (cancellation, request-watcher polls) while tokens are produced.
    """
    sentinel = object()
    while True:
        item = await asyncio.to_thread(next, streamer, sentinel)
        if item is sentinel:
            return
        yield item  # type: ignore[misc]
