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
from modelship.openai.parsers.reasoning import get_parser as get_reasoning_parser
from modelship.openai.parsers.streaming import build_chat_completion_response, stream_chat_completion
from modelship.openai.parsers.tool_calling import get_parser, request_forces_tool_call, resolve_tools_for_request
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    create_error_response,
)
from modelship.utils import base_request_id, drop_reserved_kwargs

logger = get_logger("infer.transformers.chat")


class OpenAIServingChat(OpenAIServing):
    request_id_prefix = "chatcmpl"

    def __init__(
        self,
        pipeline: Pipeline,
        model_name: str,
        config: TransformersConfig,
        capabilities: TransformersCapabilities,
        tool_call_parser: str | None = None,
        reasoning_parser: str | None = None,
        skip_special_tokens: bool | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self.pipeline = pipeline
        self.model_name = model_name
        self.config = config
        self.capabilities = capabilities
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        # Drop keys we pass to apply_chat_template ourselves — a collision would
        # crash _render_prompt (duplicate kwarg) or flip tokenize in the counter.
        # `conversation` is apply_chat_template's first positional arg (passed
        # positionally here); `messages` is the same data inside the template
        # context. The rest are keyword args we set ourselves.
        self.chat_template_kwargs = drop_reserved_kwargs(
            chat_template_kwargs or {},
            {"conversation", "messages", "tokenize", "tools", "add_generation_prompt"},
            logger=logger,
            context=f"model '{model_name}'",
        )
        assert pipeline.tokenizer is not None, "text-generation pipeline must have a tokenizer"
        self.tokenizer: PreTrainedTokenizerBase = pipeline.tokenizer
        self._lock = asyncio.Lock()
        self._logged_force_unenforceable = False
        # Validate configured parsers at startup so misconfiguration surfaces
        # before the first request rather than mid-generation. None means
        # auto-detection found no usable parser; tool calls in requests will be
        # dropped with a warning at request time, and reasoning will simply
        # not be extracted.
        tool_parser = get_parser(tool_call_parser) if tool_call_parser is not None else None
        reasoning = get_reasoning_parser(reasoning_parser) if reasoning_parser is not None else None
        # Resolve the streamer's ``skip_special_tokens`` setting. Default
        # ``True`` matches HF's example code and is correct for parsers
        # whose markers are regular text (Hermes ``<tool_call>``,
        # llama3_json ``{"name"``). For parsers whose markers are
        # registered specials (Mistral ``[TOOL_CALLS]``), the resolver
        # passes ``False`` so the marker survives detokenization, and we
        # noise-strip every OTHER registered special from each chunk
        # ourselves so clients never see ``<|im_end|>`` /
        # ``<|eot_id|>`` / etc. leak into content. Reasoning-parser
        # markers go in the keep set too — a family could register
        # ``<think>``/``</think>`` as specials, and the reasoning parser
        # needs to see them.
        self._skip_special_tokens = True if skip_special_tokens is None else skip_special_tokens
        self._noise_specials: tuple[str, ...] = ()
        if not self._skip_special_tokens:
            keep = {
                m
                for m in (
                    tool_parser.start_marker if tool_parser else "",
                    tool_parser.end_marker if tool_parser else "",
                    reasoning.start_marker if reasoning else "",
                    reasoning.end_marker if reasoning else "",
                )
                if m
            }
            # ``added_tokens_decoder`` is the authoritative list of every
            # token (special + ordinary) registered on the tokenizer, with
            # a per-entry ``special`` flag. We strip the ``special=True``
            # entries that are NOT in the keep set. Sort by length
            # descending so a longer marker (e.g. ``<|start_header_id|>``)
            # is replaced before a substring of it could cause false
            # partial matches. Use ``getattr`` because some non-fast or
            # custom tokenizers may not expose this attribute — in that
            # case noise stripping silently degrades to a no-op rather
            # than blowing up at startup.
            added_tokens = getattr(self.tokenizer, "added_tokens_decoder", {}) or {}
            self._noise_specials = tuple(
                sorted(
                    (
                        tok.content
                        for tok in added_tokens.values()
                        if tok.special and tok.content and tok.content not in keep
                    ),
                    key=len,
                    reverse=True,
                )
            )

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
        if tools and request_forces_tool_call(request.tool_choice) and not self._logged_force_unenforceable:
            self._logged_force_unenforceable = True
            logger.info(
                "tool_choice forces a tool call but the transformers loader has no constrained decoding; "
                "passing tools to the model and trusting it to comply (best-effort)"
            )

        max_tokens = request.max_tokens
        if max_tokens is None and request.max_completion_tokens is not None:
            max_tokens = request.max_completion_tokens

        parser_name = self.tool_call_parser if tools else None
        reasoning_parser_name = self.reasoning_parser

        if request.stream:
            include_usage = bool(request.stream_options and request.stream_options.include_usage)
            return self._locked_stream(
                request_id,
                messages,
                tools,
                max_tokens,
                parser_name=parser_name,
                reasoning_parser_name=reasoning_parser_name,
                include_usage=include_usage,
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
            reasoning_parser_name=reasoning_parser_name,
            noise_specials=self._noise_specials,
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
        kwargs: dict[str, Any] = {"tokenize": True, **self.chat_template_kwargs}
        if tools:
            kwargs["tools"] = tools
        token_ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        return len(token_ids)

    def _render_prompt(self, messages: list[dict], tools: list[dict[str, Any]] | None = None) -> str:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,  # type: ignore[arg-type]
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        assert isinstance(rendered, str), "apply_chat_template(tokenize=False) must return str"
        return rendered

    def _run(self, messages: list[dict], tools: list[dict[str, Any]] | None, max_tokens: int | None) -> list:
        kwargs = {**self.config.pipeline_kwargs}
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens
        # Mirror the streamer's setting on the non-streaming path so the
        # pipeline's internal detokenize doesn't pre-strip the parser's
        # marker (the pipeline defaults ``skip_special_tokens=True``).
        if not self._skip_special_tokens:
            kwargs["skip_special_tokens"] = False
        # The pipeline applies the chat template internally and forwards neither
        # `tools` nor our `chat_template_kwargs`. Render ourselves whenever either
        # is in play so both reach `apply_chat_template`; otherwise let the
        # pipeline handle the plain message list.
        if tools or self.chat_template_kwargs:
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
        reasoning_parser_name: str | None,
        include_usage: bool,
    ) -> AsyncGenerator[str, None]:
        async with self._lock:
            async for chunk in self._stream(
                request_id,
                messages,
                tools,
                max_tokens,
                parser_name=parser_name,
                reasoning_parser_name=reasoning_parser_name,
                include_usage=include_usage,
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
        reasoning_parser_name: str | None,
        include_usage: bool,
    ) -> AsyncGenerator[str, None]:
        streamer = TextIteratorStreamer(  # type: ignore[arg-type]
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=self._skip_special_tokens,
        )

        kwargs = {**self.config.pipeline_kwargs}
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens

        # Mirror _run: render ourselves whenever tools or chat_template_kwargs are
        # set, since the pipeline forwards neither to apply_chat_template.
        if tools or self.chat_template_kwargs:
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
                noise_specials=self._noise_specials,
                reasoning_parser_name=reasoning_parser_name,
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
