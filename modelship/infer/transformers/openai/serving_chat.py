import asyncio
import json
import time
from collections.abc import AsyncGenerator
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
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ErrorResponse,
    UsageInfo,
    create_error_response,
)
from modelship.openai.tool_calling import ParsedToolCalls, ToolCallStreamer, get_parser, resolve_tools_for_request
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
    ):
        self.pipeline = pipeline
        self.model_name = model_name
        self.config = config
        self.capabilities = capabilities
        assert pipeline.tokenizer is not None, "text-generation pipeline must have a tokenizer"
        self.tokenizer: PreTrainedTokenizerBase = pipeline.tokenizer
        self._lock = asyncio.Lock()
        # Validate the configured parser at startup so misconfiguration surfaces
        # before the first request rather than mid-generation.
        get_parser(self.config.tool_call_parser)

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

        max_tokens = request.max_tokens
        if max_tokens is None and request.max_completion_tokens is not None:
            max_tokens = request.max_completion_tokens

        if request.stream:
            include_usage = bool(request.stream_options and request.stream_options.include_usage)
            return self._locked_stream(request_id, messages, tools, max_tokens, include_usage=include_usage)

        async with self._lock:
            try:
                result = await self.run_in_executor(self._run, messages, tools, max_tokens)
            except Exception:
                logger.exception("chat completion inference failed for %s", request_id)
                return create_error_response("chat completion inference failed")

        prompt_tokens = self._count_prompt_tokens(messages, tools)
        completion_text = self._extract_completion_text(result)
        completion_tokens = len(self.tokenizer.encode(completion_text, add_special_tokens=False))

        parsed = self._parse_tool_calls(completion_text) if tools else ParsedToolCalls(completion_text, [])
        finish_reason = self._finish_reason(parsed, completion_tokens, max_tokens)

        response = ChatCompletionResponse(
            id=request_id,
            model=self.model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=parsed.content,
                        tool_calls=parsed.tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            created=int(time.time()),
        )
        logger.log(
            TRACE,
            "chat response %s: text=%r, tool_calls=%d, prompt_tokens=%d, completion_tokens=%d",
            request_id,
            completion_text,
            len(parsed.tool_calls),
            prompt_tokens,
            completion_tokens,
        )
        return response

    def _parse_tool_calls(self, text: str) -> ParsedToolCalls:
        return get_parser(self.config.tool_call_parser).parse(text)

    @staticmethod
    def _finish_reason(parsed: ParsedToolCalls, completion_tokens: int, max_tokens: int | None) -> str:
        if parsed.has_tool_calls:
            return "tool_calls"
        if max_tokens is not None and completion_tokens >= max_tokens:
            return "length"
        return "stop"

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
        include_usage: bool,
    ) -> AsyncGenerator[str, None]:
        async with self._lock:
            async for chunk in self._stream(request_id, messages, tools, max_tokens, include_usage=include_usage):
                yield chunk

    async def _stream(
        self,
        request_id: str,
        messages: list[dict],
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        *,
        include_usage: bool,
    ) -> AsyncGenerator[str, None]:
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore[arg-type]

        kwargs = {**self.config.pipeline_kwargs}
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens

        if tools:
            prompt: Any = self._render_prompt(messages, tools)
            tool_call_streamer: ToolCallStreamer | None = ToolCallStreamer(get_parser(self.config.tool_call_parser))
        else:
            prompt = messages
            tool_call_streamer = None

        thread = Thread(
            target=self.pipeline,
            args=(prompt,),
            kwargs={
                "streamer": streamer,
                "return_full_text": False,
                **kwargs,
            },
        )
        thread.start()

        accumulated: list[str] = []
        try:
            # Per OpenAI spec, the first delta carries `role` only.
            yield self._delta_chunk(request_id, DeltaMessage(role="assistant"))

            for text_chunk in streamer:
                if not text_chunk:
                    continue
                accumulated.append(text_chunk)
                if tool_call_streamer is None:
                    yield self._delta_chunk(request_id, DeltaMessage(content=text_chunk))
                    await asyncio.sleep(0)
                    continue
                # Re-parse the cumulative text and emit any new content / tool-call
                # fragments. Bounded held-back tail (length of the start marker) so
                # the client never sees a half-formed `<tool_call>` opening tag.
                delta = tool_call_streamer.extract_streaming("".join(accumulated))
                if delta is not None:
                    yield self._delta_chunk(request_id, delta)
                await asyncio.sleep(0)

            if tool_call_streamer is not None:
                final = tool_call_streamer.finalize()
                if final is not None:
                    yield self._delta_chunk(request_id, final)
                parsed = tool_call_streamer.result
            else:
                parsed = ParsedToolCalls("".join(accumulated), [])

            completion_text = "".join(accumulated)
            completion_tokens = len(self.tokenizer.encode(completion_text, add_special_tokens=False))
            finish_reason = self._finish_reason(parsed, completion_tokens, max_tokens)

            yield self._encode_chunk(
                ChatCompletionStreamResponse(
                    id=request_id,
                    model=self.model_name,
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=0,
                            delta=DeltaMessage(),
                            finish_reason=finish_reason,
                        )
                    ],
                    created=int(time.time()),
                )
            )

            if include_usage:
                prompt_tokens = self._count_prompt_tokens(messages, tools)
                yield self._encode_chunk(
                    ChatCompletionStreamResponse(
                        id=request_id,
                        model=self.model_name,
                        choices=[],
                        usage=UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        ),
                        created=int(time.time()),
                    )
                )

            yield "data: [DONE]\n\n"
        finally:
            thread.join()

    def _delta_chunk(self, request_id: str, delta: DeltaMessage) -> str:
        return self._encode_chunk(
            ChatCompletionStreamResponse(
                id=request_id,
                model=self.model_name,
                choices=[
                    ChatCompletionResponseStreamChoice(index=0, delta=delta),
                ],
                created=int(time.time()),
            )
        )

    @staticmethod
    def _encode_chunk(chunk: ChatCompletionStreamResponse) -> str:
        return f"data: {json.dumps(chunk.model_dump(mode='json'))}\n\n"
