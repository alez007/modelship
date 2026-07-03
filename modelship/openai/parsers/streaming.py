"""Loader-agnostic chat-completion helpers for parser-aware responses.

Loaders produce text differently — HF ``Pipeline`` for transformers, raw
``llama.create_completion`` for llama_cpp, async iterators from plugins —
but the OpenAI response shapes (streaming and non-streaming) are the
same. This module owns those shapes plus the
:class:`~modelship.openai.parsers.output.ChatOutputStreamer` driving
loop, so the loaders only deal in plain text.

Two helpers:

- :func:`build_chat_completion_response` — non-streaming. Loader hands in
  the full completion text and token counts; we parse reasoning and
  tool calls and pack the OpenAI ``ChatCompletionResponse``.
- :func:`stream_chat_completion` — streaming. Loader hands in an
  ``AsyncIterator[str]`` of new text pieces; we emit the SSE byte
  stream.

Both accept ``parser_name`` (tool-call) and ``reasoning_parser_name``
independently. Either or both may be ``None``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable

from modelship.logging import TRACE, get_logger
from modelship.openai.parsers.output import ChatOutputStreamer, ParsedChatOutput
from modelship.openai.parsers.reasoning.registry import get_parser as get_reasoning_parser
from modelship.openai.parsers.tool_calling.registry import get_parser as get_tool_call_parser
from modelship.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)

logger = get_logger("openai.parsers.streaming")


def _log_tool_calls(request_id: str, parsed: ParsedChatOutput, finish_reason: str, completion_tokens: int) -> None:
    """Log the structured tool calls handed to the client.

    Loaders already TRACE-log the model's raw text; this logs the *parsed*
    OpenAI tool calls so it's visible exactly what a downstream client (e.g.
    Home Assistant) receives and tries to execute — the key diagnostic when a
    client rejects an otherwise-200 response.
    """
    if not parsed.tool_calls or not logger.isEnabledFor(TRACE):
        return
    summary = "; ".join(f"{tc.function.name}({tc.function.arguments})" for tc in parsed.tool_calls)
    logger.log(
        TRACE,
        "chat %s -> client: %d tool call(s) [finish_reason=%s, completion_tokens=%s]: %s",
        request_id,
        len(parsed.tool_calls),
        finish_reason,
        completion_tokens,
        summary,
    )


def finish_reason_for(parsed: ParsedChatOutput, completion_tokens: int, max_tokens: int | None) -> str:
    """Compute the OpenAI ``finish_reason`` for a chat completion."""
    if parsed.has_tool_calls:
        return "tool_calls"
    if max_tokens is not None and completion_tokens >= max_tokens:
        return "length"
    return "stop"


def build_chat_completion_response(
    *,
    request_id: str,
    model_name: str,
    text: str,
    parser_name: str | None,
    reasoning_parser_name: str | None = None,
    noise_specials: tuple[str, ...] = (),
    prompt_tokens: int,
    completion_tokens: int,
    max_tokens: int | None,
    created: int,
) -> ChatCompletionResponse:
    """Parse ``text`` and pack it into an OpenAI ``ChatCompletionResponse``.

    Non-streaming counterpart of :func:`stream_chat_completion`. When
    both parser names are ``None``, ``text`` becomes the message
    content as-is with no extraction. ``noise_specials`` is forwarded to
    the streamer so loaders that decode with ``skip_special_tokens=False``
    can have unwanted special tokens (``<|eot_id|>``, ``[INST]``, etc.)
    silently dropped before parsing.
    """
    parsed = parse_chat_completion_text(
        text,
        parser_name=parser_name,
        reasoning_parser_name=reasoning_parser_name,
        noise_specials=noise_specials,
    )
    finish_reason = finish_reason_for(parsed, completion_tokens, max_tokens)
    _log_tool_calls(request_id, parsed, finish_reason, completion_tokens)
    return ChatCompletionResponse(
        id=request_id,
        model=model_name,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=parsed.content,
                    reasoning=parsed.reasoning,
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
        created=created,
    )


async def stream_chat_completion(
    *,
    request_id: str,
    model_name: str,
    text_chunks: AsyncIterator[str],
    parser_name: str | None,
    reasoning_parser_name: str | None = None,
    noise_specials: tuple[str, ...] = (),
    count_tokens: Callable[[str], int],
    prompt_tokens: int,
    max_tokens: int | None,
    include_usage: bool,
    created: int,
) -> AsyncIterator[str]:
    """Drive the OpenAI streaming-response protocol from a stream of text pieces.

    Yields SSE strings ready to be forwarded to the client. When either
    ``parser_name`` or ``reasoning_parser_name`` is set, the cumulative
    buffer is fed to a :class:`ChatOutputStreamer` so newly-emittable
    reasoning, content, and tool-call fragments show up in deltas
    without a half-formed marker ever reaching the client. When both
    are ``None``, each chunk is forwarded as a content delta unchanged.

    ``count_tokens`` is consulted only at end-of-stream to compute the
    completion-token count for ``finish_reason`` and (optionally)
    usage; pass ``lambda _: 0`` if the loader doesn't have a tokenizer
    handy.
    """
    streamer = _make_streamer(
        parser_name=parser_name,
        reasoning_parser_name=reasoning_parser_name,
        noise_specials=noise_specials,
    )
    accumulated = ""

    yield _delta_chunk(request_id, model_name, DeltaMessage(role="assistant"), created)

    async for piece in text_chunks:
        if not piece:
            continue
        accumulated += piece
        if streamer is None:
            yield _delta_chunk(request_id, model_name, DeltaMessage(content=piece), created)
            await asyncio.sleep(0)
            continue
        delta = streamer.extract_streaming(accumulated)
        if delta is not None:
            yield _delta_chunk(request_id, model_name, delta, created)
        await asyncio.sleep(0)

    if streamer is not None:
        final = streamer.finalize()
        if final is not None:
            yield _delta_chunk(request_id, model_name, final, created)
        parsed = streamer.result
    else:
        parsed = ParsedChatOutput(content=accumulated or None, reasoning=None, tool_calls=[])

    completion_tokens = count_tokens(accumulated)
    finish_reason = finish_reason_for(parsed, completion_tokens, max_tokens)
    _log_tool_calls(request_id, parsed, finish_reason, completion_tokens)

    yield _encode_chunk(
        ChatCompletionStreamResponse(
            id=request_id,
            model=model_name,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason=finish_reason,
                )
            ],
            created=created,
        )
    )

    if include_usage:
        yield _encode_chunk(
            ChatCompletionStreamResponse(
                id=request_id,
                model=model_name,
                choices=[],
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
                created=created,
            )
        )

    yield "data: [DONE]\n\n"


def parse_chat_completion_text(
    text: str,
    *,
    parser_name: str | None,
    reasoning_parser_name: str | None,
    noise_specials: tuple[str, ...] = (),
) -> ParsedChatOutput:
    """Run the shared :class:`ChatOutputStreamer` over a full completion text.

    Public because non-streaming loader paths outside this module (e.g.
    llama_cpp's ``_handle_with_parsers``) parse text themselves and feed
    the result into :func:`modelship.openai.chat_utils.build_from_parsed`
    rather than going through :func:`build_chat_completion_response`.
    """
    streamer = _make_streamer(
        parser_name=parser_name,
        reasoning_parser_name=reasoning_parser_name,
        noise_specials=noise_specials,
    )
    if streamer is None:
        return ParsedChatOutput(content=text or None, reasoning=None, tool_calls=[])
    streamer.extract_streaming(text)
    streamer.finalize()
    return streamer.result


def _make_streamer(
    *,
    parser_name: str | None,
    reasoning_parser_name: str | None,
    noise_specials: tuple[str, ...] = (),
) -> ChatOutputStreamer | None:
    if parser_name is None and reasoning_parser_name is None:
        return None
    tool_parser = get_tool_call_parser(parser_name) if parser_name else None
    reasoning_parser = get_reasoning_parser(reasoning_parser_name) if reasoning_parser_name else None
    return ChatOutputStreamer(tool_parser, reasoning_parser, noise_specials=noise_specials)


def _delta_chunk(request_id: str, model_name: str, delta: DeltaMessage, created: int) -> str:
    return _encode_chunk(
        ChatCompletionStreamResponse(
            id=request_id,
            model=model_name,
            choices=[ChatCompletionResponseStreamChoice(index=0, delta=delta)],
            created=created,
        )
    )


def _encode_chunk(chunk: ChatCompletionStreamResponse) -> str:
    return f"data: {json.dumps(chunk.model_dump(mode='json'))}\n\n"
