"""Stateless translation between the Responses API and chat completions.

Two directions:

- :func:`responses_request_to_chat` — ``ResponsesRequest`` → ``ChatCompletionRequest``.
  Translates the structurally different bits (``input``/``instructions`` →
  ``messages``, flattened tools → nested, ``text.format`` → ``response_format``,
  ``max_output_tokens`` → ``max_completion_tokens``) and rejects features Phase A
  cannot honor (``previous_response_id``, ``background``, hosted built-in tools)
  with an explicit error rather than dropping them silently.
- :func:`chat_response_to_responses` — ``ChatCompletionResponse`` → ``ResponseObject``.
  Maps the parsed message into Responses ``output[]`` items (reasoning / message /
  function_call) and remaps token usage.

Phase A is text + reasoning + (client-driven) tool calling, non-streaming.
``store`` is accepted but never persisted — the response echoes ``store=False``.
Image/audio input parts are reduced to their text for now (vision over
``/v1/responses`` is out of Phase A scope).

Imports come from the sibling ``schemas`` submodule and the ``chat`` submodule,
never from the top-level ``modelship.openai.protocol`` package — that package
imports this one, so reaching back into it would create an import cycle.
"""

from __future__ import annotations

from typing import Any, Literal

from modelship.openai.protocol.chat import ChatCompletionRequest, ChatCompletionResponse
from modelship.openai.protocol.responses.schemas import (
    ResponseFunctionToolCall,
    ResponseInputTokensDetails,
    ResponseObject,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseOutputTokensDetails,
    ResponseReasoningItem,
    ResponseReasoningSummary,
    ResponsesRequest,
    ResponseUsage,
)
from modelship.openai.protocol.usage import UsageInfo


class UnsupportedResponsesFeatureError(ValueError):
    """A Responses request used a feature Phase A does not implement.

    Subclasses ``ValueError`` so ``create_error_response`` maps it to an
    OpenAI-style 400 ``invalid_request_error``.
    """


# ---------------------------------------------------------------------------
# Request: Responses → chat
# ---------------------------------------------------------------------------


def responses_request_to_chat(request: ResponsesRequest) -> ChatCompletionRequest:
    """Translate a ``ResponsesRequest`` into a non-streaming ``ChatCompletionRequest``.

    Raises :class:`UnsupportedResponsesFeatureError` for stateful/background/hosted-tool
    features that the stateless Phase A adapter cannot fulfill.
    """
    if request.previous_response_id is not None:
        raise UnsupportedResponsesFeatureError(
            "previous_response_id is not supported: /v1/responses does not yet persist conversation "
            "state. Send the full conversation in `input` instead."
        )
    if request.background:
        raise UnsupportedResponsesFeatureError("background mode is not supported on /v1/responses.")

    messages = _messages_from_input(request.input, request.instructions)

    kwargs: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "stream": False,
    }
    if request.max_output_tokens is not None:
        kwargs["max_completion_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = request.parallel_tool_calls
    if request.user is not None:
        kwargs["user"] = request.user

    tools = _tools_to_chat(request.tools)
    if tools is not None:
        kwargs["tools"] = tools
    tool_choice = _tool_choice_to_chat(request.tool_choice)
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    response_format = _response_format_from_text(request.text)
    if response_format is not None:
        kwargs["response_format"] = response_format

    effort = (request.reasoning or {}).get("effort")
    if effort is not None:
        kwargs["reasoning_effort"] = effort

    return ChatCompletionRequest(**kwargs)


def _messages_from_input(input_: str | list[dict[str, Any]], instructions: str | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(input_, str):
        messages.append({"role": "user", "content": input_})
        return messages

    for item in input_:
        itype = item.get("type")
        if itype == "function_call":
            # A prior assistant tool call being replayed as context.
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id"),
                            "type": "function",
                            "function": {
                                "name": item.get("name"),
                                "arguments": item.get("arguments", ""),
                            },
                        }
                    ],
                }
            )
        elif itype == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": _text_of(item.get("output")),
                }
            )
        elif itype == "reasoning":
            # Don't replay raw chain-of-thought back into the prompt.
            continue
        elif itype == "message" or "role" in item:
            messages.append(
                {
                    "role": item.get("role", "user"),
                    "content": _text_of(item.get("content")),
                }
            )
        else:
            raise UnsupportedResponsesFeatureError(f"unsupported input item type {itype!r}.")

    return messages


def _text_of(content: Any) -> str | None:
    """Reduce a Responses content value to plain text (Phase A is text-only)."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)]
        return "".join(parts) if parts else None
    return str(content)


def _tools_to_chat(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        ttype = tool.get("type")
        if ttype != "function":
            raise UnsupportedResponsesFeatureError(
                f"hosted tool type {ttype!r} is not supported; only client-defined 'function' tools are."
            )
        # Responses flattens the function fields onto the tool; chat nests them.
        fn = {k: tool[k] for k in ("name", "description", "parameters", "strict") if k in tool}
        out.append({"type": "function", "function": fn})
    return out


def _tool_choice_to_chat(tool_choice: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    if tool_choice.get("type") == "function":
        return {"type": "function", "function": {"name": tool_choice.get("name")}}
    raise UnsupportedResponsesFeatureError(f"tool_choice type {tool_choice.get('type')!r} is not supported.")


def _response_format_from_text(text: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate Responses ``text.format`` into chat ``response_format``."""
    if not text:
        return None
    fmt = text.get("format")
    if not fmt:
        return None
    if not isinstance(fmt, dict):
        raise UnsupportedResponsesFeatureError(
            f"text.format must be an object with a 'type' field, got {type(fmt).__name__}."
        )
    ftype = fmt.get("type")
    if ftype == "json_schema":
        # Responses flattens name/schema/strict; chat nests them under json_schema.
        nested = {k: fmt[k] for k in ("name", "schema", "strict", "description") if k in fmt}
        return {"type": "json_schema", "json_schema": nested}
    if ftype in ("json_object", "text"):
        return {"type": ftype}
    raise UnsupportedResponsesFeatureError(f"text.format type {ftype!r} is not supported.")


# ---------------------------------------------------------------------------
# Response: chat → Responses
# ---------------------------------------------------------------------------


def chat_response_to_responses(chat: ChatCompletionResponse, request: ResponsesRequest) -> ResponseObject:
    """Translate a non-streaming ``ChatCompletionResponse`` into a ``ResponseObject``."""
    choice = chat.choices[0]
    message = choice.message

    output: list[Any] = []
    # OpenAI emits reasoning first, then the assistant message / tool calls.
    if message.reasoning:
        output.append(ResponseReasoningItem(summary=[ResponseReasoningSummary(text=message.reasoning)]))
    if message.content:
        output.append(ResponseOutputMessage(content=[ResponseOutputText(text=message.content)]))
    for call in message.tool_calls:
        output.append(
            ResponseFunctionToolCall(
                call_id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
        )

    status, incomplete = _status_for(choice.finish_reason)

    return ResponseObject(
        model=chat.model,
        status=status,
        output=output,
        usage=_usage_from_chat(chat.usage),
        incomplete_details=incomplete,
        # Echo the request settings OpenAI returns on the response object.
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        tools=request.tools or [],
        tool_choice=request.tool_choice if request.tool_choice is not None else "auto",
        parallel_tool_calls=request.parallel_tool_calls if request.parallel_tool_calls is not None else True,
        text=request.text,
        reasoning=request.reasoning,
        metadata=request.metadata or {},
        store=False,
    )


def _usage_from_chat(usage: UsageInfo) -> ResponseUsage:
    """Remap chat usage to Responses usage, preserving token details.

    ``cached_tokens`` / ``reasoning_tokens`` are surfaced by vLLM (prefix
    caching / reasoning models) under the OpenAI-standard ``prompt_tokens_details``
    / ``completion_tokens_details``. Responses uses the same sub-field names, so
    this is a direct field-to-field copy; loaders that report no details
    (llama_cpp/transformers) leave them at the zero default.
    """
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return ResponseUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens or 0,
        total_tokens=usage.total_tokens,
        input_tokens_details=ResponseInputTokensDetails(
            cached_tokens=(prompt_details.cached_tokens or 0) if prompt_details else 0
        ),
        output_tokens_details=ResponseOutputTokensDetails(
            reasoning_tokens=(completion_details.reasoning_tokens or 0) if completion_details else 0
        ),
    )


def _status_for(
    finish_reason: str | None,
) -> tuple[Literal["completed", "incomplete"], dict[str, Any] | None]:
    # chat finish_reason -> Responses status + incomplete_details.reason.
    # The Responses spec's incomplete_details.reason enum is
    # {max_output_tokens, content_filter}.
    if finish_reason == "length":
        return "incomplete", {"reason": "max_output_tokens"}
    if finish_reason == "content_filter":
        return "incomplete", {"reason": "content_filter"}
    return "completed", None
