"""Translation between the Responses API and chat completions.

:func:`responses_request_to_chat` — ``ResponsesRequest`` → ``ChatCompletionRequest``.
Translates the structurally different bits (``input``/``instructions`` →
``messages``, flattened tools → nested, ``text.format`` → ``response_format``,
``max_output_tokens`` → ``max_completion_tokens``) and rejects features not yet
supported (``background``, hosted built-in tools) with an explicit error rather
than dropping them silently. Every loader's
``create_response`` calls into this for the request side; ``vllm``/``llama_server``
then shape the response natively from their own parsed output rather than going
back through a chat-completion object (see ``utils.responses.build_responses_items_from_parsed``)
— ``build_response_object``/``_usage_from_chat``/``_status_for`` below are the shared
envelope helpers both that native path and the streaming translator build on.

``store`` and ``previous_response_id`` are acted on by the gateway (it persists the
snapshot and resolves the history); here they are echoed onto the response object
only. Image/audio input parts are reduced to their text for now.

Imports come from the sibling ``schemas`` submodule and the ``chat`` submodule,
never from the top-level ``modelship.openai.protocol`` package — that package
imports this one, so reaching back into it would create an import cycle.
"""

from __future__ import annotations

from typing import Any, Literal

from cryptography.fernet import InvalidToken

from modelship.openai import compaction_crypto
from modelship.openai.protocol.chat import ChatCompletionRequest, StreamOptions
from modelship.openai.protocol.responses.schemas import (
    ResponseInputTokensDetails,
    ResponseObject,
    ResponseOutputTokensDetails,
    ResponsesRequest,
    ResponseUsage,
)
from modelship.openai.protocol.usage import UsageInfo


class UnsupportedResponsesFeatureError(ValueError):
    """A Responses request used a feature that is not supported.

    Subclasses ``ValueError`` so ``create_error_response`` maps it to an
    OpenAI-style 400 ``invalid_request_error``.
    """


# ---------------------------------------------------------------------------
# Request: Responses → chat
# ---------------------------------------------------------------------------


def responses_request_to_chat(request: ResponsesRequest) -> ChatCompletionRequest:
    """Translate a ``ResponsesRequest`` into a ``ChatCompletionRequest``.

    ``previous_response_id`` is **already resolved** by the time this runs: the
    gateway looks the prior snapshot up and prepends its items to ``input`` before
    the Ray hop, so the field survives here only to be echoed on the response. This
    contract can't be checked from inside the loader — it holds because
    ``ModelDeployment`` is reachable only via the gateway, never over HTTP.

    Raises :class:`UnsupportedResponsesFeatureError` for background/hosted-tool
    features the adapter cannot fulfill.
    """
    if request.background:
        raise UnsupportedResponsesFeatureError("background mode is not supported on /v1/responses.")

    messages = messages_from_input(request.input, request.instructions)

    kwargs: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "stream": bool(request.stream),
    }
    if request.stream:
        # Responses always reports usage on the final event; the streaming
        # translator only gets a usage-bearing chunk out of the chat pipeline
        # when this is set (see engine_ops.stream_chat_completion). ChatCompletionRequest
        # rejects stream_options when stream=False, so this must stay conditional.
        kwargs["stream_options"] = StreamOptions(include_usage=True)
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


def messages_from_input(input_: str | list[dict[str, Any]], instructions: str | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(input_, str):
        messages.append({"role": "user", "content": input_})
        return messages

    for item in input_:
        itype = item.get("type")
        if itype == "function_call":
            # A prior assistant tool call being replayed as context. call_id and
            # name identify the call; without them the loader's chat-template /
            # tool-call handling fails downstream, so reject up front with a 400.
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name")
            if not call_id or not name:
                raise UnsupportedResponsesFeatureError(
                    "function_call input items require both 'call_id' (or 'id') and 'name'."
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": item.get("arguments", ""),
                            },
                        }
                    ],
                }
            )
        elif itype == "function_call_output":
            # call_id ties the result back to its call; a missing one can't be
            # associated, so reject rather than emit a tool message with no id.
            call_id = item.get("call_id")
            if not call_id:
                raise UnsupportedResponsesFeatureError("function_call_output input items require 'call_id'.")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _text_of(item.get("output")),
                }
            )
        elif itype == "reasoning":
            # Don't replay raw chain-of-thought back into the prompt.
            continue
        elif itype == "compaction":
            # Round-trip half of /v1/responses/compact: the blob is opaque to the
            # client, decrypted back into the items it was built from. Recursion is
            # bounded — a compaction blob never contains another compaction item.
            encrypted_content = item.get("encrypted_content")
            if not encrypted_content:
                raise UnsupportedResponsesFeatureError("compaction input items require 'encrypted_content'.")
            try:
                decoded_items = compaction_crypto.decrypt_items(encrypted_content)
            except InvalidToken:
                raise UnsupportedResponsesFeatureError("compaction item could not be decoded.") from None
            messages.extend(messages_from_input(decoded_items, None))
        elif itype == "message" or "role" in item:
            messages.append(
                {
                    "role": item.get("role", "user"),
                    "content": _content_to_chat(item.get("content")),
                }
            )
        else:
            raise UnsupportedResponsesFeatureError(f"unsupported input item type {itype!r}.")

    return messages


def _text_of(content: Any) -> str | None:
    """Reduce a Responses content value to plain text (used for tool-result content,
    which is always text)."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)]
        return "".join(parts) if parts else None
    raise UnsupportedResponsesFeatureError(f"unsupported content shape {content!r}.")


_CONTENT_TEXT_TYPES = frozenset({"input_text", "output_text", "text"})
_CONTENT_IMAGE_TYPES = frozenset({"input_image", "image_url"})


def _content_to_chat(content: Any) -> str | list[dict[str, Any]] | None:
    """Translate a Responses message ``content`` value into chat message content.

    Unlike ``_text_of`` (tool-result text), a message's content can carry images —
    dropping them here was silently discarding vision input. Text parts become chat's
    ``{"type": "text", ...}`` shape; ``input_image`` becomes chat's ``image_url`` shape,
    which ``normalize_chat_messages``/``_validate_part`` already fully validates
    (``modelship.openai.utils.chat``). Unknown part types are rejected rather than
    silently dropped.
    """
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise UnsupportedResponsesFeatureError(f"unsupported content shape {content!r}.")

    parts: list[dict[str, Any]] = []
    for p in content:
        if not isinstance(p, dict):
            raise UnsupportedResponsesFeatureError(f"unsupported content part {p!r}.")
        ptype = p.get("type")
        if ptype in _CONTENT_TEXT_TYPES:
            parts.append({"type": "text", "text": p.get("text")})
        elif ptype in _CONTENT_IMAGE_TYPES:
            # Both the Responses `input_image` and chat's own `image_url` part types
            # carry the URL/data-URI under an `image_url` key (string or {"url": ...}).
            url = p.get("image_url")
            parts.append({"type": "image_url", "image_url": {"url": url} if isinstance(url, str) else url})
        else:
            raise UnsupportedResponsesFeatureError(f"unsupported content part type {ptype!r}.")

    if all(part["type"] == "text" for part in parts):
        return "".join(part["text"] or "" for part in parts)
    return parts


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
# Response envelope: shared by every loader's native create_response and by
# the streaming translator.
# ---------------------------------------------------------------------------


def build_response_object(
    request: ResponsesRequest,
    *,
    status: str,
    output: list[Any],
    usage: ResponseUsage | None,
    incomplete: dict[str, Any] | None,
    model: str | None = None,
    response_id: str | None = None,
    created_at: int | None = None,
    completed_at: int | None = None,
    error: Any | None = None,
) -> ResponseObject:
    """Build a ``ResponseObject``, echoing the request settings OpenAI returns.

    Shared by the non-streaming adapter and the streaming translator so the
    ``response.created`` / ``response.completed`` envelopes and the
    non-streaming body carry an identical shape. ``response_id`` / ``created_at``
    let the streaming translator keep one stable id across all of its events.
    ``completed_at`` is only meaningful on a terminal (``completed``) envelope;
    every other caller leaves it ``None``. ``error`` is set only for a
    ``status="failed"`` terminal event (see ``ResponsesStreamTranslator.fail``);
    every other caller leaves it ``None``.
    """
    kwargs: dict[str, Any] = {
        "model": model or request.model or "",
        "status": status,
        "output": output,
        "usage": usage,
        "incomplete_details": incomplete,
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        # OpenAI reports the *effective* sampling values used, not a bare echo —
        # these are its own defaults when the request left them unset.
        "temperature": request.temperature if request.temperature is not None else 1.0,
        "top_p": request.top_p if request.top_p is not None else 1.0,
        "tools": _echo_tools(request.tools),
        "tool_choice": request.tool_choice if request.tool_choice is not None else "auto",
        "parallel_tool_calls": request.parallel_tool_calls if request.parallel_tool_calls is not None else True,
        "text": request.text if request.text else {"format": {"type": "text"}},
        "reasoning": request.reasoning,
        "metadata": request.metadata or {},
        # OpenAI stores by default; only an explicit `store: false` opts out. The
        # gateway reads the same field to decide whether to persist, so what the
        # response claims and what was actually stored cannot drift.
        "store": request.store is not False,
        "previous_response_id": request.previous_response_id,
    }
    if response_id is not None:
        kwargs["id"] = response_id
    if created_at is not None:
        kwargs["created_at"] = created_at
    if completed_at is not None:
        kwargs["completed_at"] = completed_at
    if error is not None:
        kwargs["error"] = error
    return ResponseObject(**kwargs)


def _echo_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Backfill echoed ``tools[]`` so every ``FunctionTool`` carries all five spec keys.

    Clients routinely omit optional keys (``description``/``parameters``/``strict``)
    on the request; the spec requires them present (nullable) on the echo. Non-function
    tool shapes are passed through untouched — the request side already rejects them
    before a response is ever built, so this only defends against an unexpected shape.
    """
    if not tools:
        return []
    out: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            out.append(tool)
            continue
        out.append(
            {
                "type": "function",
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters"),
                "strict": tool.get("strict"),
            }
        )
    return out


def _usage_from_chat(usage: UsageInfo) -> ResponseUsage:
    """Remap chat usage to Responses usage, preserving token details.

    ``cached_tokens`` / ``reasoning_tokens`` are surfaced by vLLM (prefix
    caching / reasoning models) under the OpenAI-standard ``prompt_tokens_details``
    / ``completion_tokens_details``. Responses uses the same sub-field names, so
    this is a direct field-to-field copy; loaders that report no details
    (llama_server/transformers) leave them at the zero default.
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
