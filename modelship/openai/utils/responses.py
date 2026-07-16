"""Utility helpers for ``/v1/responses``, in two halves:

- Shaping: turning a loader's parsed chat output into Responses ``output[]`` items
  (mirrors ``utils.chat``'s chat-completion equivalents; used loader-side).
- Gateway-side conversation-state plumbing: history resolution, response
  persistence, and snapshot lookup, wrapped in HTTPException so ``api.py``'s route
  handlers stay one-liners. None of this touches Ray dispatch or FastAPI routing
  itself — it's pure state-store orchestration over an explicit ``store``/``identity``.
"""

from __future__ import annotations

import json
import re
from http import HTTPStatus
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from modelship.logging import get_logger
from modelship.openai.protocol import ErrorResponse, ResponseObject, ResponsesRequest, UsageInfo, create_error_response
from modelship.openai.protocol.responses.adapter import _status_for, _usage_from_chat, build_response_object
from modelship.openai.protocol.responses.schemas import (
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningSummary,
)
from modelship.openai.protocol.responses.streaming import TERMINAL_EVENT_TYPES, store_failure_event
from modelship.openai.state import responses as responses_state
from modelship.openai.utils.chat import ParsedChatOutput
from modelship.state import StateStore, StateStoreUnavailableError

logger = get_logger("openai.utils.responses")

# Shape of the response ids we mint (`resp_<uuid>`). Ids arriving from a client are
# checked against it before becoming a state-store key segment, so a malformed id is
# a clean 404 rather than a lookup for something we could never have written.
RESPONSE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


# --- Shaping: parsed chat output -> Responses items (loader-side) ---


def build_responses_items_from_parsed(parsed: ParsedChatOutput) -> list[ResponseOutputItem]:
    """Shape one parsed choice into Responses ``output[]`` items.

    Sibling to `chat.build_from_parsed`: same DTO in, Responses items out instead
    of a `ChatCompletionResponse`. Order matches OpenAI's own: reasoning
    first, then the assistant message, then one `function_call` per tool call.
    """
    output: list[ResponseOutputItem] = []
    if parsed.reasoning:
        output.append(ResponseReasoningItem(summary=[ResponseReasoningSummary(text=parsed.reasoning)]))
    if parsed.content:
        output.append(ResponseOutputMessage(content=[ResponseOutputText(text=parsed.content)]))
    for call in parsed.tool_calls:
        output.append(
            ResponseFunctionToolCall(
                call_id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
        )
    return output


def build_response_from_parsed(
    parsed: ParsedChatOutput,
    request: ResponsesRequest,
    *,
    usage: UsageInfo,
    finish_reason: str | None,
    model: str,
) -> ResponseObject:
    """Build a non-streaming ``ResponseObject`` from one loader's parsed chat output.

    Shared by every loader's non-streaming `create_response`: each shapes its own
    `ParsedChatOutput` from its native response format, then hands it here for the
    status/usage/envelope assembly (`build_response_object` + `_status_for` +
    `_usage_from_chat`) that used to be duplicated per loader.
    """
    status, incomplete = _status_for(finish_reason)
    return build_response_object(
        request,
        status=status,
        output=build_responses_items_from_parsed(parsed),
        usage=_usage_from_chat(usage),
        incomplete=incomplete,
        model=model,
    )


def responses_validation_error(exc: ValidationError) -> ErrorResponse:
    """400 for a pydantic ``ValidationError`` surfaced by ``responses_request_to_chat``
    (e.g. a bad ``reasoning.effort`` value) — same shape as every other rejection.

    ``ValidationError.args`` is always empty (pydantic never populates it), so
    ``str(exc)`` — its full per-field error report — is the message to use.
    """
    return create_error_response(message=str(exc), err_type="invalid_request_error")


# --- Gateway-side conversation-state plumbing ---


def as_input_items(input_: str | list[Any]) -> list[Any]:
    """Normalize a Responses ``input`` to item form, so stored history and this turn's
    input concatenate."""
    if isinstance(input_, str):
        return [{"type": "message", "role": "user", "content": input_}]
    return list(input_)


def _terminal_event_payload(chunk: Any) -> dict[str, Any] | None:
    """The decoded event of *chunk* if it is a terminal Responses SSE event carrying a
    complete response, else ``None``.

    Recovers the response the gateway just forwarded so it can be stored. Re-parsing
    our own output is the cost of keeping state in the gateway; it is safe because
    `streaming._sse` is the only writer of this format. The cheap `event:` line check
    runs first so ordinary deltas never reach `json.loads`.
    """
    if not isinstance(chunk, str) or not any(chunk.startswith(f"event: {t}\n") for t in TERMINAL_EVENT_TYPES):
        return None
    _, _, data = chunk.partition("\ndata: ")
    try:
        payload = json.loads(data.strip())
    except json.JSONDecodeError:
        logger.exception("Could not decode terminal Responses event; not storing this response.")
        return None
    return payload if isinstance(payload, dict) else None


async def resolve_history(store: StateStore, identity: str, request: ResponsesRequest) -> list[Any]:
    """Prepend the conversation stored under ``previous_response_id`` to this
    turn's input. 404 if unknown, 503 if the store is unreachable — an outage must
    never masquerade as a legitimately unknown id."""
    prev_id = request.previous_response_id or ""
    if not RESPONSE_ID_RE.match(prev_id):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND.value,
            detail=f"Previous response with id '{prev_id}' not found.",
        )
    try:
        snapshot = await responses_state.read_async(store, identity, prev_id)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable resolving previous_response_id=%s", prev_id)
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Conversation state store is unavailable; retry shortly.",
        ) from None
    if snapshot is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND.value,
            detail=f"Previous response with id '{prev_id}' not found.",
        )
    return [*responses_state.history_items(snapshot), *as_input_items(request.input)]


async def persist_response(gen, store: StateStore, *, identity: str, input_items: list[Any]):
    """Tee `respond`'s output, storing the snapshot as it passes.

    Wraps the generator ahead of `_handle_response` so that stays generic. The
    response id is read back off the output rather than minted here — the loader
    already owns it, on both paths.

    Both modes are covered by one wrapper because `_handle_response` dispatches on
    the first item's type either way: a `ResponseObject` is the whole non-streaming
    body, while a stream arrives as SSE strings whose terminal event carries the
    same object. Persisting *before* yielding the terminal item is what lets a
    store failure still change what the client is told.
    """
    async for item in gen:
        if isinstance(item, ResponseObject):
            try:
                await responses_state.write_async(
                    store, identity, item.id, response=item.model_dump(mode="json"), input_items=input_items
                )
            except StateStoreUnavailableError:
                logger.exception("State store unavailable persisting response %s", item.id)
                yield create_error_response(
                    "Conversation state store is unavailable; the response was generated but not stored.",
                    err_type="api_error",
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            yield item
            continue

        payload = _terminal_event_payload(item)
        if payload is None:
            yield item
            continue

        response = payload.get("response") or {}
        try:
            await responses_state.write_async(
                store, identity, response.get("id", ""), response=response, input_items=input_items
            )
        except StateStoreUnavailableError:
            logger.exception("State store unavailable persisting streamed response %s", response.get("id"))
            yield store_failure_event(
                payload, "Conversation state store is unavailable; the response was generated but not stored."
            )
            return
        yield item


async def load_snapshot(store: StateStore, identity: str, response_id: str) -> dict:
    """The stored snapshot for *response_id*, scoped to the caller's identity.

    Isolation needs no comparison: another caller's identity builds a different
    key, so it simply misses and 404s.
    """
    if not RESPONSE_ID_RE.match(response_id):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND.value, detail=f"Response with id '{response_id}' not found."
        )
    try:
        snapshot = await responses_state.read_async(store, identity, response_id)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable reading response %s", response_id)
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Conversation state store is unavailable; retry shortly.",
        ) from None
    if snapshot is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND.value, detail=f"Response with id '{response_id}' not found."
        )
    return snapshot


async def delete_snapshot(store: StateStore, identity: str, response_id: str) -> None:
    """Delete the snapshot for *response_id*.

    Caller must confirm existence first (via :func:`load_snapshot`) — delete is
    idempotent by contract, so it alone can't tell an unknown id from a real removal.
    """
    try:
        await responses_state.delete_async(store, identity, response_id)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable deleting response %s", response_id)
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Conversation state store is unavailable; retry shortly.",
        ) from None
