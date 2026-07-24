"""Utility helpers for ``/v1/responses``, in two halves:

- Shaping: turning a loader's parsed chat output into Responses ``output[]`` items
  (mirrors ``utils.chat``'s chat-completion equivalents; used loader-side).
- Gateway-side conversation-state plumbing: history resolution, response
  persistence, and snapshot lookup, wrapped in HTTPException so ``api.py``'s route
  handlers stay one-liners. None of this touches Ray dispatch or FastAPI routing
  itself — it's pure state-store orchestration over an explicit ``store``/``identity``.
"""

from __future__ import annotations

import re
import time
from http import HTTPStatus
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from modelship.logging import get_logger
from modelship.openai import compaction_crypto
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ErrorResponse,
    ResponseObject,
    ResponsesRequest,
    UsageInfo,
    create_error_response,
)
from modelship.openai.protocol.responses.adapter import (
    _status_for,
    _usage_from_chat,
    build_response_object,
    messages_from_input,
)
from modelship.openai.protocol.responses.schemas import (
    CompactionItem,
    CompactResource,
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
        # Only a `completed` response is actually done; `incomplete`/`failed` aren't.
        completed_at=int(time.time()) if status == "completed" else None,
    )


# Structured rather than a bare "summarize this" — a flat prose summary loses the
# details a continuation actually needs. Preserve, in this order: the user's
# explicit intent; key facts, decisions, names, and identifiers; specific file
# paths, code, and command output already produced; errors hit and how they were
# resolved (including any correction the user gave); work still pending; and
# exactly what was in progress, so the next turn picks up without re-deriving it.
_COMPACTION_SYSTEM_PROMPT = (
    "Summarize this conversation so it can be continued from a fresh context window "
    "with nothing essential lost. Structure the summary with these sections, in order: "
    "(1) the user's explicit requests and intent, verbatim where it matters; "
    "(2) key facts, decisions, names, and identifiers established so far; "
    "(3) specific file paths, code, or command output already produced, in enough "
    "detail to avoid re-deriving them; "
    "(4) errors encountered and how they were fixed, including any correction the "
    "user gave about how to approach the task; "
    "(5) work explicitly still pending; "
    "(6) exactly what was being worked on immediately before this summary, so the "
    "next turn picks up from there rather than guessing."
)


def build_summarization_request(model: str, items: list[Any], instructions: str | None = None) -> ChatCompletionRequest:
    """The internal chat request ``/v1/responses/compact`` issues to summarize *items*.

    Reuses ``messages_from_input`` so a compaction item nested in *items* (a chain
    that was already compacted once) decodes the same way it would on ``/v1/responses``.
    May raise ``UnsupportedResponsesFeatureError`` for an item shape it can't translate.
    *instructions*, if given, is inserted as an additional system message alongside
    the fixed compaction prompt, so a caller can steer what the summary preserves.
    """
    messages = messages_from_input(items, None)
    messages.insert(0, {"role": "system", "content": _COMPACTION_SYSTEM_PROMPT})
    if instructions:
        messages.insert(1, {"role": "system", "content": instructions})
    return ChatCompletionRequest(model=model, messages=messages, stream=False)


def build_compaction(*, summary_items: list[Any], usage: UsageInfo) -> CompactResource:
    """Build a ``CompactResource`` from a ``/v1/responses/compact`` summarization call.

    ``id``/``created_at`` are freshly minted rather than echoing the request: a
    compaction result is never persisted under its own id, so there's nothing to key
    a future GET on (out of scope, see the compaction plan).
    """
    encrypted_content = compaction_crypto.encrypt_items(summary_items)
    return CompactResource(output=[CompactionItem(encrypted_content=encrypted_content)], usage=_usage_from_chat(usage))


def responses_validation_error(exc: ValidationError) -> ErrorResponse:
    """400 for a pydantic ``ValidationError`` surfaced by ``responses_request_to_chat``
    (e.g. a bad ``reasoning.effort`` value) — same shape as every other rejection.

    ``ValidationError.args`` is always empty (pydantic never populates it), so
    ``str(exc)`` — its full per-field error report — is the message to use.
    """
    return create_error_response(message=str(exc), err_type="invalid_request_error")


# --- Gateway-side conversation-state plumbing ---


class ResponsesApiError(HTTPException):
    """An ``HTTPException`` that also carries a full OpenAI-shaped ``ErrorResponse``.

    Lets a shared helper (this module) serve both the HTTP route (rendered via
    ``_error_response``) and the WS turn-runner (rendered via ``error_ws_frame``)
    with the *same* raise, instead of each transport needing its own error type.
    Subclasses ``HTTPException`` rather than a bare exception so existing
    ``pytest.raises(HTTPException)`` / ``.status_code`` assertions keep working, and
    so FastAPI still renders something sane if a route ever forgets to catch it.
    """

    def __init__(self, err: ErrorResponse):
        self.err = err
        super().__init__(status_code=err._http_status, detail=err.error.message)


def _not_found_error(previous_response_id: str) -> ResponsesApiError:
    # "previous_response_not_found" is OpenAI's actual code for this failure —
    # reused verbatim by load_snapshot's GET/DELETE 404s too, which fail for the
    # same underlying reason (no such response id under this identity).
    return ResponsesApiError(
        create_error_response(
            f"Previous response with id '{previous_response_id}' not found.",
            err_type="invalid_request_error",
            status_code=HTTPStatus.NOT_FOUND,
            param="previous_response_id",
            code="previous_response_not_found",
        )
    )


def _store_unavailable_error() -> ResponsesApiError:
    return ResponsesApiError(
        create_error_response(
            "Conversation state store is unavailable; retry shortly.",
            err_type="api_error",
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )
    )


def as_input_items(input_: str | list[Any]) -> list[Any]:
    """Normalize a Responses ``input`` to item form, so stored history and this turn's
    input concatenate."""
    if isinstance(input_, str):
        return [{"type": "message", "role": "user", "content": input_}]
    return list(input_)


async def resolve_history_items(
    store: StateStore, identity: str, *, previous_response_id: str | None, input_: str | list[Any] | None
) -> list[Any]:
    """Prepend the conversation stored under ``previous_response_id`` to this turn's
    input. 404 if unknown, 503 if the store is unreachable — an outage must never
    masquerade as a legitimately unknown id.

    Field-based (rather than taking a ``ResponsesRequest``) so ``/v1/responses`` and
    ``/v1/responses/compact`` share this without either faking the other's request type.
    """
    this_turn = as_input_items(input_) if input_ is not None else []
    if previous_response_id is None:
        return this_turn
    if not RESPONSE_ID_RE.match(previous_response_id):
        raise _not_found_error(previous_response_id)
    try:
        snapshot = await responses_state.read_async(store, identity, previous_response_id)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable resolving previous_response_id=%s", previous_response_id)
        raise _store_unavailable_error() from None
    if snapshot is None:
        raise _not_found_error(previous_response_id)
    return [*responses_state.history_items(snapshot), *this_turn]


async def resolve_history(store: StateStore, identity: str, request: ResponsesRequest) -> list[Any]:
    """``resolve_history_items`` for a ``ResponsesRequest``."""
    return await resolve_history_items(
        store, identity, previous_response_id=request.previous_response_id, input_=request.input
    )


async def persist_response(gen, store: StateStore, *, identity: str, input_items: list[Any]):
    """Tee `respond`'s output, storing the snapshot as it passes.

    Wraps the generator ahead of `_handle_response` so that stays generic. The
    response id is read back off the output rather than minted here — the loader
    already owns it, on both paths.

    Both modes are covered by one wrapper because `_handle_response` dispatches on
    the first item's type either way: a `ResponseObject` is the whole non-streaming
    body, while a stream arrives as event dicts whose terminal event carries the
    same object. Persisting *before* yielding the terminal item is what lets a
    store failure still change what the client is told. Operates on event dicts
    directly (no SSE-string re-parsing) — `gen` is upstream of any transport framing.
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

        if not isinstance(item, dict) or item.get("type") not in TERMINAL_EVENT_TYPES:
            yield item
            continue

        response = item.get("response")
        response_id = response.get("id") if isinstance(response, dict) else None
        if not isinstance(response, dict) or not response_id:
            logger.warning("Terminal Responses event has no usable response id; not storing.")
            yield item
            continue

        try:
            await responses_state.write_async(store, identity, response_id, response=response, input_items=input_items)
        except StateStoreUnavailableError:
            logger.exception("State store unavailable persisting streamed response %s", response.get("id"))
            yield store_failure_event(
                item, "Conversation state store is unavailable; the response was generated but not stored."
            )
            return
        yield item


async def load_snapshot(store: StateStore, identity: str, response_id: str) -> dict:
    """The stored snapshot for *response_id*, scoped to the caller's identity.

    Isolation needs no comparison: another caller's identity builds a different
    key, so it simply misses and 404s.
    """
    if not RESPONSE_ID_RE.match(response_id):
        raise _not_found_error(response_id)
    try:
        snapshot = await responses_state.read_async(store, identity, response_id)
    except StateStoreUnavailableError:
        logger.exception("State store unavailable reading response %s", response_id)
        raise _store_unavailable_error() from None
    if snapshot is None:
        raise _not_found_error(response_id)
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
        raise _store_unavailable_error() from None
