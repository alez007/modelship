"""Pydantic schemas for the ``/v1/responses`` endpoint (OpenAI Responses API).

Phase A is a *stateless* shape adapter: it accepts the Responses request shape
and returns the Responses response object, translating to/from the existing
chat-completion pipeline at the gateway edge (the translation logic lives in
:mod:`modelship.openai.protocol.responses.adapter`). Server-side conversation
state (``store`` / ``previous_response_id``), background execution, and OpenAI's
hosted built-in tools are out of scope here and rejected by the adapter.

Input items are kept as plain ``dict`` (matching ``ChatCompletionMessageParam``)
so the adapter, not pydantic, owns the role/content/typed-item translation.
"""

import time
from typing import Any, Literal

from pydantic import Field, model_serializer

from modelship.openai.protocol.base import OpenAIBaseModel, random_uuid

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

# Input items are typed dicts in the OpenAI spec (message / function_call /
# function_call_output / reasoning / ...). We keep them loose and translate in
# the adapter, mirroring how ChatCompletionRequest treats messages.
ResponseInputItem = dict[str, Any]


class ResponsesRequest(OpenAIBaseModel):
    input: str | list[ResponseInputItem]
    model: str | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    text: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    stream: bool | None = False
    metadata: dict[str, Any] | None = None
    user: str | None = None
    # Stateful / background features — accepted into the schema so the adapter
    # can reject them explicitly rather than have pydantic drop them silently.
    store: bool | None = None
    previous_response_id: str | None = None
    background: bool | None = None


# ---------------------------------------------------------------------------
# Output items
# ---------------------------------------------------------------------------


class ResponseOutputText(OpenAIBaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[Any] = Field(default_factory=list)


class ResponseOutputMessage(OpenAIBaseModel):
    type: Literal["message"] = "message"
    id: str = Field(default_factory=lambda: f"msg_{random_uuid()}")
    role: Literal["assistant"] = "assistant"
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    content: list[ResponseOutputText] = Field(default_factory=list)


class ResponseReasoningText(OpenAIBaseModel):
    type: Literal["reasoning_text"] = "reasoning_text"
    text: str


class ResponseReasoningSummary(OpenAIBaseModel):
    type: Literal["summary_text"] = "summary_text"
    text: str


class ResponseReasoningItem(OpenAIBaseModel):
    type: Literal["reasoning"] = "reasoning"
    id: str = Field(default_factory=lambda: f"rs_{random_uuid()}")
    summary: list[ResponseReasoningSummary] = Field(default_factory=list)
    content: list[ResponseReasoningText] = Field(default_factory=list)
    encrypted_content: str | None = None

    @model_serializer(mode="wrap")
    def _omit_unset_encrypted_content(self, handler: Any) -> dict[str, Any]:
        # Spec types encrypted_content as a plain (non-nullable) string that's not
        # required — the key must be absent, not null, when there's no value.
        dumped = handler(self)
        if dumped.get("encrypted_content") is None:
            dumped.pop("encrypted_content", None)
        return dumped


class ResponseFunctionToolCall(OpenAIBaseModel):
    type: Literal["function_call"] = "function_call"
    id: str = Field(default_factory=lambda: f"fc_{random_uuid()}")
    call_id: str = Field(default_factory=lambda: f"call_{random_uuid()}")
    name: str
    arguments: str
    status: Literal["in_progress", "completed", "incomplete"] = "completed"


# Discriminated only by the literal ``type``; kept as a plain union since we
# always construct these ourselves rather than parse them from the wire.
ResponseOutputItem = ResponseReasoningItem | ResponseOutputMessage | ResponseFunctionToolCall


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class ResponseOutputTokensDetails(OpenAIBaseModel):
    reasoning_tokens: int = 0


class ResponseInputTokensDetails(OpenAIBaseModel):
    cached_tokens: int = 0


class ResponseUsage(OpenAIBaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: ResponseInputTokensDetails | None = None
    output_tokens_details: ResponseOutputTokensDetails | None = None


# ---------------------------------------------------------------------------
# Response object
# ---------------------------------------------------------------------------


class ResponseObject(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"resp_{random_uuid()}")
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    completed_at: int | None = None
    status: Literal["completed", "failed", "incomplete", "in_progress"] = "completed"
    model: str
    output: list[ResponseOutputItem] = Field(default_factory=list)
    usage: ResponseUsage | None = None
    # Echoed request settings (OpenAI returns these on the response object).
    instructions: str | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_logprobs: int = 0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] = "auto"
    parallel_tool_calls: bool = True
    text: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    truncation: str = "disabled"
    background: bool = False
    service_tier: str = "default"
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    previous_response_id: str | None = None
    store: bool = False
    error: Any | None = None
    incomplete_details: Any | None = None


__all__ = [
    "ResponseFunctionToolCall",
    "ResponseInputItem",
    "ResponseInputTokensDetails",
    "ResponseObject",
    "ResponseOutputItem",
    "ResponseOutputMessage",
    "ResponseOutputText",
    "ResponseOutputTokensDetails",
    "ResponseReasoningItem",
    "ResponseReasoningSummary",
    "ResponseReasoningText",
    "ResponseUsage",
    "ResponsesRequest",
]
