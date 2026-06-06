"""Protocol-agnostic shapes returned by custom plugins.

The OpenAI serving wrappers translate these into the OpenAI-compatible
responses elsewhere in this package. Keeping engines free of OpenAI concerns
means a different protocol adapter (e.g. Anthropic, gRPC) can be added later
without changing any plugin.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RawToolCall(BaseModel):
    id: str
    name: str
    arguments: str = Field(..., description="JSON-encoded arguments as produced by the engine")


class RawChatCompletion(BaseModel):
    text: str
    tool_calls: list[RawToolCall] = Field(default_factory=list)
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class RawChatDelta(BaseModel):
    text: str | None = None
    tool_call: RawToolCall | None = None
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = None


class RawSegment(BaseModel):
    text: str
    start: float = Field(..., description="start time in seconds")
    end: float = Field(..., description="end time in seconds")


class RawTranscription(BaseModel):
    text: str
    language: str | None = None
    duration_seconds: float | None = None
    segments: list[RawSegment] = Field(default_factory=list)


RawTranslation = RawTranscription


__all__ = [
    "RawChatCompletion",
    "RawChatDelta",
    "RawSegment",
    "RawToolCall",
    "RawTranscription",
    "RawTranslation",
]
