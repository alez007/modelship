"""Embeddings request/response schemas."""

import time
from typing import Literal

from pydantic import Field

from modelship.openai.protocol.base import OpenAIBaseModel, random_uuid
from modelship.openai.protocol.usage import UsageInfo


class EmbeddingCompletionRequest(OpenAIBaseModel):
    input: list[int] | list[list[int]] | str | list[str]
    model: str = Field(..., description="ID of the model to use.")
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None


EmbeddingRequest = EmbeddingCompletionRequest


class EmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str = "embedding"
    embedding: list[float] | str


class EmbeddingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None = None
    data: list[EmbeddingResponseData]
    usage: UsageInfo


__all__ = [
    "EmbeddingCompletionRequest",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingResponseData",
]
