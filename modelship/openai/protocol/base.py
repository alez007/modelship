"""Shared base model and helpers for the OpenAI protocol schemas."""

import uuid

from pydantic import BaseModel, ConfigDict

_MASK_64_BITS = (1 << 64) - 1


def random_uuid() -> str:
    return f"{uuid.uuid4().int & _MASK_64_BITS:016x}"


class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


__all__ = [
    "OpenAIBaseModel",
    "random_uuid",
]
