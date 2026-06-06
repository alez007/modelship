"""Token-usage accounting models shared across endpoints."""

from modelship.openai.protocol.base import OpenAIBaseModel


class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: int | None = None


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0
    prompt_tokens_details: PromptTokenUsageInfo | None = None


__all__ = [
    "PromptTokenUsageInfo",
    "UsageInfo",
]
