"""
OpenAI-compatible protocol models for request/response validation.

Every backend (vllm, transformers, custom) and the API gateway import from
here instead of reaching into framework internals directly.  These are
standalone Pydantic models following the OpenAI API specification, with no
dependency on vLLM or any other inference engine.
"""

import time
import uuid
from http import HTTPStatus
from typing import Any, ClassVar, Literal

from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MASK_64_BITS = (1 << 64) - 1


def random_uuid() -> str:
    return f"{uuid.uuid4().int & _MASK_64_BITS:016x}"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorInfo(OpenAIBaseModel):
    message: str
    type: str
    param: str | None = None
    code: int


class ErrorResponse(OpenAIBaseModel):
    error: ErrorInfo


def create_error_response(
    message: str | Exception,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    param: str | None = None,
) -> ErrorResponse:
    if isinstance(message, Exception):
        exc = message
        if isinstance(exc, ValueError | TypeError | OverflowError):
            err_type = "BadRequestError"
            status_code = HTTPStatus.BAD_REQUEST
            param = None
        elif isinstance(exc, NotImplementedError):
            err_type = "NotImplementedError"
            status_code = HTTPStatus.NOT_IMPLEMENTED
            param = None
        else:
            err_type = "InternalServerError"
            status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            param = None
        message = str(exc)

    return ErrorResponse(
        error=ErrorInfo(
            message=message,
            type=err_type,
            code=status_code.value,
            param=param,
        )
    )


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: int | None = None


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0
    prompt_tokens_details: PromptTokenUsageInfo | None = None


# ---------------------------------------------------------------------------
# Chat completion — tool calls
# ---------------------------------------------------------------------------


class FunctionCall(OpenAIBaseModel):
    id: str | None = Field(default=None, exclude=True)
    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-tool-{random_uuid()}")
    type: Literal["function"] = "function"
    function: FunctionCall


class DeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(OpenAIBaseModel):
    id: str | None = None
    type: Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None


# ---------------------------------------------------------------------------
# Chat completion — logprobs
# ---------------------------------------------------------------------------


class ChatCompletionLogProb(OpenAIBaseModel):
    token: str
    logprob: float = -9999.0
    bytes: list[int] | None = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    field_names: ClassVar[set[str] | None] = None
    top_logprobs: list[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(OpenAIBaseModel):
    content: list[ChatCompletionLogProbsContent] | None = None


# ---------------------------------------------------------------------------
# Chat completion — request
# ---------------------------------------------------------------------------


class StreamOptions(OpenAIBaseModel):
    include_usage: bool | None = False
    continuous_usage_stats: bool | None = False


ChatCompletionMessageParam = dict[str, Any]


class ChatCompletionRequest(OpenAIBaseModel):
    messages: list[ChatCompletionMessageParam]
    model: str | None = None
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = False
    top_logprobs: int | None = 0
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = 1
    presence_penalty: float | None = 0.0
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = []
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    parallel_tool_calls: bool | None = True
    user: str | None = None


# ---------------------------------------------------------------------------
# Chat completion — response
# ---------------------------------------------------------------------------


class ChatMessage(OpenAIBaseModel):
    role: str
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning: str | None = None


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    logprobs: ChatCompletionLogProbs | None = None
    finish_reason: str | None = "stop"
    stop_reason: int | str | None = None


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Chat completion — streaming response
# ---------------------------------------------------------------------------


class DeltaMessage(OpenAIBaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: ChatCompletionLogProbs | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: UsageInfo | None = Field(default=None)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class EmbeddingCompletionRequest(OpenAIBaseModel):
    input: list[int] | list[list[int]] | str | list[str]
    model: str | None = None
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


# ---------------------------------------------------------------------------
# Speech-to-text (transcription)
# ---------------------------------------------------------------------------

AudioResponseFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]


class TranscriptionUsageAudio(OpenAIBaseModel):
    type: Literal["duration"] = "duration"
    seconds: int


class TranscriptionRequest(OpenAIBaseModel):
    file: UploadFile
    model: str | None = None
    language: str | None = None
    prompt: str = Field(default="")
    response_format: AudioResponseFormat = Field(default="json")
    timestamp_granularities: list[Literal["word", "segment"]] = Field(alias="timestamp_granularities[]", default=[])
    stream: bool | None = False
    temperature: float = Field(default=0.0)
    seed: int | None = None


class TranscriptionResponse(OpenAIBaseModel):
    text: str
    usage: TranscriptionUsageAudio


class TranscriptionResponseVerbose(OpenAIBaseModel):
    duration: str
    language: str
    text: str


# ---------------------------------------------------------------------------
# Speech-to-text (translation)
# ---------------------------------------------------------------------------


class TranslationRequest(OpenAIBaseModel):
    file: UploadFile
    model: str | None = None
    prompt: str = Field(default="")
    response_format: AudioResponseFormat = Field(default="json")
    stream: bool | None = False
    temperature: float = Field(default=0.0)
    seed: int | None = None
    language: str | None = None
    to_language: str | None = None


class TranslationResponse(OpenAIBaseModel):
    text: str


class TranslationResponseVerbose(OpenAIBaseModel):
    duration: str
    language: str
    text: str


# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------


class SpeechRequest(OpenAIBaseModel):
    input: str = Field(..., description="The text to generate audio for")
    model: str = Field(
        ...,
        description="The model to use for generation.",
    )
    voice: str = Field(
        ...,
        description="The voice to use for generation.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0.",
    )
    stream_format: Literal["sse", "audio"] = Field(
        default="audio",
        description="The stream format to return the audio in.",
    )


class SpeechResponse(OpenAIBaseModel):
    audio: str | None = Field(default=None, description="The generated audio data encoded in base 64")
    type: Literal["speech.audio.delta", "speech.audio.done"] = Field(
        ...,
        description="Type of audio chunk",
    )


class RawSpeechResponse(BaseModel):
    audio: bytes = Field(..., description="full audio file bytes")
    media_type: Literal["audio/wav"] = Field(default="audio/wav", description="audio bytes media type")


# ---------------------------------------------------------------------------
# Raw engine outputs
#
# Protocol-agnostic shapes returned by custom plugins. The OpenAI serving
# wrappers translate these into the OpenAI-compatible responses above. Keeping
# engines free of OpenAI concerns means a different protocol adapter (e.g.
# Anthropic, gRPC) can be added later without changing any plugin.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


class ImageGenerationRequest(OpenAIBaseModel):
    model: str = Field(..., description="The model to use for image generation.")
    prompt: str = Field(..., description="A text description of the desired image(s).")
    n: int = Field(default=1, ge=1, le=10, description="The number of images to generate.")
    size: str = Field(default="512x512", description="The size of the generated images in WxH format.")
    response_format: Literal["b64_json"] = Field(
        default="b64_json",
        description="The format in which the generated images are returned.",
    )
    num_inference_steps: int | None = Field(default=None, description="Override default inference steps.")
    guidance_scale: float | None = Field(default=None, description="Override default guidance scale.")


class ImageObject(OpenAIBaseModel):
    b64_json: str = Field(..., description="The base64-encoded JSON of the generated image.")
    revised_prompt: str | None = Field(default=None, description="The prompt that was used to generate the image.")


class ImageGenerationResponse(OpenAIBaseModel):
    created: int = Field(..., description="The Unix timestamp of when the response was created.")
    data: list[ImageObject] = Field(..., description="The list of generated images.")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "AudioResponseFormat",
    "ChatCompletionLogProb",
    "ChatCompletionLogProbs",
    "ChatCompletionLogProbsContent",
    "ChatCompletionMessageParam",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionResponseChoice",
    "ChatCompletionResponseStreamChoice",
    "ChatCompletionStreamResponse",
    "ChatMessage",
    "DeltaFunctionCall",
    "DeltaMessage",
    "DeltaToolCall",
    "EmbeddingCompletionRequest",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingResponseData",
    "ErrorInfo",
    "ErrorResponse",
    "FunctionCall",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "ImageObject",
    "OpenAIBaseModel",
    "PromptTokenUsageInfo",
    "RawChatCompletion",
    "RawChatDelta",
    "RawSegment",
    "RawSpeechResponse",
    "RawToolCall",
    "RawTranscription",
    "RawTranslation",
    "SpeechRequest",
    "SpeechResponse",
    "StreamOptions",
    "ToolCall",
    "TranscriptionRequest",
    "TranscriptionResponse",
    "TranscriptionResponseVerbose",
    "TranscriptionUsageAudio",
    "TranslationRequest",
    "TranslationResponse",
    "TranslationResponseVerbose",
    "UsageInfo",
    "create_error_response",
    "random_uuid",
]
