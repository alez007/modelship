"""
OpenAI-compatible protocol models for request/response validation.

Every backend (vllm, transformers, custom) and the API gateway import from
here instead of reaching into framework internals directly.  These are
standalone Pydantic models following the OpenAI API specification, with no
dependency on vLLM or any other inference engine.

The schemas are split across submodules by endpoint family; this package
re-exports the full public surface so ``from modelship.openai.protocol import
X`` keeps working regardless of which submodule ``X`` now lives in.
"""

from modelship.openai.protocol.audio import (
    AudioResponseFormat,
    RawSpeechResponse,
    SpeechRequest,
    SpeechResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranscriptionSegment,
    TranscriptionUsageAudio,
    TranscriptionWord,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
)
from modelship.openai.protocol.base import OpenAIBaseModel, random_uuid
from modelship.openai.protocol.chat import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionMessageParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    StreamOptions,
    ToolCall,
)
from modelship.openai.protocol.embeddings import (
    EmbeddingCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
)
from modelship.openai.protocol.error import (
    ErrorInfo,
    ErrorResponse,
    create_error_response,
)
from modelship.openai.protocol.images import (
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageObject,
    ImageVariationRequest,
)
from modelship.openai.protocol.raw import (
    RawChatCompletion,
    RawChatDelta,
    RawSegment,
    RawToolCall,
    RawTranscription,
    RawTranslation,
)
from modelship.openai.protocol.responses import (
    ResponseFunctionToolCall,
    ResponseInputItem,
    ResponseInputTokensDetails,
    ResponseObject,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseOutputTokensDetails,
    ResponseReasoningItem,
    ResponseReasoningSummary,
    ResponseReasoningText,
    ResponsesRequest,
    ResponseUsage,
)
from modelship.openai.protocol.usage import PromptTokenUsageInfo, UsageInfo

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
    "ImageEditRequest",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "ImageObject",
    "ImageVariationRequest",
    "OpenAIBaseModel",
    "PromptTokenUsageInfo",
    "RawChatCompletion",
    "RawChatDelta",
    "RawSegment",
    "RawSpeechResponse",
    "RawToolCall",
    "RawTranscription",
    "RawTranslation",
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
    "SpeechRequest",
    "SpeechResponse",
    "StreamOptions",
    "ToolCall",
    "TranscriptionRequest",
    "TranscriptionResponse",
    "TranscriptionResponseVerbose",
    "TranscriptionSegment",
    "TranscriptionUsageAudio",
    "TranscriptionWord",
    "TranslationRequest",
    "TranslationResponse",
    "TranslationResponseVerbose",
    "UsageInfo",
    "create_error_response",
    "random_uuid",
]
