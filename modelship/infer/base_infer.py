import struct
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from modelship.infer.infer_config import ModelshipModelConfig, RawRequestProxy
from modelship.logging import get_logger
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    ErrorInfo,
    ErrorResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    RawSpeechResponse,
    SpeechRequest,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
)

logger = get_logger("infer")

_NOT_SUPPORTED = ErrorResponse(
    error=ErrorInfo(message="model does not support this action", type="invalid_request_error")
)
_NOT_SUPPORTED._http_status = 404

# 44-byte WAV header + 2 bytes of silence (one 16-bit sample at 16 kHz mono)
_MINIMAL_WAV_HEADER = struct.pack(
    "<4sI4s4sIHHIIHH4sI",
    b"RIFF",
    36 + 2,
    b"WAVE",  # RIFF chunk
    b"fmt ",
    16,
    1,
    1,
    16000,
    32000,
    2,
    16,  # fmt sub-chunk: PCM, mono, 16 kHz, 16-bit
    b"data",
    2,  # data sub-chunk: 2 bytes
)
MINIMAL_WAV = _MINIMAL_WAV_HEADER + b"\x00\x00"


class BaseInfer(ABC):
    def __init__(self, model_config: ModelshipModelConfig):
        self.model_config = model_config
        self.max_context_length: int | None = None

    def _get_memory_fraction(self) -> float | None:
        """Return the GPU memory fraction if explicitly set and < 1.0, otherwise None."""
        if self.model_config.num_gpus > 0 and self.model_config.num_gpus < 1.0:
            return self.model_config.num_gpus
        return None

    def _set_max_context_length(self, length: int | None) -> None:
        self.max_context_length = length
        logger.info("max_context_length for %s: %s", self.model_config.name, self.max_context_length)

    @abstractmethod
    def shutdown(self) -> None:
        """Synchronously release resources (engine processes, GPU memory, etc.).

        Called during graceful teardown. Subclasses must implement to clean up
        loader-specific resources.
        """

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def warmup(self) -> None:
        """Run a minimal inference pass to warm up the model (CUDA kernels, caches, etc.).

        Subclasses should override this to send a tiny dummy request through
        their actual inference path. The default is a no-op for loaders that
        don't need warmup.
        """

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        return _NOT_SUPPORTED

    async def create_embedding(self, request: EmbeddingRequest, raw_request: RawRequestProxy) -> ErrorResponse:
        return _NOT_SUPPORTED

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose | AsyncGenerator[str, None]:
        return _NOT_SUPPORTED

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranslationResponse | TranslationResponseVerbose | AsyncGenerator[str, None]:
        return _NOT_SUPPORTED

    async def create_speech(
        self, request: SpeechRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | RawSpeechResponse | AsyncGenerator[str, None]:
        return _NOT_SUPPORTED

    async def create_image_generation(
        self, request: ImageGenerationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ImageGenerationResponse:
        return _NOT_SUPPORTED
