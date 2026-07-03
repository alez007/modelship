import asyncio
import contextlib
import struct
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Coroutine
from typing import Any, TypeVar

from modelship.infer.infer_config import ModelshipModelConfig, RawRequestProxy
from modelship.logging import get_logger
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    ErrorInfo,
    ErrorResponse,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageVariationRequest,
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

_DISCONNECT_POLL_INTERVAL_S = 0.1

T = TypeVar("T")


class ClientDisconnectedError(Exception):
    """Raised by `BaseInfer.run_cancellable` when the client disconnects before
    the guarded work finishes."""


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

    async def run_cancellable(self, work: Coroutine[Any, Any, T], raw_request: RawRequestProxy) -> T:
        """Run `work` to completion, or cancel it and raise `ClientDisconnectedError`
        if the client disconnects first.

        A non-streaming Ray Serve call has no socket to watch: unlike streaming
        (where Starlette's own `StreamingResponse` races disconnect against the
        body iterator and cancellation propagates down through the whole chain
        automatically), a single-shot non-stream call would otherwise run to
        completion for a client that's already gone. This polls
        `RawRequestProxy.is_disconnected()` (the same cross-process
        DisconnectRegistry signal the streaming path's disconnect ultimately
        traces back to) alongside `work` and cancels whichever loses.

        Cancelling the task is often sufficient by itself — e.g. vLLM's
        `AsyncLLM.generate()` aborts its own engine-side request when its
        consuming task is cancelled, needing no extra cleanup here. Loaders
        whose engine needs cleanup beyond task cancellation (freeing a
        connection/slot, etc.) should override `on_generation_aborted`.
        """
        task = asyncio.ensure_future(work)
        watch = asyncio.ensure_future(self._poll_disconnect(raw_request))
        done, _pending = await asyncio.wait({task, watch}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watch
            return task.result()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await self.on_generation_aborted()
        raise ClientDisconnectedError

    async def on_generation_aborted(self) -> None:
        """Hook for loaders whose engine needs cleanup beyond task cancellation
        when `run_cancellable` aborts a request on client disconnect. No-op by
        default — most engines are naturally cleaned up by cancellation alone
        (or, like a blocking call already running in a thread pool, can't be
        interrupted early regardless of what happens here)."""
        return None

    @staticmethod
    async def _poll_disconnect(raw_request: RawRequestProxy) -> None:
        while True:
            if await raw_request.is_disconnected():
                return
            await asyncio.sleep(_DISCONNECT_POLL_INTERVAL_S)

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

    async def create_image_edit(
        self,
        image_data: bytes,
        mask_data: bytes | None,
        request: ImageEditRequest,
        raw_request: RawRequestProxy,
    ) -> ErrorResponse | ImageGenerationResponse:
        return _NOT_SUPPORTED

    async def create_image_variation(
        self, image_data: bytes, request: ImageVariationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ImageGenerationResponse:
        return _NOT_SUPPORTED
