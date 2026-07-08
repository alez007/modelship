import asyncio
import contextlib
import struct
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any, TypeVar

from ray.exceptions import RayActorError

from modelship.infer import infer_config
from modelship.infer.infer_config import ModelshipModelConfig, RawRequestProxy
from modelship.logging import get_logger
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    EmbeddingRequest,
    ErrorInfo,
    ErrorResponse,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageVariationRequest,
    RawSpeechResponse,
    ResponseObject,
    ResponsesRequest,
    SpeechRequest,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
)
from modelship.openai.protocol.responses.streaming import ResponsesStreamTranslator

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
        # request_id -> local event, set by the shared disconnect pump below.
        # One pump per replica (this instance) amortizes disconnect polling
        # across every request the replica is currently serving, instead of
        # each request polling the DisconnectRegistry actor independently.
        self._watched: dict[str, asyncio.Event] = {}
        self._pump_task: asyncio.Task[None] | None = None

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
        event = self._watch_disconnect(raw_request)
        task = asyncio.ensure_future(work)
        watch = asyncio.ensure_future(event.wait())
        try:
            done, _pending = await asyncio.wait({task, watch}, return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                return task.result()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await self.on_generation_aborted()
            raise ClientDisconnectedError
        finally:
            # Unconditional, regardless of how the try exits — including this
            # coroutine's own task being cancelled from outside (e.g. replica
            # shutdown) while suspended in asyncio.wait above, which otherwise
            # leaves `task` (the actual inference work) running unobserved.
            watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watch
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            self._unwatch_disconnect(raw_request)

    async def run_cancellable_stream(
        self, work: AsyncGenerator[T, None], raw_request: RawRequestProxy
    ) -> AsyncGenerator[T, None]:
        """Streaming counterpart of `run_cancellable`.

        Races each pulled item against the same disconnect signal: cancelling
        the in-flight `__anext__()` call delivers `CancelledError` straight into
        `work`'s currently-suspended frame (and transitively into whatever it's
        awaiting, e.g. an engine's own generator), the same way cancelling a
        plain task does for `run_cancellable`. The `finally` block's `aclose()`
        is what actually guarantees `work` is closed on every exit path —
        including the consumer closing *this* generator early (`GeneratorExit`
        propagating out of the `yield`), which the disconnect branch's own
        `aclose()` above doesn't cover. It's a defensive no-op wherever `work`
        already self-terminated.

        `next_item` is tracked outside the loop so `finally` can reach it: if
        this generator is torn down (cancelled or `aclose()`d) while suspended
        in the `asyncio.wait` below rather than at the `yield`, `next_item`'s
        `__anext__()` call is still in flight and still owns `work`'s frame —
        calling `work.aclose()` before that settles raises `RuntimeError:
        aclose(): asynchronous generator is already running`. It must be
        cancelled and awaited first, same as the disconnect branch above does.
        """
        event = self._watch_disconnect(raw_request)
        watch = asyncio.ensure_future(event.wait())
        next_item: asyncio.Task[T] | None = None
        try:
            while True:
                next_item = asyncio.ensure_future(work.__anext__())
                done, _pending = await asyncio.wait({next_item, watch}, return_when=asyncio.FIRST_COMPLETED)
                if next_item in done:
                    try:
                        item = next_item.result()
                    except StopAsyncIteration:
                        return
                    yield item
                    continue
                next_item.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await next_item
                await work.aclose()
                await self.on_generation_aborted()
                raise ClientDisconnectedError
        finally:
            watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watch
            self._unwatch_disconnect(raw_request)
            if next_item is not None and not next_item.done():
                next_item.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await next_item
            await work.aclose()

    async def _stream_responses(
        self,
        request: ResponsesRequest,
        chunks: AsyncGenerator[ChatCompletionStreamResponse, None],
        *,
        request_id: str,
        client_error: Callable[[Exception], str | None] = lambda _exc: None,
    ) -> AsyncGenerator[str, None]:
        """Drive a Responses SSE event stream from a loader's typed chat chunks.

        Shared by every loader's `create_response`: each supplies its own native
        chunk source (already wrapped in `run_cancellable_stream`) and this owns
        the `ResponsesStreamTranslator` lifecycle — start, one `process()` per
        chunk, then `finish()` on a clean end or `fail()` on an error.
        `ClientDisconnectedError` (raised by `run_cancellable_stream` once the
        client is gone) ends the stream silently, matching every other endpoint.
        Any other exception is passed to `client_error`, which returns a
        client-safe message to report via `fail()`, or `None` to log a full
        stack trace and report a generic "Internal error during generation".

        The `finally` block's `chunks.aclose()` is what guarantees `chunks` (and
        transitively the native engine/subprocess stream it wraps) is torn down
        even when neither of the `except` branches runs — e.g. the consumer
        (Starlette/Ray Serve) closes *this* generator early while it's suspended
        at one of the `yield`s below. `async for` does not propagate that closure
        into the generator being iterated the way `yield from` would for a
        synchronous generator, so without this `chunks` would otherwise be left
        suspended forever, leaking its disconnect-poll task and underlying stream.
        """
        translator = ResponsesStreamTranslator(request)
        try:
            for event in translator.start():
                yield event
            try:
                async for chunk in chunks:
                    for event in translator.process(chunk):
                        yield event
            except ClientDisconnectedError:
                logger.info("responses request %s aborted: client disconnected", request_id)
                return
            except Exception as exc:
                message = client_error(exc)
                if message is None:
                    logger.exception("responses request %s failed mid-stream", request_id)
                    message = "Internal error during generation"
                for event in translator.fail(message):
                    yield event
                return
            for event in translator.finish():
                yield event
        finally:
            await chunks.aclose()

    async def on_generation_aborted(self) -> None:
        """Hook for loaders whose engine needs cleanup beyond task cancellation
        when `run_cancellable` aborts a request on client disconnect. No-op by
        default — most engines are naturally cleaned up by cancellation alone
        (or, like a blocking call already running in a thread pool, can't be
        interrupted early regardless of what happens here)."""
        return None

    def _watch_disconnect(self, raw_request: RawRequestProxy) -> asyncio.Event:
        """Register `raw_request` with the shared per-replica disconnect pump and
        return a local event that fires once it disconnects. Unwatchable proxies
        (no registry/id — e.g. an internal warmup request) get an event that
        simply never fires, matching `is_disconnected()`'s "always connected"
        behavior for them.
        """
        event = asyncio.Event()
        if not raw_request.is_watchable:
            return event
        assert raw_request.request_id is not None
        self._watched[raw_request.request_id] = event
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = asyncio.ensure_future(self._disconnect_pump())
        return event

    def _unwatch_disconnect(self, raw_request: RawRequestProxy) -> None:
        """Drop `raw_request` from the pump's watch set. Stops the pump once
        nothing is left to poll, rather than leaving it spinning idle. Mirrors
        `_watch_disconnect`'s `is_watchable` guard — an unwatchable proxy was
        never registered, so there's nothing to remove."""
        if not raw_request.is_watchable:
            return
        assert raw_request.request_id is not None
        self._watched.pop(raw_request.request_id, None)
        if not self._watched and self._pump_task is not None:
            self._pump_task.cancel()
            self._pump_task = None

    async def _disconnect_pump(self) -> None:
        """One background poller per replica, shared by every in-flight request's
        `run_cancellable`/`run_cancellable_stream` call. Batches what would
        otherwise be one DisconnectRegistry RPC per request per poll interval
        into a single `is_set_many` RPC per interval, fanning results out to
        each request's local event.
        """
        while self._watched:
            disconnected = await self._poll_disconnected_ids(list(self._watched))
            for request_id in disconnected:
                event = self._watched.get(request_id)
                if event is not None:
                    event.set()
            await asyncio.sleep(_DISCONNECT_POLL_INTERVAL_S)

    @staticmethod
    async def _poll_disconnected_ids(request_ids: list[str]) -> list[str]:
        """Injectable seam for `_disconnect_pump`: which of `request_ids` are
        disconnected right now, per the shared DisconnectRegistry. Degrades to
        "none disconnected" on a dead registry, same as `is_disconnected()`."""
        try:
            return await infer_config.get_disconnect_registry().is_set_many.remote(request_ids)
        except RayActorError:
            logger.warning("Disconnect registry unavailable; assuming clients connected")
            infer_config.reset_disconnect_registry()
            return []

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

    async def create_response(
        self, request: ResponsesRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ResponseObject | AsyncGenerator[str, None]:
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
