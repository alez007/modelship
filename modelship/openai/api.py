import asyncio
import os
import time
from http import HTTPStatus
from typing import Annotated, Any, cast

import ray
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from ray import serve
from ray.exceptions import RayTaskError
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from modelship.infer import deploy_coordinator
from modelship.infer.infer_config import RequestWatcher
from modelship.logging import get_logger
from modelship.metrics import (
    MODELS_LOADED,
    REQUEST_DURATION_SECONDS,
    REQUEST_ERRORS_TOTAL,
    REQUEST_IN_PROGRESS,
    REQUEST_TOTAL,
    STREAM_CHUNKS_TOTAL,
)
from modelship.openai.auth import ApiKeyMiddleware, get_api_keys
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
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
    TranslationRequest,
    TranslationResponse,
    create_error_response,
)
from modelship.openai.protocol.chat import StreamOptions
from modelship.openai.protocol.responses import (
    UnsupportedResponsesFeatureError,
    responses_from_chat,
    responses_request_to_chat,
    responses_stream_from_chat,
)
from modelship.utils import random_uuid

logger = get_logger("api")

_DEFAULT_MAX_BODY_BYTES = 50 * 1024 * 1024  # 50 MB
# Backoff before retrying the gateway watch loop after a transient coordinator error.
_WATCH_RETRY_S = 5.0


class PayloadSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_bytes: int = _DEFAULT_MAX_BODY_BYTES):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large (limit: {self.max_bytes} bytes)"},
            )
        return await call_next(request)


def build_app():
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    max_body_bytes = int(os.environ.get("MSHIP_MAX_REQUEST_BODY_BYTES", _DEFAULT_MAX_BODY_BYTES))
    app.add_middleware(PayloadSizeLimitMiddleware, max_bytes=max_body_bytes)
    logger.info("Payload size limit: %d bytes", max_body_bytes)

    api_keys = get_api_keys()
    if api_keys:
        app.add_middleware(ApiKeyMiddleware, api_keys=api_keys)
        logger.info("API key authentication enabled (%d key(s))", len(api_keys))
    else:
        logger.warning("API key authentication disabled (MSHIP_API_KEYS not set)")

    @app.exception_handler(RequestValidationError)
    async def log_validation_error(request: Request, exc: RequestValidationError):
        logger.warning("%s %s -> 422 validation error: %s", request.method, request.url.path, exc.errors())
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    @app.exception_handler(HTTPException)
    async def log_http_exception(request: Request, exc: HTTPException):
        logger.warning("%s %s -> %s: %s", request.method, request.url.path, exc.status_code, exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def log_unhandled_exception(request: Request, exc: Exception):
        logger.exception("%s %s -> 500: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app


app = build_app()


class OpenAiModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "modelship"


class OpenaiModelList(BaseModel):
    object: str = "list"
    data: list[OpenAiModelCard] = []


def _error_response(result: ErrorResponse) -> JSONResponse:
    return JSONResponse(content=result.model_dump(mode="json"), status_code=result._http_status)


def _validation_error_from_cause(cause: BaseException) -> ErrorResponse:
    # OpenAI-style 400 for client-side validation failures (e.g. vLLM's
    # VLLMValidationError on context overflow, which subclasses ValueError).
    base = cause.args[0] if cause.args else str(cause)
    return create_error_response(
        message=base,
        err_type="invalid_request_error",
        status_code=HTTPStatus.BAD_REQUEST,
        param=getattr(cause, "parameter", None),
    )


@serve.deployment
@serve.ingress(app)
class ModelshipAPI:
    def __init__(self, gateway_name: str):
        # model_name -> (app_name -> handle). The inner dict is keyed by app_name
        # so a specific deployment can be dropped by name in remove_deployments.
        self.models: dict[str, dict[str, DeploymentHandle]] = {}
        self._round_robin: dict[str, int] = {}
        self.model_list: list[OpenAiModelCard] = []
        self.expected_models: list[str] = []
        self._started_at = time.time()
        self._gateway_name = gateway_name
        # Routing state is reconciled from the coordinator, the cluster-wide source
        # of truth — not pushed by the driver (a push hits only one replica). Each
        # replica runs a watch loop (started lazily on first request) that pulls a
        # snapshot whenever the coordinator's per-gateway generation advances, so
        # every replica — including restarted / autoscaled ones — converges.
        self._gen = 0  # last coordinator generation this replica reconciled to
        self._watch_task: asyncio.Task | None = None
        self._coordinator = None  # cached coordinator handle
        # Timing state — the first sync with a non-empty expected set stamps a start;
        # each model's first appearance records the gap since the previous arrival as
        # an approximate load duration.
        self._expected_set_at: float | None = None
        self._last_model_at: float | None = None
        self._all_ready_at: float | None = None
        self._model_load_times: dict[str, float] = {}

    def _register_deployment(self, app_name: str, model_name: str) -> bool:
        """Wire one deployment handle into the routing table. Returns True iff the
        model was newly added. Sync so the reconcile path applies atomically."""
        try:
            handle = serve.get_app_handle(app_name)
        except Exception:
            logger.exception("Failed to get handle for app: %s", app_name)
            return False

        newly_added = model_name not in self.models
        if newly_added:
            self.models[model_name] = {}
            self._round_robin[model_name] = 0
            self.model_list.append(OpenAiModelCard(id=model_name))
        self.models[model_name][app_name] = handle
        logger.info("Registered deployment: %s (model: %s)", app_name, model_name)
        return newly_added

    def _drop_apps(self, app_names: list[str]) -> list[str]:
        """Drop the given deployment app names from the routing tables. The owning
        model is found by reverse lookup. When a model loses its last deployment its
        model entry, card, round-robin counter, expected-models entry, and load-time
        entry are also dropped. Returns the names of fully-removed models."""
        removed_models: list[str] = []
        for app_name in app_names:
            model_name = next((m for m, handles in self.models.items() if app_name in handles), None)
            if model_name is None:
                continue
            handles = self.models[model_name]
            del handles[app_name]
            logger.info("Unregistered deployment: %s (model: %s)", app_name, model_name)
            if not handles:
                del self.models[model_name]
                self._round_robin.pop(model_name, None)
                self.model_list = [c for c in self.model_list if c.id != model_name]
                self._model_load_times.pop(model_name, None)
                self.expected_models = [m for m in self.expected_models if m != model_name]
                removed_models.append(model_name)
        return removed_models

    def _apply_routing(self, desired: dict[str, str], *, allow_removals: bool) -> None:
        """Reconcile the routing table to `desired` ({app_name: model_name}): add
        handles for newly-present apps and, when `allow_removals`, drop apps no
        longer present. Sync / await-free → atomic w.r.t. in-flight requests."""
        routed = {app for handles in self.models.values() for app in handles}

        for app_name, model_name in desired.items():
            if app_name in routed or not self._register_deployment(app_name, model_name):
                continue
            base = self._last_model_at or self._started_at
            self._model_load_times[model_name] = round(time.time() - base, 2)
            self._last_model_at = time.time()

        if allow_removals:
            stale = [app for app in routed if app not in desired]
            if stale:
                self._drop_apps(stale)

    def _apply_snapshot(self, snapshot: dict) -> None:
        """Apply a coordinator routing snapshot to this replica (atomic mutation).

        Removals are honored only when the generation advances. A *lower*
        generation means the coordinator lost state (restart) — we still adopt its
        additions but never let it blank live routing; a genuine change always
        advances the generation, so real removals propagate immediately."""
        new_gen = snapshot.get("generation", self._gen)
        self._apply_routing(snapshot.get("models", {}), allow_removals=new_gen >= self._gen)
        # Prefer the coordinator's explicit expected set; fall back to the live set
        # so a restarted coordinator (empty expected) doesn't flip us to not-ready.
        self.expected_models = snapshot.get("expected") or sorted(self.models)
        if self.expected_models and self._expected_set_at is None:
            self._expected_set_at = self._last_model_at or time.time()
        if self.expected_models and self._all_ready_at is None and all(m in self.models for m in self.expected_models):
            self._all_ready_at = time.time()
        self._gen = new_gen
        MODELS_LOADED.set(len(self.models))

    def _coord(self):
        if self._coordinator is None:
            self._coordinator = deploy_coordinator.get_or_create_coordinator()
        return self._coordinator

    async def _coord_async(self):
        """Resolve (and cache) the coordinator handle without blocking the event loop.
        Cached fast path is a no-op; only after a reset (coordinator restart) does this
        do work, and get_or_create's synchronous ray.get_actor can stall on a
        recovering GCS — so hop it to a thread to keep concurrent requests flowing."""
        if self._coordinator is None:
            self._coordinator = await asyncio.to_thread(deploy_coordinator.get_or_create_coordinator)
        return self._coordinator

    def _ensure_watching(self) -> None:
        """First-request hook: do one synchronous sync so this request isn't blocked
        on the loop's first tick, then start the background watch loop (once)."""
        if self._watch_task is not None:
            return
        self._sync_routing_blocking()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no running loop yet; a later in-loop call starts it
        self._watch_task = loop.create_task(self._watch_loop())

    def _sync_routing_blocking(self) -> bool:
        """One-shot synchronous reconcile (first request). Blocking ray.get is fine
        here — it runs at most once per request until the loop is established."""
        try:
            snapshot = cast(dict, ray.get(self._coord().get_routing.remote(self._gateway_name)))
        except Exception:
            self._coordinator = None  # re-resolve next time in case the handle went stale
            logger.debug("gateway: initial routing sync deferred; coordinator unavailable", exc_info=True)
            return False
        self._apply_snapshot(snapshot)
        return True

    async def _watch_loop(self) -> None:
        """Long-poll the coordinator and reconcile on every generation change. The
        await happens on the actor call; the reconcile mutation is await-free so it
        stays atomic w.r.t. in-flight requests."""
        while True:
            try:
                coord = await self._coord_async()
                gen = await coord.wait_for_change.remote(self._gateway_name, self._gen)
                if gen != self._gen:
                    snapshot = await coord.get_routing.remote(self._gateway_name)
                    self._apply_snapshot(snapshot)
            except asyncio.CancelledError:
                return
            except Exception:
                self._coordinator = None
                logger.debug("gateway: watch iteration failed; retrying", exc_info=True)
                await asyncio.sleep(_WATCH_RETRY_S)

    def __del__(self):
        task = getattr(self, "_watch_task", None)
        if task is not None and not task.done():
            task.cancel()

    async def list_deployments(self) -> dict[str, list[str]]:
        """Return model_name -> list of deployment app_names currently registered."""
        return {model_name: list(handles.keys()) for model_name, handles in self.models.items()}

    @staticmethod
    def _set_request_id(request_id: str) -> None:
        from modelship.logging import request_id_var

        request_id_var.set(request_id)

    def _get_handle(self, model_name: str | None) -> DeploymentHandle:
        self._ensure_watching()
        if model_name is None or model_name not in self.models:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value, detail="model not found")
        handles = list(self.models[model_name].values())
        idx = self._round_robin[model_name] % len(handles)
        self._round_robin[model_name] += 1
        return handles[idx]

    async def _handle_response(
        self,
        response_gen,
        watcher: RequestWatcher,
        model: str,
        endpoint: str,
        stream_media_type: str = "text/event-stream",
    ):
        start = time.monotonic()
        REQUEST_IN_PROGRESS.set(1, tags={"model": model, "endpoint": endpoint})
        try:
            try:
                first = await response_gen.__anext__()
            except RayTaskError as e:
                # Loader code raised across the Ray boundary. Treat ValueError-family
                # causes (e.g. vLLM's VLLMValidationError on context overflow) as
                # OpenAI-style 400s rather than masking them as 500s.
                cause = e.cause if isinstance(e.cause, BaseException) else None
                if isinstance(cause, ValueError | TypeError | OverflowError):
                    REQUEST_ERRORS_TOTAL.inc(
                        tags={"model": model, "endpoint": endpoint, "error_type": "validation_error"}
                    )
                    REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
                    logger.info("Validation error for model=%s: %s", model, cause)
                    watcher.stop()
                    return _error_response(_validation_error_from_cause(cause))
                REQUEST_ERRORS_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "error_type": "unhandled"})
                REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
                logger.exception("Initial response generation failed for model=%s", model)
                watcher.stop()
                return JSONResponse(status_code=500, content={"detail": str(e)})
            except Exception as e:
                # Catch failures during initial generator creation or the very first yield
                REQUEST_ERRORS_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "error_type": "unhandled"})
                REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
                logger.exception("Initial response generation failed for model=%s", model)
                watcher.stop()
                return JSONResponse(status_code=500, content={"detail": str(e)})

            if isinstance(first, ErrorResponse):
                REQUEST_ERRORS_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "error_type": "inference_error"})
                REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
                watcher.stop()
                return _error_response(first)

            if isinstance(first, Response):
                REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "ok"})
                watcher.stop()
                return first

            if isinstance(first, RawSpeechResponse):
                REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "ok"})
                watcher.stop()
                return Response(content=first.audio, media_type=first.media_type)

            if isinstance(
                first,
                ChatCompletionResponse
                | ResponseObject
                | EmbeddingResponse
                | TranscriptionResponse
                | TranslationResponse
                | ImageGenerationResponse,
            ):
                REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "ok"})
                watcher.stop()
                return JSONResponse(content=first.model_dump(mode="json"))

            # streaming — first chunk already consumed, chain it back
            async def _stream():
                try:
                    STREAM_CHUNKS_TOTAL.inc(tags={"model": model})
                    yield first
                    async for chunk in response_gen:
                        STREAM_CHUNKS_TOTAL.inc(tags={"model": model})
                        yield chunk
                    REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "ok"})
                except Exception:
                    REQUEST_ERRORS_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "error_type": "stream_error"})
                    REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
                    raise
                finally:
                    watcher.stop()

            return StreamingResponse(content=_stream(), media_type=stream_media_type)
        except Exception:
            # Fallback for anything else not caught above
            REQUEST_ERRORS_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "error_type": "unhandled"})
            REQUEST_TOTAL.inc(tags={"model": model, "endpoint": endpoint, "status": "error"})
            raise
        finally:
            duration = time.monotonic() - start
            REQUEST_DURATION_SECONDS.observe(duration, tags={"model": model, "endpoint": endpoint})
            REQUEST_IN_PROGRESS.set(0, tags={"model": model, "endpoint": endpoint})

    @app.get("/health")
    async def health(self):
        return {
            "status": "ok",
            "uptime_s": round(time.time() - self._started_at, 1),
        }

    def _readyz_body(self) -> dict:
        self._ensure_watching()
        expected = list(self.expected_models)
        pending = [m for m in expected if m not in self.models] if expected else []
        ready = bool(expected) and len(pending) == 0
        time_to_ready: float | None = None
        if self._all_ready_at is not None and self._expected_set_at is not None:
            time_to_ready = round(self._all_ready_at - self._expected_set_at, 2)
        return {
            "status": "ok",
            "ready": ready,
            "uptime_s": round(time.time() - self._started_at, 1),
            "time_to_ready_s": time_to_ready,
            "models_loaded": sorted(self.models.keys()),
            "models_expected": expected,
            "models_pending": pending,
            "model_load_times_s": dict(self._model_load_times),
        }

    @app.get("/readyz")
    async def readyz(self):
        body = self._readyz_body()
        if body["ready"]:
            return body
        return JSONResponse(status_code=503, content=body)

    @app.get("/v1/models", response_model=OpenaiModelList)
    async def list_models(self):
        self._ensure_watching()
        return OpenaiModelList(data=self.model_list)

    @app.get("/v1/models/{model}", response_model=OpenAiModelCard)
    async def model_info(self, model: str) -> OpenAiModelCard:
        self._ensure_watching()
        for card in self.model_list:
            if card.id == model:
                return card
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value, detail="model not found")

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        model = request.model or ""
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, model=model, endpoint="create_chat_completion")
        headers = dict(raw_request.headers)
        # Materialize any lazy pydantic ValidatorIterators (from Iterable-typed fields
        # like tool_calls) in place — they can't be pickled across the Ray boundary.
        # Do NOT re-validate via model_validate/model_validate_json as pydantic will
        # re-wrap Iterable fields in a new ValidatorIterator.
        for msg in request.messages:
            if isinstance(msg, dict) and "tool_calls" in msg:
                tc = msg["tool_calls"]
                if not isinstance(tc, list):
                    msg["tool_calls"] = list(tc)  # type: ignore[arg-type]
        logger.info(
            "chat_completion model=%s messages=%d stream=%s max_tokens=%s",
            model,
            len(request.messages),
            request.stream,
            request.max_tokens,
        )
        logger.debug("chat_completion full request: %s", request.model_dump_json())
        response_gen = handle.generate.options(stream=True).remote(request, headers, watcher.event, req_id)
        return await self._handle_response(response_gen, watcher, model, "create_chat_completion")

    @app.post("/v1/responses")
    async def create_response(self, request: ResponsesRequest, raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        model = request.model or ""
        try:
            chat_request = responses_request_to_chat(request)
        except UnsupportedResponsesFeatureError as e:
            return _error_response(create_error_response(e))
        except ValidationError as e:
            # Translating into ChatCompletionRequest can surface invalid params
            # (e.g. a bad reasoning.effort value) as a pydantic ValidationError,
            # which is not a ValueError — return a 400 rather than a generic 500.
            return _error_response(_validation_error_from_cause(e))

        if request.stream:
            # Drive the chat pipeline in streaming mode and translate its SSE
            # chunks into the Responses event protocol. include_usage so the
            # terminal response.completed event carries token counts.
            chat_request.stream = True
            chat_request.stream_options = StreamOptions(include_usage=True)

        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, model=model, endpoint="create_response")
        headers = dict(raw_request.headers)
        logger.info(
            "responses model=%s input_items=%s max_output_tokens=%s stream=%s",
            model,
            1 if isinstance(request.input, str) else len(request.input),
            request.max_output_tokens,
            bool(request.stream),
        )
        # Under stream=True the remote() result is an async-iterable
        # DeploymentResponseGenerator; Ray's stub widens it to a union because it
        # doesn't overload on the stream literal, so narrow it explicitly.
        response_gen = cast(
            "DeploymentResponseGenerator[Any]",
            handle.generate.options(stream=True).remote(chat_request, headers, watcher.event, req_id),
        )
        adapted = (
            responses_stream_from_chat(response_gen, request)
            if request.stream
            else responses_from_chat(response_gen, request)
        )
        return await self._handle_response(adapted, watcher, model, "create_response")

    @app.post("/v1/embeddings")
    async def create_embeddings(self, request: EmbeddingRequest, raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        model = request.model or ""
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, model=model, endpoint="create_embeddings")
        headers = dict(raw_request.headers)
        logger.info("embeddings model=%s", model)
        # EmbeddingRequest is a UnionType — force resolution before Ray pickle boundary.
        request = type(request).model_validate_json(request.model_dump_json())
        response_gen = handle.embed.options(stream=True).remote(request, headers, watcher.event, req_id)
        return await self._handle_response(response_gen, watcher, model, "create_embeddings")

    @app.post("/v1/audio/transcriptions")
    async def create_transcriptions(self, request: Annotated[TranscriptionRequest, Form()], raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        model = request.model or ""
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, model=model, endpoint="create_transcriptions")
        headers = dict(raw_request.headers)
        logger.info("transcription model=%s", model)
        # Read audio bytes before crossing process boundary — UploadFile is not serializable.
        # The bytes are passed separately; the request is reconstructed without the file field.
        try:
            audio_data = await request.file.read()
        finally:
            await request.file.close()
        request_no_file = TranscriptionRequest.model_construct(**request.model_dump(exclude={"file"}))
        response_gen = handle.transcribe.options(stream=True).remote(
            audio_data, request_no_file, headers, watcher.event, req_id
        )
        return await self._handle_response(response_gen, watcher, model, "create_transcriptions")

    @app.post("/v1/audio/translations")
    async def create_translations(self, request: Annotated[TranslationRequest, Form()], raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        model = request.model or ""
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, model=model, endpoint="create_translations")
        headers = dict(raw_request.headers)
        logger.info("translation model=%s", model)
        # Read audio bytes before crossing process boundary — UploadFile is not serializable.
        # The bytes are passed separately; the request is reconstructed without the file field.
        try:
            audio_data = await request.file.read()
        finally:
            await request.file.close()
        request_no_file = TranslationRequest.model_construct(**request.model_dump(exclude={"file"}))
        response_gen = handle.translate.options(stream=True).remote(
            audio_data, request_no_file, headers, watcher.event, req_id
        )
        return await self._handle_response(response_gen, watcher, model, "create_translations")

    @app.post("/v1/audio/speech")
    async def create_speech(self, request: SpeechRequest, raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        logger.info("speech model=%s voice=%s format=%s", request.model, request.voice, request.response_format)
        logger.debug("speech full request: %s", request.model_dump_json())
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, model=request.model, endpoint="create_speech")
        headers = dict(raw_request.headers)
        response_gen = handle.speak.options(stream=True).remote(request, headers, watcher.event, req_id)
        return await self._handle_response(response_gen, watcher, request.model, "create_speech")

    @app.post("/v1/images/generations")
    async def create_image(self, request: ImageGenerationRequest, raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        logger.info(
            "image_generation model=%s prompt=%r n=%d size=%s", request.model, request.prompt, request.n, request.size
        )
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, model=request.model, endpoint="create_image")
        headers = dict(raw_request.headers)
        response_gen = handle.imagine.options(stream=True).remote(request, headers, watcher.event, req_id)
        return await self._handle_response(response_gen, watcher, request.model, "create_image")

    @app.post("/v1/images/edits")
    async def create_image_edit(self, request: Annotated[ImageEditRequest, Form()], raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        logger.info(
            "image_edit model=%s prompt=%r n=%d size=%s mask=%s",
            request.model,
            request.prompt,
            request.n,
            request.size,
            request.mask is not None,
        )
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, model=request.model, endpoint="create_image_edit")
        headers = dict(raw_request.headers)
        # Read image bytes before crossing the process boundary — UploadFile is not serializable.
        # The bytes are passed separately; the request is reconstructed without the file fields.
        # `image` is Optional on the model (to accept the `image[]` alias) but the validator
        # guarantees it is set post-validation.
        assert request.image is not None
        try:
            image_data = await request.image.read()
            mask_data = await request.mask.read() if request.mask is not None else None
        finally:
            await request.image.close()
            if request.mask is not None:
                await request.mask.close()
        request_no_file = ImageEditRequest.model_construct(**request.model_dump(exclude={"image", "mask"}))
        response_gen = handle.edit_image.options(stream=True).remote(
            image_data, mask_data, request_no_file, headers, watcher.event, req_id
        )
        return await self._handle_response(response_gen, watcher, request.model, "create_image_edit")

    @app.post("/v1/images/variations")
    async def create_image_variation(self, request: Annotated[ImageVariationRequest, Form()], raw_request: Request):
        req_id = random_uuid()
        self._set_request_id(req_id)
        logger.info("image_variation model=%s n=%d size=%s", request.model, request.n, request.size)
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request, model=request.model, endpoint="create_image_variation")
        headers = dict(raw_request.headers)
        # Read image bytes before crossing the process boundary — UploadFile is not serializable.
        # `image` is Optional on the model (to accept the `image[]` alias) but the validator
        # guarantees it is set post-validation.
        assert request.image is not None
        try:
            image_data = await request.image.read()
        finally:
            await request.image.close()
        request_no_file = ImageVariationRequest.model_construct(**request.model_dump(exclude={"image"}))
        response_gen = handle.vary_image.options(stream=True).remote(
            image_data, request_no_file, headers, watcher.event, req_id
        )
        return await self._handle_response(response_gen, watcher, request.model, "create_image_variation")
