import os
import time
from http import HTTPStatus
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from ray import serve
from ray.exceptions import RayTaskError
from ray.serve.handle import DeploymentHandle
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

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
    ImageGenerationRequest,
    ImageGenerationResponse,
    RawSpeechResponse,
    SpeechRequest,
    TranscriptionRequest,
    TranscriptionResponse,
    TranslationRequest,
    TranslationResponse,
    create_error_response,
)
from modelship.utils import random_uuid

logger = get_logger("api")

_DEFAULT_MAX_BODY_BYTES = 50 * 1024 * 1024  # 50 MB


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
    def __init__(self):
        # model_name -> (app_name -> handle). Keyed by app_name so
        # remove_deployments can drop a specific deployment by its
        # fingerprint-suffixed app name without scanning handles.
        self.models: dict[str, dict[str, DeploymentHandle]] = {}
        self._round_robin: dict[str, int] = {}
        self.model_list: list[OpenAiModelCard] = []
        self.expected_models: list[str] = []
        self._started_at = time.time()
        # Timing state — set_expected_models stamps a start, each add_models
        # arrival records the gap since the previous arrival as that model's
        # load duration (mship_deploy.py deploys sequentially so the gap ≈ load time).
        self._expected_set_at: float | None = None
        self._last_model_at: float | None = None
        self._all_ready_at: float | None = None
        self._model_load_times: dict[str, float] = {}

    async def set_expected_models(self, names: list[str]):
        self.expected_models = list(names)
        now = time.time()
        self._expected_set_at = now
        self._last_model_at = now

    async def add_models(self, deployments: dict[str, str]):
        """Register new model deployments with the gateway.

        Args:
            deployments: mapping of deployment_app_name -> model_name.
        """
        for app_name, model_name in deployments.items():
            try:
                handle = serve.get_app_handle(app_name)
            except Exception:
                logger.exception("Failed to get handle for app: %s", app_name)
                continue

            newly_added = model_name not in self.models
            if newly_added:
                self.models[model_name] = {}
                self._round_robin[model_name] = 0
                self.model_list.append(OpenAiModelCard(id=model_name))
            self.models[model_name][app_name] = handle
            logger.info("Registered deployment: %s (model: %s)", app_name, model_name)

            if newly_added:
                now = time.time()
                base = self._last_model_at or self._started_at
                self._model_load_times[model_name] = round(now - base, 2)
                self._last_model_at = now

        if self.expected_models and self._all_ready_at is None and all(m in self.models for m in self.expected_models):
            self._all_ready_at = time.time()

        MODELS_LOADED.set(len(self.models))

    async def remove_deployments(self, app_names: list[str]) -> list[str]:
        """Drop the given deployment app names from the routing tables.

        Deployment names follow `{model_name}-{fingerprint}`, so the owning
        model name is derived by stripping the fingerprint suffix. If a model
        has no remaining deployments after removal, the model entry, model
        card, round-robin counter, expected-models entry, and load-time entry
        are also dropped. Returns the names of fully-removed models.
        """
        removed_models: list[str] = []
        for app_name in app_names:
            model_name = app_name.rsplit("-", 1)[0]
            handles = self.models.get(model_name)
            if handles is None or app_name not in handles:
                logger.warning("remove_deployments: no deployment named %s", app_name)
                continue

            del handles[app_name]
            logger.info("Unregistered deployment: %s (model: %s)", app_name, model_name)

            if not handles:
                del self.models[model_name]
                self._round_robin.pop(model_name, None)
                self.model_list = [c for c in self.model_list if c.id != model_name]
                self._model_load_times.pop(model_name, None)
                self.expected_models = [m for m in self.expected_models if m != model_name]
                removed_models.append(model_name)

        MODELS_LOADED.set(len(self.models))
        return removed_models

    async def list_deployments(self) -> dict[str, list[str]]:
        """Return model_name -> list of deployment app_names currently registered."""
        return {model_name: list(handles.keys()) for model_name, handles in self.models.items()}

    @staticmethod
    def _set_request_id(request_id: str) -> None:
        from modelship.logging import request_id_var

        request_id_var.set(request_id)

    def _get_handle(self, model_name: str | None) -> DeploymentHandle:
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

    def _status_body(self) -> dict:
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

    @app.get("/status")
    async def status(self):
        body = self._status_body()
        if body["ready"]:
            return body
        return JSONResponse(status_code=503, content=body)

    @app.get("/v1/models", response_model=OpenaiModelList)
    async def list_models(self):
        return OpenaiModelList(data=self.model_list)

    @app.get("/v1/models/{model}", response_model=OpenAiModelCard)
    async def model_info(self, model: str) -> OpenAiModelCard:
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
        audio_data = await request.file.read()
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
        audio_data = await request.file.read()
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
