import io
from collections.abc import AsyncGenerator
from http import HTTPStatus
from typing import Any, ClassVar, cast

from fastapi import UploadFile
from starlette.requests import Request
from starlette.responses import Response
from vllm.config.model import ModelDType
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as VllmChatCompletionRequest,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse as VllmChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse as VllmErrorResponse,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionRequest as VllmTranscriptionRequest,
)
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionResponse as VllmTranscriptionResponse,
)
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranscriptionResponseVerbose as VllmTranscriptionResponseVerbose,
)
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranslationRequest as VllmTranslationRequest,
)
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranslationResponse as VllmTranslationResponse,
)
from vllm.entrypoints.openai.speech_to_text.protocol import (
    TranslationResponseVerbose as VllmTranslationResponseVerbose,
)
from vllm.entrypoints.openai.speech_to_text.serving import OpenAIServingTranscription, OpenAIServingTranslation
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingCompletionRequest as VllmEmbeddingCompletionRequest,
)
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.exceptions import VLLMValidationError
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from modelship.infer.base_infer import MINIMAL_WAV, BaseInfer
from modelship.infer.infer_config import ModelshipModelConfig, ModelUsecase, RawRequestProxy, VllmEngineConfig
from modelship.infer.vllm.capabilities import VllmCapabilities
from modelship.logging import get_logger
from modelship.metrics import _ENABLED as _METRICS_ENABLED
from modelship.openai.chat_utils import UnsupportedContentError, normalize_chat_messages
from modelship.openai.parsers.reasoning import resolve_active_reasoning_parser
from modelship.openai.parsers.utils import render_generation_prompt
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingCompletionRequest,
    EmbeddingRequest,
    ErrorResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
    create_error_response,
)
from modelship.preflight import discover_hardware, merge_with_user_overrides, run_preflight

logger = get_logger("infer.vllm")


def _validation_error(exc: VLLMValidationError) -> ErrorResponse:
    # VLLMValidationError.__str__ appends "(parameter=..., value=...)"; keep the
    # original message and surface the offending field via the OpenAI `param` slot.
    base = exc.args[0] if exc.args else str(exc)
    return create_error_response(
        message=base,
        err_type="invalid_request_error",
        status_code=HTTPStatus.BAD_REQUEST,
        param=exc.parameter,
    )


class VllmInfer(BaseInfer):
    _vllm_usecases: ClassVar[list[ModelUsecase]] = [
        ModelUsecase.generate,
        ModelUsecase.embed,
        ModelUsecase.transcription,
        ModelUsecase.translation,
    ]

    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)

        if not model_config._resolved_path:
            raise ValueError(
                f"vllm deployment '{model_config.name}' is missing a resolved model path. "
                f"Check driver logs for resolution errors."
            )

        user_overrides = model_config.vllm_engine_kwargs.model_dump(exclude_unset=True)

        # Preflight: hardware-aware safe defaults the user can override.
        # User-supplied values always win; divergences are logged so
        # misconfigured deploys are visible without spelunking vLLM logs.
        recommendation = run_preflight(model_config, discover_hardware())
        if recommendation:
            logger.info("preflight recommendation for '%s': %s", model_config.name, recommendation)
        else:
            logger.info("preflight recommendation for '%s': none", model_config.name)
        config_engine_kwargs = merge_with_user_overrides(recommendation, user_overrides, model_name=model_config.name)
        config_engine_kwargs["model"] = model_config._resolved_path

        # gpu_memory_utilization for a fractional num_gpus is resolved once at
        # config normalization (normalize_num_gpus_and_tp), so it's already in
        # user_overrides here — no runtime override needed. Folding it in here
        # instead would (a) be invisible to the preflight, which reads the config
        # model, and (b) clobber an explicitly-declared value. (diffusers and
        # transformers still derive a per-process torch fraction via
        # _get_memory_fraction; they have no equivalent engine knob.)
        self.vllm_engine_kwargs: VllmEngineConfig = VllmEngineConfig(**config_engine_kwargs)
        logger.info("initialising vllm engine with args: %s", self.vllm_engine_kwargs.model_dump())

        # Force the ray executor for multi-slot deploys: the outer actor sits
        # in a 0-GPU PG bundle, but vLLM's ParallelConfig validates world_size
        # against the actor's visible GPUs before consulting the backend. The
        # ray backend skips that check (workers claim their own bundles via the
        # inherited placement group).
        world_size = self.vllm_engine_kwargs.tensor_parallel_size * self.vllm_engine_kwargs.pipeline_parallel_size
        distributed_executor_backend = "ray" if world_size > 1 else None

        # Multimodal knobs: only forward when the user set them so we inherit
        # vLLM's own defaults (empty dict / None) otherwise.
        mm_kwargs: dict[str, Any] = {}
        if self.vllm_engine_kwargs.limit_mm_per_prompt is not None:
            mm_kwargs["limit_mm_per_prompt"] = self.vllm_engine_kwargs.limit_mm_per_prompt
        if self.vllm_engine_kwargs.mm_processor_kwargs is not None:
            mm_kwargs["mm_processor_kwargs"] = self.vllm_engine_kwargs.mm_processor_kwargs

        engine_args = AsyncEngineArgs(
            model=self.vllm_engine_kwargs.model,
            tensor_parallel_size=self.vllm_engine_kwargs.tensor_parallel_size,
            pipeline_parallel_size=self.vllm_engine_kwargs.pipeline_parallel_size,
            max_model_len=cast("int", self.vllm_engine_kwargs.max_model_len),
            dtype=cast("ModelDType", self.vllm_engine_kwargs.dtype),
            tokenizer=self.vllm_engine_kwargs.tokenizer,
            trust_remote_code=self.vllm_engine_kwargs.trust_remote_code,
            gpu_memory_utilization=self.vllm_engine_kwargs.gpu_memory_utilization,
            distributed_executor_backend=distributed_executor_backend,
            enable_log_requests=self.vllm_engine_kwargs.enable_log_requests
            if self.vllm_engine_kwargs.enable_log_requests is not None
            else False,
            disable_log_stats=self.vllm_engine_kwargs.disable_log_stats
            if self.vllm_engine_kwargs.disable_log_stats is not None
            else False,
            quantization=self.vllm_engine_kwargs.quantization,
            kv_cache_dtype=self.vllm_engine_kwargs.kv_cache_dtype or "auto",  # type: ignore[arg-type]
            enforce_eager=self.vllm_engine_kwargs.enforce_eager or False,
            max_num_batched_tokens=self.vllm_engine_kwargs.max_num_batched_tokens,
            **mm_kwargs,
        )

        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)

        stat_loggers: list | None = None
        if _METRICS_ENABLED:
            from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger

            stat_loggers = [RayPrometheusStatLogger]

        self.engine = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def shutdown(self) -> None:
        try:
            if engine := getattr(self, "engine", None):
                logger.info("Shutting down vllm engine for %s", self.model_config.name)
                engine.shutdown()
        except Exception:
            from modelship.metrics import RESOURCE_CLEANUP_ERRORS_TOTAL

            RESOURCE_CLEANUP_ERRORS_TOTAL.inc(tags={"model": self.model_config.name, "component": "vllm_engine"})
            logger.exception("Failed to shutdown vllm engine for %s", self.model_config.name)

    def __del__(self):
        self.shutdown()

    async def start(self):
        logger.info("Start vllm infer for model: %s", self.model_config)
        self.vllm_config = self.engine.vllm_config
        self._set_max_context_length(self.vllm_config.model_config.max_model_len)
        self.supported_tasks = await self.engine.get_supported_tasks()
        logger.info("Supported_tasks: %s", self.supported_tasks)

        self._caps = VllmCapabilities.detect(self.vllm_config.model_config)
        logger.info("vllm capabilities for '%s': %s", self.model_config.name, self._caps)

        self.serving_chat = await self.init_serving_chat()
        self.serving_embedding = await self.init_serving_embeding()
        self.serving_transcription = await self.init_serving_transcription()
        self.serving_translation = await self.init_serving_translation()

    async def warmup(self) -> None:
        logger.info("Warming up vllm model: %s", self.model_config.name)
        dummy_proxy = RawRequestProxy(None, {})

        if self.serving_chat is not None:
            request = ChatCompletionRequest(
                model=self.model_config.name, messages=[{"role": "user", "content": "warmup"}], max_tokens=1, seed=-1
            )
            result = await self.create_chat_completion(request, dummy_proxy)
            if isinstance(result, AsyncGenerator):
                async for _ in result:
                    pass
            logger.info("Warmup chat completion done for %s", self.model_config.name)

        elif self.serving_embedding is not None:
            request = EmbeddingCompletionRequest(
                model=self.model_config.name,
                input="warmup",
            )
            await self.create_embedding(request, dummy_proxy)
            logger.info("Warmup embedding done for %s", self.model_config.name)

        elif self.serving_transcription is not None:
            request = TranscriptionRequest(
                model=self.model_config.name, file=UploadFile(file=io.BytesIO(MINIMAL_WAV)), seed=-1
            )
            audio_data = MINIMAL_WAV
            result = await self.create_transcription(audio_data, request, dummy_proxy)
            if isinstance(result, AsyncGenerator):
                async for _ in result:
                    pass
            logger.info("Warmup transcription done for %s", self.model_config.name)

        elif self.serving_translation is not None:
            request = TranslationRequest(
                model=self.model_config.name, file=UploadFile(file=io.BytesIO(MINIMAL_WAV)), seed=-1
            )
            audio_data = MINIMAL_WAV
            result = await self.create_translation(audio_data, request, dummy_proxy)
            if isinstance(result, AsyncGenerator):
                async for _ in result:
                    pass
            logger.info("Warmup translation done for %s", self.model_config.name)

    async def init_serving_chat(self) -> OpenAIServingChat | None:
        logger.info("init_serving_chat: %s, %s", self.supported_tasks, self.model_config.usecase)
        if not (self.model_config.usecase is ModelUsecase.generate and "generate" in self.supported_tasks):
            return None

        models = OpenAIServingModels(
            engine_client=self.engine,
            base_model_paths=[BaseModelPath(name=self.model_config.name, model_path=self.vllm_engine_kwargs.model)],
        )

        # Driver-side `resolve_all_{tool,reasoning}_parsers` populated these
        # with the final parser name (explicit user setting or auto-detected),
        # or left them None to signal "disabled". Don't redo the precedence here.
        tool_parser_name = self.model_config._resolved_tool_call_parser
        enable_tools = tool_parser_name is not None
        # Confirm the capability-detected reasoning parser against the real render: a
        # deployment that suppressed reasoning via chat_template_kwargs downgrades to
        # None. vLLM owns reasoning parsing, so this mainly keeps reported state honest.
        template = self.model_config._resolved_chat_template
        reasoning_parser_name = (
            resolve_active_reasoning_parser(
                self.model_config._resolved_reasoning_parser,
                lambda: render_generation_prompt(template or "", self.model_config.chat_template_kwargs),
            )
            or ""
        )

        openai_serving_render = OpenAIServingRender(
            model_config=self.engine.model_config,
            renderer=self.engine.renderer,
            model_registry=models.registry,
            request_logger=RequestLogger(max_log_len=None),
            chat_template=None,
            chat_template_content_format=self.vllm_engine_kwargs.chat_template_content_format,
            enable_auto_tools=enable_tools,
            tool_parser=tool_parser_name,
        )

        return OpenAIServingChat(
            engine_client=self.engine,
            models=models,
            openai_serving_render=openai_serving_render,
            response_role="assistant",
            request_logger=RequestLogger(max_log_len=None),
            chat_template=None,
            chat_template_content_format=self.vllm_engine_kwargs.chat_template_content_format,
            enable_auto_tools=enable_tools,
            tool_parser=tool_parser_name,
            reasoning_parser=reasoning_parser_name,
        )

    async def init_serving_embeding(self) -> ServingEmbedding | None:
        logger.info("init_serving_embeding: %s, %s", self.supported_tasks, self.model_config.usecase)
        return (
            ServingEmbedding(
                engine_client=self.engine,
                models=OpenAIServingModels(
                    engine_client=self.engine,
                    base_model_paths=[
                        BaseModelPath(name=self.model_config.name, model_path=self.vllm_engine_kwargs.model)
                    ],
                ),
                request_logger=RequestLogger(max_log_len=None),
                chat_template=None,
                chat_template_content_format="auto",
            )
            if self.model_config.usecase is ModelUsecase.embed
            and any(task in self.supported_tasks for task in ["embed", "embedding"])
            else None
        )

    async def init_serving_transcription(self) -> OpenAIServingTranscription | None:
        logger.info("init_serving_transcription: %s, %s", self.supported_tasks, self.model_config.usecase)
        return (
            OpenAIServingTranscription(
                engine_client=self.engine,
                models=OpenAIServingModels(
                    engine_client=self.engine,
                    base_model_paths=[
                        BaseModelPath(name=self.model_config.name, model_path=self.vllm_engine_kwargs.model)
                    ],
                ),
                request_logger=RequestLogger(max_log_len=None),
            )
            if (self.model_config.usecase in [ModelUsecase.transcription, ModelUsecase.translation])
            and "transcription" in self.supported_tasks
            else None
        )

    async def init_serving_translation(self) -> OpenAIServingTranslation | None:
        logger.info("init_serving_translation: %s, %s", self.supported_tasks, self.model_config.usecase)
        return (
            OpenAIServingTranslation(
                engine_client=self.engine,
                models=OpenAIServingModels(
                    engine_client=self.engine,
                    base_model_paths=[
                        BaseModelPath(name=self.model_config.name, model_path=self.vllm_engine_kwargs.model)
                    ],
                ),
                request_logger=RequestLogger(max_log_len=None),
            )
            if (self.model_config.usecase in [ModelUsecase.transcription, ModelUsecase.translation])
            and "transcription" in self.supported_tasks
            else None
        )

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        if self.serving_chat is None:
            return await super().create_chat_completion(request, raw_request)
        try:
            request.messages = normalize_chat_messages(
                request.messages,
                supports_image=self._caps.supports_image,
                supports_audio=self._caps.supports_audio,
            )
        except UnsupportedContentError as exc:
            return create_error_response(exc)
        request_data = request.model_dump()
        # vLLM renders the chat template internally; merge the model's default
        # kwargs under any per-request values (request wins).
        if self.model_config.chat_template_kwargs:
            request_data["chat_template_kwargs"] = {
                **self.model_config.chat_template_kwargs,
                **(request_data.get("chat_template_kwargs") or {}),
            }
        vllm_request = VllmChatCompletionRequest(**request_data)
        try:
            result = await self.serving_chat.create_chat_completion(vllm_request, cast("Request", raw_request))
        except VLLMValidationError as exc:
            return _validation_error(exc)
        if isinstance(result, VllmErrorResponse):
            return ErrorResponse.model_validate(result.model_dump())
        if isinstance(result, VllmChatCompletionResponse):
            return ChatCompletionResponse.model_validate(result.model_dump())
        return cast("AsyncGenerator[str, None]", result)

    async def create_embedding(
        self, request: EmbeddingRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | Response:
        if self.serving_embedding is None:
            return await super().create_embedding(request, raw_request)
        vllm_request = VllmEmbeddingCompletionRequest(**request.model_dump())
        try:
            result = await self.serving_embedding(vllm_request, cast("Request", raw_request))
        except VLLMValidationError as exc:
            return _validation_error(exc)
        if isinstance(result, VllmErrorResponse):
            return ErrorResponse.model_validate(result.model_dump())
        return cast("ErrorResponse | Response", result)

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose | AsyncGenerator[str, None]:
        if self.serving_transcription is None:
            return await super().create_transcription(audio_data, request, raw_request)
        vllm_request = VllmTranscriptionRequest(**request.model_dump())
        vllm_request.timestamp_granularities = []
        try:
            result = await self.serving_transcription.create_transcription(
                audio_data, vllm_request, cast("Request", raw_request)
            )
        except VLLMValidationError as exc:
            return _validation_error(exc)
        if isinstance(result, VllmErrorResponse):
            return ErrorResponse.model_validate(result.model_dump())
        if isinstance(result, VllmTranscriptionResponseVerbose):
            return TranscriptionResponseVerbose.model_validate(result.model_dump())
        if isinstance(result, VllmTranscriptionResponse):
            return TranscriptionResponse.model_validate(result.model_dump())
        if isinstance(result, AsyncGenerator):
            return cast("AsyncGenerator[str, None]", result)
        raise TypeError(f"Unexpected transcription result type: {type(result).__name__}")

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranslationResponse | TranslationResponseVerbose | AsyncGenerator[str, None]:
        if self.serving_translation is None:
            return await super().create_translation(audio_data, request, raw_request)
        vllm_request = VllmTranslationRequest(**request.model_dump())
        try:
            result = await self.serving_translation.create_translation(
                audio_data, vllm_request, cast("Request", raw_request)
            )
        except VLLMValidationError as exc:
            return _validation_error(exc)
        if isinstance(result, VllmErrorResponse):
            return ErrorResponse.model_validate(result.model_dump())
        if isinstance(result, VllmTranslationResponseVerbose):
            return TranslationResponseVerbose.model_validate(result.model_dump())
        if isinstance(result, VllmTranslationResponse):
            return TranslationResponse.model_validate(result.model_dump())
        if isinstance(result, AsyncGenerator):
            return cast("AsyncGenerator[str, None]", result)
        raise TypeError(f"Unexpected translation result type: {type(result).__name__}")
