import io
from collections.abc import AsyncGenerator
from typing import ClassVar, cast

from fastapi import UploadFile
from starlette.requests import Request
from starlette.responses import Response
from vllm.config.model import ModelDType
from vllm.config.parallel import DistributedExecutorBackend
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
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from modelship.infer.base_infer import MINIMAL_WAV, BaseInfer
from modelship.infer.infer_config import ModelshipModelConfig, ModelUsecase, RawRequestProxy, VllmEngineConfig
from modelship.logging import get_logger
from modelship.metrics import _ENABLED as _METRICS_ENABLED
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
)

logger = get_logger("infer.vllm")


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

        config_engine_kwargs = model_config.vllm_engine_kwargs.model_dump(exclude_unset=True)
        config_engine_kwargs["model"] = model_config._resolved_path

        mem_fraction = self._get_memory_fraction()
        if mem_fraction is not None:
            config_engine_kwargs["gpu_memory_utilization"] = mem_fraction

        self.vllm_engine_kwargs: VllmEngineConfig = VllmEngineConfig(**config_engine_kwargs)
        logger.info("initialising vllm engine with args: %s", self.vllm_engine_kwargs.model_dump())

        engine_args = AsyncEngineArgs(
            model=self.vllm_engine_kwargs.model,
            tensor_parallel_size=self.vllm_engine_kwargs.tensor_parallel_size,
            max_model_len=cast("int", self.vllm_engine_kwargs.max_model_len),
            dtype=cast("ModelDType", self.vllm_engine_kwargs.dtype),
            tokenizer=self.vllm_engine_kwargs.tokenizer,
            trust_remote_code=self.vllm_engine_kwargs.trust_remote_code,
            gpu_memory_utilization=self.vllm_engine_kwargs.gpu_memory_utilization,
            distributed_executor_backend=cast(
                "DistributedExecutorBackend", self.vllm_engine_kwargs.distributed_executor_backend
            ),
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
        reasoning_parser_name = self.model_config._resolved_reasoning_parser or ""

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
        vllm_request = VllmChatCompletionRequest(**request.model_dump())
        result = await self.serving_chat.create_chat_completion(vllm_request, cast("Request", raw_request))
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
        result = await self.serving_embedding(vllm_request, cast("Request", raw_request))
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
        result = await self.serving_transcription.create_transcription(
            audio_data, vllm_request, cast("Request", raw_request)
        )
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
        result = await self.serving_translation.create_translation(
            audio_data, vllm_request, cast("Request", raw_request)
        )
        if isinstance(result, VllmErrorResponse):
            return ErrorResponse.model_validate(result.model_dump())
        if isinstance(result, VllmTranslationResponseVerbose):
            return TranslationResponseVerbose.model_validate(result.model_dump())
        if isinstance(result, VllmTranslationResponse):
            return TranslationResponse.model_validate(result.model_dump())
        if isinstance(result, AsyncGenerator):
            return cast("AsyncGenerator[str, None]", result)
        raise TypeError(f"Unexpected translation result type: {type(result).__name__}")
