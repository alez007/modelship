import io
import json
from collections.abc import AsyncGenerator
from http import HTTPStatus
from typing import Any, ClassVar, cast

from fastapi import UploadFile
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import Response
from vllm.config.model import ModelDType
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.chat_utils import ChatTemplateConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as VllmChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse as VllmErrorResponse,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.embed.protocol import (
    EmbeddingCompletionRequest as VllmEmbeddingCompletionRequest,
)
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest as VllmTranscriptionRequest,
)
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionResponse as VllmTranscriptionResponse,
)
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionResponseVerbose as VllmTranscriptionResponseVerbose,
)
from vllm.entrypoints.speech_to_text.transcription.serving import OpenAIServingTranscription
from vllm.entrypoints.speech_to_text.translation.protocol import (
    TranslationRequest as VllmTranslationRequest,
)
from vllm.entrypoints.speech_to_text.translation.protocol import (
    TranslationResponse as VllmTranslationResponse,
)
from vllm.entrypoints.speech_to_text.translation.protocol import (
    TranslationResponseVerbose as VllmTranslationResponseVerbose,
)
from vllm.entrypoints.speech_to_text.translation.serving import OpenAIServingTranslation
from vllm.exceptions import VLLMValidationError
from vllm.inputs import EngineInput
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from modelship.infer.base_infer import MINIMAL_WAV, BaseInfer, ClientDisconnectedError
from modelship.infer.infer_config import (
    ModelshipModelConfig,
    ModelUsecase,
    RawRequestProxy,
    VllmEngineConfig,
    split_vllm_user_overrides,
)
from modelship.infer.vllm import engine_ops
from modelship.infer.vllm.capabilities import VllmCapabilities
from modelship.infer.vllm.parsing.detect import resolve_reasoning_parser, resolve_tool_parser
from modelship.logging import TRACE, get_logger
from modelship.metrics import _ENABLED as _METRICS_ENABLED
from modelship.openai.chat_utils import (
    UnsupportedContentError,
    build_from_parsed,
    build_responses_items_from_parsed,
    normalize_chat_messages,
)
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    EmbeddingCompletionRequest,
    EmbeddingRequest,
    ErrorResponse,
    ResponseObject,
    ResponsesRequest,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
    UsageInfo,
    create_error_response,
)
from modelship.openai.protocol.responses.adapter import (
    UnsupportedResponsesFeatureError,
    _status_for,
    _usage_from_chat,
    build_response_object,
    responses_request_to_chat,
)
from modelship.openai.protocol.responses.streaming import ResponsesStreamTranslator
from modelship.preflight import discover_hardware, merge_with_user_overrides, run_preflight
from modelship.utils import base_request_id

logger = get_logger("infer.vllm")


def _encode_chunk(chunk: ChatCompletionStreamResponse) -> str:
    return f"data: {json.dumps(chunk.model_dump(mode='json'))}\n\n"


def _encode_error(error: ErrorResponse) -> str:
    return f"data: {json.dumps(error.model_dump(mode='json'))}\n\n"


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


def _responses_validation_error(exc: ValidationError) -> ErrorResponse:
    # Same shape as _validation_error, for pydantic ValidationErrors surfaced by
    # responses_request_to_chat (e.g. a bad reasoning.effort value).
    base = exc.args[0] if exc.args else str(exc)
    return create_error_response(message=base, err_type="invalid_request_error", status_code=HTTPStatus.BAD_REQUEST)


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

        user_overrides, auto_defaults = split_vllm_user_overrides(model_config)

        # Preflight: hardware-aware safe defaults the user can override.
        # User-supplied values always win; divergences are logged so
        # misconfigured deploys are visible without spelunking vLLM logs.
        recommendation = run_preflight(model_config, discover_hardware())
        if recommendation:
            logger.info("preflight recommendation for '%s': %s", model_config.name, recommendation)
        else:
            logger.info("preflight recommendation for '%s': none", model_config.name)
        config_engine_kwargs = merge_with_user_overrides(recommendation, user_overrides, model_name=model_config.name)
        for key, value in auto_defaults.items():
            config_engine_kwargs.setdefault(key, value)
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

        await self.init_serving_chat()
        self.serving_embedding = await self.init_serving_embeding()
        self.serving_transcription = await self.init_serving_transcription()
        self.serving_translation = await self.init_serving_translation()

    async def warmup(self) -> None:
        logger.info("Warming up vllm model: %s", self.model_config.name)
        dummy_proxy = RawRequestProxy(None, {})

        if hasattr(self, "openai_serving_render"):
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

    async def init_serving_chat(self) -> None:
        """Sets up the render/parse pipeline `create_chat_completion` drives directly
        (see engine_ops), if the model supports it. Leaves `openai_serving_render`
        unset otherwise — callers gate on `hasattr(self, "openai_serving_render")`."""
        logger.info("init_serving_chat: %s, %s", self.supported_tasks, self.model_config.usecase)
        if not (self.model_config.usecase is ModelUsecase.generate and "generate" in self.supported_tasks):
            return

        models = OpenAIServingModels(
            engine_client=self.engine,
            base_model_paths=[BaseModelPath(name=self.model_config.name, model_path=self.vllm_engine_kwargs.model)],
        )

        # get_chat_template isn't in vLLM's TokenizerLike protocol (it's a plain
        # HF PreTrainedTokenizer method the real tokenizer always has).
        template = cast(Any, self.engine.get_tokenizer()).get_chat_template()
        tool_parser_name = resolve_tool_parser(self.model_config, template)
        enable_tools = tool_parser_name is not None
        reasoning_parser_name = resolve_reasoning_parser(self.model_config, template) or ""

        self._enable_auto_tools = enable_tools
        self.openai_serving_render = OpenAIServingRender(
            model_config=self.engine.model_config,
            renderer=self.engine.renderer,
            model_registry=models.registry,
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
                chat_template_config=ChatTemplateConfig(chat_template=None, chat_template_content_format="auto"),
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
        if not hasattr(self, "openai_serving_render"):
            return await super().create_chat_completion(request, raw_request)
        try:
            request.messages = normalize_chat_messages(
                request.messages,
                supports_image=self._caps.supports_image,
                supports_audio=self._caps.supports_audio,
            )
        except UnsupportedContentError as exc:
            return create_error_response(exc)
        vllm_request = engine_ops.build_vllm_request(request, self.model_config.chat_template_kwargs)

        if request.stream:
            return self._create_chat_completion_stream(request, vllm_request, raw_request)

        return await self._create_chat_completion_no_stream(request, vllm_request, raw_request)

    async def _create_chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        vllm_request: VllmChatCompletionRequest,
        raw_request: RawRequestProxy,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat path via `engine_ops`, bypassing `OpenAIServingChat`."""
        request_id = f"chatcmpl-{base_request_id(raw_request)}"
        try:
            render_result = await engine_ops.render_and_params(self.openai_serving_render, vllm_request)
        except VLLMValidationError as exc:
            yield _encode_error(_validation_error(exc))
            yield "data: [DONE]\n\n"
            return
        if isinstance(render_result, VllmErrorResponse):
            yield _encode_error(ErrorResponse.model_validate(render_result.model_dump()))
            yield "data: [DONE]\n\n"
            return
        engine_input, sampling_params = render_result

        tokenizer = self.openai_serving_render.renderer.tokenizer
        assert tokenizer is not None, "vllm renderer has no tokenizer (skip_tokenizer_init=True is unsupported here)"

        stream = engine_ops.stream_chat_completion(
            self.engine,
            self.openai_serving_render,
            vllm_request,
            engine_input,
            sampling_params,
            request_id,
            self.model_config.name,
            tokenizer,
            enable_auto_tools=self._enable_auto_tools,
            want_logprobs=bool(request.logprobs),
            num_output_top_logprobs=request.top_logprobs,
        )
        buffered: list[str] = []
        try:
            async for chunk in self.run_cancellable_stream(stream, raw_request):
                for choice in chunk.choices:
                    if choice.delta.content:
                        buffered.append(choice.delta.content)
                yield _encode_chunk(chunk)
        except ClientDisconnectedError:
            logger.info("chat request %s aborted: client disconnected", request_id)
            return
        except VLLMValidationError as exc:
            yield _encode_error(_validation_error(exc))
            yield "data: [DONE]\n\n"
            return
        except Exception:
            logger.exception("chat request %s failed mid-stream", request_id)
            yield _encode_error(
                create_error_response("Internal error during generation", err_type="api_error", status_code=500)
            )
            yield "data: [DONE]\n\n"
            return
        finally:
            logger.log(TRACE, "chat response %s (stream): %r", request_id, "".join(buffered))
        yield "data: [DONE]\n\n"

    async def _create_chat_completion_no_stream(
        self,
        request: ChatCompletionRequest,
        vllm_request: VllmChatCompletionRequest,
        raw_request: RawRequestProxy,
    ) -> ErrorResponse | ChatCompletionResponse:
        """Non-stream chat path via `engine_ops`, bypassing `OpenAIServingChat`."""
        try:
            render_result = await engine_ops.render_and_params(self.openai_serving_render, vllm_request)
        except VLLMValidationError as exc:
            return _validation_error(exc)
        if isinstance(render_result, VllmErrorResponse):
            return ErrorResponse.model_validate(render_result.model_dump())
        engine_input, sampling_params = render_result

        tokenizer = self.openai_serving_render.renderer.tokenizer
        assert tokenizer is not None, "vllm renderer has no tokenizer (skip_tokenizer_init=True is unsupported here)"

        # Non-streaming reuses one parser instance across every choice (see
        # engine_ops.build_choices), so only one is needed here regardless of n.
        parser = engine_ops.make_parsers(
            self.openai_serving_render, tokenizer, vllm_request, vllm_request.chat_template_kwargs, n=1
        )[0]
        prompt_token_ids = engine_ops.extract_prompt_token_ids(self.openai_serving_render, engine_input)
        reasoning_ended = engine_ops.derive_reasoning_ended(vllm_request, parser, prompt_token_ids)

        request_id = f"chatcmpl-{base_request_id(raw_request)}"
        try:
            final_res = await self.run_cancellable(
                engine_ops.consume_final_output(
                    self.engine,
                    engine_input,
                    sampling_params,
                    request_id,
                    reasoning_ended=reasoning_ended,
                    parser=parser,
                    chat_template_kwargs=vllm_request.chat_template_kwargs,
                ),
                raw_request,
            )
        except ClientDisconnectedError:
            return create_error_response("Client disconnected")
        except VLLMValidationError as exc:
            return _validation_error(exc)

        choices, finish_reasons, logprobs_list = engine_ops.build_choices(
            final_res,
            vllm_request,
            parser,
            tokenizer,
            enable_auto_tools=self._enable_auto_tools,
            want_logprobs=bool(request.logprobs),
            num_output_top_logprobs=request.top_logprobs,
        )

        if final_res.prompt_token_ids is None:
            return create_error_response("vllm returned no prompt_token_ids for a completed request", status_code=502)
        prompt_tokens = len(final_res.prompt_token_ids)
        completion_tokens = sum(len(output.token_ids) for output in final_res.outputs)

        return build_from_parsed(
            request_id=request_id,
            model_name=self.model_config.name,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reasons=finish_reasons,
            logprobs=logprobs_list,
        )

    async def create_response(
        self, request: ResponsesRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ResponseObject | AsyncGenerator[str, None]:
        if not hasattr(self, "openai_serving_render"):
            return await super().create_response(request, raw_request)

        try:
            chat_request = responses_request_to_chat(request)
        except UnsupportedResponsesFeatureError as e:
            return create_error_response(e)
        except ValidationError as e:
            return _responses_validation_error(e)

        try:
            chat_request.messages = normalize_chat_messages(
                chat_request.messages,
                supports_image=self._caps.supports_image,
                supports_audio=self._caps.supports_audio,
            )
        except UnsupportedContentError as exc:
            return create_error_response(exc)
        chat_request.stream = request.stream or False
        vllm_request = engine_ops.build_vllm_request(chat_request, self.model_config.chat_template_kwargs)

        if request.stream:
            return await self._create_response_stream_or_error(request, vllm_request, raw_request)
        return await self._create_response_no_stream(request, vllm_request, raw_request)

    async def _create_response_stream_or_error(
        self,
        request: ResponsesRequest,
        vllm_request: VllmChatCompletionRequest,
        raw_request: RawRequestProxy,
    ) -> ErrorResponse | AsyncGenerator[str, None]:
        """Render + derive sampling params before committing to a Responses event
        stream, so a pre-generation failure (e.g. context overflow) can still be
        returned as a plain `ErrorResponse` instead of a mid-stream `response.failed`."""
        try:
            render_result = await engine_ops.render_and_params(self.openai_serving_render, vllm_request)
        except VLLMValidationError as exc:
            return _validation_error(exc)
        if isinstance(render_result, VllmErrorResponse):
            return ErrorResponse.model_validate(render_result.model_dump())
        engine_input, sampling_params = render_result
        return self._create_response_stream(request, vllm_request, engine_input, sampling_params, raw_request)

    async def _create_response_no_stream(
        self,
        request: ResponsesRequest,
        vllm_request: VllmChatCompletionRequest,
        raw_request: RawRequestProxy,
    ) -> ErrorResponse | ResponseObject:
        """Non-stream Responses path via `engine_ops`, shaping items directly from
        `ParsedChatOutput` instead of round-tripping through a `ChatCompletionResponse`."""
        try:
            render_result = await engine_ops.render_and_params(self.openai_serving_render, vllm_request)
        except VLLMValidationError as exc:
            return _validation_error(exc)
        if isinstance(render_result, VllmErrorResponse):
            return ErrorResponse.model_validate(render_result.model_dump())
        engine_input, sampling_params = render_result

        tokenizer = self.openai_serving_render.renderer.tokenizer
        assert tokenizer is not None, "vllm renderer has no tokenizer (skip_tokenizer_init=True is unsupported here)"

        parser = engine_ops.make_parsers(
            self.openai_serving_render, tokenizer, vllm_request, vllm_request.chat_template_kwargs, n=1
        )[0]
        prompt_token_ids = engine_ops.extract_prompt_token_ids(self.openai_serving_render, engine_input)
        reasoning_ended = engine_ops.derive_reasoning_ended(vllm_request, parser, prompt_token_ids)

        request_id = f"resp-{base_request_id(raw_request)}"
        try:
            final_res = await self.run_cancellable(
                engine_ops.consume_final_output(
                    self.engine,
                    engine_input,
                    sampling_params,
                    request_id,
                    reasoning_ended=reasoning_ended,
                    parser=parser,
                    chat_template_kwargs=vllm_request.chat_template_kwargs,
                ),
                raw_request,
            )
        except ClientDisconnectedError:
            return create_error_response("Client disconnected")
        except VLLMValidationError as exc:
            return _validation_error(exc)

        choices, finish_reasons, _logprobs_list = engine_ops.build_choices(
            final_res,
            vllm_request,
            parser,
            tokenizer,
            enable_auto_tools=self._enable_auto_tools,
            want_logprobs=False,
            num_output_top_logprobs=None,
        )

        if final_res.prompt_token_ids is None:
            return create_error_response("vllm returned no prompt_token_ids for a completed request", status_code=502)
        prompt_tokens = len(final_res.prompt_token_ids)
        completion_tokens = sum(len(output.token_ids) for output in final_res.outputs)

        status, incomplete = _status_for(finish_reasons[0])
        return build_response_object(
            request,
            status=status,
            output=build_responses_items_from_parsed(choices[0]),
            usage=_usage_from_chat(
                UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
            ),
            incomplete=incomplete,
            model=self.model_config.name,
        )

    async def _create_response_stream(
        self,
        request: ResponsesRequest,
        vllm_request: VllmChatCompletionRequest,
        engine_input: EngineInput,
        sampling_params: SamplingParams,
        raw_request: RawRequestProxy,
    ) -> AsyncGenerator[str, None]:
        """Native streaming Responses path: feeds `ResponsesStreamTranslator` directly
        from `engine_ops.stream_chat_completion`'s typed chunks — no chat SSE text
        round trip like the (unused-by-this-loader) `BaseInfer` default."""
        request_id = f"resp-{base_request_id(raw_request)}"
        tokenizer = self.openai_serving_render.renderer.tokenizer
        assert tokenizer is not None, "vllm renderer has no tokenizer (skip_tokenizer_init=True is unsupported here)"

        stream = engine_ops.stream_chat_completion(
            self.engine,
            self.openai_serving_render,
            vllm_request,
            engine_input,
            sampling_params,
            request_id,
            self.model_config.name,
            tokenizer,
            enable_auto_tools=self._enable_auto_tools,
            want_logprobs=False,
            num_output_top_logprobs=None,
        )
        translator = ResponsesStreamTranslator(request)
        for event in translator.start():
            yield event
        try:
            async for chunk in self.run_cancellable_stream(stream, raw_request):
                for event in translator.process(chunk):
                    yield event
        except ClientDisconnectedError:
            logger.info("responses request %s aborted: client disconnected", request_id)
            return
        except VLLMValidationError as exc:
            base = exc.args[0] if exc.args else str(exc)
            for event in translator.fail(str(base)):
                yield event
            return
        except Exception:
            logger.exception("responses request %s failed mid-stream", request_id)
            for event in translator.fail("Internal error during generation"):
                yield event
            return
        for event in translator.finish():
            yield event

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
        # `file` is a required field on vLLM's own request schema but unused by
        # create_transcription (audio_data is passed separately) — model_construct
        # skips validation instead of raising on the field modelship never populates.
        vllm_request = VllmTranscriptionRequest.model_construct(**request.model_dump())
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
        # `file` is a required field on vLLM's own request schema but unused by
        # create_translation (audio_data is passed separately) — model_construct
        # skips validation instead of raising on the field modelship never populates.
        vllm_request = VllmTranslationRequest.model_construct(**request.model_dump())
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
