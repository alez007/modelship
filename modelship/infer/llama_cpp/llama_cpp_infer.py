import asyncio
import os
from collections.abc import AsyncGenerator

from llama_cpp import Llama

from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import LlamaCppConfig, ModelshipModelConfig, ModelUsecase, RawRequestProxy
from modelship.infer.llama_cpp.capabilities import LlamaCppCapabilities
from modelship.infer.llama_cpp.openai.serving_chat import OpenAIServingChat
from modelship.infer.llama_cpp.openai.serving_embedding import OpenAIServingEmbedding
from modelship.infer.llama_cpp.utils import build_tool_call_renderer
from modelship.infer.preflight import discover_hardware, merge_with_user_overrides, run_preflight
from modelship.logging import get_logger
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
)

logger = get_logger("infer.llama_cpp")


class LlamaCppInfer(BaseInfer):
    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        user_config = model_config.llama_cpp_config or LlamaCppConfig()
        user_overrides = user_config.model_dump(exclude_unset=True)

        # Preflight: hardware-aware safe defaults the user can override.
        # User-supplied values always win; divergences are logged so
        # misconfigured deploys are visible without spelunking llama.cpp logs.
        recommendation = run_preflight(model_config, discover_hardware())
        if recommendation:
            logger.info("preflight recommendation for '%s': %s", model_config.name, recommendation)
        else:
            logger.info("preflight recommendation for '%s': none", model_config.name)
        merged = merge_with_user_overrides(recommendation, user_overrides, model_name=model_config.name)
        self.config = user_config.model_copy(update=merged)

        # Automatically enable verbose mode if MSHIP_LOG_LEVEL is TRACE.
        # Other log levels are handled via the 'llama_cpp' Python logger (configured in logging.py).
        mship_log_level = os.environ.get("MSHIP_LOG_LEVEL", "INFO").upper()
        self._verbose = mship_log_level == "TRACE"

        # Force CPU-only as llama_cpp is currently compiled without GPU support in this environment.
        if self.config.n_gpu_layers != 0:
            logger.warning(
                "n_gpu_layers=%s is ignored for model '%s': llama_cpp currently only supports CPU.",
                self.config.n_gpu_layers,
                self.model_config.name,
            )
        self._n_gpu_layers = 0

        self.llamacpp: Llama | None = None
        self.serving_chat: OpenAIServingChat | None = None
        self.serving_embedding: OpenAIServingEmbedding | None = None
        logger.info(
            "initialising llama.cpp engine (verbose=%s) with config: %s",
            self._verbose,
            self.config.model_dump(),
        )

    def shutdown(self) -> None:
        if self.llamacpp:
            logger.info("Shutting down llama.cpp engine for %s", self.model_config.name)
            # llama-cpp-python relies on __del__ for resource cleanup.
            del self.llamacpp
            self.llamacpp = None
        self.serving_chat = None
        self.serving_embedding = None

    def __del__(self):
        self.shutdown()

    async def start(self) -> None:
        logger.info("Start llama.cpp infer for model: %s", self.model_config)
        loop = asyncio.get_event_loop()

        model_path = self.model_config._resolved_path
        if not model_path:
            raise ValueError(
                f"LlamaCpp deployment '{self.model_config.name}' is missing a resolved model path. "
                f"Check driver logs for resolution errors."
            )

        self.llamacpp = await loop.run_in_executor(
            None,
            lambda: Llama(
                model_path=model_path,
                n_gpu_layers=self._n_gpu_layers,
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                chat_format=self.config.chat_format,
                verbose=self._verbose,
                embedding=self.model_config.usecase == ModelUsecase.embed,
                **self.config.model_kwargs,
            ),
        )

        self._set_max_context_length(self.config.n_ctx)

        assert self.llamacpp is not None
        capabilities = LlamaCppCapabilities.detect(self.llamacpp)
        if capabilities.supports_image:
            logger.info("Multimodal (vision) capability detected for model: %s", self.model_config.name)

        if self.model_config.usecase == ModelUsecase.generate:
            parser_name = self.model_config._resolved_tool_call_parser
            reasoning_name = self.model_config._resolved_reasoning_parser
            template = self.model_config._resolved_chat_template
            # Default to driving the model through raw `create_completion`
            # so reasoning + tool-call markers in the model's output are
            # visible to ChatOutputStreamer regardless of whether parsers
            # are configured for this request. Fall back to llama-cpp's
            # native chat handler only when:
            #   - the user explicitly set `chat_format` (presumed opt-in
            #     to llama-cpp's templating; our parsers are bypassed), or
            #   - no chat template was resolvable (we'd have nothing to
            #     render with).
            if self.config.chat_format is None and template is not None:
                renderer = build_tool_call_renderer(self.llamacpp, template)
            else:
                renderer = None
                if parser_name is not None or reasoning_name is not None:
                    logger.warning(
                        "model '%s' has parsers resolved (tool=%s, reasoning=%s) but `chat_format` is set "
                        "or no chat template is available; falling back to llama-cpp's native chat handler "
                        "and our parsers will not run.",
                        self.model_config.name,
                        parser_name,
                        reasoning_name,
                    )
                    parser_name = None
                    reasoning_name = None
            self.serving_chat = OpenAIServingChat(
                self.llamacpp,
                self.model_config.name,
                capabilities,
                tool_call_parser=parser_name,
                reasoning_parser=reasoning_name,
                renderer=renderer,
            )
        elif self.model_config.usecase == ModelUsecase.embed:
            self.serving_embedding = OpenAIServingEmbedding(self.llamacpp, self.model_config.name)

    async def warmup(self) -> None:
        if self.serving_chat is not None:
            await self.serving_chat.warmup()
        elif self.serving_embedding is not None:
            await self.serving_embedding.warmup()

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        if self.serving_chat is None:
            return await super().create_chat_completion(request, raw_request)
        return await self.serving_chat.create_chat_completion(request, raw_request)

    async def create_embedding(
        self, request: EmbeddingRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | EmbeddingResponse:
        if self.serving_embedding is None:
            return await super().create_embedding(request, raw_request)
        return await self.serving_embedding.create_embedding(request, raw_request)
