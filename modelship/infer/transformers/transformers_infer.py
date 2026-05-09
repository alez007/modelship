from collections.abc import AsyncGenerator

import torch

from modelship.infer.base_infer import BaseInfer
from modelship.infer.base_serving import OpenAIServing
from modelship.infer.infer_config import ModelshipModelConfig, ModelUsecase, RawRequestProxy, TransformersConfig
from modelship.logging import get_logger
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    RawSpeechResponse,
    SpeechRequest,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranslationRequest,
    TranslationResponse,
    TranslationResponseVerbose,
)

logger = get_logger("infer.transformers")

_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class TransformersInfer(BaseInfer):
    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        self.config = model_config.transformers_config or TransformersConfig()
        self.device = self.config.device
        self.torch_dtype = _TORCH_DTYPES.get(self.config.torch_dtype) if self.config.torch_dtype != "auto" else "auto"

        mem_frac = self._get_memory_fraction()
        if torch.cuda.is_available() and mem_frac is not None:
            torch.cuda.set_per_process_memory_fraction(mem_frac)

        self._serving: list[OpenAIServing] = []

    def shutdown(self) -> None:
        pass

    def __del__(self):
        try:
            for serving in getattr(self, "_serving", []):
                del serving
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            from modelship.metrics import RESOURCE_CLEANUP_ERRORS_TOTAL

            RESOURCE_CLEANUP_ERRORS_TOTAL.inc(tags={"model": self.model_config.name, "component": "transformers_model"})

    async def start(self):
        usecase = self.model_config.usecase
        model_name = self.model_config._resolved_path
        if not model_name:
            raise ValueError(
                f"Transformers deployment '{self.model_config.name}' is missing a resolved model path. "
                f"Check driver logs for resolution errors."
            )

        if usecase is ModelUsecase.generate:
            from transformers import pipeline

            from modelship.infer.transformers.capabilities import TransformersCapabilities
            from modelship.infer.transformers.openai.serving_chat import OpenAIServingChat

            pipe = pipeline(
                "text-generation",
                model=model_name,
                device=self.device,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.config.trust_remote_code,
                model_kwargs=self.config.model_kwargs,
            )
            capabilities = TransformersCapabilities.detect(pipe)
            if capabilities.supports_image:
                logger.info("Multimodal (vision) capability detected for model: %s", self.model_config.name)
            # `resolve_all_tool_parsers` / `resolve_all_reasoning_parsers`
            # already honored loader-specific opt-outs (leaving the resolved
            # field None) and captured explicit overrides. Read both directly.
            tool_call_parser = self.model_config._resolved_tool_call_parser
            reasoning_parser = self.model_config._resolved_reasoning_parser
            self.serving_chat = OpenAIServingChat(
                pipe,
                self.model_config.name,
                self.config,
                capabilities,
                tool_call_parser=tool_call_parser,
                reasoning_parser=reasoning_parser,
            )
            self._serving.append(self.serving_chat)

        elif usecase is ModelUsecase.embed:
            from sentence_transformers import SentenceTransformer

            from modelship.infer.transformers.openai.serving_embedding import OpenAIServingEmbedding

            model = SentenceTransformer(
                model_name,
                device=self.device,
                trust_remote_code=self.config.trust_remote_code,
                model_kwargs=self.config.model_kwargs,
            )
            self.serving_embedding = OpenAIServingEmbedding(model, self.model_config.name)
            self._serving.append(self.serving_embedding)

        elif usecase in (ModelUsecase.transcription, ModelUsecase.translation):
            from transformers import pipeline

            from modelship.infer.transformers.openai.serving_transcription import (
                OpenAIServingTranscription,
                OpenAIServingTranslation,
            )

            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=self.device,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.config.trust_remote_code,
                model_kwargs=self.config.model_kwargs,
            )
            self.serving_transcription = OpenAIServingTranscription(pipe, self.model_config.name, self.config)
            self.serving_translation = OpenAIServingTranslation(pipe, self.model_config.name, self.config)
            self._serving.append(self.serving_transcription)
            self._serving.append(self.serving_translation)

        elif usecase is ModelUsecase.tts:
            from transformers import pipeline

            from modelship.infer.transformers.openai.serving_speech import OpenAIServingSpeech

            pipe = pipeline(
                "text-to-audio",
                model=model_name,
                device=self.device,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.config.trust_remote_code,
                model_kwargs=self.config.model_kwargs,
            )
            self.serving_speech = OpenAIServingSpeech(pipe, self.model_config.name, self.config)
            self._serving.append(self.serving_speech)

        logger.info(
            "TransformersInfer started for %s (usecase=%s, device=%s)", self.model_config.name, usecase, self.device
        )

    async def warmup(self) -> None:
        for serving in self._serving:
            await serving.warmup()

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ChatCompletionResponse | AsyncGenerator[str, None]:
        if not hasattr(self, "serving_chat"):
            return await super().create_chat_completion(request, raw_request)
        return await self.serving_chat.create_chat_completion(request, raw_request)

    async def create_embedding(
        self, request: EmbeddingRequest, raw_request: RawRequestProxy
    ) -> EmbeddingResponse | ErrorResponse:
        if not hasattr(self, "serving_embedding"):
            return await super().create_embedding(request, raw_request)
        return await self.serving_embedding.create_embedding(request, raw_request)

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose | AsyncGenerator[str, None]:
        if not hasattr(self, "serving_transcription"):
            return await super().create_transcription(audio_data, request, raw_request)
        return await self.serving_transcription.create_transcription(audio_data, request, raw_request)

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranslationResponse | TranslationResponseVerbose | AsyncGenerator[str, None]:
        if not hasattr(self, "serving_translation"):
            return await super().create_translation(audio_data, request, raw_request)
        return await self.serving_translation.create_translation(audio_data, request, raw_request)

    async def create_speech(
        self, request: SpeechRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | RawSpeechResponse | AsyncGenerator[str, None]:
        if not hasattr(self, "serving_speech"):
            return await super().create_speech(request, raw_request)
        return await self.serving_speech.create_speech(request, raw_request)
