import importlib
from collections.abc import AsyncGenerator
from typing import cast

from modelship.infer.base_infer import BaseInfer
from modelship.infer.custom.openai.serving_speech import OpenAIServingSpeech
from modelship.infer.custom.openai.serving_transcription import (
    OpenAIServingTranscription,
    OpenAIServingTranslation,
)
from modelship.infer.infer_config import ModelshipModelConfig, ModelUsecase, RawRequestProxy
from modelship.logging import get_logger
from modelship.openai.protocol import (
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
from modelship.plugins.base_plugin import BasePlugin, PluginProto
from modelship.preflight import discover_hardware, merge_with_user_overrides, run_preflight

logger = get_logger("infer.custom")


class CustomInfer(BaseInfer):
    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        self.custom_engine: BasePlugin | None = None
        self.serving_speech: OpenAIServingSpeech | None = None
        self.serving_transcription: OpenAIServingTranscription | None = None
        self.serving_translation: OpenAIServingTranslation | None = None

    def shutdown(self) -> None:
        pass

    async def start(self):
        plugin = self.model_config.plugin
        if plugin is not None:
            module = cast("PluginProto", importlib.import_module(plugin))
            recommendation = run_preflight(self.model_config, discover_hardware())
            if recommendation:
                logger.info("preflight recommendation for '%s': %s", self.model_config.name, recommendation)
                self.model_config.plugin_config = merge_with_user_overrides(
                    recommendation,
                    self.model_config.plugin_config or {},
                    model_name=self.model_config.name,
                )
            else:
                logger.info("preflight recommendation for '%s': none", self.model_config.name)
            self.custom_engine = module.ModelPlugin(model_config=self.model_config)
            await self.custom_engine.start()
            self._set_max_context_length(self.custom_engine.max_context_length())

        usecase = self.model_config.usecase
        if usecase is ModelUsecase.tts:
            self.serving_speech = OpenAIServingSpeech(serving_engine=self.custom_engine)
        elif usecase is ModelUsecase.transcription:
            self.serving_transcription = OpenAIServingTranscription(serving_engine=self.custom_engine)
        elif usecase is ModelUsecase.translation:
            self.serving_translation = OpenAIServingTranslation(serving_engine=self.custom_engine)

    async def warmup(self) -> None:
        pass

    async def create_speech(
        self, request: SpeechRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | RawSpeechResponse | AsyncGenerator[str, None]:
        if self.serving_speech is None:
            return await super().create_speech(request, raw_request)
        return await self.serving_speech.create_speech(request, raw_request)

    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranscriptionResponse | TranscriptionResponseVerbose | AsyncGenerator[str, None]:
        if self.serving_transcription is None:
            return await super().create_transcription(audio_data, request, raw_request)
        return await self.serving_transcription.create_transcription(audio_data, request, raw_request)

    async def create_translation(
        self, audio_data: bytes, request: TranslationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | TranslationResponse | TranslationResponseVerbose | AsyncGenerator[str, None]:
        if self.serving_translation is None:
            return await super().create_translation(audio_data, request, raw_request)
        return await self.serving_translation.create_translation(audio_data, request, raw_request)
