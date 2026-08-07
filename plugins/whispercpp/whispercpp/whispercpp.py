"""
whisper.cpp STT plugin for Modelship.

Uses pywhispercpp (Python bindings for whisper.cpp) to run Whisper locally
without PyTorch. Ideal for low-resource hosts (Intel N100 mini-PCs, ARM boards)
where a full transformers stack is too heavy.

Plugin config options (via plugin_config in models.yaml):

    models_dir:   Directory to store/load ggml model files (default: plugins dir)
    n_threads:    CPU threads for inference (default: pywhispercpp default)

Example models.yaml entry:

    - name: "whisper-base"
      model: "base.en"
      usecase: "stt"
      loader: "custom"
      plugin: "whispercpp"
      num_gpus: 0
      num_cpus: 2
      plugin_config:
        n_threads: 4

Example request:

    curl http://localhost:8000/v1/audio/transcriptions \\
      -H "Content-Type: multipart/form-data" \\
      -F file=@audio.wav \\
      -F model=whisper-base
"""

import os

from pywhispercpp.model import Model  # type: ignore[import-unresolved]

from modelship.infer.infer_config import ModelshipModelConfig
from modelship.logging import get_logger
from modelship.openai.protocol import ErrorResponse, RawSegment, RawTranscription, RawTranslation
from modelship.plugins.base_plugin import BasePlugin
from modelship.utils import plugins_dir
from modelship.utils.audio import decode_audio

logger = get_logger("plugin.whispercpp")

_WHISPER_SAMPLE_RATE = 16000


class ModelPlugin(BasePlugin):
    def __init__(self, model_config: ModelshipModelConfig):
        self.model_config = model_config
        if not model_config.model:
            raise ValueError(f"whispercpp plugin requires `model:` (deployment '{model_config.name}')")
        model_name = model_config.model
        plugin_config = model_config.plugin_config or {}

        models_dir = plugin_config.get("models_dir", f"{plugins_dir()}/whispercpp")
        os.makedirs(models_dir, exist_ok=True)

        kwargs: dict = {"models_dir": models_dir}
        if (n_threads := plugin_config.get("n_threads")) is not None:
            kwargs["n_threads"] = n_threads

        logger.info("loading whisper.cpp model: %s (dir=%s)", model_name, models_dir)
        self.model = Model(model_name, **kwargs)

    async def start(self):
        pass

    async def create_transcription(
        self,
        audio_data: bytes,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        request_id: str | None = None,
    ) -> RawTranscription | ErrorResponse:
        return self._run(audio_data, language=language, translate=False)

    async def create_translation(
        self,
        audio_data: bytes,
        prompt: str | None = None,
        temperature: float | None = None,
        request_id: str | None = None,
    ) -> RawTranslation | ErrorResponse:
        return self._run(audio_data, language=None, translate=True)

    def _run(self, audio_data: bytes, language: str | None, translate: bool) -> RawTranscription:
        samples, duration_seconds = decode_audio(audio_data, _WHISPER_SAMPLE_RATE)
        params: dict = {"translate": translate}
        if language:
            params["language"] = language

        segments = self.model.transcribe(samples, **params)

        text = "".join(s.text for s in segments).strip()
        raw_segments = [RawSegment(text=s.text.strip(), start=s.t0 / 100.0, end=s.t1 / 100.0) for s in segments]
        detected_lang = "en" if translate else language

        return RawTranscription(
            text=text,
            language=detected_lang,
            duration_seconds=float(duration_seconds),
            segments=raw_segments,
        )
