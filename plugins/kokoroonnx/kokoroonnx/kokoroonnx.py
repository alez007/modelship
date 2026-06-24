"""
Kokoro ONNX TTS plugin for Modelship.

Selected voices (full list in voices-v1.0.bin):

    American Female: af_heart, af_bella, af_nicole, af_sarah, af_sky
    American Male:   am_adam, am_michael
    British Female:  bf_emma, bf_isabella
    British Male:    bm_george, bm_lewis

Plugin config options (via plugin_config in models.yaml):

    onnx_provider: ONNX execution provider (default: "CUDAExecutionProvider")
                   Other options: "CPUExecutionProvider", "TensorrtExecutionProvider"
    sample_rate:   Resample output to this rate in Hz (default: model native ~24000)

Example request:

    curl http://localhost:8000/v1/audio/speech \\
      -H "Content-Type: application/json" \\
      -d '{"model": "kokoroonnx", "input": "Hello world", "voice": "af_heart", "response_format": "wav"}' \\
      --output speech.wav
"""

import ctypes.util
import os
import shutil
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro  # type: ignore[import-unresolved]
from kokoro_onnx.config import EspeakConfig  # type: ignore[import-unresolved]

from modelship.infer.infer_config import ModelshipModelConfig
from modelship.logging import get_logger
from modelship.openai.protocol import ErrorResponse, RawSpeechResponse
from modelship.plugins.base_plugin import BasePlugin
from modelship.utils import download, plugins_dir
from modelship.utils.audio import resample, to_pcm16, to_wav


def _resolve_system_espeak() -> EspeakConfig | None:
    """Locate the system-installed espeak-ng library + data.

    espeakng-loader bundles a library with a CI-runner data path baked in that
    SIGSEGVs at runtime; prefer the apt-installed espeak-ng when present. Falls
    back to espeakng-loader's bundled copy if the system install is missing.
    """
    lib = ctypes.util.find_library("espeak-ng") or ctypes.util.find_library("espeak")
    if not lib:
        return None
    # Typical Debian/Ubuntu layout: lib in /usr/lib/<triple>/, data next to it.
    for candidate in (
        "/usr/lib/x86_64-linux-gnu/espeak-ng-data",
        "/usr/lib/aarch64-linux-gnu/espeak-ng-data",
        "/usr/share/espeak-ng-data",
    ):
        if Path(candidate, "phontab").is_file():
            return EspeakConfig(lib_path=lib, data_path=candidate)
    return None


logger = get_logger("plugin.kokoroonnx")


class ModelPlugin(BasePlugin):
    def __init__(self, model_config: ModelshipModelConfig):
        if not shutil.which("espeak-ng") and not shutil.which("espeak"):
            raise RuntimeError(
                "espeak/espeak-ng is required by kokoroonnx but not found on this system. "
                "Install it with: apt-get install -y espeak-ng (Debian/Ubuntu) "
                "or brew install espeak (macOS)"
            )
        logger.info("onnxruntime device: %s", ort.get_device())  # pyright: ignore[reportAttributeAccessIssue]
        logger.info("available providers: %s", ort.get_available_providers())  # pyright: ignore[reportAttributeAccessIssue]

        plugin_dir = f"{plugins_dir()}/kokoroonnx"
        os.makedirs(plugin_dir, exist_ok=True)

        model_path = f"{plugin_dir}/kokoro-v1.0.onnx"
        voices_path = f"{plugin_dir}/voices-v1.0.bin"
        download(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
            model_path,
        )
        download(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
            voices_path,
        )

        onnx_provider = (model_config.plugin_config or {}).get("onnx_provider", "CUDAExecutionProvider")
        os.environ["ONNX_PROVIDER"] = onnx_provider
        logger.info("ONNX_PROVIDER=%s", onnx_provider)
        espeak_config = _resolve_system_espeak()
        if espeak_config:
            logger.info("using system espeak-ng: lib=%s data=%s", espeak_config.lib_path, espeak_config.data_path)
        else:
            logger.warning("system espeak-ng not found; falling back to bundled espeakng-loader (may crash)")
        self.kokoro = Kokoro(model_path, voices_path, espeak_config=espeak_config)
        logger.info("kokoro session providers: %s", self.kokoro.sess.get_providers())
        self.target_sample_rate: int | None = (model_config.plugin_config or {}).get("sample_rate")

    def __del__(self):
        try:
            if kokoro := getattr(self, "kokoro", None):
                del kokoro
                self.kokoro = None
        except Exception:
            pass

    async def start(self):
        pass

    def _maybe_resample(self, audio: np.ndarray, from_rate: int) -> tuple[np.ndarray, int]:
        if self.target_sample_rate is None:
            return audio, from_rate
        return resample(audio, from_rate, self.target_sample_rate), self.target_sample_rate

    async def create_speech(
        self,
        input: str,
        voice: str | None = None,
        speed: float | None = None,
        stream: bool = False,
        request_id: str | None = None,
    ) -> RawSpeechResponse | AsyncGenerator[tuple[bytes, int], None] | ErrorResponse:
        voice = voice or "af_heart"
        # Callers (e.g. the OpenAI-compatible Home Assistant integration) may request
        # OpenAI voice names like "alloy" that Kokoro doesn't have. Rather than let
        # kokoro-onnx assert, fall back to the default voice.
        if voice not in self.kokoro.voices:  # type: ignore[union-attr]
            logger.warning("voice %r not available in Kokoro; falling back to af_heart", voice)
            voice = "af_heart"
        speed = speed or 1.0
        logger.info("started generation: %s with voice: %s", input, voice)

        if stream:
            return self._stream(input, voice, speed)

        chunks: list[np.ndarray] = []
        sample_rate = self.target_sample_rate or 0
        async for audio, sr in self.kokoro.create_stream(input, voice=voice, speed=speed, lang="en-us"):  # type: ignore[union-attr]
            audio, sample_rate = self._maybe_resample(audio, sr)
            chunks.append(audio)

        logger.info("got %d chunks (sample rate %s) for input: %s", len(chunks), sample_rate, input)
        combined = np.concatenate(chunks)
        return RawSpeechResponse(audio=to_wav(combined, sample_rate), media_type="audio/wav")

    async def _stream(self, input: str, voice: str, speed: float) -> AsyncGenerator[tuple[bytes, int], None]:
        async for audio, sr in self.kokoro.create_stream(input, voice=voice, speed=speed, lang="en-us"):  # type: ignore[union-attr]
            audio, sr = self._maybe_resample(audio, sr)
            yield to_pcm16(audio), sr
