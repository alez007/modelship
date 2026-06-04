"""
Base class for custom model plugins.

Plugins are protocol-agnostic engines. Each `create_*` method takes raw
inputs and returns raw outputs. OpenAI (or any other) protocol shaping is
done by the serving wrappers, not the plugin itself. Override only the
methods matching the plugin's `usecase`; unimplemented methods return a 404
error.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Protocol

from modelship.infer.infer_config import ModelshipModelConfig
from modelship.openai.protocol import (
    ErrorInfo,
    ErrorResponse,
    RawChatCompletion,
    RawChatDelta,
    RawSpeechResponse,
    RawTranscription,
    RawTranslation,
)

if TYPE_CHECKING:
    from modelship.infer.preflight import HardwareProfile

_NOT_SUPPORTED = ErrorResponse(
    error=ErrorInfo(message="plugin does not support this action", type="invalid_request_error")
)
_NOT_SUPPORTED._http_status = 404


class BasePlugin(ABC):
    @abstractmethod
    def __init__(self, model_config: ModelshipModelConfig):
        pass

    @abstractmethod
    async def start(self):
        pass

    def max_context_length(self) -> int | None:
        return None

    @classmethod
    def preflight(
        cls,
        config: ModelshipModelConfig,
        hw: "HardwareProfile",
    ) -> dict[str, Any]:
        """Optional hook: inspect hardware and recommend safe defaults for
        this plugin's configuration. Returns a dict keyed on `plugin_config`
        field names. User-supplied values in `models.yaml` always override
        these recommendations. Default no-op."""
        return {}

    async def create_chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        stream: bool = False,
        request_id: str | None = None,
    ) -> RawChatCompletion | AsyncGenerator[RawChatDelta, None] | ErrorResponse:
        return _NOT_SUPPORTED

    async def create_embedding(
        self,
        input: list[str],
        request_id: str | None = None,
    ) -> list[list[float]] | ErrorResponse:
        return _NOT_SUPPORTED

    async def create_transcription(
        self,
        audio_data: bytes,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        request_id: str | None = None,
    ) -> RawTranscription | ErrorResponse:
        return _NOT_SUPPORTED

    async def create_translation(
        self,
        audio_data: bytes,
        prompt: str | None = None,
        temperature: float | None = None,
        request_id: str | None = None,
    ) -> RawTranslation | ErrorResponse:
        return _NOT_SUPPORTED

    async def create_speech(
        self,
        input: str,
        voice: str | None = None,
        speed: float | None = None,
        stream: bool = False,
        request_id: str | None = None,
    ) -> RawSpeechResponse | AsyncGenerator[tuple[bytes, int], None] | ErrorResponse:
        """Synthesize speech. Non-stream returns a full `RawSpeechResponse`;
        stream yields `(pcm_bytes, sample_rate)` tuples where `pcm_bytes` is
        signed 16-bit little-endian mono PCM."""
        return _NOT_SUPPORTED

    async def create_image_generation(
        self,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        request_id: str | None = None,
    ) -> list[bytes] | ErrorResponse:
        """Returns a list of PNG-encoded image bytes."""
        return _NOT_SUPPORTED

    async def create_image_edit(
        self,
        image_data: bytes,
        mask_data: bytes | None = None,
        prompt: str | None = None,
        n: int = 1,
        size: str | None = None,
        strength: float | None = None,
        request_id: str | None = None,
    ) -> list[bytes] | ErrorResponse:
        """Returns a list of PNG-encoded image bytes."""
        return _NOT_SUPPORTED

    async def create_image_variation(
        self,
        image_data: bytes,
        n: int = 1,
        size: str | None = None,
        strength: float | None = None,
        request_id: str | None = None,
    ) -> list[bytes] | ErrorResponse:
        """Returns a list of PNG-encoded image bytes."""
        return _NOT_SUPPORTED


class PluginProto(Protocol):
    ModelPlugin: type[BasePlugin]
