"""Modality capability detection for a loaded vLLM engine."""

from dataclasses import dataclass

from vllm.config.model import ModelConfig


@dataclass(frozen=True)
class VllmCapabilities:
    """Modalities the underlying vLLM model can ingest."""

    supports_image: bool
    supports_audio: bool = False  # audio chat input not wired through the vLLM serving_chat path

    @classmethod
    def detect(cls, model_config: ModelConfig) -> "VllmCapabilities":
        # vLLM's own ModelConfig inspects the loaded HF config and registry to
        # decide whether the architecture is multimodal — far more reliable than
        # sniffing model names or task strings.
        return cls(supports_image=bool(model_config.is_multimodal_model))
