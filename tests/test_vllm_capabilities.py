from types import SimpleNamespace

from modelship.infer.vllm.capabilities import VllmCapabilities


def test_multimodal_model_supports_image():
    mc = SimpleNamespace(is_multimodal_model=True)
    caps = VllmCapabilities.detect(mc)  # type: ignore[arg-type]
    assert caps.supports_image is True
    assert caps.supports_audio is False


def test_text_only_model_is_text_only():
    mc = SimpleNamespace(is_multimodal_model=False)
    caps = VllmCapabilities.detect(mc)  # type: ignore[arg-type]
    assert caps.supports_image is False
    assert caps.supports_audio is False
