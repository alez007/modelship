"""Tests for the StableDiffusionCppPreflight estimator."""

from __future__ import annotations

from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
    StableDiffusionCppConfig,
)
from modelship.infer.preflight import HardwareProfile
from modelship.infer.preflight.stable_diffusion_cpp import (
    _VAE_TILING_RAM_THRESHOLD_BYTES,
    StableDiffusionCppPreflight,
)

_GiB = 1024**3


def _make_config(**sdcpp_kwargs) -> ModelshipModelConfig:
    return ModelshipModelConfig(
        name="test-image",
        model="org/test-sd-gguf",
        usecase=ModelUsecase.image,
        loader=ModelLoader.stable_diffusion_cpp,
        stable_diffusion_cpp_config=StableDiffusionCppConfig(**sdcpp_kwargs),
    )


def test_no_ram_returns_empty():
    rec = StableDiffusionCppPreflight().recommend(_make_config(), HardwareProfile(ram_bytes=0))
    assert rec == {}


def test_low_ram_recommends_vae_tiling():
    hw = HardwareProfile(ram_bytes=_VAE_TILING_RAM_THRESHOLD_BYTES - _GiB)
    rec = StableDiffusionCppPreflight().recommend(_make_config(), hw)
    assert rec == {"vae_tiling": True}


def test_high_ram_no_recommendation():
    hw = HardwareProfile(ram_bytes=_VAE_TILING_RAM_THRESHOLD_BYTES + _GiB)
    rec = StableDiffusionCppPreflight().recommend(_make_config(), hw)
    assert rec == {}
