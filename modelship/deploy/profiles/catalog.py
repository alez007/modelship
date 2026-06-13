"""The curated model catalog behind `MSHIP_MODEL_STACK`.

Shape (kept deliberately small and hand-verifiable):
- **profiles** = a capability set (which `ModelUsecase`s to serve).
- **satellites** (embed / tts / transcription) = constants, CPU-pinned, the same
  on every box (transcription steps base→small once off the smallest CPU tier).
- **generate** and **image** = the only real variables — one ladder per
  accelerator, indexed by `Tier`.

Every `model:` here was verified live against HF: it resolves AND is ungated, so
the one-click path needs no HF token. Gated models (FLUX, SD3.5) are deliberately
excluded; a user can swap them in by editing the generated yaml + setting
HF_TOKEN. Each spec carries a coarse `footprint_bytes` (RAM for CPU loaders, VRAM
for vllm/diffusers) the budget-aware selector uses to bin-pack; the per-loader
preflight does the fine runtime tuning (n_ctx / vae_tiling) at deploy time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modelship.deploy.profiles.tiers import Accelerator, Tier
from modelship.infer.infer_config import ModelLoader, ModelUsecase

_GiB = 1024**3


@dataclass(frozen=True)
class ModelSpec:
    """One catalog entry — enough to build a `models.yaml` model minus the
    resource allocation, which the generator fills from the budget.

    `loader_config` is the inner loader-config dict (e.g. sd.cpp's
    `sample_steps`/`cfg_scale`); the generator wraps it under the loader's field
    name. `footprint_bytes` is coarse — it gates bin-packing, not runtime
    sizing. CPU-loader footprints draw from the RAM budget; vllm/diffusers from
    VRAM. Satellites (llama_cpp embed, custom tts/stt) always draw from RAM."""

    model: str
    loader: ModelLoader
    usecase: ModelUsecase
    footprint_bytes: int
    plugin: str | None = None
    plugin_config: dict[str, Any] | None = None
    loader_config: dict[str, Any] | None = None

    @property
    def draws_from_vram(self) -> bool:
        """vllm/diffusers consume VRAM on a GPU box; every other loader (the
        CPU-pinned ones) consumes system RAM."""
        return self.loader in (ModelLoader.vllm, ModelLoader.diffusers)


# --- Profiles: capability sets -------------------------------------------------

PROFILES: dict[str, tuple[ModelUsecase, ...]] = {
    "chat": (ModelUsecase.generate, ModelUsecase.embed),
    "assistant": (ModelUsecase.generate, ModelUsecase.transcription, ModelUsecase.tts),
    "studio": (ModelUsecase.generate, ModelUsecase.image, ModelUsecase.embed),
    "everything": (
        ModelUsecase.generate,
        ModelUsecase.image,
        ModelUsecase.embed,
        ModelUsecase.transcription,
        ModelUsecase.tts,
    ),
}


# --- Satellites: constant, CPU-pinned -----------------------------------------

_EMBED = ModelSpec(
    model="nomic-ai/nomic-embed-text-v1.5-GGUF:*f16.gguf",
    loader=ModelLoader.llama_cpp,
    usecase=ModelUsecase.embed,
    footprint_bytes=int(0.6 * _GiB),
)

_TTS = ModelSpec(
    model="hexgrad/Kokoro-82M",
    loader=ModelLoader.custom,
    usecase=ModelUsecase.tts,
    plugin="kokoroonnx",
    plugin_config={"onnx_provider": "CPUExecutionProvider"},
    footprint_bytes=int(0.5 * _GiB),
)


def _transcription(tier: Tier) -> ModelSpec:
    # whisper.cpp model *name* (pywhispercpp downloads ggml by name). base on the
    # smallest CPU tier, small everywhere else — the only tier-sensitive satellite.
    name = "base" if tier == Tier.small else "small"
    return ModelSpec(
        model=name,
        loader=ModelLoader.custom,
        usecase=ModelUsecase.transcription,
        plugin="whispercpp",
        plugin_config={"n_threads": 2},
        footprint_bytes=int((0.4 if name == "base" else 0.7) * _GiB),
    )


# --- Generate ladder (the anchor) ---------------------------------------------

_GENERATE: dict[Accelerator, tuple[ModelSpec, ModelSpec, ModelSpec]] = {
    Accelerator.cpu: (
        ModelSpec(
            model="bartowski/Llama-3.2-3B-Instruct-GGUF:*Q4_K_M.gguf",
            loader=ModelLoader.llama_cpp,
            usecase=ModelUsecase.generate,
            footprint_bytes=int(3.0 * _GiB),
        ),
        ModelSpec(
            model="bartowski/Qwen2.5-7B-Instruct-GGUF:*Q4_K_M.gguf",
            loader=ModelLoader.llama_cpp,
            usecase=ModelUsecase.generate,
            footprint_bytes=int(6.0 * _GiB),
        ),
        ModelSpec(
            model="bartowski/Qwen2.5-14B-Instruct-GGUF:*Q4_K_M.gguf",
            loader=ModelLoader.llama_cpp,
            usecase=ModelUsecase.generate,
            footprint_bytes=int(11.0 * _GiB),
        ),
    ),
    Accelerator.gpu: (
        ModelSpec(
            model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            loader=ModelLoader.vllm,
            usecase=ModelUsecase.generate,
            footprint_bytes=int(6.0 * _GiB),
        ),
        ModelSpec(
            model="Qwen/Qwen2.5-14B-Instruct-AWQ",
            loader=ModelLoader.vllm,
            usecase=ModelUsecase.generate,
            footprint_bytes=int(11.0 * _GiB),
        ),
        ModelSpec(
            model="Qwen/Qwen2.5-32B-Instruct-AWQ",
            loader=ModelLoader.vllm,
            usecase=ModelUsecase.generate,
            footprint_bytes=int(20.0 * _GiB),
        ),
    ),
}

# --- Image ladder -------------------------------------------------------------
# Turbo checkpoints are distilled for ~4 steps with no CFG; SDXL-base/playground
# want the full ~30 steps. Step counts matter most on CPU (fewer = faster).

_IMAGE: dict[Accelerator, tuple[ModelSpec, ModelSpec, ModelSpec]] = {
    Accelerator.cpu: (
        ModelSpec(
            model="stabilityai/sd-turbo:sd_turbo.safetensors",
            loader=ModelLoader.stable_diffusion_cpp,
            usecase=ModelUsecase.image,
            loader_config={"sample_steps": 4, "cfg_scale": 1.0, "wtype": "q8_0"},
            footprint_bytes=int(2.5 * _GiB),
        ),
        ModelSpec(
            model="stabilityai/sdxl-turbo:sd_xl_turbo_1.0_fp16.safetensors",
            loader=ModelLoader.stable_diffusion_cpp,
            usecase=ModelUsecase.image,
            loader_config={"sample_steps": 4, "cfg_scale": 1.0, "wtype": "q8_0"},
            footprint_bytes=int(6.0 * _GiB),
        ),
        ModelSpec(
            model="stabilityai/stable-diffusion-xl-base-1.0:sd_xl_base_1.0.safetensors",
            loader=ModelLoader.stable_diffusion_cpp,
            usecase=ModelUsecase.image,
            loader_config={"sample_steps": 30, "cfg_scale": 7.0, "wtype": "q8_0"},
            footprint_bytes=int(7.0 * _GiB),
        ),
    ),
    Accelerator.gpu: (
        ModelSpec(
            model="stabilityai/sd-turbo",
            loader=ModelLoader.diffusers,
            usecase=ModelUsecase.image,
            loader_config={"num_inference_steps": 4, "guidance_scale": 0.0},
            footprint_bytes=int(4.0 * _GiB),
        ),
        ModelSpec(
            model="stabilityai/sdxl-turbo",
            loader=ModelLoader.diffusers,
            usecase=ModelUsecase.image,
            loader_config={"num_inference_steps": 4, "guidance_scale": 0.0},
            footprint_bytes=int(7.0 * _GiB),
        ),
        ModelSpec(
            model="playgroundai/playground-v2.5-1024px-aesthetic",
            loader=ModelLoader.diffusers,
            usecase=ModelUsecase.image,
            loader_config={"num_inference_steps": 30, "guidance_scale": 3.0},
            footprint_bytes=int(8.0 * _GiB),
        ),
    ),
}


def generate_ladder(accel: Accelerator) -> tuple[ModelSpec, ModelSpec, ModelSpec]:
    """The generate anchor ladder for this accelerator, ascending S/M/L."""
    return _GENERATE[accel]


def image_ladder(accel: Accelerator) -> tuple[ModelSpec, ModelSpec, ModelSpec]:
    """The image ladder for this accelerator, ascending S/M/L."""
    return _IMAGE[accel]


def generate_at(accel: Accelerator, tier: Tier) -> ModelSpec:
    """Tier-exact generate model (no shrinking — the selector steps whole tiers)."""
    return _GENERATE[accel][tier]


def image_at(accel: Accelerator, tier: Tier) -> ModelSpec:
    """Tier-exact image model."""
    return _IMAGE[accel][tier]


def satellite(usecase: ModelUsecase, tier: Tier) -> ModelSpec:
    """The constant satellite spec for a non-generate, non-image capability."""
    if usecase == ModelUsecase.embed:
        return _EMBED
    if usecase == ModelUsecase.tts:
        return _TTS
    if usecase == ModelUsecase.transcription:
        return _transcription(tier)
    raise ValueError(f"{usecase!r} is not a satellite capability")


# Capabilities the budget-aware selector treats as size-variable ladders; every
# other capability in a profile is a constant satellite.
LADDER_USECASES = (ModelUsecase.generate, ModelUsecase.image)
