"""The curated model catalog behind `MSHIP_MODEL_STACK`.

Shape (kept deliberately small and hand-verifiable):
- **profiles** = a capability set (which `ModelUsecase`s to serve).
- For each usecase there is a **candidate pool** per accelerator. The selector runs
  a weighted knapsack over these pools (one pick per capability) — there is no
  "ladder vs satellite" split; embed/tts/transcription are just smaller pools.

Each spec carries:
- `footprint_bytes` — coarse VRAM (vllm/diffusers) or RAM (CPU loaders); the GPU
  figure still gates VRAM bin-packing in the selector/generator.
- `req_min` / `req_rec` — each a `ModeReq(cpu cores, ram bytes, weight)`: the
  resources the model needs running at minimum vs. comfortably, plus that mode's
  quality `weight`. The knapsack fits the cpu/ram scalars against the host's free
  cpu/RAM and maximises total weight. Weights are hand-set so a smaller model at
  `req_rec` can outscore a bigger model at `req_min` (the tuning lever) — keep that
  ordering in mind when editing them.

Every `model:` here was verified live against HF: it resolves AND is ungated, so
the one-click path needs no HF token. Gated models (FLUX, SD3.5) are deliberately
excluded; a user can swap them in by editing the generated yaml + setting HF_TOKEN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modelship.deploy.profiles.tiers import Accelerator
from modelship.infer.infer_config import ModelLoader, ModelUsecase

_GiB = 1024**3


@dataclass(frozen=True)
class ModeReq:
    """One resource mode (minimum or recommended) of a model: the host resources it
    needs plus that mode's quality `weight`. cpu = cores; ram in bytes. VRAM is NOT
    here — on a GPU box vllm/diffusers expand to whatever the card allows, so the
    VRAM check stays the coarse `footprint_bytes` gate, not a min/rec scalar."""

    cpu: float
    ram_bytes: int
    weight: float


def _req(cpu: float, ram_gib: float, weight: float) -> ModeReq:
    return ModeReq(cpu=cpu, ram_bytes=int(ram_gib * _GiB), weight=weight)


@dataclass(frozen=True)
class ModelSpec:
    """One catalog entry — enough to build a `models.yaml` model minus the resource
    allocation, which the generator fills from the budget.

    `loader_config` is the inner loader-config dict (e.g. sd.cpp's
    `sample_steps`/`cfg_scale`); the generator wraps it under the loader's field
    name. `footprint_bytes` is coarse — it gates VRAM bin-packing, not runtime
    sizing (the per-loader preflight does that at deploy time)."""

    model: str
    loader: ModelLoader
    usecase: ModelUsecase
    footprint_bytes: int
    req_min: ModeReq
    req_rec: ModeReq
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


# --- Generate pool ------------------------------------------------------------

_GENERATE_CPU = (
    ModelSpec(
        model="bartowski/Qwen2.5-1.5B-Instruct-GGUF:*Q4_K_M.gguf",
        loader=ModelLoader.llama_server,
        usecase=ModelUsecase.generate,
        footprint_bytes=int(1.5 * _GiB),
        req_min=_req(1, 1.8, 18),
        req_rec=_req(2, 2.5, 35),
    ),
    ModelSpec(
        model="bartowski/Llama-3.2-3B-Instruct-GGUF:*Q4_K_M.gguf",
        loader=ModelLoader.llama_server,
        usecase=ModelUsecase.generate,
        footprint_bytes=int(3.0 * _GiB),
        req_min=_req(2, 3.0, 30),
        req_rec=_req(4, 4.5, 50),
    ),
    ModelSpec(
        model="bartowski/Qwen2.5-7B-Instruct-GGUF:*Q4_K_M.gguf",
        loader=ModelLoader.llama_server,
        usecase=ModelUsecase.generate,
        footprint_bytes=int(6.0 * _GiB),
        req_min=_req(4, 5.0, 45),
        req_rec=_req(6, 7.0, 80),
    ),
    ModelSpec(
        model="bartowski/Qwen2.5-14B-Instruct-GGUF:*Q4_K_M.gguf",
        loader=ModelLoader.llama_server,
        usecase=ModelUsecase.generate,
        footprint_bytes=int(11.0 * _GiB),
        req_min=_req(6, 9.5, 70),
        req_rec=_req(8, 12.0, 120),
    ),
)

_GENERATE_GPU = (
    ModelSpec(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        loader=ModelLoader.vllm,
        usecase=ModelUsecase.generate,
        footprint_bytes=int(6.0 * _GiB),
        req_min=_req(1, 2.0, 45),
        req_rec=_req(2, 4.0, 80),
    ),
    ModelSpec(
        model="Qwen/Qwen2.5-14B-Instruct-AWQ",
        loader=ModelLoader.vllm,
        usecase=ModelUsecase.generate,
        footprint_bytes=int(11.0 * _GiB),
        req_min=_req(1, 3.0, 70),
        req_rec=_req(2, 5.0, 120),
    ),
    ModelSpec(
        model="Qwen/Qwen2.5-32B-Instruct-AWQ",
        loader=ModelLoader.vllm,
        usecase=ModelUsecase.generate,
        footprint_bytes=int(20.0 * _GiB),
        req_min=_req(2, 4.0, 100),
        req_rec=_req(2, 6.0, 170),
    ),
)


# --- Image pool ---------------------------------------------------------------
# Turbo checkpoints are distilled for ~4 steps with no CFG; SDXL-base wants the
# full ~30 steps. Step counts matter most on CPU (fewer = faster).

_IMAGE_CPU = (
    ModelSpec(
        model="stabilityai/sd-turbo:sd_turbo.safetensors",
        loader=ModelLoader.stable_diffusion_cpp,
        usecase=ModelUsecase.image,
        footprint_bytes=int(2.5 * _GiB),
        req_min=_req(2, 2.5, 25),
        req_rec=_req(4, 4.0, 40),
        loader_config={"sample_steps": 4, "cfg_scale": 1.0, "wtype": "q8_0"},
    ),
    ModelSpec(
        model="stabilityai/sdxl-turbo:sd_xl_turbo_1.0_fp16.safetensors",
        loader=ModelLoader.stable_diffusion_cpp,
        usecase=ModelUsecase.image,
        footprint_bytes=int(6.0 * _GiB),
        req_min=_req(4, 6.0, 50),
        req_rec=_req(6, 8.0, 80),
        loader_config={"sample_steps": 4, "cfg_scale": 1.0, "wtype": "q8_0"},
    ),
    ModelSpec(
        model="stabilityai/stable-diffusion-xl-base-1.0:sd_xl_base_1.0.safetensors",
        loader=ModelLoader.stable_diffusion_cpp,
        usecase=ModelUsecase.image,
        footprint_bytes=int(7.0 * _GiB),
        req_min=_req(4, 7.0, 70),
        req_rec=_req(8, 10.0, 110),
        loader_config={"sample_steps": 30, "cfg_scale": 7.0, "wtype": "q8_0"},
    ),
)

_IMAGE_GPU = (
    ModelSpec(
        model="stabilityai/sd-turbo",
        loader=ModelLoader.diffusers,
        usecase=ModelUsecase.image,
        footprint_bytes=int(4.0 * _GiB),
        req_min=_req(1, 2.0, 25),
        req_rec=_req(2, 3.0, 40),
        loader_config={"num_inference_steps": 4, "guidance_scale": 0.0},
    ),
    ModelSpec(
        model="stabilityai/sdxl-turbo",
        loader=ModelLoader.diffusers,
        usecase=ModelUsecase.image,
        footprint_bytes=int(7.0 * _GiB),
        req_min=_req(1, 2.0, 50),
        req_rec=_req(2, 3.0, 80),
        loader_config={"num_inference_steps": 4, "guidance_scale": 0.0},
    ),
    ModelSpec(
        model="playgroundai/playground-v2.5-1024px-aesthetic",
        loader=ModelLoader.diffusers,
        usecase=ModelUsecase.image,
        footprint_bytes=int(8.0 * _GiB),
        req_min=_req(1, 3.0, 75),
        req_rec=_req(2, 4.0, 120),
        loader_config={"num_inference_steps": 30, "guidance_scale": 3.0},
    ),
)


# --- Satellite pools (CPU-pinned, same on every accelerator) -------------------

_EMBED = (
    ModelSpec(
        model="nomic-ai/nomic-embed-text-v1.5-GGUF:*f16.gguf",
        loader=ModelLoader.llama_server,
        usecase=ModelUsecase.embed,
        footprint_bytes=int(0.6 * _GiB),
        req_min=_req(0.5, 0.6, 7),
        req_rec=_req(1, 1.0, 10),
    ),
)

_TTS = (
    ModelSpec(
        model="hexgrad/Kokoro-82M",
        loader=ModelLoader.custom,
        usecase=ModelUsecase.tts,
        footprint_bytes=int(0.5 * _GiB),
        req_min=_req(1, 0.5, 7),
        req_rec=_req(2, 1.0, 10),
        plugin="kokoroonnx",
        plugin_config={"onnx_provider": "CPUExecutionProvider"},
    ),
)

# whisper.cpp model *names* (pywhispercpp downloads ggml by name). tiny is the floor
# rung for the smallest boxes (~75 MB; swap to `tiny.en` for English-only HA voice);
# base/small are better and outweigh it, so the knapsack only falls to tiny when it
# must.
_TRANSCRIPTION = (
    ModelSpec(
        model="tiny",
        loader=ModelLoader.custom,
        usecase=ModelUsecase.transcription,
        footprint_bytes=int(0.2 * _GiB),
        req_min=_req(1, 0.3, 10),
        req_rec=_req(1, 0.5, 18),
        plugin="whispercpp",
        plugin_config={"n_threads": 2},
    ),
    ModelSpec(
        model="base",
        loader=ModelLoader.custom,
        usecase=ModelUsecase.transcription,
        footprint_bytes=int(0.4 * _GiB),
        req_min=_req(1, 0.4, 20),
        req_rec=_req(2, 0.8, 30),
        plugin="whispercpp",
        plugin_config={"n_threads": 2},
    ),
    ModelSpec(
        model="small",
        loader=ModelLoader.custom,
        usecase=ModelUsecase.transcription,
        footprint_bytes=int(0.7 * _GiB),
        req_min=_req(2, 0.7, 40),
        req_rec=_req(2, 1.2, 60),
        plugin="whispercpp",
        plugin_config={"n_threads": 2},
    ),
)


# Per-usecase candidate pools, indexed by accelerator. Satellites list the same
# tuple under both accelerators (they're always CPU-pinned).
_CANDIDATES: dict[ModelUsecase, dict[Accelerator, tuple[ModelSpec, ...]]] = {
    ModelUsecase.generate: {Accelerator.cpu: _GENERATE_CPU, Accelerator.gpu: _GENERATE_GPU},
    ModelUsecase.image: {Accelerator.cpu: _IMAGE_CPU, Accelerator.gpu: _IMAGE_GPU},
    ModelUsecase.embed: {Accelerator.cpu: _EMBED, Accelerator.gpu: _EMBED},
    ModelUsecase.tts: {Accelerator.cpu: _TTS, Accelerator.gpu: _TTS},
    ModelUsecase.transcription: {Accelerator.cpu: _TRANSCRIPTION, Accelerator.gpu: _TRANSCRIPTION},
}


def candidates(usecase: ModelUsecase, accel: Accelerator) -> tuple[ModelSpec, ...]:
    """The candidate model pool for `usecase` on this accelerator (ascending
    footprint). Empty tuple if the catalog has no models for it."""
    return _CANDIDATES.get(usecase, {}).get(accel, ())
