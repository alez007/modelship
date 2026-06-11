from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Protocol

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig
from modelship.logging import get_logger

logger = get_logger("infer.preflight")


@dataclass(frozen=True)
class GPUInfo:
    index: int
    available_bytes: int  # free VRAM at preflight time, not the device's total capacity
    name: str


@dataclass(frozen=True)
class HardwareProfile:
    """Per-actor view of the hardware Ray has assigned. GPU indices here are
    CUDA-visible indices (i.e. already filtered through `CUDA_VISIBLE_DEVICES`)."""

    gpus: list[GPUInfo] = field(default_factory=list)
    ram_bytes: int = 0
    cpu_count: int = 0


class BasePreflight(Protocol):
    def recommend(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        """Return a dict keyed on the loader config's field names. Empty dict
        means no recommendation (estimator can't reason about this config)."""
        ...


_REGISTRY: dict[ModelLoader, BasePreflight] = {}


def register(loader: ModelLoader, impl: BasePreflight) -> None:
    _REGISTRY[loader] = impl


def get_preflight(loader: ModelLoader) -> BasePreflight | None:
    return _REGISTRY.get(loader)


def discover_hardware() -> HardwareProfile:
    """Snapshot the hardware available to this deployment.

    Tries two layers, in order:
    1. `torch.cuda` (honors `CUDA_VISIBLE_DEVICES`) — accurate when Ray
       gave the actor direct GPU ownership (single-GPU, or vLLM mp backend).
    2. `pynvml` at the node level — needed when the actor itself owns no
       GPUs because vLLM ray-backend spawns worker sub-actors that hold them
       (see `deploy/actor_options.py`). Falls back to physical-node GPUs
       because TP workers are co-located on the same node anyway.
    """
    import os

    gpus = _torch_cuda_discover()
    if not gpus:
        gpus = _pynvml_node_discover()
        if gpus:
            logger.debug(
                "preflight: actor has no direct GPU ownership; using node-level pynvml view (%d GPU(s))", len(gpus)
            )

    ram_bytes = 0
    try:
        import psutil

        ram_bytes = int(psutil.virtual_memory().total)
    except Exception:
        logger.debug("preflight: psutil probe failed; ram_bytes=0", exc_info=True)

    return HardwareProfile(gpus=gpus, ram_bytes=ram_bytes, cpu_count=os.cpu_count() or 0)


def _torch_cuda_discover() -> list[GPUInfo]:
    try:
        import torch
    except Exception:
        logger.debug("preflight: torch import failed", exc_info=True)
        return []
    try:
        if not torch.cuda.is_available():
            return []
        gpus: list[GPUInfo] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            try:
                free, _total = torch.cuda.mem_get_info(i)
                available = int(free)
            except Exception:
                # mem_get_info needs a CUDA context. If it fails (e.g. no
                # context yet and lazy init refuses), fall back to total —
                # better than nothing, the runtime ValueError will catch it.
                available = int(props.total_memory)
            gpus.append(GPUInfo(index=i, available_bytes=available, name=props.name))
        return gpus
    except Exception:
        logger.debug("preflight: torch.cuda probe failed", exc_info=True)
        return []


def _pynvml_node_discover() -> list[GPUInfo]:
    """Query the physical node's GPUs via NVML. Ignores `CUDA_VISIBLE_DEVICES`
    so we can see GPUs Ray will hand to vLLM worker sub-actors.

    Imports `pynvml`, which on modern installs resolves to NVIDIA's official
    `nvidia-ml-py` package (the abandoned third-party `pynvml` package was
    deprecated in 2023; both register the same module name). `nvidia-ml-py`
    is already pinned transitively by vllm/torch."""
    try:
        import pynvml
    except Exception:
        logger.debug("preflight: pynvml not installed; node GPU discovery unavailable")
        return []
    try:
        pynvml.nvmlInit()
    except Exception:
        logger.debug("preflight: nvmlInit failed; node GPU discovery unavailable", exc_info=True)
        return []
    try:
        gpus: list[GPUInfo] = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            gpus.append(GPUInfo(index=i, available_bytes=int(mem.free), name=name))
        return gpus
    except Exception:
        logger.debug("preflight: pynvml node discovery failed", exc_info=True)
        return []
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def run_preflight(config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
    """Look up the loader's estimator and run it. Returns `{}` if no estimator
    is registered or the estimator declines (no resolved path, missing config,
    etc.). For `loader='custom'`, dispatches to the plugin's
    `ModelPlugin.preflight()` classmethod via a registered adapter.
    Never raises — preflight failures must not block a deploy."""
    # Register-on-first-call so importing this module doesn't pull in
    # backend-specific deps (vllm, transformers) when they're not installed.
    _ensure_registered()

    impl = get_preflight(config.loader)
    if impl is None:
        return {}
    try:
        return impl.recommend(config, hw)
    except Exception:
        logger.exception("preflight estimator raised for '%s'; ignoring recommendation", config.name)
        return {}


def merge_with_user_overrides(
    recommendation: dict[str, Any],
    user_overrides: dict[str, Any],
    *,
    model_name: str,
) -> dict[str, Any]:
    """`final = {**recommendation, **user_overrides}` with a warning logged
    for every key the user overrode to a different value."""
    for key, rec_value in recommendation.items():
        if key in user_overrides and user_overrides[key] != rec_value:
            logger.warning(
                "preflight: '%s' suggested %s=%r based on hardware budget, "
                "user config specifies %r — proceeding with user value",
                model_name,
                key,
                rec_value,
                user_overrides[key],
            )
    return {**recommendation, **user_overrides}


class _CustomPluginPreflight:
    """Dispatch adapter for `loader='custom'`. Imports `config.plugin` and
    delegates to its `ModelPlugin.preflight()` classmethod. The outer
    `run_preflight()` already swallows exceptions, so import or attribute
    errors propagate up to be logged there."""

    def recommend(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        if config.plugin is None:
            return {}
        import importlib

        module = importlib.import_module(config.plugin)
        plugin_cls = getattr(module, "ModelPlugin", None)
        if plugin_cls is None:
            return {}
        return plugin_cls.preflight(config, hw) or {}


def _ensure_registered() -> None:
    if ModelLoader.custom not in _REGISTRY:
        register(ModelLoader.custom, _CustomPluginPreflight())
    if ModelLoader.llama_cpp not in _REGISTRY:
        try:
            from modelship.infer.preflight.llama_cpp import LlamaCppPreflight

            register(ModelLoader.llama_cpp, LlamaCppPreflight())
        except Exception:
            logger.debug("preflight: LlamaCppPreflight registration skipped", exc_info=True)
    if ModelLoader.stable_diffusion_cpp not in _REGISTRY:
        try:
            from modelship.infer.preflight.stable_diffusion_cpp import StableDiffusionCppPreflight

            register(ModelLoader.stable_diffusion_cpp, StableDiffusionCppPreflight())
        except Exception:
            logger.debug("preflight: StableDiffusionCppPreflight registration skipped", exc_info=True)
    if ModelLoader.vllm in _REGISTRY:
        return
    try:
        from modelship.infer.preflight.vllm import VllmPreflight

        register(ModelLoader.vllm, VllmPreflight())
    except Exception:
        logger.debug("preflight: VllmPreflight registration skipped", exc_info=True)
