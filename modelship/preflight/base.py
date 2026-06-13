from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Protocol

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig
from modelship.logging import get_logger

logger = get_logger("preflight")


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

    return HardwareProfile(gpus=detect_gpus(), ram_bytes=detect_ram_bytes(), cpu_count=os.cpu_count() or 0)


def detect_gpus() -> list[GPUInfo]:
    """GPUs visible to this process, with free VRAM.

    `torch.cuda` first (honors `CUDA_VISIBLE_DEVICES`, i.e. the actor's assigned
    GPUs); `pynvml` node-level fallback when the actor owns no GPU directly
    (vLLM ray-backend spawns worker sub-actors that hold them). On the driver
    there's no mask, so this sees all physical GPUs — which is what the profile
    generator wants for VRAM tiering. Split out of `discover_hardware` so deploy
    code can read just the GPUs."""
    gpus = _torch_cuda_discover()
    if not gpus:
        gpus = _pynvml_node_discover()
        if gpus:
            logger.debug(
                "preflight: actor has no direct GPU ownership; using node-level pynvml view (%d GPU(s))", len(gpus)
            )
    return gpus


def detect_ram_bytes() -> int:
    """Total RAM available to *this* process, honoring a container memory cap.

    psutil reads /proc/meminfo, which the kernel does NOT namespace per
    container — so inside a memory-capped container it reports the HOST's RAM,
    not the cgroup limit. Sizing a model against host RAM would OOM-kill a capped
    container. The real ceiling lives in the cgroup pseudo-files; we take the
    tighter of psutil and the cgroup limit. Returns 0 if RAM can't be read.

    """
    ram_bytes = 0
    try:
        import psutil

        ram_bytes = int(psutil.virtual_memory().total)
    except Exception:
        logger.debug("preflight: psutil probe failed; ram_bytes=0", exc_info=True)

    cgroup_limit = _cgroup_memory_limit_bytes()
    if cgroup_limit is not None:
        if ram_bytes > 0:
            capped = min(ram_bytes, cgroup_limit)
            if capped < ram_bytes:
                logger.debug(
                    "preflight: applying cgroup memory limit %.2f GiB (host reports %.2f GiB)",
                    capped / 1024**3,
                    ram_bytes / 1024**3,
                )
            ram_bytes = capped
        else:
            # psutil failed but the cgroup limit is readable — use it rather than
            # returning 0 (which would fail sizing / refuse the deploy).
            logger.debug("preflight: psutil unavailable; using cgroup memory limit %.2f GiB", cgroup_limit / 1024**3)
            ram_bytes = cgroup_limit

    return ram_bytes


def _cgroup_memory_limit_bytes(
    paths: tuple[str, ...] = (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ),
) -> int | None:
    """Return the container's memory ceiling from cgroup, or None if unlimited
    or not containerized. Checks cgroup v2 (`memory.max`) then v1
    (`memory.limit_in_bytes`). Mirrors Ray's `get_system_memory()`: the caller
    takes `min()` with the psutil host value, which naturally discards cgroup
    v1's astronomically large "unlimited" sentinel — so no magic threshold is
    needed. Returns None on any read/parse failure so the caller keeps the host
    value. `paths` is a parameter only so tests can point it at temp files."""
    for path in paths:
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw == "max":  # cgroup v2 "unlimited"
            return None
        try:
            return int(raw)
        except ValueError:
            continue
    return None


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
            from modelship.preflight.llama_cpp import LlamaCppPreflight

            register(ModelLoader.llama_cpp, LlamaCppPreflight())
        except Exception:
            logger.debug("preflight: LlamaCppPreflight registration skipped", exc_info=True)
    if ModelLoader.stable_diffusion_cpp not in _REGISTRY:
        try:
            from modelship.preflight.stable_diffusion_cpp import StableDiffusionCppPreflight

            register(ModelLoader.stable_diffusion_cpp, StableDiffusionCppPreflight())
        except Exception:
            logger.debug("preflight: StableDiffusionCppPreflight registration skipped", exc_info=True)
    if ModelLoader.vllm in _REGISTRY:
        return
    try:
        from modelship.preflight.vllm import VllmPreflight

        register(ModelLoader.vllm, VllmPreflight())
    except Exception:
        logger.debug("preflight: VllmPreflight registration skipped", exc_info=True)
