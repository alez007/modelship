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
    available_ram_bytes: int = 0
    cpu_count: int = 0

    @property
    def sizing_ram_bytes(self) -> int:
        """RAM a loader should size itself against: free RAM when known, else total.
        Free reflects what's left after co-resident models, so a model deployed last
        doesn't oversize and OOM its neighbours; total is the fallback when the
        available probe read nothing."""
        return self.available_ram_bytes or self.ram_bytes


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

    return HardwareProfile(
        gpus=detect_gpus(),
        ram_bytes=detect_ram_bytes(),
        available_ram_bytes=detect_available_ram_bytes(),
        cpu_count=os.cpu_count() or 0,
    )


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
    tighter of psutil and the cgroup limit. Returns 0 if RAM can't be read."""
    host_total = 0
    try:
        import psutil

        host_total = int(psutil.virtual_memory().total)
    except Exception:
        logger.debug("preflight: psutil total-RAM probe failed", exc_info=True)
    return _tighter_ram(host_total, _cgroup_memory_limit_bytes(), what="memory limit")


def detect_available_ram_bytes() -> int:
    """RAM currently *free* for new allocations, honoring a container memory cap.

    Same host-vs-cgroup reconciliation as `detect_ram_bytes`, but for headroom
    rather than the ceiling — lets a model size against what's left after
    co-resident models, not the whole box. `psutil.virtual_memory().available`
    is cache-aware (counts reclaimable page cache as free) but NOT
    cgroup-namespaced, so inside a cap it reads the host's headroom and
    overestimates; we take the tighter of it and the cgroup's own headroom.
    Returns 0 only if neither signal is readable."""
    host_available = 0
    try:
        import psutil

        host_available = int(psutil.virtual_memory().available)
    except Exception:
        logger.debug("preflight: psutil available-RAM probe failed", exc_info=True)
    return _tighter_ram(host_available, _cgroup_memory_available_bytes(), what="memory headroom")


def _tighter_ram(host_bytes: int, cgroup_bytes: int | None, *, what: str) -> int:
    """Reconcile a host psutil reading with the cgroup's: take the tighter when
    both are present, fall back to whichever is readable, 0 if neither is. Shared
    by the total and available probes (only the two inputs differ)."""
    if cgroup_bytes is None:
        return host_bytes  # uncapped or unreadable cgroup — trust the host value
    if host_bytes <= 0:
        # psutil failed but the cgroup value is readable — use it rather than 0.
        logger.debug("preflight: psutil unavailable; using cgroup %s %.2f GiB", what, cgroup_bytes / 1024**3)
        return cgroup_bytes
    if cgroup_bytes < host_bytes:
        logger.debug(
            "preflight: cgroup %s %.2f GiB binds (host reports %.2f GiB)",
            what,
            cgroup_bytes / 1024**3,
            host_bytes / 1024**3,
        )
    return min(host_bytes, cgroup_bytes)


# cgroup v1 reports "unlimited" as a near-INT64_MAX sentinel: PAGE_COUNTER_MAX
# (LONG_MAX rounded down to the page size) = 0x7FFFFFFFFFFFF000, and some kernels
# report LONG_MAX itself. Both are >= this value. No real machine has ~9.2 EiB of
# RAM, so treating anything this large as "no limit" has zero false positives.
_CGROUP_V1_UNLIMITED = 0x7FFFFFFFFFFFF000


def _cgroup_memory_limit_bytes(
    paths: tuple[str, ...] = (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ),
) -> int | None:
    """Return the container's memory ceiling from cgroup, or None if unlimited
    or not containerized. Checks cgroup v2 (`memory.max` == "max") then v1
    (`memory.limit_in_bytes` == the near-INT64_MAX sentinel). Detecting the v1
    sentinel here — rather than relying on the caller's `min()` with psutil — keeps
    the value safe even when psutil is unavailable (e.g. `detect_ram_bytes`'s
    fallback). Returns None on any read/parse failure so the caller keeps the host
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
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= _CGROUP_V1_UNLIMITED:  # cgroup v1 "unlimited" / nonsensical
            return None
        return value
    return None


def _cgroup_memory_available_bytes(
    usage_paths: tuple[str, ...] = (
        "/sys/fs/cgroup/memory.current",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",  # cgroup v1
    ),
    stat_paths: tuple[str, ...] = (
        "/sys/fs/cgroup/memory.stat",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.stat",  # cgroup v1
    ),
) -> int | None:
    """Free RAM inside the container's memory cgroup, or None when uncapped/unreadable.

    `limit - current + reclaimable`: current usage counts page cache, but the kernel
    will evict reclaimable file cache under pressure so it isn't really "used". We
    add back `inactive_file + active_file` (v2; `total_*_file` v1) from memory.stat.
    If memory.stat is unreadable we treat reclaimable as 0 — conservative (smaller
    headroom). Each pseudo-file read is isolated; a parse failure skips that signal
    rather than raising. `*_paths` are parameters only so tests can use temp files."""
    limit = _cgroup_memory_limit_bytes()
    if limit is None:  # uncapped — defer to the host (psutil) reading
        return None
    current = _read_first_int(usage_paths)
    if current is None:
        return None
    reclaimable = _cgroup_reclaimable_cache_bytes(stat_paths) or 0
    return max(0, limit - current + reclaimable)


def _read_first_int(paths: tuple[str, ...]) -> int | None:
    """Read the first readable path as a single integer, else None."""
    for path in paths:
        try:
            with open(path) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            continue
    return None


def _cgroup_reclaimable_cache_bytes(stat_paths: tuple[str, ...]) -> int | None:
    """Sum the evictable file-cache from memory.stat. None if no memory.stat is
    readable; 0 if it's readable but lists no cache keys.

    cgroup v1 lists BOTH the hierarchical `total_*_file` and the per-cgroup
    `*_file` lines, so summing all keys double-counts. We prefer the `total_*`
    pair when present (v1, hierarchical — the right figure under a cap) and fall
    back to the plain `inactive_file`/`active_file` pair (v2 has only those)."""
    for path in stat_paths:
        try:
            with open(path) as f:
                raw = f.read()
        except OSError:
            continue
        values: dict[str, int] = {}
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) == 2:
                with contextlib.suppress(ValueError):
                    values[parts[0]] = int(parts[1])
        if "total_inactive_file" in values:  # cgroup v1 — use the hierarchical pair only
            return values.get("total_inactive_file", 0) + values.get("total_active_file", 0)
        return values.get("inactive_file", 0) + values.get("active_file", 0)
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
    if ModelLoader.llama_server not in _REGISTRY:
        try:
            from modelship.preflight.llama_cpp import LlamaServerPreflight

            register(ModelLoader.llama_server, LlamaServerPreflight())
        except Exception:
            logger.debug("preflight: LlamaServerPreflight registration skipped", exc_info=True)
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
