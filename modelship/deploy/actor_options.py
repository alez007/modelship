"""Ray Serve actor option construction for model deployments.

Centralises the GPU-allocation decisions and the plugin-wheel runtime_env
injection for custom-loader models. Multi-slot vLLM deploys always use a
Ray Serve placement group (one whole-GPU bundle per slot) that vLLM
inherits via its ray distributed executor.
"""

from __future__ import annotations

import os
from pathlib import Path

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig
from modelship.logging import get_logger

logger = get_logger("startup")

# Forwarded from the driver to each replica's runtime_env: logging vars, the gateway
# name (metrics.py stamps every metric with it), and MSHIP_METRICS so --no-metrics on
# the driver also disables metrics in the replicas (else they'd default to on).
_PASSTHROUGH_ENV_VARS = (
    "MSHIP_LOG_LEVEL",
    "MSHIP_LOG_FORMAT",
    "MSHIP_LOG_TARGET",
    "MSHIP_GATEWAY_NAME",
    "MSHIP_METRICS",
)


def build_passthrough_env_vars() -> dict[str, str]:
    """Driver→replica env vars (logging, gateway name, metrics) read off the
    driver's environment. Shared by model and gateway deployments so both
    replicas inherit the same logging/metrics config."""
    return {var: os.environ[var] for var in _PASSTHROUGH_ENV_VARS if os.environ.get(var) is not None}


def build_cache_env_vars() -> dict[str, str]:
    """Resolve HF / vLLM / FlashInfer cache dirs, all rooted at MSHIP_CACHE_DIR."""
    base_cache = os.environ.get("MSHIP_CACHE_DIR", "/.cache")
    return {
        "HF_HOME": os.environ.get("HF_HOME", f"{base_cache}/huggingface"),
        "VLLM_CACHE_ROOT": os.environ.get("VLLM_CACHE_ROOT", f"{base_cache}/vllm"),
        "FLASHINFER_CACHE_DIR": os.environ.get("FLASHINFER_CACHE_DIR", f"{base_cache}/flashinfer"),
    }


def _plugin_wheel_dir() -> Path:
    return Path(os.environ.get("MSHIP_PLUGIN_WHEEL_DIR", ".build/plugin-wheels"))


def resolve_plugin_wheel(plugin: str) -> Path:
    wheel_dir = _plugin_wheel_dir()
    normalized_name = plugin.replace("-", "_")
    wheels = sorted(wheel_dir.glob(f"{normalized_name}-*.whl"))
    if not wheels:
        raise RuntimeError(
            f"No wheel found for plugin '{plugin}' (normalized: '{normalized_name}') in {wheel_dir}. "
            f"Build wheels with `make plugin-wheels` (or rebuild the Docker image), "
            f"or set MSHIP_PLUGIN_WHEEL_DIR to the directory containing them."
        )
    # Absolute path required: Ray workers run with a different cwd
    # (/tmp/ray/session_*/runtime_resources/.../exec_cwd), so a relative wheel
    # path in runtime_env.pip would fail to resolve on the worker.
    return wheels[-1].resolve()


def _world_size(config: ModelshipModelConfig) -> int:
    if config.loader != ModelLoader.vllm:
        return 1
    tp = config.vllm_engine_kwargs.tensor_parallel_size
    pp = config.vllm_engine_kwargs.pipeline_parallel_size
    return tp * pp


def total_gpu_reservation(deploy_opts: dict) -> float:
    """Sum the GPU units this deployment (actor + any PG bundles) will consume.

    Used by the coordinator's resource tracker, which can't read the PG
    bundle list as a single scalar.
    """
    return _total_reservation(deploy_opts, "GPU", "num_gpus")


def total_cpu_reservation(deploy_opts: dict) -> float:
    """Sum the CPU units this deployment (actor + any PG bundles) will consume.

    For multi-slot deploys the outer actor sits in bundle 0 and its CPU
    request is satisfied from that bundle's reservation, so summing the
    bundles gives the correct total — same shape as the GPU helper.
    """
    return _total_reservation(deploy_opts, "CPU", "num_cpus")


def _total_reservation(deploy_opts: dict, bundle_key: str, actor_key: str) -> float:
    if "placement_group_bundles" in deploy_opts:
        return float(sum(b.get(bundle_key, 0) for b in deploy_opts["placement_group_bundles"]))
    return float(deploy_opts.get("ray_actor_options", {}).get(actor_key, 0) or 0)


def build_deployment_options(config: ModelshipModelConfig, plugin_wheel: Path | None = None) -> dict:
    """Return a kwargs dict for `Deployment.options(**...)`.

    Always contains ``ray_actor_options``; for multi-slot vLLM deploys also
    contains ``placement_group_bundles`` and ``placement_group_strategy`` so
    Ray Serve allocates one whole-GPU bundle per slot and vLLM's ray executor
    inherits the PG. When the model config sets ``max_ongoing_requests`` it is
    forwarded as the per-replica Ray Serve concurrency cap.
    """
    env_vars = build_cache_env_vars()
    env_vars.update(build_passthrough_env_vars())

    runtime_env: dict = {"env_vars": env_vars}
    if plugin_wheel is not None:
        # Ship the plugin to the Ray worker via runtime_env. Ray content-hashes
        # and caches the resulting per-job venv, so repeat deploys of the same
        # wheel reuse the install.
        runtime_env["pip"] = [str(plugin_wheel)]

    if config.loader in (ModelLoader.llama_cpp, ModelLoader.stable_diffusion_cpp):
        if config.num_gpus > 0:
            logger.warning(
                "num_gpus=%s is ignored for model '%s': %s loader currently only supports CPU.",
                config.num_gpus,
                config.name,
                config.loader.value,
            )
        opts: dict = {"ray_actor_options": {"num_gpus": 0, "num_cpus": config.num_cpus, "runtime_env": runtime_env}}
    else:
        world_size = _world_size(config)
        if world_size == 1:
            # Single slot: scalar Ray allocation. Fractional num_gpus (0 < n < 1)
            # lets Ray pack other actors onto the same physical GPU.
            opts = {
                "ray_actor_options": {
                    "num_gpus": config.num_gpus,
                    "num_cpus": config.num_cpus,
                    "runtime_env": runtime_env,
                }
            }
        else:
            # Multi-slot: one PG bundle per slot, STRICT_PACK keeps them on the
            # same node (NVLink). Outer actor sits in bundle 0 with 0 GPU; vLLM's
            # ray executor reuses the PG via get_current_placement_group() and
            # pins each worker actor to its bundle. Each bundle requests a whole
            # GPU, so Ray spreads across distinct physical GPUs.
            bundles = [{"GPU": 1, "CPU": config.num_cpus} for _ in range(world_size)]
            opts = {
                "ray_actor_options": {"num_gpus": 0, "num_cpus": config.num_cpus, "runtime_env": runtime_env},
                "placement_group_bundles": bundles,
                "placement_group_strategy": "STRICT_PACK",
            }

    # Per-model Ray Serve concurrency cap; only override the default when set.
    # The reservation helpers read only the GPU/CPU keys, so this is inert there.
    if config.max_ongoing_requests is not None:
        opts["max_ongoing_requests"] = config.max_ongoing_requests
    return opts
