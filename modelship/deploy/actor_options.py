"""Ray Serve actor option construction for model deployments.

Centralises the GPU-allocation decisions for vLLM tensor parallelism (mp vs ray
backends) and the plugin-wheel runtime_env injection for custom-loader models.
"""

from __future__ import annotations

import os
from pathlib import Path

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig
from modelship.logging import get_logger

logger = get_logger("startup")

_LOG_PASSTHROUGH_ENV_VARS = ("MSHIP_LOG_LEVEL", "MSHIP_LOG_FORMAT", "MSHIP_LOG_TARGET")


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


def _resolve_num_gpus(config: ModelshipModelConfig, env_vars: dict[str, str]) -> float:
    """Decide how many GPU units the outer Ray actor should request.

    The result depends on loader, vLLM tensor- and pipeline-parallel sizes, and
    the chosen distributed-executor backend. Mutates *env_vars* in place to add
    VLLM_RAY_PER_WORKER_GPUS when the ray-backend path needs it, and may set
    config.vllm_engine_kwargs.distributed_executor_backend to "ray" as the
    default when the world size (tp * pp) is greater than 1.
    """
    if config.loader == ModelLoader.llama_cpp:
        if config.num_gpus > 0:
            logger.warning(
                "num_gpus=%s is ignored for model '%s': llama_cpp loader currently only supports CPU.",
                config.num_gpus,
                config.name,
            )
        return 0

    tp = config.vllm_engine_kwargs.tensor_parallel_size if config.vllm_engine_kwargs else 1
    pp = config.vllm_engine_kwargs.pipeline_parallel_size if config.vllm_engine_kwargs else 1
    world_size = tp * pp
    tp_backend = config.vllm_engine_kwargs.distributed_executor_backend if config.vllm_engine_kwargs else None

    # vLLM validates world_size against the outer actor's visible GPUs before consulting
    # the executor backend. With world_size>1 and no backend set, the outer actor has 0 GPUs
    # (ray-backend path below) and ParallelConfig blows up. Default to ray so the outer
    # actor is a coordinator and vLLM uses Ray V2 executor (new default in 0.20.0).
    if world_size > 1 and tp_backend is None and config.vllm_engine_kwargs is not None:
        config.vllm_engine_kwargs.distributed_executor_backend = "ray"
        tp_backend = "ray"
        logger.info(
            "Defaulting distributed_executor_backend='ray' for model '%s' "
            "(tensor_parallel_size=%d, pipeline_parallel_size=%d).",
            config.name,
            tp,
            pp,
        )

    if config.num_gpus == 0:
        return 0
    if world_size > 1 and tp_backend == "mp":
        # mp backend: the main actor forks worker subprocesses, each owning one
        # physical GPU. Allocate world_size whole units so Ray exposes all devices via
        # CUDA_VISIBLE_DEVICES. num_gpus is not used for Ray allocation in this path —
        # gpu_memory_utilization is passed directly to vLLM instead.
        if config.num_gpus > 0:
            logger.warning(
                "num_gpus=%s is ignored for model '%s': with vllm mp backend and "
                "tensor_parallel_size=%d pipeline_parallel_size=%d, Ray GPU allocation "
                "is determined by their product.",
                config.num_gpus,
                config.name,
                tp,
                pp,
            )
        return float(world_size)
    if world_size > 1:
        # ray backend: vLLM spawns world_size worker Ray actors that each claim
        # their own fractional GPU. The outer actor is a coordinator only and needs
        # no GPU units. VLLM_RAY_PER_WORKER_GPUS tells vLLM what fraction each
        # worker actor should request.
        if config.num_gpus > 0:
            env_vars["VLLM_RAY_PER_WORKER_GPUS"] = str(config.num_gpus)
        return 0
    return config.num_gpus


def build_actor_options(config: ModelshipModelConfig, plugin_wheel: Path | None = None) -> dict:
    env_vars = build_cache_env_vars()
    for log_var in _LOG_PASSTHROUGH_ENV_VARS:
        val = os.environ.get(log_var)
        if val is not None:
            env_vars[log_var] = val

    num_gpus = _resolve_num_gpus(config, env_vars)

    runtime_env: dict = {"env_vars": env_vars}
    if plugin_wheel is not None:
        # Ship the plugin to the Ray worker via runtime_env. Ray content-hashes
        # and caches the resulting per-job venv, so repeat deploys of the same
        # wheel reuse the install.
        runtime_env["pip"] = [str(plugin_wheel)]

    return {
        "num_gpus": num_gpus,
        "num_cpus": config.num_cpus,
        "runtime_env": runtime_env,
    }
