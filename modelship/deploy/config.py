import os
from pathlib import Path

import yaml
from pydantic_yaml import parse_yaml_raw_as

from modelship.deploy.actor_options import resolve_plugin_wheel
from modelship.infer.infer_config import ModelLoader, ModelshipConfig
from modelship.infer.model_resolver import resolve_model_source
from modelship.logging import get_logger

logger = get_logger("startup")


def resolve_config_path(arg_path: str | None, config_dir: Path | None = None) -> str:
    """Resolve the models.yaml to deploy.

    Precedence:
    1. An explicit ``--config`` path always wins (most specific signal); it must exist.
    2. Otherwise the default ``config/models.yaml`` must exist.
    """
    config_dir = config_dir or Path(__file__).resolve().parent.parent.parent / "config"

    if arg_path:
        if not os.path.exists(arg_path):
            raise FileNotFoundError(f"--config {arg_path} not found.")
        return arg_path

    default = config_dir / "models.yaml"
    if default.exists():
        return str(default)

    raise FileNotFoundError(f"{default} not found. Copy an example config from config/examples/ to config/models.yaml.")


def load_yaml_config(arg_path: str | None) -> ModelshipConfig:
    with open(resolve_config_path(arg_path)) as f:
        return parse_yaml_raw_as(ModelshipConfig, f)


def load_raw_models(arg_path: str | None) -> list[dict]:
    """Read the user's models.yaml as raw, pre-validation dicts.

    The effective-config store keeps raw dicts (not validated configs, which don't
    round-trip through num_gpus/tp normalization), so the deploy path merges at the
    raw-dict level and validates only the merged result."""
    with open(resolve_config_path(arg_path)) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("models.yaml: top-level document must be a mapping with a 'models' key.")
    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError("models.yaml: 'models' must be a list.")
    return models


def resolve_all_plugin_wheels(yml_conf: ModelshipConfig) -> dict[str, Path]:
    """Pre-flight: resolve every referenced plugin wheel up front so a missing
    wheel fails the whole startup before any Ray deploy is attempted."""
    wheels: dict[str, Path] = {}
    for cfg in yml_conf.models:
        if cfg.loader == ModelLoader.custom and cfg.plugin and cfg.plugin not in wheels:
            wheels[cfg.plugin] = resolve_plugin_wheel(cfg.plugin)
    return wheels


def resolve_all_model_sources(yml_conf: ModelshipConfig) -> None:
    """Pre-flight: resolve every built-in-loader model to a local path.

    Populates `_resolved_path` on each config in place. Raises on the first
    failure (auth, missing repo, missing file, glob-no-match) so the operator
    sees the error before any Ray actor spins up.

    Plugins (`loader=custom`) are skipped — they manage their own download.

    Note: HF_HOME / VLLM_CACHE_ROOT / FLASHINFER_CACHE_DIR are set at module
    load time in mship_deploy.py — `huggingface_hub.HF_HOME` is latched at
    import, so setting them later doesn't take effect.
    """
    for cfg in yml_conf.models:
        if cfg.loader == ModelLoader.custom:
            continue
        assert cfg.model is not None  # validator guarantees this for built-in loaders
        trust_remote_code = bool(cfg.vllm_engine_kwargs and cfg.vllm_engine_kwargs.trust_remote_code)
        logger.info("Resolving model source for '%s': %s", cfg.name, cfg.model)
        cfg._resolved_path = resolve_model_source(cfg.model, trust_remote_code=trust_remote_code)
        logger.info("Resolved '%s' -> %s", cfg.name, cfg._resolved_path)

        if cfg.loader == ModelLoader.llama_server and cfg.llama_server_config and cfg.llama_server_config.mmproj:
            logger.info("Resolving mmproj source for '%s': %s", cfg.name, cfg.llama_server_config.mmproj)
            cfg.llama_server_config.mmproj = resolve_model_source(
                cfg.llama_server_config.mmproj, trust_remote_code=trust_remote_code
            )
            logger.info("Resolved mmproj -> %s", cfg.llama_server_config.mmproj)

        # GGUF is not supported on the vllm loader: vLLM 0.24 moved GGUF out of
        # tree, and the only external plugin is incompatible with 0.24's
        # quantization API. Reject early with a pointer to llama_server instead
        # of letting vLLM misparse the .gguf as a config.json deep in engine init.
        if cfg.loader == ModelLoader.vllm and cfg._resolved_path.lower().endswith(".gguf"):
            raise ValueError(
                f"Model '{cfg.name}' resolves to a GGUF file, which the vllm loader does not support "
                f"(vLLM 0.24 dropped in-tree GGUF). Use `loader: llama_server` for GGUF models, or point "
                f"the vllm loader at a non-GGUF checkpoint (safetensors, or an AWQ/GPTQ/FP8 quant)."
            )
