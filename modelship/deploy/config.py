import os
from pathlib import Path

from pydantic_yaml import parse_yaml_raw_as

from modelship.deploy.actor_options import resolve_plugin_wheel
from modelship.infer.infer_config import ModelLoader, ModelshipConfig, ModelUsecase
from modelship.infer.model_resolver import resolve_model_source
from modelship.logging import get_logger
from modelship.openai.parsers.tool_calling.registry import available_parsers
from modelship.openai.parsers.tool_calling.utils import classify_template
from modelship.openai.parsers.utils import read_chat_template

logger = get_logger("startup")


def _is_explicit_tool_opt_out(cfg) -> bool:
    """Loader-specific explicit "no auto-detection" signal.

    Auto-detection is skipped when the user has signalled an explicit choice
    that excludes our parser:

    - vllm: ``enable_auto_tool_choice: false`` — user disabled tool calling.
    - transformers: ``tool_calls_enabled: false`` — user disabled tool calling.
    - llama_cpp: ``tool_calls_enabled: false`` (disabled) OR ``chat_format`` set
      (user wants llama-cpp-python's own function-calling handler — we must not
      also wire up our parser).
    """
    if cfg.loader == ModelLoader.vllm:
        return cfg.vllm_engine_kwargs is not None and cfg.vllm_engine_kwargs.enable_auto_tool_choice is False
    if cfg.loader == ModelLoader.transformers:
        return cfg.transformers_config is not None and cfg.transformers_config.tool_calls_enabled is False
    if cfg.loader == ModelLoader.llama_cpp:
        if cfg.llama_cpp_config is None:
            return False
        if cfg.llama_cpp_config.tool_calls_enabled is False:
            return True
        if cfg.llama_cpp_config.chat_format is not None:
            return True
    return False


def resolve_config_path(arg_path: str | None) -> str:
    path = arg_path or str(Path(__file__).resolve().parent.parent.parent / "config" / "models.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Copy one of the example configs from config/ to config/models.yaml."
        )
    return path


def load_yaml_config(arg_path: str | None) -> ModelshipConfig:
    with open(resolve_config_path(arg_path)) as f:
        return parse_yaml_raw_as(ModelshipConfig, f)


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
        trust_remote_code = bool(
            (cfg.vllm_engine_kwargs and cfg.vllm_engine_kwargs.trust_remote_code)
            or (cfg.transformers_config and cfg.transformers_config.trust_remote_code)
        )
        logger.info("Resolving model source for '%s': %s", cfg.name, cfg.model)
        cfg._resolved_path = resolve_model_source(cfg.model, trust_remote_code=trust_remote_code)
        logger.info("Resolved '%s' -> %s", cfg.name, cfg._resolved_path)


def resolve_all_tool_parsers(yml_conf: ModelshipConfig) -> None:
    """Pre-flight: auto-detect a tool-call parser for each generative model.

    Runs after `resolve_all_model_sources` so `_resolved_path` is populated.
    Reads the model's chat template once (from GGUF metadata or
    `tokenizer_config.json`), stores it on `_resolved_chat_template`, and
    classifies it onto `_resolved_tool_call_parser`. Per-loader code reads
    these as a fallback when the loader-specific parser/format field is
    unset, so an explicit user setting always wins.

    Auto-detection runs for vllm, transformers, and llama_cpp. diffusers has
    no chat path; custom is plugin-managed.

    Behavior per model:
    - Loader-specific opt-out (see `_is_explicit_tool_opt_out`): skipped.
    - Explicit parser name configured (vllm/transformers `tool_call_parser`):
      validated against the registry; raises if unknown so misconfiguration
      fails startup, not mid-request.
    - Auto-detected, registered: stored on `_resolved_tool_call_parser`.
    - Auto-detected, known format but no parser implemented yet: warn, leave
      unset — tool calling will be disabled for this model.
    - Auto-detected as `unknown` (template uses tools but no recognized
      markers): warn, leave unset.
    - Not detected: tool calling silently disabled (model has no template
      tool-call affordance to begin with).
    """
    registered = set(available_parsers())
    for cfg in yml_conf.models:
        if cfg.loader not in (ModelLoader.vllm, ModelLoader.transformers, ModelLoader.llama_cpp):
            continue
        if cfg.usecase != ModelUsecase.generate:
            continue
        if _is_explicit_tool_opt_out(cfg):
            logger.info("Tool-call auto-detection skipped for '%s' (explicit opt-out).", cfg.name)
            continue

        explicit = None
        if cfg.loader == ModelLoader.transformers and cfg.transformers_config:
            explicit = cfg.transformers_config.tool_call_parser
        elif cfg.loader == ModelLoader.vllm and cfg.vllm_engine_kwargs:
            explicit = cfg.vllm_engine_kwargs.tool_call_parser

        if explicit is not None:
            if explicit not in registered:
                raise ValueError(
                    f"Model '{cfg.name}' configures tool_call_parser={explicit!r} "
                    f"which is not registered. Available: {sorted(registered) or '(none)'}."
                )
            logger.info("Using explicit tool_call_parser=%r for '%s'", explicit, cfg.name)
            continue

        assert cfg._resolved_path is not None  # populated by resolve_all_model_sources
        template = read_chat_template(cfg._resolved_path)
        if template is None:
            logger.info("No chat template found for '%s'; tool-call detection skipped.", cfg.name)
            continue
        cfg._resolved_chat_template = template
        detected = classify_template(template)
        if detected is None:
            logger.info("No tool-calling support detected for '%s'; tools disabled.", cfg.name)
            continue
        if detected == "unknown":
            logger.warning(
                "Model '%s' chat template references tools but uses unrecognized markers; tool calling disabled.",
                cfg.name,
            )
            continue
        if detected not in registered:
            logger.warning(
                "Model '%s' uses tool format %r but no parser is registered; tool calling disabled.",
                cfg.name,
                detected,
            )
            continue
        cfg._resolved_tool_call_parser = detected
        logger.info("Auto-detected tool_call_parser=%r for '%s'", detected, cfg.name)
