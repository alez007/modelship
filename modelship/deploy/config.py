import os
from pathlib import Path

from pydantic_yaml import parse_yaml_raw_as

from modelship.deploy.actor_options import resolve_plugin_wheel
from modelship.infer.infer_config import ModelLoader, ModelshipConfig, ModelUsecase
from modelship.infer.model_resolver import resolve_model_source
from modelship.logging import get_logger
from modelship.openai.tool_calling.registry import available_parsers
from modelship.openai.tool_calling.utils import detect_tool_parser

# Tool-call formats `detect_tool_parser` can identify but for which a parser
# may or may not yet be registered. Used to differentiate "format unknown" from
# "format known but parser not implemented yet" in the auto-detect warning.
_KNOWN_TOOL_FORMATS = frozenset({"hermes", "mistral", "llama3_json"})

# Loaders that emit raw model text and rely on the tool_calling registry to
# parse tool calls. vLLM and llama.cpp have native tool-call handling and are
# excluded; diffusers/custom don't do chat completion through this path.
_LOADERS_USING_TEXT_PARSER = frozenset({ModelLoader.transformers})

logger = get_logger("startup")


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
    """Pre-flight: pick a tool-call parser for each text-parser loader model.

    Runs after `resolve_all_model_sources` so `_resolved_path` is populated.
    Inspects the model's `tokenizer_config.json` chat template and stores the
    result on `_resolved_tool_call_parser`. Effective parser at request time
    is `transformers_config.tool_call_parser or _resolved_tool_call_parser`,
    so an explicit user setting always wins.

    Behavior:
    - Explicit parser configured: validated against the registry; raises if
      the name is unknown so misconfiguration fails startup, not mid-request.
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
        if cfg.loader not in _LOADERS_USING_TEXT_PARSER:
            continue
        if cfg.usecase != ModelUsecase.generate:
            continue

        explicit = cfg.transformers_config.tool_call_parser if cfg.transformers_config else None
        if explicit is not None:
            if explicit not in registered:
                raise ValueError(
                    f"Model '{cfg.name}' configures tool_call_parser={explicit!r} "
                    f"which is not registered. Available: {sorted(registered) or '(none)'}."
                )
            logger.info("Using explicit tool_call_parser=%r for '%s'", explicit, cfg.name)
            continue

        assert cfg._resolved_path is not None  # populated by resolve_all_model_sources
        detected = detect_tool_parser(cfg._resolved_path)
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
            known_note = "" if detected in _KNOWN_TOOL_FORMATS else " (format not in known list)"
            logger.warning(
                "Model '%s' uses tool format %r%s but no parser is registered; tool calling disabled.",
                cfg.name,
                detected,
                known_note,
            )
            continue
        cfg._resolved_tool_call_parser = detected
        logger.info("Auto-detected tool_call_parser=%r for '%s'", detected, cfg.name)
