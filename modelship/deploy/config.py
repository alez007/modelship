import os
from pathlib import Path

import yaml
from pydantic_yaml import parse_yaml_raw_as

from modelship.deploy.actor_options import resolve_plugin_wheel
from modelship.infer.infer_config import ModelLoader, ModelshipConfig, ModelUsecase
from modelship.infer.model_resolver import resolve_model_source
from modelship.logging import get_logger
from modelship.openai.parsers.reasoning import classify_template as classify_reasoning_template
from modelship.openai.parsers.tool_calling import classify_template
from modelship.openai.parsers.utils import read_chat_template

logger = get_logger("startup")


def _is_explicit_tool_opt_out(cfg) -> bool:
    """Loader-specific explicit "no auto-detection" signal.

    Auto-detection is skipped when the user has signalled an explicit choice
    that excludes our parser:

    - vllm: ``enable_auto_tool_choice: false`` — user disabled tool calling.
    """
    if cfg.loader == ModelLoader.vllm:
        return cfg.vllm_engine_kwargs is not None and cfg.vllm_engine_kwargs.enable_auto_tool_choice is False
    return False


def resolve_config_path(arg_path: str | None, config_dir: Path | None = None) -> str:
    """Resolve the models.yaml to deploy.

    Precedence:
    1. An explicit ``--config`` path always wins (most specific signal); it must exist.
    2. ``MSHIP_MODEL_STACK=<profile>`` (or ``--model-stack``) → regenerate
       ``models_stack_<profile>.yaml`` from scratch on every start, sized to the
       detected hardware, and deploy that. Regenerating fresh each time lets the
       user switch profiles by just changing the value — no stale file to delete by
       hand. Refuses with a clean exit (no partial deploy) if the profile can't fit.
    3. Otherwise the default ``config/models.yaml`` must exist.
    """
    config_dir = config_dir or Path(__file__).resolve().parent.parent.parent / "config"
    stack = os.environ.get("MSHIP_MODEL_STACK")

    if arg_path:
        if not os.path.exists(arg_path):
            raise FileNotFoundError(f"--config {arg_path} not found.")
        return arg_path

    if stack:
        from modelship.deploy.profiles.catalog import PROFILES
        from modelship.deploy.profiles.generator import generate_models_yaml
        from modelship.deploy.profiles.selector import ProfileDoesNotFitError

        # Validate against the known profiles BEFORE building a path or touching the
        # filesystem — `stack` is operator-supplied (env var / CLI), and feeding it
        # into the filename unchecked would allow path traversal on the unlink below.
        if stack not in PROFILES:
            raise SystemExit(f"MSHIP_MODEL_STACK={stack!r}: unknown profile; choose one of {sorted(PROFILES)}.")

        path = config_dir / f"models_stack_{stack}.yaml"
        logger.info("MSHIP_MODEL_STACK=%s: generating %s for the detected hardware...", stack, path)
        try:
            # Remove any prior generation first so a refusal never leaves a stale
            # file behind that a later run could mistake for hand-authored config.
            path.unlink(missing_ok=True)
            generate_models_yaml(stack, str(path))
        except (ProfileDoesNotFitError, ValueError) as e:
            raise SystemExit(f"MSHIP_MODEL_STACK={stack}: {e}") from e
        except OSError as e:
            # Read-only / permission-denied config dir, etc. — fail cleanly instead
            # of dumping a traceback.
            raise SystemExit(f"MSHIP_MODEL_STACK={stack}: cannot write {path}: {e}") from e
        return str(path)

    default = config_dir / "models.yaml"
    if default.exists():
        return str(default)

    raise FileNotFoundError(
        f"{default} not found. Set MSHIP_MODEL_STACK=<profile> (or pass --model-stack) to "
        f"auto-generate one, or copy an example config from config/examples/ to config/models.yaml."
    )


def load_yaml_config(arg_path: str | None) -> ModelshipConfig:
    with open(resolve_config_path(arg_path)) as f:
        return parse_yaml_raw_as(ModelshipConfig, f)


def load_raw_models(arg_path: str | None) -> list[dict]:
    """Read the user's models.yaml as raw, pre-validation dicts.

    The effective-config store keeps raw dicts (not validated configs, which don't
    round-trip through num_gpus/tp normalization), so the deploy path merges at the
    raw-dict level and validates only the merged result. Shares resolve_config_path
    with load_yaml_config so MSHIP_MODEL_STACK generation runs at most once per
    deploy (callers should use one or the other, not both)."""
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
                f"the vllm loader at a non-GGUF checkpoint (safetensors, or an AWQ/GPTQ/FP8 quant — this is "
                f"also the supported path for gemma models, whose tool calling llama.cpp's parsers can't "
                f"handle; see config/examples/vllm-cpu.yaml)."
            )


def resolve_all_tool_parsers(yml_conf: ModelshipConfig) -> None:
    """Pre-flight: resolve the final tool-call parser name per generative model.

    Runs after `resolve_all_model_sources` so `_resolved_path` is populated.
    Captures the FINAL parser name (explicit user setting or auto-detection)
    onto `_resolved_tool_call_parser` so loader code has a single source of
    truth and never re-implements the precedence.

    Auto-detection runs for vllm. llama_server does its own tool-call
    detection internally; diffusers has no chat path; custom is plugin-managed.

    Behavior per model:
    - Loader-specific opt-out (see `_is_explicit_tool_opt_out`): leaves
      `_resolved_tool_call_parser` as None.
    - Explicit parser name configured: validated against vLLM's own
      `ToolParserManager`, stored on `_resolved_tool_call_parser`. Raises if
      unknown.
    - Auto-detected, registered: stored on `_resolved_tool_call_parser`.
    - Auto-detected as `unknown` / known-but-unregistered: warn, leave None.
    - Not detected: leave None (no template tool-call affordance).
    """
    vllm_generate_cfgs = [
        cfg for cfg in yml_conf.models if cfg.loader == ModelLoader.vllm and cfg.usecase == ModelUsecase.generate
    ]
    if not vllm_generate_cfgs:
        return

    from vllm.tool_parsers import ToolParserManager

    registered = set(ToolParserManager.list_registered())
    for cfg in vllm_generate_cfgs:
        if _is_explicit_tool_opt_out(cfg):
            logger.info("Tool-call resolution skipped for '%s' (explicit opt-out).", cfg.name)
            continue

        explicit = cfg.vllm_engine_kwargs.tool_call_parser if cfg.vllm_engine_kwargs else None

        if explicit is not None:
            if explicit not in registered:
                raise ValueError(
                    f"Model '{cfg.name}' configures tool_call_parser={explicit!r} "
                    f"which is not registered. Available: {sorted(registered) or '(none)'}."
                )
            cfg._resolved_tool_call_parser = explicit
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


def _is_explicit_reasoning_opt_out(cfg) -> bool:
    """Loader-specific explicit "no reasoning auto-detection" signal.

    - vllm: ``enable_reasoning: false`` — user disabled reasoning.
    """
    if cfg.loader == ModelLoader.vllm:
        return cfg.vllm_engine_kwargs is not None and cfg.vllm_engine_kwargs.enable_reasoning is False
    return False


def resolve_all_reasoning_parsers(yml_conf: ModelshipConfig) -> None:
    """Pre-flight: resolve the final reasoning parser name per generative model.

    Mirrors `resolve_all_tool_parsers`: captures the FINAL parser name onto
    `_resolved_reasoning_parser` so loader code has a single source of truth.
    Reuses `_resolved_chat_template` if populated by the tool-parser pass.

    Behavior per model:
    - Loader-specific opt-out: leaves `_resolved_reasoning_parser` as None.
    - Explicit ``reasoning_parser`` on the loader config: stored as-is.
    - Auto-detected from chat template: stored.
    - Not detected: leaves None (reasoning disabled).
    """
    for cfg in yml_conf.models:
        if cfg.loader != ModelLoader.vllm:
            continue
        if cfg.usecase != ModelUsecase.generate:
            continue
        if _is_explicit_reasoning_opt_out(cfg):
            logger.info("Reasoning resolution skipped for '%s' (explicit opt-out).", cfg.name)
            continue

        explicit = None
        if cfg.loader == ModelLoader.vllm and cfg.vllm_engine_kwargs:
            explicit = cfg.vllm_engine_kwargs.reasoning_parser

        if explicit is not None:
            cfg._resolved_reasoning_parser = explicit
            logger.info("Using explicit reasoning_parser=%r for '%s'", explicit, cfg.name)
            continue

        template = cfg._resolved_chat_template
        if template is None:
            assert cfg._resolved_path is not None  # populated by resolve_all_model_sources
            template = read_chat_template(cfg._resolved_path)
            if template is None:
                logger.info("No chat template found for '%s'; reasoning detection skipped.", cfg.name)
                continue
            cfg._resolved_chat_template = template

        detected = classify_reasoning_template(template)
        if detected is None:
            continue
        cfg._resolved_reasoning_parser = detected
        logger.info("Auto-detected reasoning_parser=%r for '%s'", detected, cfg.name)
