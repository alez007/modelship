"""vLLM tool-call / reasoning parser-name detection, run inside the vllm actor.

Auto-detects which vLLM-native parser (``vllm.tool_parsers.ToolParserManager`` /
``vllm.reasoning.ReasoningParserManager``) a model's chat template calls for, by
marker sniffing, with explicit config always taking precedence. Names returned
here must match vLLM's own registered parser names exactly — both resolvers
validate against vLLM's registry directly rather than a modelship-side one.
"""

from __future__ import annotations

from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig
from modelship.logging import get_logger

logger = get_logger("infer.vllm.parsing.detect")


def classify_tool_template(template: str) -> str | None:
    """Map a chat-template string to a vLLM tool-call parser name based on markers.

    - Returns "gemma4" if `<|tool_call>` markers are found.
    - Returns "functiongemma" if `<start_function_call>` markers are found.
    - Returns "qwen3_coder" if `<function=` or `<parameter=` markers are found.
    - Returns "hermes" if `<tool_call>` markers are found.
    - Returns "mistral" if `[TOOL_CALLS]` markers are found.
    - Returns "llama3_json" if `<|python_tag|>` markers are found.
    - Returns "unknown" if tool logic is found but format markers are not recognized.
    - Returns ``None`` if no chat template / no tool logic is detected.
    """
    if "tools" not in template and "tool_calls" not in template and "function" not in template:
        return None

    if "<|tool_call>" in template:
        return "gemma4"

    if "<start_function_call>" in template:
        return "functiongemma"

    # Qwen3-Coder shares the ``<tool_call>`` envelope with Hermes but the
    # body is XML rather than JSON. The ``<function=`` / ``<parameter=``
    # markers only appear in Qwen3-Coder templates, so check them before
    # the ``<tool_call>`` -> hermes branch below.
    if "<function=" in template or "<parameter=" in template:
        return "qwen3_coder"

    if "<tool_call>" in template:
        return "hermes"

    if "[TOOL_CALLS]" in template:
        return "mistral"

    if "<|python_tag|>" in template or "<|eom_id|>" in template:
        return "llama3_json"

    return "unknown"


def classify_reasoning_template(template: str) -> str | None:
    """Map a chat-template string to a vLLM reasoning parser name based on markers."""
    if "<|channel>thought" in template:
        return "gemma4"
    if "<think>" in template or "</think>" in template:
        return "deepseek_r1"
    return None


def _is_tool_opt_out(cfg: ModelshipModelConfig) -> bool:
    """Explicit "no auto tool-call detection" signal: ``enable_auto_tool_choice: false``."""
    return cfg.loader == ModelLoader.vllm and cfg.vllm_engine_kwargs.enable_auto_tool_choice is False


def _is_reasoning_opt_out(cfg: ModelshipModelConfig) -> bool:
    """Explicit "no auto reasoning detection" signal: ``enable_reasoning: false``."""
    return cfg.loader == ModelLoader.vllm and cfg.vllm_engine_kwargs.enable_reasoning is False


def resolve_tool_parser(cfg: ModelshipModelConfig, template: str | None) -> str | None:
    """Resolve the tool-call parser name to hand to ``OpenAIServingRender``.

    Precedence: opt-out -> None; explicit ``tool_call_parser`` (validated against
    vLLM's registry, raises if unknown) -> that name; else classify the chat
    template and validate the detected name against the registry (warn +
    disable if unrecognized or unregistered); else None.
    """
    if _is_tool_opt_out(cfg):
        logger.info("Tool-call resolution skipped for '%s' (explicit opt-out).", cfg.name)
        return None

    from vllm.tool_parsers import ToolParserManager

    registered = set(ToolParserManager.list_registered())

    explicit = cfg.vllm_engine_kwargs.tool_call_parser
    if explicit is not None:
        if explicit not in registered:
            raise ValueError(
                f"Model '{cfg.name}' configures tool_call_parser={explicit!r} "
                f"which is not registered. Available: {sorted(registered) or '(none)'}."
            )
        logger.info("Using explicit tool_call_parser=%r for '%s'", explicit, cfg.name)
        return explicit

    if template is None:
        logger.info("No chat template found for '%s'; tool-call detection skipped.", cfg.name)
        return None

    detected = classify_tool_template(template)
    if detected is None:
        logger.info("No tool-calling support detected for '%s'; tools disabled.", cfg.name)
        return None
    if detected == "unknown":
        logger.warning(
            "Model '%s' chat template references tools but uses unrecognized markers; tool calling disabled.",
            cfg.name,
        )
        return None
    if detected not in registered:
        logger.warning(
            "Model '%s' uses tool format %r but no parser is registered; tool calling disabled.", cfg.name, detected
        )
        return None
    logger.info("Auto-detected tool_call_parser=%r for '%s'", detected, cfg.name)
    return detected


def resolve_reasoning_parser(cfg: ModelshipModelConfig, template: str | None) -> str | None:
    """Resolve the reasoning parser name to hand to ``OpenAIServingRender``.

    Same precedence/validation shape as ``resolve_tool_parser``.
    """
    if _is_reasoning_opt_out(cfg):
        logger.info("Reasoning resolution skipped for '%s' (explicit opt-out).", cfg.name)
        return None

    from vllm.reasoning import ReasoningParserManager

    registered = set(ReasoningParserManager.list_registered())

    explicit = cfg.vllm_engine_kwargs.reasoning_parser
    if explicit is not None:
        if explicit not in registered:
            raise ValueError(
                f"Model '{cfg.name}' configures reasoning_parser={explicit!r} "
                f"which is not registered. Available: {sorted(registered) or '(none)'}."
            )
        logger.info("Using explicit reasoning_parser=%r for '%s'", explicit, cfg.name)
        return explicit

    if template is None:
        logger.info("No chat template found for '%s'; reasoning detection skipped.", cfg.name)
        return None

    detected = classify_reasoning_template(template)
    if detected is None:
        return None
    if detected not in registered:
        logger.warning(
            "Model '%s' uses reasoning format %r but no parser is registered; reasoning disabled.",
            cfg.name,
            detected,
        )
        return None
    logger.info("Auto-detected reasoning_parser=%r for '%s'", detected, cfg.name)
    return detected
