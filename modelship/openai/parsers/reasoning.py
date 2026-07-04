"""Reasoning parser name detection by chat-template inspection.

Auto-detects which vLLM-native reasoning parser (``vllm.reasoning.ReasoningParserManager``)
a model's chat template calls for, by marker sniffing.
"""

from __future__ import annotations

from collections.abc import Callable

from modelship.logging import get_logger

logger = get_logger("openai.reasoning.detect")

# (start_marker, end_marker) for the two auto-detectable families, used by
# resolve_active_reasoning_parser to probe whether a deployment's rendered
# prompt shows reasoning pre-suppressed (e.g. Qwen3 enable_thinking=false).
_MARKERS: dict[str, tuple[str, str]] = {
    "deepseek_r1": ("<think>", "</think>"),
    "gemma4": ("<|channel>thought\n", "<channel|>"),
}


def classify_template(template: str) -> str | None:
    """Map a chat-template string to a vLLM reasoning parser name based on markers."""
    if "<|channel>thought" in template:
        return "gemma4"
    if "<think>" in template or "</think>" in template:
        return "deepseek_r1"
    return None


def reasoning_active_in_render(rendered_prompt: str, start_marker: str, end_marker: str) -> bool:
    """Whether a reasoning-capable model will actually emit reasoning.

    Used post-load against the generation prompt rendered under the deployment's
    effective ``chat_template_kwargs``. Suppression is declared only on *positive
    evidence* — a balanced (closed) reasoning block prefilled into the prompt, as
    templates do when reasoning is switched off (e.g. Qwen3 ``enable_thinking=false``
    prefills ``<think></think>``). Everything else is treated as active:

    - no ``start_marker`` in the render: a thinking-on template typically primes
      nothing and lets the model emit ``<think>`` itself, so assume active.
    - an unclosed ``start_marker``: the model continues inside the open block.

    Errs active: the only unsafe verdict is a false OFF (it would let a tool-call
    grammar block ``<think>``); a false ON merely forgoes constraining.
    """
    if not start_marker:
        return True
    opens = rendered_prompt.count(start_marker)
    if opens == 0:
        return True
    closes = rendered_prompt.count(end_marker) if end_marker else 0
    # opens == closes ⇒ a closed block (suppressed); opens > closes ⇒ open block (active).
    return opens > closes


def resolve_active_reasoning_parser(candidate: str | None, render_prompt: Callable[[], str]) -> str | None:
    """Downgrade a candidate reasoning parser to ``None`` if it is suppressed.

    ``candidate`` is the parser detected from the template *markers* (capability),
    or an explicit user-configured vLLM reasoning parser name. ``render_prompt``
    renders a generation prompt through the loader's real render path, so the
    deployment's effective ``chat_template_kwargs`` are already applied. Returns
    the candidate when reasoning is (or might be) active, ``None`` when the render
    shows positive evidence of suppression. Falls back to the candidate if no known
    markers exist for it, or if the render raises — never invents a parser.
    """
    if candidate is None:
        return None
    markers = _MARKERS.get(candidate)
    if markers is None:
        # No known start/end markers for this name (e.g. an explicitly configured
        # vLLM reasoning parser modelship doesn't auto-detect) — nothing to probe.
        return candidate
    start_marker, end_marker = markers
    try:
        rendered = render_prompt()
    except Exception as exc:
        logger.warning("reasoning probe failed (%s); keeping reasoning_parser=%r", exc, candidate)
        return candidate
    if reasoning_active_in_render(rendered, start_marker, end_marker):
        return candidate
    logger.info("reasoning_parser %r disabled: render shows reasoning suppressed for this deployment", candidate)
    return None
