"""Reasoning parser detection by chat-template inspection."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from modelship.logging import get_logger
from modelship.openai.parsers.utils import read_chat_template

logger = get_logger("openai.reasoning.detect")


def detect_reasoning_parser(model_path: str | Path) -> str | None:
    """Return the name of the reasoning parser the model needs, or None."""
    template = read_chat_template(model_path)
    if template is None:
        return None
    return classify_template(template)


def classify_template(template: str) -> str | None:
    """Map a chat-template string to a reasoning parser name based on markers."""
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

    ``candidate`` is the parser detected from the template *markers* (capability).
    ``render_prompt`` renders a generation prompt through the loader's real render
    path, so the deployment's effective ``chat_template_kwargs`` are already applied.
    Returns the candidate when reasoning is (or might be) active, ``None`` when the
    render shows positive evidence of suppression. Falls back to the candidate if the
    render raises — never invents a parser.
    """
    if candidate is None:
        return None
    from modelship.openai.parsers.reasoning.registry import get_parser

    parser = get_parser(candidate)
    try:
        rendered = render_prompt()
    except Exception as exc:
        logger.warning("reasoning probe render failed (%s); keeping reasoning_parser=%r", exc, candidate)
        return candidate
    if reasoning_active_in_render(rendered, parser.start_marker, parser.end_marker):
        return candidate
    logger.info("reasoning_parser %r disabled: render shows reasoning suppressed for this deployment", candidate)
    return None
