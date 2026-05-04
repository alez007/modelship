"""Input-side helpers for tool calling.

Loaders that hand a chat template a list of OpenAI messages plus a list of
tool schemas use these helpers to interpret the request's ``tool_choice``
(``"none"`` suppresses tools, ``"required"`` / specific-function downgrade
to ``"auto"`` with a warning) and to validate that a parser exists for the
configured family before generation starts.
"""

from __future__ import annotations

from typing import Any

from modelship.logging import get_logger

logger = get_logger("openai.tool_calling.input")


def resolve_tools_for_request(
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Apply OpenAI ``tool_choice`` semantics to the request's ``tools`` list.

    Returns the list of tools to render into the prompt, or ``None`` when
    the request should be served without any tool-calling affordance.

    - ``tool_choice == "none"`` — suppress tools entirely.
    - ``tool_choice == "auto"`` (or unset) — pass all tools through.
    - ``tool_choice == "required"`` — pass tools through, log that we cannot
      strictly enforce a tool call without constrained decoding.
    - ``tool_choice == {"type": "function", "function": {"name": "X"}}`` —
      filter ``tools`` to that single function and warn that the call cannot
      be strictly enforced.
    """
    if not tools:
        return None
    if tool_choice in (None, "auto"):
        return tools
    if tool_choice == "none":
        return None
    if tool_choice == "required":
        logger.warning(
            "tool_choice='required' requested but this loader cannot enforce a tool call; "
            "passing all tools to the model and trusting it to call one"
        )
        return tools
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name:
            filtered = [t for t in tools if (t.get("function") or {}).get("name") == name]
            if not filtered:
                logger.warning("tool_choice names function %r which is not in the request's tools list", name)
                return tools
            logger.warning(
                "tool_choice forcing function %r is not strictly enforced by this loader; "
                "passing only that tool to the model",
                name,
            )
            return filtered
    logger.warning("unrecognized tool_choice value %r; falling back to 'auto' semantics", tool_choice)
    return tools
