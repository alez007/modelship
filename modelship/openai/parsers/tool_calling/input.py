"""Input-side helpers for tool calling.

Loaders that hand a chat template a list of OpenAI messages plus a list of
tool schemas use these helpers to interpret the request's ``tool_choice``:
``resolve_tools_for_request`` shapes the tools list passed into the prompt and
``request_forces_tool_call`` reports whether the request mandates a tool call.
Enforcement of that mandate is the loader's job (e.g. the llama_cpp tool-call
grammar) — these helpers only express intent.
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
    - ``tool_choice == "auto"`` / ``"required"`` (or unset) — pass all tools
      through. Whether ``"required"`` is enforced is the loader's call (see
      :func:`request_forces_tool_call`), not this helper's.
    - ``tool_choice == {"type": "function", "function": {"name": "X"}}`` —
      filter ``tools`` to that single function.
    """
    if not tools:
        return None
    if tool_choice in (None, "auto", "required"):
        return tools
    if tool_choice == "none":
        return None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name:
            filtered = [t for t in tools if (t.get("function") or {}).get("name") == name]
            if not filtered:
                logger.warning("tool_choice names function %r which is not in the request's tools list", name)
                return tools
            return filtered
    logger.warning("unrecognized tool_choice value %r; falling back to 'auto' semantics", tool_choice)
    return tools


def request_forces_tool_call(tool_choice: str | dict[str, Any] | None) -> bool:
    """Whether ``tool_choice`` mandates a tool call (vs. allowing free text).

    True for ``"required"`` and the ``{"type": "function", "function":
    {"name": "X"}}`` named-function form. Loaders use this to pick the
    tool-only grammar root.
    """
    if tool_choice == "required":
        return True
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        name = fn.get("name") if isinstance(fn, dict) else None
        return isinstance(name, str) and bool(name)
    return False
