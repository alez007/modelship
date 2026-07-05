"""Tool-call parser name detection by chat-template inspection.

Auto-detects which vLLM-native tool-call parser (``vllm.tool_parsers.ToolParserManager``)
a model's chat template calls for, by marker sniffing. Names returned here must match
vLLM's own registered parser names exactly — ``deploy/config.py`` validates them against
vLLM's registry directly rather than a modelship-side one.
"""

from __future__ import annotations


def classify_template(template: str) -> str | None:
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
