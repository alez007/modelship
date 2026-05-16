"""Qwen3-Coder XML-style tool call parser.

Qwen3-Coder reuses the same ``<tool_call>`` / ``</tool_call>`` envelope as
Hermes but the body is an XML-ish nested structure instead of a JSON
object:

    <tool_call>
    <function=function_name>
    <parameter=param_a>
    value_a
    </parameter>
    <parameter=param_b>
    42
    </parameter>
    </function>
    </tool_call>

Disambiguation from Hermes happens upstream in
:func:`modelship.openai.parsers.tool_calling.utils.classify_template`: a
chat template that mentions ``<function=`` or ``<parameter=`` is routed
to this parser; otherwise the same ``<tool_call>`` marker falls through
to Hermes.

Each ``<tool_call>...</tool_call>`` envelope holds exactly one
``<function=...>`` block; multiple tool calls arrive as multiple
back-to-back envelopes, which the streamer already treats as independent
regions, so no :meth:`split_payload` override is needed.

Parameter values are kept as strings — Qwen3-Coder treats them as
textual blobs without explicit type markers. Callers that need typed
arguments (numbers, booleans, nested JSON) can JSON-decode the value in
their tool handler. Streaming-safe coercion would require schema-aware
disambiguation we don't have at this layer.
"""

from __future__ import annotations

import json
import re

from modelship.openai.parsers.tool_calling.parsers.base import ToolCallParser

_FUNC_OPEN_RE = re.compile(r"<function=([^>]+)>")
_PARAM_OPEN_RE = re.compile(r"<parameter=([^>]+)>")
_PARAM_CLOSE = "</parameter>"
_FUNC_CLOSE = "</function>"

# Structural suffix that ``json.dumps`` always emits for a non-empty dict
# whose last value is a string: the closing quote of that value plus the
# closing brace of the object. Stripping exactly these two bytes exposes
# the in-flight value as a monotonic prefix without touching content
# inside the value — a character-wise loop would eat trailing ``,``,
# ``}``, or ``"`` that are actually part of the parameter value.
_STREAM_STRIP_TAIL = '"}'


class Qwen3CoderToolCallParser(ToolCallParser):
    name = "qwen3_coder"
    start_marker = "<tool_call>"
    end_marker = "</tool_call>"

    def extract_partial_name(self, partial_payload: str) -> str | None:
        m = _FUNC_OPEN_RE.search(partial_payload)
        return m.group(1).strip() if m else None

    def extract_partial_args(self, partial_payload: str, is_complete: bool = False) -> str | None:
        fm = _FUNC_OPEN_RE.search(partial_payload)
        if fm is None:
            return None

        body = partial_payload[fm.end() :]
        end_func = body.find(_FUNC_CLOSE)
        if end_func != -1:
            body = body[:end_func]

        args: dict[str, str] = {}
        pos = 0
        while True:
            pm = _PARAM_OPEN_RE.search(body, pos)
            if pm is None:
                break
            key = pm.group(1).strip()
            val_start = pm.end()
            close = body.find(_PARAM_CLOSE, val_start)
            if close == -1:
                # In-flight parameter value. Withhold any trailing bytes
                # that could be a proper prefix of ``</parameter>`` or of
                # the next ``<parameter=`` opener, so the streamed value
                # is a monotonic prefix of its eventual final form.
                raw = body[val_start:]
                cut = min(_safe_overlap(raw, _PARAM_CLOSE), _safe_overlap(raw, "<parameter="))
                raw = raw[:cut]
                stripped = raw.strip()
                if stripped:
                    args[key] = stripped
                break
            args[key] = body[val_start:close].strip()
            pos = close + len(_PARAM_CLOSE)

        if not args:
            return "{}" if is_complete else ""

        full_json = json.dumps(args, ensure_ascii=False)
        if is_complete:
            return full_json

        if full_json.endswith(_STREAM_STRIP_TAIL):
            return full_json[: -len(_STREAM_STRIP_TAIL)]
        return full_json


def _safe_overlap(buf: str, marker: str) -> int:
    """Index up to which ``buf`` can be flushed without splitting ``marker``.

    Returns the position past the last byte safe to commit; trailing bytes
    that could be a proper prefix of ``marker`` are excluded so partial
    output stays a monotonic prefix of the eventual final value.
    """
    if not marker:
        return len(buf)
    max_overlap = min(len(buf), len(marker) - 1)
    for k in range(max_overlap, 0, -1):
        if buf.endswith(marker[:k]):
            return len(buf) - k
    return len(buf)
