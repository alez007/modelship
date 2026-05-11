"""Hermes-style ``<tool_call>{json}</tool_call>`` parser.

Used by Hermes-2-Pro, Qwen2.5-Instruct, and a large family of NousResearch /
community fine-tunes whose chat templates wrap each tool call in the literal
tags ``<tool_call>`` / ``</tool_call>`` around a JSON object of the shape
``{"name": "...", "arguments": {...}}``.
"""

from __future__ import annotations

import re

from modelship.openai.parsers.tool_calling.parsers.base import ToolCallParser


class HermesToolCallParser(ToolCallParser):
    name = "hermes"
    start_marker = "<tool_call>"
    end_marker = "</tool_call>"

    _NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
    _ARGS_RE = re.compile(r'"arguments"\s*:\s*')

    def extract_partial_name(self, partial_payload: str) -> str | None:
        m = self._NAME_RE.search(partial_payload)
        return m.group(1) if m else None

    def extract_partial_args(self, partial_payload: str, is_complete: bool = False) -> str | None:
        m = self._ARGS_RE.search(partial_payload)
        if m is None:
            return None
        args = partial_payload[m.end() :].rstrip()
        if args.endswith("}"):
            # The block envelope is `{"name":"x","arguments":<args>}`. The
            # closing brace of the envelope arrives in the byte stream before
            # `</tool_call>` does, so we cannot tell whether any given
            # trailing `}` belongs to the args object or to the envelope.
            # Withholding one trailing `}` keeps the args stream well-formed:
            # if the model goes on to emit more args bytes, the held brace is
            # recovered on the next pass; if instead it goes on to emit
            # `</tool_call>`, the held brace was the envelope closer and
            # discarding it was correct. At ``is_complete=True`` the trailing
            # `}` is always the envelope closer (the args object's own closer
            # is preceded by it), so the strip still applies.
            args = args[:-1].rstrip()
        return args or None
