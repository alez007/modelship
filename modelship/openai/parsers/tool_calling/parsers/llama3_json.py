"""Llama-3.1/3.2 JSON tool-call parser.

Used by Llama-3.1 / Llama-3.2 Instruct models whose chat template renders a
custom JSON tool call as a bare object — ``{"name": "fn", "parameters":
{...}}`` followed by ``<|eot_id|>`` (the official Meta / vLLM
``tool_chat_template_llama3.1_json.jinja`` shape). A leading
``<|python_tag|>`` is *not* part of the JSON variant; it is reserved for
built-in code-interpreter calls and would be stripped by HF's default
``skip_special_tokens=True`` anyway. The parser starts at the literal
substring ``{"name"`` so it sees the JSON regardless of whether the tag
prefix was emitted upstream.

Differences from Hermes / Mistral:

- The marker ``{"name"`` is *part of* the JSON object, not an envelope
  around it, so the parser sets :attr:`consume_start_marker` to False so
  the streamer keeps the marker bytes at the head of the payload.
- The arguments field name is ``parameters`` per Meta's spec; some
  community fine-tunes emit ``arguments`` instead. The parser accepts
  either and the streamer forwards the field bytes verbatim — clients
  see the JSON the model produced.
- Multi-call output uses ``; `` (semicolon-space) between top-level
  objects, matching vLLM's convention.
"""

from __future__ import annotations

import re

from modelship.openai.parsers.tool_calling.parsers.base import ToolCallParser


class Llama3JsonToolCallParser(ToolCallParser):
    name = "llama3_json"
    start_marker = '{"name"'
    end_marker = ""
    consume_start_marker = False

    _NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
    _ARGS_RE = re.compile(r'"(?:parameters|arguments)"\s*:\s*')

    def split_payload(self, payload: str, is_complete: bool) -> list[tuple[str, bool]]:
        """Split ``{...}; {...}`` into per-call ``({...}, complete)`` entries.

        Walks the payload once tracking string state and brace depth. Each
        balanced top-level ``{...}`` becomes one complete sub-block; a
        trailing partially-open object becomes one incomplete sub-block so
        the streamer can begin emitting its name/args as bytes arrive. The
        ``; `` separator between objects is skipped along with any other
        whitespace at depth 0.

        ``is_complete`` is the region's completion flag (True at finalize).
        It only affects whether a trailing partial object is reported as
        complete — closed objects are always complete on their own merits.
        """
        results: list[tuple[str, bool]] = []
        depth = 0
        in_str = False
        escape = False
        start: int | None = None
        for i, c in enumerate(payload):
            if escape:
                escape = False
                continue
            if in_str:
                if c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    results.append((payload[start : i + 1], True))
                    start = None

        if start is not None and depth > 0:
            # Trailing partial object — only complete if the region is
            # complete (finalize) AND we somehow ended balanced, which the
            # closed-on-its-own-merits branch above would have caught.
            results.append((payload[start:], is_complete and depth == 0))
        return results

    def extract_partial_name(self, partial_payload: str) -> str | None:
        m = self._NAME_RE.search(partial_payload)
        return m.group(1) if m else None

    def extract_partial_args(self, partial_payload: str) -> str | None:
        m = self._ARGS_RE.search(partial_payload)
        if m is None:
            return None
        args = partial_payload[m.end() :].rstrip()
        if args.endswith("}"):
            # Mirror Hermes's trailing-`}` withhold: the per-call envelope
            # is ``{"name": "x", "parameters": <args>}``. The closing brace
            # of the envelope arrives in the byte stream alongside (or
            # before) the args object's closer. Withhold one trailing `}`
            # so the streamed args view never contains the envelope's
            # closer; if more args bytes follow, the held brace is
            # recovered on the next pass.
            args = args[:-1].rstrip()
        return args or None
