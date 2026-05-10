"""Mistral-style ``[TOOL_CALLS][{...}, {...}]`` parser.

Used by Mistral 7B Instruct v0.3+, Mistral Small/Large, and downstream
fine-tunes whose chat templates emit a single ``[TOOL_CALLS]`` envelope
followed by a JSON array of one or more ``{"name": "...", "arguments":
{...}}`` objects, with no trailing closing marker — the array runs to
end-of-stream.

The per-call JSON shape is identical to Hermes; the differences are
that all calls share one envelope (we split the array into per-call
sub-blocks via :meth:`split_payload`) and that the region is
EOS-bounded (the streamer marks it complete at finalize time when
``end_marker`` is empty).
"""

from __future__ import annotations

import re

from modelship.openai.parsers.tool_calling.parsers.base import ToolCallParser


class MistralToolCallParser(ToolCallParser):
    name = "mistral"
    start_marker = "[TOOL_CALLS]"
    end_marker = ""
    # ``[TOOL_CALLS]`` is registered in Mistral v3+ tokenizers'
    # ``added_tokens_decoder`` with ``special=True`` — alongside ``[INST]``,
    # ``[/INST]``, ``[AVAILABLE_TOOLS]``, etc. ``skip_special_tokens=True``
    # strips it on the way out of detokenization, so loaders must opt in to
    # keep it. See ``tests/test_mistral_specials_smoketest.py`` for the
    # round-trip evidence.
    markers_are_specials = True

    _NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
    _ARGS_RE = re.compile(r'"arguments"\s*:\s*')

    def split_payload(self, payload: str, is_complete: bool) -> list[tuple[str, bool]]:
        """Split ``[{...}, {...}]`` into per-call ``({...}, complete)`` entries.

        Walks the payload once tracking string state and brace depth. Each
        balanced top-level ``{...}`` becomes one complete sub-block; a
        trailing partially-open object becomes one incomplete sub-block (so
        the streamer can start emitting its name/args as bytes arrive).

        ``is_complete`` is the region's completion flag (True at finalize).
        It only affects whether a trailing partial object is reported as
        complete — closed objects are always complete on their own merits.
        """
        s = payload.lstrip()
        # Skip the leading `[` of the JSON array if present. Some templates
        # render `[TOOL_CALLS] [{...}]` with whitespace; lstrip handled that.
        if s.startswith("["):
            s = s[1:]

        results: list[tuple[str, bool]] = []
        depth = 0
        in_str = False
        escape = False
        start: int | None = None
        for i, c in enumerate(s):
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
                    results.append((s[start : i + 1], True))
                    start = None

        if start is not None and depth > 0:
            # Partial trailing object. Mark it complete only if the whole
            # region is complete AND somehow we landed inside an object — in
            # practice the model never closes the array without closing the
            # object, so a complete region means a clean split above.
            results.append((s[start:], is_complete and depth == 0))
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
            # Mirror Hermes: the per-call envelope is
            # `{"name":"x","arguments":<args>}` and the closing brace of
            # the envelope arrives in the byte stream alongside (or before)
            # the args object's closer. Withhold one trailing `}` so the
            # streamed args view never contains the envelope's closer.
            args = args[:-1].rstrip()
        return args or None
