"""Gemma 4 "Thinking Mode" reasoning parser.

Gemma 4 uses a "channel" mechanism to separate internal reasoning from
the final answer. Reasoning is wrapped in `<|channel>thought\\n...<channel|>`.
"""

from __future__ import annotations

from modelship.openai.parsers.reasoning.parsers.base import ReasoningParser


class Gemma4ReasoningParser(ReasoningParser):
    name = "gemma4"
    # We include 'thought\n' in the start marker so ChatOutputStreamer
    # strips it from the reasoning payload, consistent with vLLM's
    # implementation.
    start_marker = "<|channel>thought\n"
    end_marker = "<channel|>"
    markers_are_specials = True
