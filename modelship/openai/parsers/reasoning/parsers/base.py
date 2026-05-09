"""Base class for model-family-specific reasoning output parsers.

Reasoning parsers are *marker-based*: each family wraps its
chain-of-thought in a fixed pair of literal strings (``<think>`` /
``</think>`` for DeepSeek-R1, QwQ, Qwen3, Phi-4-reasoning, ...).
Unlike tool-call parsers, the payload between markers is opaque text —
no per-family extractors are needed, just the marker pair.

The unified :class:`ChatOutputStreamer` (over in
``modelship.openai.parsers.tool_calling.parsers``) consumes a parser
instance and walks the cumulative model output once, splitting it into
content / reasoning / tool-call regions in a single pass.
"""

from __future__ import annotations

from abc import ABC


class ReasoningParser(ABC):
    """Family-specific marker pair for reasoning extraction.

    Subclasses set ``name``, ``start_marker``, ``end_marker``. They hold
    no per-request state — that all lives on :class:`ChatOutputStreamer`.
    """

    name: str
    start_marker: str
    end_marker: str
