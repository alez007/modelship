"""DeepSeek-R1-style ``<think>...</think>`` reasoning parser.

Used by DeepSeek-R1 (and distilled variants), QwQ, Qwen3, and
Phi-4-reasoning, all of which wrap their chain-of-thought in the
literal tags ``<think>`` / ``</think>``.
"""

from __future__ import annotations

from modelship.openai.parsers.reasoning.parsers.base import ReasoningParser


class DeepseekR1ReasoningParser(ReasoningParser):
    name = "deepseek_r1"
    start_marker = "<think>"
    end_marker = "</think>"
