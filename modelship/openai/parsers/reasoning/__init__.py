"""Cross-loader reasoning parser toolkit.

Loaders without native reasoning support (transformers, llama.cpp) use
the parsers here, dispatched via the unified
:class:`ChatOutputStreamer`, to surface ``<think>...</think>`` blocks
in the protocol-level ``reasoning`` field. vLLM has its own built-in
reasoning parsers and uses only the auto-detected parser name.
"""

from modelship.openai.parsers.reasoning.parsers import DeepseekR1ReasoningParser, ReasoningParser
from modelship.openai.parsers.reasoning.registry import available_parsers, get_parser, register_parser
from modelship.openai.parsers.reasoning.utils import classify_template, detect_reasoning_parser

__all__ = [
    "DeepseekR1ReasoningParser",
    "ReasoningParser",
    "available_parsers",
    "classify_template",
    "detect_reasoning_parser",
    "get_parser",
    "register_parser",
]
