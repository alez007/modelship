"""Utilities for the llama_cpp loader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelship.logging import get_logger

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = get_logger("infer.llama_cpp.utils")


@dataclass
class LlamaCppToolCallRenderer:
    """Renders tool-aware chat prompts and counts tokens for a llama_cpp model.

    The chat template is pre-resolved on the driver (``_resolved_chat_template``)
    so detection and rendering see the exact same string — no chance of drift
    if the GGUF reader and llama-cpp-python's metadata loader interpret the
    field differently.

    Rendering uses ``transformers.utils.chat_template_utils.render_jinja_template``
    so we inherit the polyfills HF templates rely on (``raise_exception``,
    ``tojson``, ``strftime_now``, ``ChainableUndefined``).

    Token counting goes through ``Llama.tokenize`` so it reflects the actual
    GGUF vocab — no second tokenizer to keep in sync.
    """

    chat_template: str
    bos_token: str
    eos_token: str
    _llama: Llama

    def render(self, messages: list[dict], tools: list[dict] | None) -> str:
        from transformers.utils.chat_template_utils import render_jinja_template

        # Reasoning chat templates (e.g. Qwen3) test `'</think>' in message.content`,
        # which raises "argument of type 'NoneType' is not iterable" on assistant
        # tool-call messages that legitimately carry content=None (replayed multi-turn
        # tool calls). Coerce None to "" so the membership test is a safe no-op.
        safe_messages = [{**m, "content": ""} if m.get("content") is None else m for m in messages]

        rendered, _ = render_jinja_template(
            conversations=[safe_messages],
            tools=tools,  # type: ignore[arg-type]  # HF types it as list[dict | Callable]; we only pass dicts
            chat_template=self.chat_template,
            add_generation_prompt=True,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
        )
        return rendered[0]

    def count_tokens(self, text: str) -> int:
        try:
            tokens = self._llama.tokenize(text.encode("utf-8"), add_bos=False, special=True)
            return len(tokens)
        except Exception as e:
            logger.warning("llama_cpp tokenize failed (%s); returning 0", e)
            return 0


def build_tool_call_renderer(llama: Llama, chat_template: str) -> LlamaCppToolCallRenderer:
    """Build a renderer from a loaded ``Llama`` and the pre-resolved template."""
    return LlamaCppToolCallRenderer(
        chat_template=chat_template,
        bos_token=_detokenize_special(llama, llama.token_bos()),
        eos_token=_detokenize_special(llama, llama.token_eos()),
        _llama=llama,
    )


def _detokenize_special(llama: Llama, token_id: int) -> str:
    if token_id < 0:
        return ""
    try:
        return llama.detokenize([token_id], special=True).decode("utf-8", errors="replace")
    except Exception:
        return ""
