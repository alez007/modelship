"""Render-based reasoning suppression detection."""

from modelship.openai.parsers.reasoning import reasoning_active_in_render, resolve_active_reasoning_parser
from modelship.openai.parsers.utils import render_generation_prompt

# Qwen3-style: prefills a closed <think></think> only when enable_thinking is false.
_QWEN3_LIKE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant: "
    "{% if enable_thinking is defined and not enable_thinking %}<think></think>{% endif %}{% endif %}"
)


class TestRenderGenerationProbe:
    def test_thinking_on_keeps_parser(self):
        parser = resolve_active_reasoning_parser("deepseek_r1", lambda: render_generation_prompt(_QWEN3_LIKE, {}))
        assert parser == "deepseek_r1"

    def test_thinking_off_downgrades(self):
        parser = resolve_active_reasoning_parser(
            "deepseek_r1", lambda: render_generation_prompt(_QWEN3_LIKE, {"enable_thinking": False})
        )
        assert parser is None


class TestReasoningActiveInRender:
    def test_closed_block_is_suppressed(self):
        # Qwen3 enable_thinking=false prefills a closed block.
        assert reasoning_active_in_render("...prompt...<think></think>", "<think>", "</think>") is False

    def test_open_block_is_active(self):
        assert reasoning_active_in_render("...prompt...<think>", "<think>", "</think>") is True

    def test_no_marker_is_active(self):
        # Thinking-on templates often prime nothing; the model emits <think> itself.
        assert reasoning_active_in_render("plain prompt", "<think>", "</think>") is True

    def test_prior_closed_plus_open_is_active(self):
        assert reasoning_active_in_render("<think>a</think> ... <think>", "<think>", "</think>") is True

    def test_empty_start_marker_is_active(self):
        assert reasoning_active_in_render("anything", "", "") is True


class TestResolveActiveReasoningParser:
    def test_none_candidate_short_circuits(self):
        called = False

        def render():
            nonlocal called
            called = True
            return ""

        assert resolve_active_reasoning_parser(None, render) is None
        assert called is False

    def test_suppressed_downgrades_to_none(self):
        assert resolve_active_reasoning_parser("deepseek_r1", lambda: "p <think></think>") is None

    def test_active_keeps_candidate(self):
        assert resolve_active_reasoning_parser("deepseek_r1", lambda: "p <think>") == "deepseek_r1"

    def test_render_failure_falls_back_to_candidate(self):
        def boom():
            raise RuntimeError("render exploded")

        assert resolve_active_reasoning_parser("deepseek_r1", boom) == "deepseek_r1"

    def test_unknown_candidate_skips_probe(self):
        # An explicitly configured vLLM reasoning parser modelship doesn't
        # auto-detect (no known start/end markers) — nothing to probe against,
        # so the candidate is trusted as-is and the render is never called.
        called = False

        def render():
            nonlocal called
            called = True
            return ""

        assert resolve_active_reasoning_parser("qwen3", render) == "qwen3"
        assert called is False
