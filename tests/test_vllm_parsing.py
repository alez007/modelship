"""vLLM tool-call / reasoning parser-name detection, run inside the vllm actor.

Covers `resolve_tool_parser` / `resolve_reasoning_parser` resolving the parser
name `init_serving_chat` hands to `OpenAIServingRender`.
"""

from __future__ import annotations

import pytest

from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
    VllmEngineConfig,
)
from modelship.infer.vllm.parsing.detect import (
    classify_reasoning_template,
    classify_tool_template,
    resolve_reasoning_parser,
    resolve_tool_parser,
)


def _make_cfg(**overrides) -> ModelshipModelConfig:
    base = {
        "name": "m",
        "model": "some/model",
        "usecase": ModelUsecase.generate,
        "loader": ModelLoader.vllm,
    }
    base.update(overrides)
    return ModelshipModelConfig(**base)


class TestClassifyToolTemplate:
    """``classify_tool_template`` must return names that match vLLM's own
    ``ToolParserManager`` registry exactly — ``resolve_tool_parser``
    validates auto-detected names against it directly."""

    def test_no_tool_markers_returns_none(self):
        assert classify_tool_template("plain template with no markers") is None

    def test_gemma4_marker(self):
        assert classify_tool_template("{% if tools %}<|tool_call>{% endif %}") == "gemma4"

    def test_function_gemma_marker_matches_vllm_name(self):
        # Regression: vLLM registers this parser as "functiongemma" (no
        # underscore); modelship's own class used to be named "function_gemma"
        # and this detector returned that mismatched name, which would fail
        # validation against vLLM's real registry (or, worse, be handed
        # straight to vLLM's OpenAIServingRender and fail there instead).
        assert classify_tool_template("{% if tools %}<start_function_call>{% endif %}") == "functiongemma"

    def test_qwen3_coder_function_marker_routes_ahead_of_hermes(self):
        # The chat template mentions tools (gating clause) and contains
        # ``<function=`` — must not fall through to Hermes.
        template = "{% if tools %}<tool_call>\n<function={{ name }}>{% endif %}"
        assert classify_tool_template(template) == "qwen3_coder"

    def test_qwen3_coder_parameter_marker(self):
        template = "{% if tools %}<parameter={{ key }}>value</parameter>{% endif %}"
        assert classify_tool_template(template) == "qwen3_coder"

    def test_hermes_template_without_function_marker_stays_hermes(self):
        template = '{% if tools %}<tool_call>{"name": "x"}</tool_call>{% endif %}'
        assert classify_tool_template(template) == "hermes"

    def test_mistral_marker(self):
        assert classify_tool_template("{% if tools %}[TOOL_CALLS]{% endif %}") == "mistral"

    def test_llama3_json_marker(self):
        assert classify_tool_template("{% if tools %}<|python_tag|>{% endif %}") == "llama3_json"

    def test_unrecognized_markers_returns_unknown(self):
        assert classify_tool_template("{% if tools %}some tool syntax{% endif %}") == "unknown"


class TestClassifyReasoningTemplate:
    def test_open_think_tag(self):
        assert classify_reasoning_template("...<think>...") == "deepseek_r1"

    def test_close_think_tag(self):
        assert classify_reasoning_template("...</think>...") == "deepseek_r1"

    def test_no_markers_returns_none(self):
        assert classify_reasoning_template("plain template with no markers") is None

    def test_empty_returns_none(self):
        assert classify_reasoning_template("") is None


class TestResolveReasoningParsers:
    def test_explicit_parser_stored(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(reasoning_parser="deepseek_r1"))
        assert resolve_reasoning_parser(cfg, None) == "deepseek_r1"

    def test_explicit_opt_out_leaves_none(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(enable_reasoning=False))
        # Would auto-detect if not opted out.
        assert resolve_reasoning_parser(cfg, "<think>x</think>") is None

    def test_auto_detect_from_template(self):
        cfg = _make_cfg()
        assert resolve_reasoning_parser(cfg, "blah <think>...</think> blah") == "deepseek_r1"

    def test_no_markers_leaves_none(self):
        cfg = _make_cfg()
        assert resolve_reasoning_parser(cfg, "no reasoning markers here") is None

    def test_no_template_leaves_none(self):
        cfg = _make_cfg()
        assert resolve_reasoning_parser(cfg, None) is None

    def test_explicit_wins_over_auto(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(reasoning_parser="deepseek_r1"))
        assert resolve_reasoning_parser(cfg, "<think>x</think>") == "deepseek_r1"

    def test_unknown_explicit_raises(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(reasoning_parser="not-a-real-parser"))
        with pytest.raises(ValueError, match="not-a-real-parser"):
            resolve_reasoning_parser(cfg, None)


class TestResolveToolParsersStoresExplicit:
    """Regression: explicit `tool_call_parser` must be returned as-is."""

    def test_vllm_explicit_stored(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(tool_call_parser="hermes"))
        assert resolve_tool_parser(cfg, None) == "hermes"

    def test_unknown_explicit_raises(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(tool_call_parser="not-a-real-parser"))
        with pytest.raises(ValueError, match="not-a-real-parser"):
            resolve_tool_parser(cfg, None)

    def test_vllm_opt_out_leaves_none(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(enable_auto_tool_choice=False, tool_call_parser="hermes"))
        assert resolve_tool_parser(cfg, None) is None
