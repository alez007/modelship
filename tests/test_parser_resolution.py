"""Driver-side parser resolution: tool-calling and reasoning.

Covers `resolve_all_tool_parsers` / `resolve_all_reasoning_parsers` populating
`_resolved_tool_call_parser` / `_resolved_reasoning_parser` as the single
source of truth for loader code.
"""

from __future__ import annotations

import pytest

from modelship.deploy.config import resolve_all_reasoning_parsers, resolve_all_tool_parsers
from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipConfig,
    ModelshipModelConfig,
    ModelUsecase,
    VllmEngineConfig,
)
from modelship.openai.parsers.reasoning import classify_template as classify_reasoning
from modelship.openai.parsers.tool_calling import classify_template as classify_tool_calling


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
    """``classify_template`` must return names that match vLLM's own
    ``ToolParserManager`` registry exactly — ``resolve_all_tool_parsers``
    validates auto-detected names against it directly."""

    def test_no_tool_markers_returns_none(self):
        assert classify_tool_calling("plain template with no markers") is None

    def test_gemma4_marker(self):
        assert classify_tool_calling("{% if tools %}<|tool_call>{% endif %}") == "gemma4"

    def test_function_gemma_marker_matches_vllm_name(self):
        # Regression: vLLM registers this parser as "functiongemma" (no
        # underscore); modelship's own class used to be named "function_gemma"
        # and this detector returned that mismatched name, which would fail
        # validation against vLLM's real registry (or, worse, be handed
        # straight to vLLM's OpenAIServingRender and fail there instead).
        assert classify_tool_calling("{% if tools %}<start_function_call>{% endif %}") == "functiongemma"

    def test_qwen3_coder_function_marker_routes_ahead_of_hermes(self):
        # The chat template mentions tools (gating clause) and contains
        # ``<function=`` — must not fall through to Hermes.
        template = "{% if tools %}<tool_call>\n<function={{ name }}>{% endif %}"
        assert classify_tool_calling(template) == "qwen3_coder"

    def test_qwen3_coder_parameter_marker(self):
        template = "{% if tools %}<parameter={{ key }}>value</parameter>{% endif %}"
        assert classify_tool_calling(template) == "qwen3_coder"

    def test_hermes_template_without_function_marker_stays_hermes(self):
        template = '{% if tools %}<tool_call>{"name": "x"}</tool_call>{% endif %}'
        assert classify_tool_calling(template) == "hermes"

    def test_mistral_marker(self):
        assert classify_tool_calling("{% if tools %}[TOOL_CALLS]{% endif %}") == "mistral"

    def test_llama3_json_marker(self):
        assert classify_tool_calling("{% if tools %}<|python_tag|>{% endif %}") == "llama3_json"

    def test_unrecognized_markers_returns_unknown(self):
        assert classify_tool_calling("{% if tools %}some tool syntax{% endif %}") == "unknown"


class TestClassifyReasoningTemplate:
    def test_open_think_tag(self):
        assert classify_reasoning("...<think>...") == "deepseek_r1"

    def test_close_think_tag(self):
        assert classify_reasoning("...</think>...") == "deepseek_r1"

    def test_no_markers_returns_none(self):
        assert classify_reasoning("plain template with no markers") is None

    def test_empty_returns_none(self):
        assert classify_reasoning("") is None


class TestResolveReasoningParsers:
    def test_explicit_parser_stored(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(reasoning_parser="deepseek_r1"))
        resolve_all_reasoning_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_reasoning_parser == "deepseek_r1"

    def test_explicit_opt_out_leaves_none(self, monkeypatch):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(enable_reasoning=False))
        cfg._resolved_chat_template = "<think>x</think>"  # would auto-detect if not opted out
        resolve_all_reasoning_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_reasoning_parser is None

    def test_auto_detect_from_template(self, monkeypatch):
        cfg = _make_cfg()
        cfg._resolved_chat_template = "blah <think>...</think> blah"
        resolve_all_reasoning_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_reasoning_parser == "deepseek_r1"

    def test_no_markers_leaves_none(self):
        cfg = _make_cfg()
        cfg._resolved_chat_template = "no reasoning markers here"
        resolve_all_reasoning_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_reasoning_parser is None

    def test_explicit_wins_over_auto(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(reasoning_parser="deepseek_r1"))
        cfg._resolved_chat_template = "<think>x</think>"
        resolve_all_reasoning_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_reasoning_parser == "deepseek_r1"

    def test_skips_non_generate_usecase(self):
        cfg = _make_cfg(usecase=ModelUsecase.embed)
        cfg._resolved_chat_template = "<think>x</think>"
        resolve_all_reasoning_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_reasoning_parser is None

    def test_skips_non_applicable_loader(self):
        cfg = _make_cfg(loader=ModelLoader.diffusers, usecase=ModelUsecase.image)
        cfg._resolved_chat_template = "<think>x</think>"
        resolve_all_reasoning_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_reasoning_parser is None

    def test_reads_template_from_disk_when_not_cached(self, monkeypatch):
        cfg = _make_cfg()
        cfg._resolved_path = "/fake/path"
        monkeypatch.setattr(
            "modelship.deploy.config.read_chat_template",
            lambda p: "{% if tools %}<think>{% endif %}",
        )
        resolve_all_reasoning_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_reasoning_parser == "deepseek_r1"
        # And the template gets cached on the cfg.
        assert cfg._resolved_chat_template is not None

    def test_no_template_leaves_none(self, monkeypatch):
        cfg = _make_cfg()
        cfg._resolved_path = "/fake/path"
        monkeypatch.setattr("modelship.deploy.config.read_chat_template", lambda p: None)
        resolve_all_reasoning_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_reasoning_parser is None


class TestResolveToolParsersStoresExplicit:
    """Regression: explicit `tool_call_parser` must populate `_resolved_tool_call_parser`."""

    def test_vllm_explicit_stored(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(tool_call_parser="hermes"))
        resolve_all_tool_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_tool_call_parser == "hermes"

    def test_unknown_explicit_raises(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(tool_call_parser="not-a-real-parser"))
        with pytest.raises(ValueError, match="not-a-real-parser"):
            resolve_all_tool_parsers(ModelshipConfig(models=[cfg]))

    def test_vllm_opt_out_leaves_none(self):
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(enable_auto_tool_choice=False, tool_call_parser="hermes"))
        resolve_all_tool_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_tool_call_parser is None
