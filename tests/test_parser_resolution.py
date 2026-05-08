"""Driver-side parser resolution: tool-calling and reasoning.

Covers `resolve_all_tool_parsers` / `resolve_all_reasoning_parsers` populating
`_resolved_tool_call_parser` / `_resolved_reasoning_parser` as the single
source of truth for loader code.
"""

from __future__ import annotations

import pytest

from modelship.deploy.config import resolve_all_reasoning_parsers, resolve_all_tool_parsers
from modelship.infer.infer_config import (
    LlamaCppConfig,
    ModelLoader,
    ModelshipConfig,
    ModelshipModelConfig,
    ModelUsecase,
    TransformersConfig,
    VllmEngineConfig,
)
from modelship.openai.parsers.reasoning.utils import classify_template as classify_reasoning


def _make_cfg(**overrides) -> ModelshipModelConfig:
    base = {
        "name": "m",
        "model": "some/model",
        "usecase": ModelUsecase.generate,
        "loader": ModelLoader.vllm,
    }
    base.update(overrides)
    return ModelshipModelConfig(**base)


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
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(reasoning_parser="qwen3"))
        resolve_all_reasoning_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_reasoning_parser == "qwen3"

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
        cfg = _make_cfg(vllm_engine_kwargs=VllmEngineConfig(reasoning_parser="custom_x"))
        cfg._resolved_chat_template = "<think>x</think>"
        resolve_all_reasoning_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_reasoning_parser == "custom_x"

    def test_skips_non_generate_usecase(self):
        cfg = _make_cfg(usecase=ModelUsecase.embed)
        cfg._resolved_chat_template = "<think>x</think>"
        resolve_all_reasoning_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_reasoning_parser is None

    def test_skips_non_applicable_loader(self):
        cfg = _make_cfg(loader=ModelLoader.diffusers)
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

    def test_transformers_explicit_stored(self):
        cfg = _make_cfg(
            loader=ModelLoader.transformers,
            transformers_config=TransformersConfig(tool_call_parser="hermes"),
        )
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

    def test_llama_cpp_chat_format_opts_out(self):
        cfg = _make_cfg(
            loader=ModelLoader.llama_cpp,
            llama_cpp_config=LlamaCppConfig(chat_format="chatml-function-calling"),
        )
        cfg._resolved_chat_template = "{% if tools %}<tool_call>{% endif %}"
        resolve_all_tool_parsers(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_tool_call_parser is None
