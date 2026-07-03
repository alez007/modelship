"""Guard tests for modelship/infer/vllm/engine_ops.py, the vLLM-internal quarantine layer.

Two tiers:
- Fast, always-run unit tests for the pure branching logic (`build_vllm_request`,
  `derive_reasoning_ended`, `make_parsers`) using real vLLM request/parser *types*
  but no engine, no tokenizer download, no GPU.
- `TestVllmParserAcceptsOurRequest`: real (non-mocked) vLLM render pipeline against
  real cached tokenizers (hermes+qwen3 and mistral tool parsers — the two families
  the A2 spike validated), built via `renderer_from_config` the same way vLLM's own
  GPU-less render server does. No engine/weights load, so no GPU is needed, but it
  does need the tokenizer files reachable (skips cleanly if not). This exercises the
  actual vLLM call signatures engine_ops depends on, so a vLLM version bump that
  renames/removes any of them fails here loudly instead of surfacing downstream.
"""

import inspect
from typing import Any
from unittest.mock import Mock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as VllmChatCompletionRequest,
)
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.parser import Parser
from vllm.tokenizers import TokenizerLike

from modelship.infer.vllm import engine_ops
from modelship.openai.protocol import ChatCompletionRequest


def _vllm_req(**overrides: Any) -> VllmChatCompletionRequest:
    base: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    base.update(overrides)
    return VllmChatCompletionRequest(**base)


class TestBuildVllmRequest:
    def test_request_chat_template_kwargs_wins_over_model_default(self):
        # chat_template_kwargs isn't a declared ChatCompletionRequest field — it
        # arrives as a client-supplied extra (model_config extra="allow") — so
        # build it via model_validate rather than the constructor kwargs.
        request = ChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        vllm_req = engine_ops.build_vllm_request(request, chat_template_kwargs={"enable_thinking": True, "x": 1})
        assert vllm_req.chat_template_kwargs == {"enable_thinking": False, "x": 1}

    def test_no_model_defaults_leaves_request_value_untouched(self):
        request = ChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"a": 1},
            }
        )
        vllm_req = engine_ops.build_vllm_request(request, chat_template_kwargs=None)
        assert vllm_req.chat_template_kwargs == {"a": 1}


class TestDeriveReasoningEnded:
    def test_include_reasoning_false_forces_true(self):
        vllm_req = _vllm_req(include_reasoning=False)
        assert engine_ops.derive_reasoning_ended(vllm_req, parser=None, prompt_token_ids=[]) is True

    def test_grammar_from_tool_parser_forces_true(self):
        vllm_req = _vllm_req()
        vllm_req._grammar_from_tool_parser = True
        assert engine_ops.derive_reasoning_ended(vllm_req, parser=None, prompt_token_ids=[]) is True

    def test_reasoning_parser_present_defers_to_is_reasoning_end(self):
        vllm_req = _vllm_req()
        parser = Mock(spec=Parser)
        parser.reasoning_parser = Mock()
        parser.is_reasoning_end.return_value = True
        result = engine_ops.derive_reasoning_ended(vllm_req, parser=parser, prompt_token_ids=[1, 2, 3])
        assert result is True
        parser.is_reasoning_end.assert_called_once_with([1, 2, 3])

    def test_no_reasoning_parser_and_no_grammar_flag_is_none(self):
        vllm_req = _vllm_req()
        parser = Mock(spec=Parser)
        parser.reasoning_parser = None
        assert engine_ops.derive_reasoning_ended(vllm_req, parser=parser, prompt_token_ids=[]) is None

    def test_no_parser_at_all_is_none(self):
        vllm_req = _vllm_req()
        assert engine_ops.derive_reasoning_ended(vllm_req, parser=None, prompt_token_ids=[]) is None


class TestMakeParsers:
    def test_no_parser_class_returns_n_nones(self):
        render = Mock(spec=OpenAIServingRender)
        render.parser = None
        result = engine_ops.make_parsers(render, tokenizer=Mock(), vllm_req=_vllm_req(), chat_template_kwargs=None, n=3)
        assert result == [None, None, None]

    def test_instantiates_one_independent_parser_per_choice(self):
        render = Mock(spec=OpenAIServingRender)
        parser_cls = Mock()
        instances = [Mock(), Mock(), Mock()]
        parser_cls.side_effect = instances
        render.parser = parser_cls
        vllm_req = _vllm_req(tools=None)
        tokenizer = Mock()

        result = engine_ops.make_parsers(render, tokenizer, vllm_req, chat_template_kwargs={"k": "v"}, n=3)

        assert result == instances
        assert parser_cls.call_count == 3
        for call in parser_cls.call_args_list:
            args, kwargs = call
            assert args == (tokenizer, vllm_req.tools)
            assert kwargs == {"chat_template_kwargs": {"k": "v"}}


class TestSignaturesGuardVllmBump:
    """No engine/tokenizer needed — pure import-time signature checks.

    These fail immediately (not silently) if a vLLM version bump renames or
    drops a kwarg engine_ops relies on, instead of only surfacing as a runtime
    AttributeError/TypeError deep in a real request path.
    """

    def test_async_llm_generate_accepts_reasoning_kwargs(self):
        from vllm.v1.engine.async_llm import AsyncLLM

        params = inspect.signature(AsyncLLM.generate).parameters
        assert "reasoning_ended" in params
        assert "reasoning_parser_kwargs" in params

    def test_chat_completion_request_has_grammar_from_tool_parser_private_attr(self):
        vllm_req = _vllm_req()
        assert hasattr(vllm_req, "_grammar_from_tool_parser")
        assert vllm_req._grammar_from_tool_parser is False

    def test_to_sampling_params_signature_unchanged(self):
        params = inspect.signature(VllmChatCompletionRequest.to_sampling_params).parameters
        assert list(params)[1:] == ["max_tokens", "default_sampling_params"]

    def test_render_chat_signature_unchanged(self):
        params = inspect.signature(OpenAIServingRender.render_chat).parameters
        assert "request" in params


class TestVllmParserAcceptsOurRequest:
    """Real vLLM render pipeline, real cached tokenizers, no engine/GPU.

    Mirrors the A2 spike's two model families. Built via `renderer_from_config`
    the same way vLLM's own GPU-less render server does (`init_render_app_state`
    in vllm/entrypoints/openai/api_server.py) — this needs the tokenizer files,
    not weights, so it stays fast and GPU-free. Skips cleanly if the tokenizer
    can't be fetched (no network / no HF auth for gated repos).
    """

    def _build_render(
        self, model: str, *, tokenizer_mode: str = "auto", **tool_reasoning_kwargs: Any
    ) -> tuple[OpenAIServingRender, TokenizerLike]:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.openai.models.protocol import BaseModelPath
        from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
        from vllm.entrypoints.serve.utils.request_logger import RequestLogger
        from vllm.renderers import renderer_from_config
        from vllm.usage.usage_lib import UsageContext

        try:
            engine_args = AsyncEngineArgs(
                model=model, tokenizer_mode=tokenizer_mode, max_model_len=4096, enforce_eager=True
            )
            vllm_config = engine_args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
            renderer = renderer_from_config(vllm_config)
        except Exception as e:
            pytest.skip(f"could not build a GPU-free render pipeline for {model!r}: {e}")

        registry = OpenAIModelRegistry(
            model_config=vllm_config.model_config,
            base_model_paths=[BaseModelPath(name="test-model", model_path=model)],
        )
        render = OpenAIServingRender(
            model_config=vllm_config.model_config,
            renderer=renderer,
            model_registry=registry,
            request_logger=RequestLogger(max_log_len=None),
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=True,
            **tool_reasoning_kwargs,
        )
        assert renderer.tokenizer is not None
        return render, renderer.tokenizer

    def _tool_request(self, **overrides: Any) -> VllmChatCompletionRequest:
        base: dict[str, Any] = dict(
            model="test-model",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            tool_choice="auto",
            max_tokens=64,
        )
        base.update(overrides)
        return VllmChatCompletionRequest(**base)

    @pytest.mark.asyncio
    async def test_hermes_qwen3_reasoning_and_tools(self):
        render, tokenizer = self._build_render("Qwen/Qwen3-0.6B", tool_parser="hermes", reasoning_parser="qwen3")
        vllm_req = self._tool_request()

        result = await engine_ops.render_and_params(render, vllm_req)
        assert not isinstance(result, engine_ops.VllmErrorResponse)
        engine_input, sampling_params = result
        assert sampling_params.max_tokens == 64

        prompt_ids = engine_ops.extract_prompt_token_ids(render, engine_input)
        assert prompt_ids

        n = 2
        parsers = engine_ops.make_parsers(render, tokenizer, vllm_req, chat_template_kwargs=None, n=n)
        assert len(parsers) == n
        assert all(p is not None for p in parsers)

        reasoning_ended = engine_ops.derive_reasoning_ended(vllm_req, parsers[0], prompt_ids)
        assert reasoning_ended is False  # prompt has no prior reasoning to have already ended

        fake_output = (
            "<think>I should check the weather.</think>\n"
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'
        )
        for parser in parsers:
            assert parser is not None
            reasoning, content, tool_calls = parser.parse(
                fake_output, vllm_req, enable_auto_tools=True, model_output_token_ids=[]
            )
            assert reasoning == "I should check the weather."
            assert content is None
            assert tool_calls is not None and len(tool_calls) == 1
            assert tool_calls[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_mistral_tool_only_no_grammar_flag(self):
        # A2's confirmed correction: a tool-only request (no structured-outputs
        # constraint active) takes MistralToolParser.adjust_request's early-return
        # branch and never sets _grammar_from_tool_parser, so reasoning_ended must
        # NOT assume the flag is reliably True whenever a mistral tool parser is
        # in play.
        render, tokenizer = self._build_render(
            "mistralai/Mistral-7B-Instruct-v0.3", tokenizer_mode="mistral", tool_parser="mistral"
        )
        vllm_req = self._tool_request()

        result = await engine_ops.render_and_params(render, vllm_req)
        assert not isinstance(result, engine_ops.VllmErrorResponse)
        engine_input, _sampling_params = result

        assert vllm_req._grammar_from_tool_parser is False

        prompt_ids = engine_ops.extract_prompt_token_ids(render, engine_input)
        parsers = engine_ops.make_parsers(render, tokenizer, vllm_req, chat_template_kwargs=None, n=1)
        assert len(parsers) == 1
        parser = parsers[0]
        assert parser is not None

        reasoning_ended = engine_ops.derive_reasoning_ended(vllm_req, parser, prompt_ids)
        assert reasoning_ended is None  # no reasoning parser configured for this model

        # No AttributeError on a plain-text (non-tool-call) output.
        reasoning, content, tool_calls = parser.parse(
            "Just a plain text response.", vllm_req, enable_auto_tools=True, model_output_token_ids=[]
        )
        assert reasoning is None
        assert content == "Just a plain text response."
        assert tool_calls in (None, [])
