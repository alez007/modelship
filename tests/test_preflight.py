"""Tests for the preflight estimator framework and VllmPreflight."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
    VllmEngineConfig,
)
from modelship.preflight import (
    GPUInfo,
    HardwareProfile,
    merge_with_user_overrides,
    run_preflight,
)
from modelship.preflight.vllm import VllmPreflight, _is_moe


def _make_config(
    *,
    resolved_path: str | None = None,
    vllm_kwargs: dict | None = None,
    num_gpus: float = 0,
) -> ModelshipModelConfig:
    cfg = ModelshipModelConfig(
        name="test-model",
        model="org/test-model",
        usecase=ModelUsecase.generate,
        loader=ModelLoader.vllm,
        num_gpus=num_gpus,
        vllm_engine_kwargs=VllmEngineConfig(**(vllm_kwargs or {})),
    )
    cfg._resolved_path = resolved_path
    return cfg


def _write_model_snapshot(
    tmp_path: Path,
    *,
    config_json: dict,
    weight_bytes: int,
) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps(config_json))
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": weight_bytes}, "weight_map": {}})
    )
    return snapshot


class TestMergeWithUserOverrides:
    def test_recommendation_fills_missing(self):
        result = merge_with_user_overrides({"max_model_len": 4096}, {}, model_name="m")
        assert result == {"max_model_len": 4096}

    def test_user_value_wins(self):
        result = merge_with_user_overrides(
            {"max_model_len": 4096},
            {"max_model_len": 32000},
            model_name="m",
        )
        assert result == {"max_model_len": 32000}

    def test_disjoint_keys_merge(self):
        result = merge_with_user_overrides(
            {"max_model_len": 4096},
            {"tensor_parallel_size": 2},
            model_name="m",
        )
        assert result == {"max_model_len": 4096, "tensor_parallel_size": 2}

    def test_warning_emitted_on_divergence(self):
        with patch("modelship.preflight.base.logger") as mock_logger:
            merge_with_user_overrides(
                {"max_model_len": 4096},
                {"max_model_len": 32000},
                model_name="gemma4-coder",
            )
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "gemma4-coder" in call_args.args
        assert "max_model_len" in call_args.args

    def test_matching_value_no_warning(self):
        with patch("modelship.preflight.base.logger") as mock_logger:
            merge_with_user_overrides(
                {"max_model_len": 4096},
                {"max_model_len": 4096},
                model_name="m",
            )
        mock_logger.warning.assert_not_called()


class TestRunPreflightDispatch:
    def test_returns_empty_for_unregistered_loader(self):
        cfg = _make_config()
        cfg.loader = ModelLoader.transformers
        result = run_preflight(cfg, HardwareProfile())
        assert result == {}

    def test_swallows_estimator_exceptions(self):
        cfg = _make_config()
        with patch.object(VllmPreflight, "recommend", side_effect=RuntimeError("boom")):
            result = run_preflight(cfg, HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")]))
        assert result == {}


class TestVllmPreflight:
    def test_no_gpus_returns_empty(self):
        cfg = _make_config(resolved_path="/nonexistent")
        assert VllmPreflight().recommend(cfg, HardwareProfile()) == {}

    def test_no_resolved_path_returns_empty(self):
        cfg = _make_config()
        hw = HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")])
        assert VllmPreflight().recommend(cfg, hw) == {}

    def test_missing_config_json_returns_empty(self, tmp_path):
        cfg = _make_config(resolved_path=str(tmp_path))
        hw = HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")])
        assert VllmPreflight().recommend(cfg, hw) == {}

    def test_constrained_budget_recommends_lower_max_model_len(self, tmp_path):
        # Mimic the gemma4-coder failure: 31B model + small KV budget.
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 48,
                "num_attention_heads": 32,
                "num_key_value_heads": 16,
                "hidden_size": 5120,
                "head_dim": 160,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 32768,
            },
            weight_bytes=19 * 1024**3,
        )
        cfg = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"tensor_parallel_size": 2, "gpu_memory_utilization": 0.9},
        )
        # Two small GPUs (typical of the failure scenario)
        hw = HardwareProfile(gpus=[GPUInfo(0, 16 * 1024**3, "test"), GPUInfo(1, 16 * 1024**3, "test")])
        rec = VllmPreflight().recommend(cfg, hw)
        assert "max_model_len" in rec
        assert rec["max_model_len"] < 32768
        assert rec["max_model_len"] % 16 == 0

    def test_roomy_budget_caps_at_max_position_embeddings(self, tmp_path):
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 8192,
            },
            weight_bytes=(15 * 1024**3),
        )
        cfg = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9},
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "test")])
        rec = VllmPreflight().recommend(cfg, hw)
        # Roomy GPU: should cap at max_position_embeddings (8192), not over-suggest.
        assert rec["max_model_len"] == 8192

    def test_missing_geometry_returns_empty(self, tmp_path):
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={"torch_dtype": "bfloat16"},
            weight_bytes=1024,
        )
        cfg = _make_config(resolved_path=str(snapshot))
        hw = HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "test")])
        assert VllmPreflight().recommend(cfg, hw) == {}

    def test_budget_below_zero_returns_empty(self, tmp_path):
        # Tiny GPU, huge model: budget goes negative.
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 80,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "hidden_size": 8192,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 32768,
            },
            weight_bytes=(140 * 1024**3),
        )
        cfg = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9},
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")])
        assert VllmPreflight().recommend(cfg, hw) == {}

    def test_fractional_num_gpus_sizes_max_model_len_to_share(self, tmp_path):
        # The studio failure: a dense 7B at num_gpus=0.5. Normalization sets
        # gpu_memory_utilization=0.5, and the lighter dense overhead keeps the KV
        # budget positive, so preflight sizes max_model_len DOWN to fit instead of
        # bailing (which left the model at its 32768 default → vLLM KV OOM).
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "hidden_size": 3584,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 32768,
            },
            weight_bytes=5 * 1024**3,
        )
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0.5)
        assert cfg.vllm_engine_kwargs.gpu_memory_utilization == 0.5  # normalization fed the share through
        hw = HardwareProfile(gpus=[GPUInfo(0, 16 * 1024**3, "test")])
        rec = VllmPreflight().recommend(cfg, hw)
        assert "max_model_len" in rec
        assert 0 < rec["max_model_len"] < 32768
        assert rec["max_model_len"] % 16 == 0

    def test_fractional_share_recommends_less_than_whole_gpu(self, tmp_path):
        # Same model, same card: a 0.5 share must yield a smaller max_model_len
        # than the whole GPU — proving the effective util drives the estimate.
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "hidden_size": 3584,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 131072,
            },
            weight_bytes=5 * 1024**3,
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 16 * 1024**3, "test")])
        shared = VllmPreflight().recommend(_make_config(resolved_path=str(snapshot), num_gpus=0.5), hw)
        whole = VllmPreflight().recommend(_make_config(resolved_path=str(snapshot), num_gpus=1), hw)
        assert shared["max_model_len"] < whole["max_model_len"]

    def test_is_moe_detection_and_non_dict_safety(self):
        assert _is_moe({"num_experts": 8}, {}) is True
        assert _is_moe({}, {"n_routed_experts": 64}) is True
        assert _is_moe({"num_hidden_layers": 28}, {"num_hidden_layers": 28}) is False
        # malformed config.json: a sub-config that isn't a dict must not raise
        assert _is_moe(None, {}) is False  # type: ignore[arg-type]
        assert _is_moe("not-a-dict", None) is False  # type: ignore[arg-type]
        assert _is_moe({"num_experts": "bogus"}, {}) is False  # non-int value ignored

    def test_moe_pays_heavier_overhead_than_dense(self, tmp_path):
        # An otherwise-identical MoE model carries the heavier fixed overhead, so
        # it gets a smaller KV budget → smaller max_model_len than the dense one.
        base = {
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "hidden_size": 3584,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 131072,
        }
        (tmp_path / "dense").mkdir()
        (tmp_path / "moe").mkdir()
        dense = _write_model_snapshot(tmp_path / "dense", config_json=base, weight_bytes=5 * 1024**3)
        moe = _write_model_snapshot(tmp_path / "moe", config_json={**base, "num_experts": 8}, weight_bytes=5 * 1024**3)
        hw = HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")])
        rec_dense = VllmPreflight().recommend(_make_config(resolved_path=str(dense), num_gpus=0.5), hw)
        rec_moe = VllmPreflight().recommend(_make_config(resolved_path=str(moe), num_gpus=0.5), hw)
        assert rec_moe["max_model_len"] < rec_dense["max_model_len"]

    def test_fp8_kv_halves_per_token_bytes(self, tmp_path):
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 1_000_000,
            },
            weight_bytes=15 * 1024**3,
        )
        cfg_fp16 = _make_config(resolved_path=str(snapshot), vllm_kwargs={"gpu_memory_utilization": 0.9})
        cfg_fp8 = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"gpu_memory_utilization": 0.9, "kv_cache_dtype": "fp8_e4m3"},
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")])
        rec_fp16 = VllmPreflight().recommend(cfg_fp16, hw)
        rec_fp8 = VllmPreflight().recommend(cfg_fp8, hw)
        # fp8 stores KV in half the bytes, so the suggested context roughly doubles.
        assert rec_fp8["max_model_len"] >= rec_fp16["max_model_len"]


class TestMultimodal:
    @pytest.mark.parametrize(
        "model_cfg,expected",
        [
            ({"vision_config": {"image_size": 224}}, True),
            ({"audio_config": {}}, True),
            ({"architectures": ["LlavaForConditionalGeneration"]}, True),
            ({"architectures": ["Qwen2VLForConditionalGeneration"]}, True),
            ({"architectures": ["LlamaForCausalLM"]}, False),
            ({}, False),
        ],
    )
    def test_multimodal_detection(self, model_cfg, expected):
        from modelship.preflight.vllm import _is_multimodal

        assert _is_multimodal(model_cfg) == expected

    def test_mm_tokens_per_item_estimate(self):
        from modelship.preflight.vllm import _estimate_mm_tokens_per_item

        # 224 / 14 = 16 → 16² = 256 patches per image
        assert _estimate_mm_tokens_per_item({"vision_config": {"image_size": 224, "patch_size": 14}}) == 256
        # Missing geometry → None
        assert _estimate_mm_tokens_per_item({"vision_config": {}}) is None
        assert _estimate_mm_tokens_per_item({}) is None

    def test_multimodal_recommends_max_num_batched_tokens(self, tmp_path):
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 8192,
                "architectures": ["LlavaForConditionalGeneration"],
                "vision_config": {"image_size": 336, "patch_size": 14},
            },
            weight_bytes=15 * 1024**3,
        )
        cfg = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9},
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "test")])
        rec = VllmPreflight().recommend(cfg, hw)
        assert "max_num_batched_tokens" in rec
        # 336/14 = 24 → 24² = 576 patches → 2x headroom = 1152, capped at the 8192 floor.
        # MNBT must match the value the cudagraph budget was sized against; vLLM's
        # chunked prefill handles prompts longer than MNBT, so it stays at the floor
        # rather than scaling up to max_model_len.
        assert rec["max_num_batched_tokens"] == 8192

    def test_nested_text_config_is_unwrapped(self, tmp_path):
        # The point of this test is that we can READ geometry from a nested
        # `text_config` (Gemma 3/4, LLaVA, Qwen2-VL, etc.) — sized with roomy
        # GPUs so the budget actually produces a recommendation rather than
        # returning `{}` for hardware reasons.
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "architectures": ["Gemma3ForConditionalGeneration"],
                "torch_dtype": "bfloat16",
                "text_config": {
                    "num_hidden_layers": 48,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 16,
                    "hidden_size": 5120,
                    "head_dim": 160,
                    "max_position_embeddings": 32768,
                },
                "vision_config": {"image_size": 896, "patch_size": 14},
            },
            weight_bytes=19 * 1024**3,
        )
        cfg = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"tensor_parallel_size": 2, "gpu_memory_utilization": 0.9},
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test"), GPUInfo(1, 24 * 1024**3, "test")])
        rec = VllmPreflight().recommend(cfg, hw)
        # Should produce a real recommendation now, not bail with `{}`.
        assert "max_model_len" in rec
        assert rec["max_model_len"] > 0
        # Also recognised as multimodal → max_num_batched_tokens is set.
        assert "max_num_batched_tokens" in rec

    @pytest.mark.parametrize(
        "nesting_key",
        ["text_config", "language_config", "llm_config", "language_model_config"],
    )
    def test_resolve_text_config_handles_known_nestings(self, nesting_key):
        from modelship.preflight.vllm import _resolve_text_config

        nested = {"num_hidden_layers": 32, "hidden_size": 4096}
        resolved = _resolve_text_config({nesting_key: nested, "architectures": ["X"]})
        assert resolved is nested

    def test_resolve_text_config_passes_through_top_level(self):
        from modelship.preflight.vllm import _resolve_text_config

        top = {"num_hidden_layers": 32, "hidden_size": 4096}
        assert _resolve_text_config(top) is top

    def test_text_only_no_max_num_batched_tokens(self, tmp_path):
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 8192,
                "architectures": ["LlamaForCausalLM"],
            },
            weight_bytes=15 * 1024**3,
        )
        cfg = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9},
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "test")])
        rec = VllmPreflight().recommend(cfg, hw)
        assert "max_num_batched_tokens" not in rec


@pytest.mark.parametrize(
    "tp_size,num_kv_heads,expect_kv_shrinkage",
    [
        (1, 8, False),
        (2, 8, True),
        (4, 8, True),
        (3, 8, False),  # GQA edge case: 8 not divisible by 3, KV replicated
    ],
)
def test_kv_shrinks_per_gpu_only_when_tp_divides_heads(tp_size, num_kv_heads, expect_kv_shrinkage):
    """Unit-level: per-GPU KV bytes shrink by tp_size only when num_kv_heads is divisible."""
    from modelship.preflight.vllm import _divide_kv_by_tp

    kv_full = 100_000
    result = _divide_kv_by_tp(kv_full, {"num_key_value_heads": num_kv_heads}, tp_size)
    if expect_kv_shrinkage:
        assert result == kv_full / tp_size
    else:
        assert result == kv_full


class TestCudagraphEstimation:
    """Unit-level tests for `_estimate_cudagraph_bytes_per_gpu`."""

    def test_matches_formula_on_gemma4_measurement(self):
        # Anchor against the Gemma-4 31B AWQ run: vLLM measured 2.23 GiB CUDA
        # graph memory; the formula predicts 2.46 GiB (within ~10%, slight
        # over-estimate is the safe direction).
        from modelship.preflight.vllm import _estimate_cudagraph_bytes_per_gpu

        text_cfg = {"hidden_size": 5376, "num_hidden_layers": 60, "torch_dtype": "bfloat16"}
        cfg = _make_config()
        estimate = _estimate_cudagraph_bytes_per_gpu(text_cfg, text_cfg, cfg, 8192, 2, 1)
        # Within 10% of vLLM's measured 2.23 GiB.
        measured = 2.23 * 1024**3
        assert 0.9 * measured <= estimate <= 1.2 * measured

    def test_zero_when_enforce_eager(self):
        from modelship.preflight.vllm import _estimate_cudagraph_bytes_per_gpu

        text_cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "torch_dtype": "bfloat16"}
        cfg = _make_config(vllm_kwargs={"enforce_eager": True})
        assert _estimate_cudagraph_bytes_per_gpu(text_cfg, text_cfg, cfg, 8192, 1, 1) == 0

    def test_nonzero_when_enforce_eager_unset(self):
        # None and False both mean "CUDA graphs enabled"; estimator should run.
        from modelship.preflight.vllm import _estimate_cudagraph_bytes_per_gpu

        text_cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "torch_dtype": "bfloat16"}
        for eager in (None, False):
            cfg = _make_config(vllm_kwargs={} if eager is None else {"enforce_eager": eager})
            assert _estimate_cudagraph_bytes_per_gpu(text_cfg, text_cfg, cfg, 8192, 1, 1) > 0

    def test_zero_when_geometry_missing(self):
        # Without hidden_size / num_layers the formula can't run; return 0
        # rather than guessing — the caller already over-subtracts other
        # overheads, and a 0 here is the right "unknown" signal.
        from modelship.preflight.vllm import _estimate_cudagraph_bytes_per_gpu

        cfg = _make_config()
        assert _estimate_cudagraph_bytes_per_gpu({}, {}, cfg, 8192, 1, 1) == 0

    def test_scales_linearly_with_max_num_batched_tokens(self):
        from modelship.preflight.vllm import _estimate_cudagraph_bytes_per_gpu

        text_cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "torch_dtype": "bfloat16"}
        cfg = _make_config()
        a = _estimate_cudagraph_bytes_per_gpu(text_cfg, text_cfg, cfg, 2048, 1, 1)
        b = _estimate_cudagraph_bytes_per_gpu(text_cfg, text_cfg, cfg, 8192, 1, 1)
        assert b == 4 * a

    def test_divides_by_tp_size(self):
        from modelship.preflight.vllm import _estimate_cudagraph_bytes_per_gpu

        text_cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "torch_dtype": "bfloat16"}
        cfg = _make_config()
        single = _estimate_cudagraph_bytes_per_gpu(text_cfg, text_cfg, cfg, 8192, 1, 1)
        tp2 = _estimate_cudagraph_bytes_per_gpu(text_cfg, text_cfg, cfg, 8192, 2, 1)
        assert tp2 == single // 2

    def test_divides_by_pp_size(self):
        from modelship.preflight.vllm import _estimate_cudagraph_bytes_per_gpu

        text_cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "torch_dtype": "bfloat16"}
        cfg = _make_config()
        single = _estimate_cudagraph_bytes_per_gpu(text_cfg, text_cfg, cfg, 8192, 1, 1)
        pp2 = _estimate_cudagraph_bytes_per_gpu(text_cfg, text_cfg, cfg, 8192, 1, 2)
        assert pp2 == single // 2

    def test_enforce_eager_widens_max_model_len_recommendation(self, tmp_path):
        """End-to-end: turning on enforce_eager should give a larger budget
        (no CUDA-graph overhead subtracted) and therefore a higher max_model_len."""
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 1_000_000,
            },
            weight_bytes=15 * 1024**3,
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")])
        cfg_graphs = _make_config(resolved_path=str(snapshot), vllm_kwargs={"gpu_memory_utilization": 0.9})
        cfg_eager = _make_config(
            resolved_path=str(snapshot),
            vllm_kwargs={"gpu_memory_utilization": 0.9, "enforce_eager": True},
        )
        rec_graphs = VllmPreflight().recommend(cfg_graphs, hw)
        rec_eager = VllmPreflight().recommend(cfg_eager, hw)
        assert rec_eager["max_model_len"] > rec_graphs["max_model_len"]
