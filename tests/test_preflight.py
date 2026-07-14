"""Tests for the preflight estimator framework and VllmPreflight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest

from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
    VllmEngineConfig,
    default_gpu_memory_utilization,
)
from modelship.preflight import (
    GPUInfo,
    HardwareProfile,
    merge_with_user_overrides,
    run_preflight,
)
from modelship.preflight.vllm import VllmPreflight


def _make_config(
    *,
    resolved_path: str | None = None,
    vllm_kwargs: dict | None = None,
    num_gpus: float = 1,
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
        cfg.loader = ModelLoader.diffusers
        result = run_preflight(cfg, HardwareProfile())
        assert result == {}

    def test_swallows_estimator_exceptions(self):
        cfg = _make_config()
        with patch.object(VllmPreflight, "recommend", side_effect=RuntimeError("boom")):
            result = run_preflight(cfg, HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")]))
        assert result == {}

    def test_disabled_via_env_returns_empty_even_with_recommendation(self, monkeypatch):
        monkeypatch.setenv("MSHIP_PREFLIGHT", "false")
        cfg = _make_config()
        with patch.object(VllmPreflight, "recommend", return_value={"max_model_len": 4096}):
            result = run_preflight(cfg, HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")]))
        assert result == {}

    def test_enabled_by_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("MSHIP_PREFLIGHT", raising=False)
        cfg = _make_config()
        with patch.object(VllmPreflight, "recommend", return_value={"max_model_len": 4096}):
            result = run_preflight(cfg, HardwareProfile(gpus=[GPUInfo(0, 24 * 1024**3, "test")]))
        assert result == {"max_model_len": 4096}


class TestEstimateWeightFootprint:
    def test_no_real_files_trusts_index_total(self, tmp_path):
        # _write_model_snapshot's fixtures (used throughout this file) only ever
        # write the index JSON, never real .safetensors files — this must keep
        # returning the declared total_size unchanged.
        from modelship.preflight.vllm import _estimate_weight_footprint

        snapshot = _write_model_snapshot(tmp_path, config_json={}, weight_bytes=5 * 1024**3)
        assert _estimate_weight_footprint(str(snapshot)) == 5 * 1024**3

    def test_unindexed_safetensors_file_is_not_dropped(self, tmp_path):
        # A vision tower/projector shipped as a separate safetensors file that
        # the index doesn't reference (common for VLM checkpoints) must still
        # be counted, not silently missed because the index total looked complete.
        # The indexed shard's real on-disk size matches what the index declares
        # for it; only the unindexed file is "extra" — so if the fix is working,
        # the directory sum (indexed + unindexed) exceeds the index's total and
        # wins the max().
        from modelship.preflight.vllm import _estimate_weight_footprint

        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        indexed_shard_bytes = 8 * 1024
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": indexed_shard_bytes}, "weight_map": {}})
        )
        (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"\0" * indexed_shard_bytes)
        unindexed_bytes = 2 * 1024
        (snapshot / "vision_tower.safetensors").write_bytes(b"\0" * unindexed_bytes)
        assert _estimate_weight_footprint(str(snapshot)) == indexed_shard_bytes + unindexed_bytes

    def test_path_is_a_file_not_a_directory_returns_zero(self, tmp_path):
        # os.listdir() raises NotADirectoryError (an OSError subclass, not
        # FileNotFoundError) when model_path resolves to a file — must degrade
        # gracefully like the missing-path case, not crash preflight.
        from modelship.preflight.vllm import _estimate_weight_footprint

        not_a_dir = tmp_path / "not_a_directory"
        not_a_dir.write_text("oops")
        assert _estimate_weight_footprint(str(not_a_dir)) == 0


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
            num_gpus=2,
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
        # gpu_memory_utilization=0.5, and the halved budget is still too small for
        # the full 32768 context, so preflight sizes max_model_len DOWN to fit
        # instead of bailing (which left the model at its default → vLLM KV OOM).
        snapshot = _write_model_snapshot(
            tmp_path,
            config_json={
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 8,
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


class TestVllmPreflightCpu:
    """`config.num_gpus == 0` routes to `_recommend_cpu`, sized against system
    RAM rather than VRAM. `_raw_host_ram_bytes` is patched to a fixed value so
    the hand-checked math doesn't depend on the actual test machine's RAM."""

    _SMALL_MODEL_CFG: ClassVar[dict] = {
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_size": 1024,
        "head_dim": 128,
        "torch_dtype": "float16",
        "max_position_embeddings": 2048,
    }

    def test_cpu_only_node_caps_at_mpe_and_clamps_gmu(self, tmp_path):
        # Roomy RAM: the KV budget vastly exceeds what's needed for the model's
        # own max_position_embeddings (2048), so max_model_len caps there, and
        # gpu_memory_utilization is sized to just the clamped 4-sequence KV
        # budget rather than the raw (huge) headroom.
        snapshot = _write_model_snapshot(tmp_path, config_json=self._SMALL_MODEL_CFG, weight_bytes=1 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0)
        assert cfg.vllm_engine_kwargs.gpu_memory_utilization is None
        hw = HardwareProfile(ram_bytes=256 * 1024**3, available_ram_bytes=256 * 1024**3)
        with patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=256 * 1024**3):
            rec = VllmPreflight().recommend(cfg, hw)
        assert rec["max_model_len"] == 2048
        # kv_per_token=32768B; 4 seqs * 2048 tokens = 256 MiB, / 256 GiB denom ≈ 0.001 → clamped to the 0.01 floor.
        assert rec["gpu_memory_utilization"] == 0.01

    def test_mixed_node_ignores_discoverable_gpus(self, tmp_path):
        # Same config, but the node-level pynvml view reports GPUs Ray didn't
        # actually assign to this num_gpus=0 deploy — must not affect sizing.
        snapshot = _write_model_snapshot(tmp_path, config_json=self._SMALL_MODEL_CFG, weight_bytes=1 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0)
        hw_cpu_only = HardwareProfile(ram_bytes=256 * 1024**3, available_ram_bytes=256 * 1024**3)
        hw_mixed = HardwareProfile(
            gpus=[GPUInfo(0, 80 * 1024**3, "test")], ram_bytes=256 * 1024**3, available_ram_bytes=256 * 1024**3
        )
        with patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=256 * 1024**3):
            rec_cpu_only = VllmPreflight().recommend(cfg, hw_cpu_only)
            rec_mixed = VllmPreflight().recommend(cfg, hw_mixed)
        assert rec_mixed == rec_cpu_only

    def test_unknown_context_length_falls_back_to_cap(self, tmp_path):
        cfg_json = {k: v for k, v in self._SMALL_MODEL_CFG.items() if k != "max_position_embeddings"}
        snapshot = _write_model_snapshot(tmp_path, config_json=cfg_json, weight_bytes=1 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0)
        hw = HardwareProfile(ram_bytes=1024 * 1024**3, available_ram_bytes=1024 * 1024**3)
        with patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=1024 * 1024**3):
            rec = VllmPreflight().recommend(cfg, hw)
        assert rec["max_model_len"] == 32768

    def test_weights_exceed_ram_returns_empty(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=self._SMALL_MODEL_CFG, weight_bytes=64 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0)
        hw = HardwareProfile(ram_bytes=32 * 1024**3, available_ram_bytes=32 * 1024**3)
        with patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=32 * 1024**3):
            assert VllmPreflight().recommend(cfg, hw) == {}

    def test_undiscoverable_host_ram_returns_empty(self, tmp_path):
        # _raw_host_ram_bytes reads raw psutil total independently of hw.ram_bytes
        # (needed to match vLLM's own cgroup-blind denominator) — if that probe
        # comes back 0, dividing by it in _recommend_cpu_auto_gmu must not raise.
        snapshot = _write_model_snapshot(tmp_path, config_json=self._SMALL_MODEL_CFG, weight_bytes=1 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0)
        hw = HardwareProfile(ram_bytes=256 * 1024**3, available_ram_bytes=256 * 1024**3)
        with patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=0):
            assert VllmPreflight().recommend(cfg, hw) == {}

    def test_user_pinned_gmu_sizes_max_model_len_without_recommending_gmu(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=self._SMALL_MODEL_CFG, weight_bytes=1 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0, vllm_kwargs={"gpu_memory_utilization": 0.5})
        assert cfg.vllm_engine_kwargs.gpu_memory_utilization == 0.5
        hw = HardwareProfile(ram_bytes=256 * 1024**3, available_ram_bytes=256 * 1024**3)
        with patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=256 * 1024**3):
            rec = VllmPreflight().recommend(cfg, hw)
        assert "gpu_memory_utilization" not in rec
        # Pinned at 0.5 * 256 GiB is far beyond the 2048 mpe cap.
        assert rec["max_model_len"] == 2048


class TestDefaultGpuMemoryUtilization:
    """`default_gpu_memory_utilization()` + the setdefault merge in
    vllm_infer.py replace the old auto-default/split mechanism: the config
    field itself stays None until an explicit user value, a fractional
    num_gpus derivation, or a preflight recommendation resolves it, and only
    the loader-appropriate fallback (0.9 GPU / 0.4 CPU) is applied last."""

    def test_gpu_deploy_default(self):
        cfg = _make_config(num_gpus=1)
        assert cfg.vllm_engine_kwargs.gpu_memory_utilization is None
        assert default_gpu_memory_utilization(cfg) == 0.9

    def test_cpu_deploy_default(self):
        cfg = _make_config(num_gpus=0)
        assert cfg.vllm_engine_kwargs.gpu_memory_utilization is None
        assert default_gpu_memory_utilization(cfg) == 0.4

    def test_explicit_gmu_on_cpu_deploy_is_a_user_override(self):
        cfg = _make_config(num_gpus=0, vllm_kwargs={"gpu_memory_utilization": 0.6})
        user_overrides = cfg.vllm_engine_kwargs.model_dump(exclude_unset=True)
        assert user_overrides["gpu_memory_utilization"] == 0.6

    def test_precedence_user_over_recommendation_over_default(self):
        # Mirrors the merge in vllm_infer.py: {**rec, **user_overrides}, then
        # the loader-appropriate default setdefault'd in last.
        cfg = _make_config(num_gpus=0, vllm_kwargs={"max_model_len": 4096})
        user_overrides = cfg.vllm_engine_kwargs.model_dump(exclude_unset=True)
        recommendation = {"gpu_memory_utilization": 0.2, "max_model_len": 8192}
        merged = merge_with_user_overrides(recommendation, user_overrides, model_name=cfg.name)
        merged.setdefault("gpu_memory_utilization", default_gpu_memory_utilization(cfg))
        assert merged["max_model_len"] == 4096  # user wins over recommendation
        assert merged["gpu_memory_utilization"] == 0.2  # recommendation wins over the default

    def test_default_survives_preflight_decline(self):
        cfg = _make_config(num_gpus=0)
        user_overrides = cfg.vllm_engine_kwargs.model_dump(exclude_unset=True)
        merged = merge_with_user_overrides({}, user_overrides, model_name=cfg.name)
        merged.setdefault("gpu_memory_utilization", default_gpu_memory_utilization(cfg))
        assert merged["gpu_memory_utilization"] == 0.4


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
            num_gpus=2,
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


# Hybrid config mirroring Qwen3.5-4B: 32 layers, 8 full-attention + 24 linear.
_HYBRID_CFG: dict = {
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "head_dim": 128,
    "torch_dtype": "bfloat16",
    "max_position_embeddings": 262144,
}


def _mamba_info(**overrides):
    from modelship.preflight.vllm import MambaStateInfo

    base = dict(
        per_seq_state_bytes=49 * 1024**2,
        n_state_layers=24,
        n_full_attention_layers=8,
        n_total_layers=32,
        default_max_num_seqs=128,
    )
    base.update(overrides)
    return MambaStateInfo(**base)


class TestApplyHybridFit:
    """Pure arithmetic of the device-agnostic fit ladder — no vLLM, no config
    building. `kv_pool` is the bytes available for KV cache + mamba state."""

    PER_SEQ = 50 * 1024**2  # 50 MiB per concurrent slot
    KV_PER_TOKEN = 32768  # 32 KiB/token (full-attention layers only)
    TARGET = 100_000

    def _fit(self, kv_pool, user_seqs=None):
        from modelship.preflight.vllm import _apply_hybrid_fit

        return _apply_hybrid_fit("test", kv_pool, self.PER_SEQ, self.KV_PER_TOKEN, self.TARGET, user_seqs, 128)

    def test_tight_pool_floors_seqs_and_trims_context(self):
        rec = self._fit(1 * 1024**3)  # 1 GiB: full 100k context can't fit
        assert rec["max_num_seqs"] == 8  # floor concurrency
        assert 0 < rec["max_model_len"] < self.TARGET  # context trimmed
        assert rec["max_model_len"] % 16 == 0

    def test_roomy_pool_keeps_context_and_climbs_seqs(self):
        rec = self._fit(10 * 1024**3)  # 10 GiB: full context fits with surplus
        assert rec["max_model_len"] == self.TARGET  # capability preserved
        assert rec["max_num_seqs"] == 128  # surplus spent, capped at vLLM default

    def test_user_pinned_seqs_sizes_context_and_omits_seq_recommendation(self):
        rec = self._fit(10 * 1024**3, user_seqs=64)
        assert rec["max_model_len"] == self.TARGET
        assert "max_num_seqs" not in rec  # honor the user's contract, don't recommend

    def test_pool_too_small_for_floor_state_returns_empty(self):
        # 0.3 GiB < mamba state at the floor of 8 seqs (8 * 50 MiB = 0.39 GiB).
        assert self._fit(int(0.3 * 1024**3)) == {}


class TestCorrectKvForHybrid:
    def test_scales_by_full_attention_fraction(self):
        from modelship.preflight.vllm import _correct_kv_for_hybrid

        # Only 8 of 32 layers hold a token-growing KV cache.
        assert _correct_kv_for_hybrid(32000, _mamba_info()) == 32000 * 8 / 32


class TestHybridIntegration:
    """End-to-end through recommend(), with `_resolve_mamba_state` patched to a
    synthetic MambaStateInfo so no real vLLM config gets built."""

    def test_gpu_hybrid_floors_seqs_and_trims_vs_dense_baseline(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=_HYBRID_CFG, weight_bytes=8 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), vllm_kwargs={"gpu_memory_utilization": 0.9})
        hw = HardwareProfile(gpus=[GPUInfo(0, int(15.45 * 1024**3), "test")])
        with patch("modelship.preflight.vllm._resolve_mamba_state", return_value=_mamba_info()):
            hybrid = VllmPreflight().recommend(cfg, hw)
        # Same model/GPU but treated as a plain transformer (no state term).
        with patch("modelship.preflight.vllm._resolve_mamba_state", return_value=None):
            dense = VllmPreflight().recommend(cfg, hw)
        assert hybrid["max_num_seqs"] == 8
        assert 0 < hybrid["max_model_len"] < _HYBRID_CFG["max_position_embeddings"]
        assert "max_num_seqs" not in dense  # non-hybrid never emits it

    def test_gpu_roomy_keeps_full_context_and_climbs(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=_HYBRID_CFG, weight_bytes=8 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), vllm_kwargs={"gpu_memory_utilization": 0.9})
        hw = HardwareProfile(gpus=[GPUInfo(0, 40 * 1024**3, "test")])
        with patch("modelship.preflight.vllm._resolve_mamba_state", return_value=_mamba_info()):
            rec = VllmPreflight().recommend(cfg, hw)
        assert rec["max_model_len"] == _HYBRID_CFG["max_position_embeddings"]
        assert rec["max_num_seqs"] > 8

    def test_cpu_auto_gmu_hybrid_folds_state_into_gmu(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=_HYBRID_CFG, weight_bytes=8 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0)
        hw = HardwareProfile(ram_bytes=16 * 1024**3, available_ram_bytes=16 * 1024**3)
        with (
            patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=16 * 1024**3),
            patch("modelship.preflight.vllm._resolve_mamba_state", return_value=_mamba_info()),
        ):
            rec = VllmPreflight().recommend(cfg, hw)
        assert rec["max_num_seqs"] == 8  # tight RAM → floor
        assert 0 < rec["max_model_len"] < _HYBRID_CFG["max_position_embeddings"]
        assert "gpu_memory_utilization" in rec  # auto path still sizes the fraction

    def test_cpu_pinned_gmu_hybrid_recommends_seqs_without_overriding_gmu(self, tmp_path):
        snapshot = _write_model_snapshot(tmp_path, config_json=_HYBRID_CFG, weight_bytes=8 * 1024**3)
        cfg = _make_config(resolved_path=str(snapshot), num_gpus=0, vllm_kwargs={"gpu_memory_utilization": 0.5})
        hw = HardwareProfile(ram_bytes=64 * 1024**3, available_ram_bytes=64 * 1024**3)
        with (
            patch("modelship.preflight.vllm._raw_host_ram_bytes", return_value=64 * 1024**3),
            patch("modelship.preflight.vllm._resolve_mamba_state", return_value=_mamba_info()),
        ):
            rec = VllmPreflight().recommend(cfg, hw)
        assert rec["max_model_len"] > 0
        assert "max_num_seqs" in rec  # a small pinned gmu can't be blown by default concurrency
        assert "gpu_memory_utilization" not in rec  # never override the user's pin
