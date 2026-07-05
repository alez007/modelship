"""Tests for the LlamaServerPreflight estimator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from modelship.infer.infer_config import (
    LlamaServerConfig,
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
)
from modelship.preflight import GPUInfo, HardwareProfile, merge_with_user_overrides
from modelship.preflight.llama_cpp import LlamaServerPreflight, _GGUFMeta, _weight_bytes

_LLAMA_META = _GGUFMeta(block_count=32, head_count_kv=8, head_dim=128, context_length=131072)


def _make_config(
    *,
    resolved_path: str | None = None,
    llama_server_kwargs: dict | None = None,
    num_gpus: float = 0,
    num_cpus: float = 0.1,
) -> ModelshipModelConfig:
    cfg = ModelshipModelConfig(
        name="test-model",
        model="org/test-model",
        usecase=ModelUsecase.generate,
        loader=ModelLoader.llama_server,
        llama_server_config=LlamaServerConfig(**(llama_server_kwargs or {})),
        num_gpus=num_gpus,
        num_cpus=num_cpus,
    )
    cfg._resolved_path = resolved_path
    return cfg


def _write_dummy_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"\0" * 1024)
    return path


class TestLlamaServerPreflightCpu:
    def test_single_slot_matches_ram_budget_math(self, tmp_path):
        # parallel=1 (default): no division.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        hw = HardwareProfile(ram_bytes=4 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert "n_ctx" in rec
        assert rec["n_ctx"] % 256 == 0

    def test_parallel_divides_total_budget(self, tmp_path):
        # Same hardware/model, parallel=4 must yield a per-slot n_ctx roughly
        # 1/4 of the single-slot recommendation (the launch command
        # re-multiplies by parallel to reconstruct the RAM-safe total).
        cfg_single = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), llama_server_kwargs={"parallel": 1})
        cfg_parallel = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), llama_server_kwargs={"parallel": 4})
        hw = HardwareProfile(ram_bytes=4 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec_single = LlamaServerPreflight().recommend(cfg_single, hw)
            rec_parallel = LlamaServerPreflight().recommend(cfg_parallel, hw)

        assert rec_parallel["n_ctx"] * 4 <= rec_single["n_ctx"] + 256 * 4
        assert rec_parallel["n_ctx"] < rec_single["n_ctx"]

    def test_parallel_too_high_for_budget_returns_empty(self, tmp_path):
        # A tiny budget divided across many slots drops below the minimum
        # usable n_ctx; the estimator should decline rather than recommend
        # something unusably small.
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)), llama_server_kwargs={"parallel": 64}, num_gpus=0
        )
        hw = HardwareProfile(ram_bytes=2 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.9 * 1024**3)),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec == {}

    def test_run_preflight_dispatches_to_llama_server(self, tmp_path):
        from modelship.preflight import run_preflight

        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        hw = HardwareProfile(ram_bytes=4 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec = run_preflight(cfg, hw)
        assert "n_ctx" in rec


class TestLlamaServerPreflightGpu:
    """`config.num_gpus >= 1` routes to `_recommend_gpu`. `_LLAMA_META` has
    block_count=32 (total_layers=33), head_count_kv=8, head_dim=128 → kv/token
    = 2*32*8*128*2 = 131072 B, kv/layer = 131072/32 = 4096 B."""

    def test_no_discoverable_gpus_returns_empty(self, tmp_path):
        # Ray only sets CUDA_VISIBLE_DEVICES for GPU-owning actors, and the
        # node-level pynvml view can be empty too — nothing to size against.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(ram_bytes=64 * 1024**3)
        with patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META):
            assert LlamaServerPreflight().recommend(cfg, hw) == {}

    def test_full_offload_when_vram_is_roomy(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "test")], ram_bytes=64 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        # block_count(32) + 1 non-block layer equivalent = full offload.
        assert rec["n_gpu_layers"] == 33
        # VRAM budget vastly exceeds what the model's own context_length needs,
        # so n_ctx caps at context_length (131072, already 256-aligned).
        assert rec["n_ctx"] == 131072

    def test_partial_offload_fits_fewer_layers_at_default_target(self, tmp_path):
        # VRAM too small to fit every layer's weights, but RAM is roomy enough
        # that the default 8192-token partial-offload target doesn't shrink.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(gpus=[GPUInfo(0, 2 * 1024**3, "test")], ram_bytes=64 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec
        assert 0 < rec["n_gpu_layers"] < 33
        assert rec["n_ctx"] == 8192

    def test_ram_constrained_partial_shrinks_ctx(self, tmp_path):
        # Same tight VRAM as the previous test, but RAM is now also tight —
        # the CPU-resident layers' KV cache doesn't fit at the 8192 target, so
        # the context shrinks (and ngl is refit against the smaller context).
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(gpus=[GPUInfo(0, 2 * 1024**3, "test")], ram_bytes=5 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec
        assert rec["n_ctx"] < 8192
        assert rec["n_ctx"] % 256 == 0
        assert rec["n_gpu_layers"] > 0

    def test_doesnt_fit_anywhere_returns_empty(self, tmp_path):
        # Neither VRAM nor RAM can absorb even a minimal context.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(gpus=[GPUInfo(0, 512 * 1024**2, "test")], ram_bytes=512 * 1024**2)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        assert rec == {}

    def test_multi_gpu_picks_smallest_n_as_lower_bound(self, tmp_path):
        # 4 GPUs discoverable at the node level but only 2 reserved: picking
        # the 2 smallest-free is a lower bound over any 2-subset, and should
        # match sizing against exactly those two directly.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=2)
        hw_all_four = HardwareProfile(
            gpus=[
                GPUInfo(0, 40 * 1024**3, "test"),
                GPUInfo(1, 10 * 1024**3, "test"),
                GPUInfo(2, 30 * 1024**3, "test"),
                GPUInfo(3, 20 * 1024**3, "test"),
            ],
            ram_bytes=64 * 1024**3,
        )
        hw_two_smallest = HardwareProfile(
            gpus=[GPUInfo(0, 10 * 1024**3, "test"), GPUInfo(1, 20 * 1024**3, "test")],
            ram_bytes=64 * 1024**3,
        )

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec_all_four = LlamaServerPreflight().recommend(cfg, hw_all_four)
            rec_two_smallest = LlamaServerPreflight().recommend(cfg, hw_two_smallest)

        assert rec_all_four == rec_two_smallest

    def test_parallel_division_preserves_n_gpu_layers(self, tmp_path):
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1, llama_server_kwargs={"parallel": 4}
        )
        hw = HardwareProfile(gpus=[GPUInfo(0, 80 * 1024**3, "test")], ram_bytes=64 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=4 * 1024**3),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)

        # n_gpu_layers is per-process, not per-slot — must survive the
        # parallel-division step untouched, unlike n_ctx.
        assert rec["n_gpu_layers"] == 33
        assert rec["n_ctx"] == 131072 // 4 // 256 * 256


class TestShardedGgufWeightBytes:
    def test_sums_sibling_shards(self, tmp_path):
        shard1 = tmp_path / "model-00001-of-00002.gguf"
        shard2 = tmp_path / "model-00002-of-00002.gguf"
        shard1.write_bytes(b"\0" * 1000)
        shard2.write_bytes(b"\0" * 2000)

        assert _weight_bytes(str(shard1)) == 3000

    def test_missing_sibling_shard_is_skipped(self, tmp_path):
        shard1 = tmp_path / "model-00001-of-00002.gguf"
        shard1.write_bytes(b"\0" * 1000)
        # shard 2 is missing entirely.

        assert _weight_bytes(str(shard1)) == 1000

    def test_non_sharded_file_uses_plain_size(self, tmp_path):
        path = tmp_path / "model.gguf"
        path.write_bytes(b"\0" * 4096)
        assert _weight_bytes(str(path)) == 4096


class TestLlamaServerThreadsRecommendation:
    def test_whole_num_cpus_recommends_threads(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_cpus=4)
        hw = HardwareProfile(ram_bytes=4 * 1024**3)
        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)
        assert rec["threads"] == 4

    def test_default_fractional_num_cpus_has_no_threads_recommendation(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))  # num_cpus defaults to 0.1
        hw = HardwareProfile(ram_bytes=4 * 1024**3)
        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)
        assert "threads" not in rec

    def test_threads_recommended_even_when_gguf_metadata_unreadable(self, tmp_path):
        # Thread alignment doesn't depend on being able to size n_ctx.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_cpus=8)
        hw = HardwareProfile(ram_bytes=4 * 1024**3)
        with patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=None):
            rec = LlamaServerPreflight().recommend(cfg, hw)
        assert rec == {"threads": 8}

    def test_user_set_threads_wins_at_merge_level(self):
        merged = merge_with_user_overrides({"threads": 4}, {"threads": 16}, model_name="m")
        assert merged["threads"] == 16

    def test_threads_declined_when_it_would_undercut_parallel_slots(self, tmp_path):
        # num_cpus=2 with parallel=4: capping to 2 threads would starve the 4
        # concurrent slots of compute and defeat the loader's headline
        # concurrency feature — decline and let llama-server keep all cores.
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)),
            num_cpus=2,
            llama_server_kwargs={"parallel": 4},
        )
        hw = HardwareProfile(ram_bytes=4 * 1024**3)
        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)
        assert "threads" not in rec

    def test_threads_recommended_when_it_covers_parallel_slots(self, tmp_path):
        cfg = _make_config(
            resolved_path=str(_write_dummy_gguf(tmp_path)),
            num_cpus=4,
            llama_server_kwargs={"parallel": 4},
        )
        hw = HardwareProfile(ram_bytes=4 * 1024**3)
        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec = LlamaServerPreflight().recommend(cfg, hw)
        assert rec["threads"] == 4
