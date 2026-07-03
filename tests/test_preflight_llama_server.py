"""Tests for the LlamaServerPreflight estimator (wraps LlamaCppPreflight)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from modelship.infer.infer_config import (
    LlamaServerConfig,
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
)
from modelship.preflight import HardwareProfile
from modelship.preflight.llama_cpp import LlamaServerPreflight, _GGUFMeta

_LLAMA_META = _GGUFMeta(block_count=32, head_count_kv=8, head_dim=128, context_length=131072)


def _make_config(
    *,
    resolved_path: str | None = None,
    llama_server_kwargs: dict | None = None,
    num_gpus: float = 0,
) -> ModelshipModelConfig:
    cfg = ModelshipModelConfig(
        name="test-model",
        model="org/test-model",
        usecase=ModelUsecase.generate,
        loader=ModelLoader.llama_server,
        llama_server_config=LlamaServerConfig(**(llama_server_kwargs or {})),
        num_gpus=num_gpus,
    )
    cfg._resolved_path = resolved_path
    return cfg


def _write_dummy_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"\0" * 1024)
    return path


class TestLlamaServerPreflight:
    def test_single_slot_matches_llama_cpp_math(self, tmp_path):
        # parallel=1 (default): no division, identical to LlamaCppPreflight.
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

    def test_gpu_offload_skips_ram_sizing(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(ram_bytes=64 * 1024**3)
        assert LlamaServerPreflight().recommend(cfg, hw) == {}

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
