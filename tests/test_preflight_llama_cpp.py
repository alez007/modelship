"""Tests for the LlamaCppPreflight estimator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from modelship.infer.infer_config import (
    LlamaCppConfig,
    ModelLoader,
    ModelshipModelConfig,
    ModelUsecase,
)
from modelship.infer.preflight import HardwareProfile
from modelship.infer.preflight.llama_cpp import (
    LlamaCppPreflight,
    _ggml_type_bytes,
    _GGUFMeta,
    _resolve_kv_dtype_bytes,
)


def _make_config(
    *,
    resolved_path: str | None = None,
    llama_cpp_kwargs: dict | None = None,
) -> ModelshipModelConfig:
    cfg = ModelshipModelConfig(
        name="test-model",
        model="org/test-model",
        usecase=ModelUsecase.generate,
        loader=ModelLoader.llama_cpp,
        llama_cpp_config=LlamaCppConfig(**(llama_cpp_kwargs or {})),
    )
    cfg._resolved_path = resolved_path
    return cfg


def _write_dummy_gguf(tmp_path: Path) -> Path:
    """Write a tiny placeholder file with a `.gguf` suffix. Existence + path
    are all that's checked before parsing; tests mock both `_read_gguf_metadata`
    and `_weight_bytes` to inject the scenarios that matter."""
    path = tmp_path / "model.gguf"
    path.write_bytes(b"\0" * 1024)
    return path


_LLAMA_META = _GGUFMeta(block_count=32, head_count_kv=8, head_dim=128, context_length=8192)


class TestLlamaCppPreflightSkips:
    def test_no_ram_returns_empty(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        assert LlamaCppPreflight().recommend(cfg, HardwareProfile(ram_bytes=0)) == {}

    def test_no_resolved_path_returns_empty(self):
        cfg = _make_config()
        hw = HardwareProfile(ram_bytes=64 * 1024**3)
        assert LlamaCppPreflight().recommend(cfg, hw) == {}

    def test_missing_file_returns_empty(self, tmp_path):
        cfg = _make_config(resolved_path=str(tmp_path / "nope.gguf"))
        hw = HardwareProfile(ram_bytes=64 * 1024**3)
        assert LlamaCppPreflight().recommend(cfg, hw) == {}

    def test_unparseable_gguf_returns_empty(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        hw = HardwareProfile(ram_bytes=64 * 1024**3)
        # Real GGUFReader will fail on our zero-bytes file; preflight must skip.
        assert LlamaCppPreflight().recommend(cfg, hw) == {}


class TestLlamaCppPreflightRecommends:
    def test_roomy_budget_caps_at_context_length(self, tmp_path):
        # 1 GiB weights, 64 GiB RAM → plenty of headroom; cap at model's max.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        hw = HardwareProfile(ram_bytes=64 * 1024**3)

        with (
            patch("modelship.infer.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.infer.preflight.llama_cpp._weight_bytes", return_value=1 * 1024**3),
        ):
            rec = LlamaCppPreflight().recommend(cfg, hw)

        assert rec == {"n_ctx": 8192}

    def test_constrained_budget_recommends_lower_nctx(self, tmp_path):
        # Tight: 4 GiB RAM, 1.75 GiB weights → budget < model's max context.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        hw = HardwareProfile(ram_bytes=4 * 1024**3)

        # Giant-context model so the cap doesn't kick in first.
        big_meta = _GGUFMeta(
            block_count=_LLAMA_META.block_count,
            head_count_kv=_LLAMA_META.head_count_kv,
            head_dim=_LLAMA_META.head_dim,
            context_length=131072,
        )
        with (
            patch("modelship.infer.preflight.llama_cpp._read_gguf_metadata", return_value=big_meta),
            patch("modelship.infer.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
        ):
            rec = LlamaCppPreflight().recommend(cfg, hw)

        assert "n_ctx" in rec
        assert rec["n_ctx"] < 131072
        assert rec["n_ctx"] % 256 == 0

    def test_missing_context_length_applies_safety_cap(self, tmp_path):
        # GGUF lacks `{arch}.context_length`. On a huge-RAM host the math
        # would otherwise hand out a context far beyond the model's training
        # window; the cap (32768) must kick in instead.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        hw = HardwareProfile(ram_bytes=1024 * 1024**3)  # 1 TiB

        no_ctx_meta = _GGUFMeta(
            block_count=_LLAMA_META.block_count,
            head_count_kv=_LLAMA_META.head_count_kv,
            head_dim=_LLAMA_META.head_dim,
            context_length=None,
        )
        with (
            patch("modelship.infer.preflight.llama_cpp._read_gguf_metadata", return_value=no_ctx_meta),
            patch("modelship.infer.preflight.llama_cpp._weight_bytes", return_value=1 * 1024**3),
        ):
            rec = LlamaCppPreflight().recommend(cfg, hw)

        assert rec == {"n_ctx": 32768}

    def test_oversubscribed_budget_returns_empty(self, tmp_path):
        # Weights >> available RAM → no budget left; skip recommendation
        # rather than ship something the user can't actually run.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        hw = HardwareProfile(ram_bytes=8 * 1024**3)

        with (
            patch("modelship.infer.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.infer.preflight.llama_cpp._weight_bytes", return_value=32 * 1024**3),
        ):
            rec = LlamaCppPreflight().recommend(cfg, hw)

        assert rec == {}


class TestKvDtypeResolution:
    def test_default_is_fp16(self):
        cfg = _make_config()
        assert _resolve_kv_dtype_bytes(cfg) == 2

    def test_string_alias_q8_0(self):
        cfg = _make_config(llama_cpp_kwargs={"model_kwargs": {"type_k": "q8_0", "type_v": "q8_0"}})
        assert _resolve_kv_dtype_bytes(cfg) == 1

    def test_mixed_types_uses_larger(self):
        cfg = _make_config(llama_cpp_kwargs={"model_kwargs": {"type_k": "f16", "type_v": "q8_0"}})
        # Conservative: pick the larger of the two so we don't under-estimate.
        assert _resolve_kv_dtype_bytes(cfg) == 2

    def test_unknown_falls_back_to_fp16(self):
        cfg = _make_config(llama_cpp_kwargs={"model_kwargs": {"type_k": "q3_k_xxs"}})
        assert _resolve_kv_dtype_bytes(cfg) == 2

    def test_ggml_enum_int(self):
        # 0 = F32, 1 = F16 in ggml's type enum
        assert _ggml_type_bytes(0) == 4
        assert _ggml_type_bytes(1) == 2
        assert _ggml_type_bytes(999) is None


class TestRegistration:
    def test_run_preflight_dispatches_to_llama_cpp(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        hw = HardwareProfile(ram_bytes=64 * 1024**3)

        from modelship.infer.preflight import run_preflight

        with (
            patch("modelship.infer.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.infer.preflight.llama_cpp._weight_bytes", return_value=1 * 1024**3),
        ):
            rec = run_preflight(cfg, hw)
        assert rec == {"n_ctx": 8192}
