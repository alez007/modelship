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
from modelship.preflight import HardwareProfile
from modelship.preflight.llama_cpp import (
    LlamaCppPreflight,
    _ggml_type_bytes,
    _GGUFMeta,
    _read_int,
    _read_string,
    _resolve_kv_dtype_bytes,
)


def _make_config(
    *,
    resolved_path: str | None = None,
    llama_cpp_kwargs: dict | None = None,
    num_gpus: float = 0,
) -> ModelshipModelConfig:
    cfg = ModelshipModelConfig(
        name="test-model",
        model="org/test-model",
        usecase=ModelUsecase.generate,
        loader=ModelLoader.llama_cpp,
        llama_cpp_config=LlamaCppConfig(**(llama_cpp_kwargs or {})),
        num_gpus=num_gpus,
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

    def test_gpu_offload_skips_ram_sizing(self, tmp_path):
        # num_gpus > 0 means weights live in VRAM; the RAM-based sizer must not
        # run at all, regardless of GGUF metadata or RAM available.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)), num_gpus=1)
        hw = HardwareProfile(ram_bytes=64 * 1024**3)
        assert LlamaCppPreflight().recommend(cfg, hw) == {}


class TestLlamaCppPreflightRecommends:
    def test_roomy_budget_caps_at_context_length(self, tmp_path):
        # 1 GiB weights, 64 GiB RAM → plenty of headroom; cap at model's max.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        hw = HardwareProfile(ram_bytes=64 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=1 * 1024**3),
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
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=big_meta),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.75 * 1024**3)),
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
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=no_ctx_meta),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=1 * 1024**3),
        ):
            rec = LlamaCppPreflight().recommend(cfg, hw)

        assert rec == {"n_ctx": 32768}

    def test_sizes_against_available_ram_not_total(self, tmp_path):
        # Same total RAM, different *free* RAM → the box with more free RAM gets a
        # larger n_ctx. This is the multi-model fix: the generate model sizes against
        # what's left after the satellites, not the box's total.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        big_meta = _GGUFMeta(
            block_count=_LLAMA_META.block_count,
            head_count_kv=_LLAMA_META.head_count_kv,
            head_dim=_LLAMA_META.head_dim,
            context_length=131072,  # giant so the budget, not the cap, decides
        )
        tight = HardwareProfile(ram_bytes=64 * 1024**3, available_ram_bytes=6 * 1024**3)
        roomy = HardwareProfile(ram_bytes=64 * 1024**3, available_ram_bytes=24 * 1024**3)
        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=big_meta),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=int(1.5 * 1024**3)),
        ):
            rec_tight = LlamaCppPreflight().recommend(cfg, tight)
            rec_roomy = LlamaCppPreflight().recommend(cfg, roomy)
        assert rec_roomy["n_ctx"] > rec_tight["n_ctx"]

    def test_zero_available_falls_back_to_total(self, tmp_path):
        # available_ram_bytes == 0 (probe read nothing) → size against total, matching
        # the pre-change behaviour (no regression for single-model deploys).
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        only_total = HardwareProfile(ram_bytes=64 * 1024**3, available_ram_bytes=0)
        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=1 * 1024**3),
        ):
            rec = LlamaCppPreflight().recommend(cfg, only_total)
        assert rec == {"n_ctx": 8192}  # caps at the model's context_length, as with total

    def test_oversubscribed_budget_returns_empty(self, tmp_path):
        # Weights >> available RAM → no budget left; skip recommendation
        # rather than ship something the user can't actually run.
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        hw = HardwareProfile(ram_bytes=8 * 1024**3)

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=32 * 1024**3),
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


class _FakeField:
    """Imitates gguf.gguf_reader.ReaderField just enough for _read_field_value."""

    def __init__(self, value: object) -> None:
        self._value = value

    def contents(self) -> object:
        return self._value


class _FakeReader:
    def __init__(self, fields: dict[str, object]) -> None:
        self._fields = {k: _FakeField(v) for k, v in fields.items()}

    def get_field(self, key: str) -> _FakeField | None:
        return self._fields.get(key)


class TestFieldExtraction:
    """gguf hands back several shapes for `ReaderField.contents()`; these
    tests pin the behavior so future gguf bumps don't silently regress."""

    def test_python_int_scalar(self):
        reader = _FakeReader({"llama.block_count": 32})
        assert _read_int(reader, "llama.block_count") == 32

    def test_numpy_scalar(self):
        import numpy as np

        reader = _FakeReader({"llama.block_count": np.uint32(32)})
        assert _read_int(reader, "llama.block_count") == 32

    def test_numpy_array_single_element(self):
        import numpy as np

        # Some gguf versions return numpy arrays even for scalar metadata.
        reader = _FakeReader({"llama.block_count": np.array([32], dtype=np.uint32)})
        assert _read_int(reader, "llama.block_count") == 32

    def test_python_list(self):
        reader = _FakeReader({"llama.block_count": [32]})
        assert _read_int(reader, "llama.block_count") == 32

    def test_string_as_bytes_array(self):
        import numpy as np

        reader = _FakeReader({"general.architecture": np.array([b"llama"], dtype=object)})
        assert _read_string(reader, "general.architecture") == "llama"

    def test_missing_key_returns_none(self):
        reader = _FakeReader({})
        assert _read_int(reader, "llama.block_count") is None
        assert _read_string(reader, "general.architecture") is None


class TestRegistration:
    def test_run_preflight_dispatches_to_llama_cpp(self, tmp_path):
        cfg = _make_config(resolved_path=str(_write_dummy_gguf(tmp_path)))
        hw = HardwareProfile(ram_bytes=64 * 1024**3)

        from modelship.preflight import run_preflight

        with (
            patch("modelship.preflight.llama_cpp._read_gguf_metadata", return_value=_LLAMA_META),
            patch("modelship.preflight.llama_cpp._weight_bytes", return_value=1 * 1024**3),
        ):
            rec = run_preflight(cfg, hw)
        assert rec == {"n_ctx": 8192}
