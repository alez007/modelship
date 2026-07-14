from __future__ import annotations

import contextlib
import json
import logging
import math
import os
from typing import Any, NamedTuple, cast

from modelship.infer.infer_config import ModelshipModelConfig, default_gpu_memory_utilization
from modelship.logging import get_logger
from modelship.preflight.base import HardwareProfile

logger = get_logger("preflight.vllm")

# vLLM default; KV cache is allocated in pages of `block_size` tokens.
_DEFAULT_BLOCK_SIZE = 16

# Concurrency floor for hybrid/SSM models. vLLM parks a fixed recurrent-state
# buffer per concurrent sequence slot (sized by max_num_seqs, not max_model_len),
# so we start low to keep that state memory minimal and grow only from surplus.
# Matches vLLM's own hybrid conservatism.
_MIN_MAX_NUM_SEQS = 8

# Fractional overhead added on top of weight bytes. Captures the runtime
# inflation between safetensors `total_size` and the bytes a backend actually
# parks on the GPU: AWQ/Marlin transposed packs, quant scales not in the file,
# embedding tables, AOT/torch.compile artifacts. Measured at ~11-14% on AWQ
# runs; 14% leans conservative.
_OVERHEAD_WEIGHT_FRACTION = 0.14

# vLLM v1 engine's default `max_num_batched_tokens` for text-only models when
# the user doesn't set one. Used as the baseline batch size for CUDA-graph
# memory estimation in non-multimodal cases.
_DEFAULT_TEXT_BATCHED_TOKENS = 2048

# torch_dtype string -> bytes per element. Quantized weight formats keep KV
# cache in the model's *compute* dtype (typically bf16/fp16), not the storage
# dtype, so AWQ/GPTQ models are still 2 bytes per KV element.
_DTYPE_BYTES = {
    "float16": 2,
    "half": 2,
    "bfloat16": 2,
    "float32": 4,
    "float": 4,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
}

# Safe floor for `max_num_batched_tokens` on multimodal models. vLLM refuses
# to start if a single MM item (one image, one audio clip) tokenizes to more
# than this. Modern VLMs (LLaVA-NeXT, Qwen2-VL, Gemma3 multimodal) sit
# comfortably under 8192 per item; we'll widen if telemetry shows otherwise.
_MULTIMODAL_BATCHED_TOKENS_FLOOR = 8192

# CPU backend constants. vLLM's CPU worker reserves `gpu_memory_utilization *
# total_memory` (raw psutil total, cgroup-blind) for the KV cache and hard-
# raises at startup if that exceeds available memory — see `_recommend_cpu`.
_CPU_RAM_UTILIZATION = 0.8
_CPU_OVERHEAD_FIXED_BYTES = 2 * 1024**3
# Clamp the auto-picked KV budget to ~4 full-length sequences so a large RAM
# box doesn't reserve an absurd utilization fraction just because the model's
# context cap is small.
_CPU_KV_SEQUENCES = 4
# Context-length cap used when the model config doesn't declare
# max_position_embeddings.
_UNKNOWN_CONTEXT_LENGTH_CAP = 32768


class VllmPreflight:
    def recommend(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        # Branch on the reservation (the intent signal), never on hardware
        # discoverability: the pynvml node-level fallback in discover_hardware()
        # can report GPUs Ray didn't actually assign to this num_gpus=0 deploy.
        if config.num_gpus == 0:
            return self._recommend_cpu(config, hw)
        return self._recommend_gpu(config, hw)

    def _recommend_gpu(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        if not hw.gpus:
            # discover_hardware()'s pynvml fallback should have found node-level
            # GPUs even when the actor itself owns none (PG-coordinator case).
            # An empty list here means the node is genuinely GPU-less or NVML
            # discovery failed — nothing to recommend either way.
            logger.info("preflight '%s': skipping — no GPUs discoverable on this node", config.name)
            return {}

        model_path = config._resolved_path
        if not model_path:
            logger.info("preflight '%s': skipping — no resolved model path", config.name)
            return {}

        model_cfg = _load_model_config_json(model_path)
        if model_cfg is None:
            logger.info(
                "preflight '%s': skipping — config.json not found or unreadable at %s",
                config.name,
                model_path,
            )
            return {}

        # For multimodal models (Gemma 3+, LLaVA-NeXT, Qwen2-VL, etc.) the
        # text-model geometry is nested under `text_config`. Unwrap before
        # computing KV-cache size.
        text_cfg = _resolve_text_config(model_cfg)

        kv_per_token, max_position_embeddings = _kv_bytes_per_token(text_cfg, model_cfg, config)
        if kv_per_token is None:
            logger.warning(
                "preflight '%s': skipping — config.json missing KV-cache geometry "
                "(num_hidden_layers/num_key_value_heads/head_dim). Top-level keys=%s, "
                "architectures=%s",
                config.name,
                sorted(model_cfg.keys()),
                model_cfg.get("architectures"),
            )
            return {}

        tp_size = max(config.vllm_engine_kwargs.tensor_parallel_size, 1)
        pp_size = max(config.vllm_engine_kwargs.pipeline_parallel_size, 1)
        # PP shards layers across stages. KV cache is per-layer, so per-GPU KV
        # bytes shrink by 1/pp on top of any TP-driven shrinking of KV heads.
        kv_per_token_per_gpu = _divide_kv_by_tp(kv_per_token, text_cfg, tp_size) / pp_size

        # Hybrid/SSM models park a fixed recurrent-state buffer per sequence slot
        # (sized by max_num_seqs) and only their full-attention layers hold a
        # token-growing KV cache. None for ordinary transformers.
        mamba = _resolve_mamba_state(config, model_path)
        if mamba is not None:
            kv_per_token_per_gpu = _correct_kv_for_hybrid(kv_per_token_per_gpu, mamba)

        weight_bytes = _estimate_weight_footprint(model_path)
        weight_bytes_per_gpu = weight_bytes / (tp_size * pp_size) if weight_bytes else 0.0

        # Resolve multimodal status + the `max_num_batched_tokens` we expect
        # vLLM to use — both are inputs to the CUDA-graph estimate.
        is_mm = _is_multimodal(model_cfg)
        mm_tokens_per_item = _estimate_mm_tokens_per_item(model_cfg) if is_mm else None
        mm_recommended_mnbt = _recommended_mm_batched_tokens(mm_tokens_per_item) if is_mm else None
        effective_mnbt = (
            config.vllm_engine_kwargs.max_num_batched_tokens or mm_recommended_mnbt or _DEFAULT_TEXT_BATCHED_TOKENS
        )

        cudagraph_bytes_per_gpu = _estimate_cudagraph_bytes_per_gpu(
            text_cfg, model_cfg, config, effective_mnbt, tp_size, pp_size
        )

        # vLLM requires homogeneous GPUs for TP; take the smallest free-memory
        # GPU to be safe. `available_bytes` is free VRAM at preflight time
        # (not device total), so the budget reflects what vLLM will actually
        # see when it measures KV cache headroom.
        gpu_available = min(g.available_bytes for g in hw.gpus)
        # gpu_memory_utilization already reflects a fractional num_gpus: it's
        # resolved at config normalization, so we read the effective value here.
        # An unset field (whole-GPU deploy, no user override) falls back to
        # vLLM's own default.
        gpu_util = config.vllm_engine_kwargs.gpu_memory_utilization or default_gpu_memory_utilization(config)
        budget = (
            gpu_available * gpu_util
            - weight_bytes_per_gpu
            - _OVERHEAD_WEIGHT_FRACTION * weight_bytes_per_gpu
            - cudagraph_bytes_per_gpu
        )
        if budget <= 0:
            logger.warning(
                "preflight: '%s' has no KV-cache budget on the assigned GPU "
                "(free=%.2f GiB, util=%.2f, est. weights/GPU=%.2f GiB, "
                "cudagraph/GPU=%.2f GiB). Model likely won't fit; deploy will be attempted anyway.",
                config.name,
                gpu_available / 1024**3,
                gpu_util,
                weight_bytes_per_gpu / 1024**3,
                cudagraph_bytes_per_gpu / 1024**3,
            )
            return {}

        rec: dict[str, Any]
        if mamba is not None:
            # `budget` is the KV+state pool; the shared ladder splits it between
            # the mamba state (max_num_seqs) and attention KV (max_model_len).
            target_len = config.vllm_engine_kwargs.max_model_len or max_position_embeddings
            rec = _apply_hybrid_fit(
                config.name,
                budget,
                mamba.per_seq_state_bytes,
                kv_per_token_per_gpu,
                target_len,
                config.vllm_engine_kwargs.max_num_seqs,
                mamba.default_max_num_seqs,
            )
            if not rec:
                return {}
        else:
            max_tokens = int(budget // kv_per_token_per_gpu)
            suggested = (max_tokens // _DEFAULT_BLOCK_SIZE) * _DEFAULT_BLOCK_SIZE
            if max_position_embeddings:
                suggested = min(suggested, max_position_embeddings)
            if suggested < _DEFAULT_BLOCK_SIZE:
                logger.warning(
                    "preflight: '%s' budget yields max_model_len=%d (< block_size); skipping recommendation",
                    config.name,
                    suggested,
                )
                return {}
            rec = {"max_model_len": suggested}

        logger.info(
            "preflight vllm '%s': gpu_free=%.2f GiB util=%.2f tp=%d pp=%d "
            "weights/GPU≈%.2f GiB cudagraph/GPU≈%.2f GiB kv/token=%d B%s → %s",
            config.name,
            gpu_available / 1024**3,
            gpu_util,
            tp_size,
            pp_size,
            weight_bytes_per_gpu / 1024**3,
            cudagraph_bytes_per_gpu / 1024**3,
            int(kv_per_token_per_gpu),
            f" hybrid(state {mamba.per_seq_state_bytes / 1024**2:.1f} MiB/seq)" if mamba else "",
            rec,
        )

        # Multimodal models: bump `max_num_batched_tokens` so vLLM can fit a
        # single image/audio item in one batch. The exact per-item token
        # count is computed inside vLLM's vision tower (architecture-
        # specific) — we pick a conservative floor that covers common VLMs.
        # Must equal `effective_mnbt` so the cudagraph estimate above stays
        # accurate: vLLM's CUDA-graph capture scales linearly with MNBT, and
        # any larger value here would invalidate the KV-cache budget. vLLM's
        # chunked prefill handles prompts longer than MNBT.
        if is_mm:
            rec["max_num_batched_tokens"] = effective_mnbt
            logger.info(
                "preflight vllm '%s': multimodal detected → suggested max_num_batched_tokens=%d "
                "(mm_tokens_per_item≈%s)",
                config.name,
                effective_mnbt,
                mm_tokens_per_item if mm_tokens_per_item is not None else "unknown",
            )

        return rec

    def _recommend_cpu(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        model_path = config._resolved_path
        if not model_path:
            logger.info("preflight '%s': skipping — no resolved model path", config.name)
            return {}

        model_cfg = _load_model_config_json(model_path)
        if model_cfg is None:
            logger.info(
                "preflight '%s': skipping — config.json not found or unreadable at %s",
                config.name,
                model_path,
            )
            return {}

        text_cfg = _resolve_text_config(model_cfg)
        kv_per_token, max_position_embeddings = _kv_bytes_per_token(text_cfg, model_cfg, config)
        if kv_per_token is None:
            logger.warning(
                "preflight '%s': skipping — config.json missing KV-cache geometry "
                "(num_hidden_layers/num_key_value_heads/head_dim). Top-level keys=%s, "
                "architectures=%s",
                config.name,
                sorted(model_cfg.keys()),
                model_cfg.get("architectures"),
            )
            return {}

        weight_bytes = _estimate_weight_footprint(model_path)
        weight_overhead = _OVERHEAD_WEIGHT_FRACTION * weight_bytes
        ctx_cap = max_position_embeddings or _UNKNOWN_CONTEXT_LENGTH_CAP
        # vLLM's CPU worker multiplies gpu_memory_utilization by the raw,
        # cgroup-blind host total — matching that denominator here keeps our
        # recommended fraction faithful to what vLLM will actually reserve.
        denom_ram = _raw_host_ram_bytes(hw)
        if denom_ram <= 0:
            logger.info("preflight '%s': skipping — system RAM not discoverable", config.name)
            return {}

        # Hybrid/SSM state accounting is device-agnostic (see _resolve_mamba_state).
        # The CPU worker draws the mamba state out of the same gmu*RAM KV pool,
        # so the fit ladder handles it identically to GPU — only the pool source
        # differs. Correct kv/token to full-attention layers only.
        mamba = _resolve_mamba_state(config, model_path)
        if mamba is not None:
            kv_per_token = _correct_kv_for_hybrid(kv_per_token, mamba)

        gmu = config.vllm_engine_kwargs.gpu_memory_utilization
        if gmu is not None:
            return self._recommend_cpu_pinned_gmu(
                config, hw, kv_per_token, weight_bytes, weight_overhead, ctx_cap, denom_ram, gmu, mamba
            )
        return self._recommend_cpu_auto_gmu(
            config, hw, kv_per_token, weight_bytes, weight_overhead, ctx_cap, denom_ram, mamba
        )

    def _recommend_cpu_pinned_gmu(
        self,
        config: ModelshipModelConfig,
        hw: HardwareProfile,
        kv_per_token: float,
        weight_bytes: int,
        weight_overhead: float,
        ctx_cap: int,
        denom_ram: int,
        gmu: float,
        mamba: MambaStateInfo | None,
    ) -> dict[str, Any]:
        """The user explicitly set gpu_memory_utilization: vLLM's CPU worker
        reserves exactly `gmu * denom_ram` for the KV cache regardless of what
        we'd otherwise pick, so size max_model_len against that instead of our
        own utilization target. We can't change gmu here, only warn if the
        combined footprint won't fit."""
        kv_budget = gmu * denom_ram
        # Mamba state is allocated *within* the gmu*RAM pool, so it's already
        # covered by kv_budget here — don't add it again.
        total_footprint = kv_budget + weight_bytes + weight_overhead + _CPU_OVERHEAD_FIXED_BYTES
        if total_footprint > hw.sizing_ram_bytes:
            logger.warning(
                "preflight '%s': user-pinned gpu_memory_utilization=%.3f reserves %.2f GiB for "
                "the KV cache; combined with an estimated %.2f GiB of weights this exceeds the "
                "%.2f GiB of RAM available — vLLM's CPU worker will likely hard-raise at startup. "
                "Lower gpu_memory_utilization or free up RAM.",
                config.name,
                gmu,
                kv_budget / 1024**3,
                (weight_bytes + weight_overhead) / 1024**3,
                hw.sizing_ram_bytes / 1024**3,
            )

        if mamba is not None:
            target_len = config.vllm_engine_kwargs.max_model_len or ctx_cap
            return _apply_hybrid_fit(
                config.name,
                kv_budget,
                mamba.per_seq_state_bytes,
                kv_per_token,
                target_len,
                config.vllm_engine_kwargs.max_num_seqs,
                mamba.default_max_num_seqs,
            )

        max_tokens = int(kv_budget // kv_per_token)
        suggested = min((max_tokens // _DEFAULT_BLOCK_SIZE) * _DEFAULT_BLOCK_SIZE, ctx_cap)
        if suggested < _DEFAULT_BLOCK_SIZE:
            logger.warning(
                "preflight '%s': user-pinned gpu_memory_utilization=%.3f yields max_model_len=%d "
                "(< block_size); skipping recommendation",
                config.name,
                gmu,
                suggested,
            )
            return {}

        logger.info(
            "preflight vllm cpu '%s': user-pinned util=%.3f denom_ram=%.2f GiB kv_budget=%.2f GiB "
            "kv/token=%d B → suggested max_model_len=%d",
            config.name,
            gmu,
            denom_ram / 1024**3,
            kv_budget / 1024**3,
            int(kv_per_token),
            suggested,
        )
        return {"max_model_len": suggested}

    def _recommend_cpu_auto_gmu(
        self,
        config: ModelshipModelConfig,
        hw: HardwareProfile,
        kv_per_token: float,
        weight_bytes: int,
        weight_overhead: float,
        ctx_cap: int,
        denom_ram: int,
        mamba: MambaStateInfo | None,
    ) -> dict[str, Any]:
        """gpu_memory_utilization is still at its CPU-deploy auto default: we're
        free to size both max_model_len and the utilization fraction. Target
        using up to `_CPU_RAM_UTILIZATION` of the RAM actually free right now,
        setting weight bytes and a fixed overhead aside first."""
        kv_budget = (
            hw.sizing_ram_bytes * _CPU_RAM_UTILIZATION - weight_bytes - weight_overhead - _CPU_OVERHEAD_FIXED_BYTES
        )
        if kv_budget <= 0:
            logger.warning(
                "preflight '%s': no KV-cache budget on CPU (available=%.2f GiB, est. weights=%.2f GiB); "
                "model likely won't fit; deploy will be attempted anyway.",
                config.name,
                hw.sizing_ram_bytes / 1024**3,
                (weight_bytes + weight_overhead) / 1024**3,
            )
            return {}

        if mamba is not None:
            return self._recommend_cpu_auto_gmu_hybrid(
                config, kv_budget, kv_per_token, ctx_cap, denom_ram, weight_bytes, weight_overhead, mamba
            )

        max_tokens = int(kv_budget // kv_per_token)
        suggested = min((max_tokens // _DEFAULT_BLOCK_SIZE) * _DEFAULT_BLOCK_SIZE, ctx_cap)
        if suggested < _DEFAULT_BLOCK_SIZE:
            logger.warning(
                "preflight '%s': CPU budget yields max_model_len=%d (< block_size); skipping recommendation",
                config.name,
                suggested,
            )
            return {}

        clamped_kv_bytes = min(kv_budget, _CPU_KV_SEQUENCES * kv_per_token * suggested)
        recommended_gmu = round(clamped_kv_bytes / denom_ram, 3)
        recommended_gmu = min(max(recommended_gmu, 0.01), 0.9)

        logger.info(
            "preflight vllm cpu '%s': sizing_ram=%.2f GiB weights≈%.2f GiB kv/token=%d B "
            "→ suggested max_model_len=%d gpu_memory_utilization=%.3f",
            config.name,
            hw.sizing_ram_bytes / 1024**3,
            (weight_bytes + weight_overhead) / 1024**3,
            int(kv_per_token),
            suggested,
            recommended_gmu,
        )
        return {"max_model_len": suggested, "gpu_memory_utilization": recommended_gmu}

    def _recommend_cpu_auto_gmu_hybrid(
        self,
        config: ModelshipModelConfig,
        kv_budget: float,
        kv_per_token: float,
        ctx_cap: int,
        denom_ram: int,
        weight_bytes: int,
        weight_overhead: float,
        mamba: MambaStateInfo,
    ) -> dict[str, Any]:
        """Hybrid model on the auto-gmu path: the shared ladder splits kv_budget
        between mamba state and attention KV, then we back-compute a gmu large
        enough to hold the *actual* reservation. The mamba state is mandatory and
        fixed, and vLLM's CPU worker sizes the KV budget as `gmu*RAM - RSS`
        (RSS ≈ weights) — so the fraction must cover weights + state + a healthy
        KV budget, else the state won't fit and startup hard-raises."""
        target_len = config.vllm_engine_kwargs.max_model_len or ctx_cap
        rec = _apply_hybrid_fit(
            config.name,
            kv_budget,
            mamba.per_seq_state_bytes,
            kv_per_token,
            target_len,
            config.vllm_engine_kwargs.max_num_seqs,
            mamba.default_max_num_seqs,
        )
        if not rec:
            return {}

        chosen_len = rec["max_model_len"]
        chosen_seqs = rec.get("max_num_seqs", config.vllm_engine_kwargs.max_num_seqs or _MIN_MAX_NUM_SEQS)
        state_bytes = mamba.per_seq_state_bytes * chosen_seqs
        # Clamp attention KV to ~a few full-length sequences (kv_budget already
        # had weights set aside, so its remainder after state is the KV room).
        attn_kv = min(kv_budget - state_bytes, _CPU_KV_SEQUENCES * kv_per_token * chosen_len)
        # Add weights back: vLLM subtracts RSS from gmu*RAM, so the fraction must
        # cover them for the mandatory state to fit.
        reservation = weight_bytes + weight_overhead + _CPU_OVERHEAD_FIXED_BYTES + state_bytes + max(attn_kv, 0)
        recommended_gmu = min(max(round(reservation / denom_ram, 3), 0.01), 0.9)
        rec["gpu_memory_utilization"] = recommended_gmu

        logger.info(
            "preflight vllm cpu '%s': hybrid state %.1f MiB/seq x %d seqs = %.2f GiB + weights + attn KV -> "
            "gpu_memory_utilization=%.3f (max_model_len=%d)",
            config.name,
            mamba.per_seq_state_bytes / 1024**2,
            chosen_seqs,
            state_bytes / 1024**3,
            recommended_gmu,
            chosen_len,
        )
        return rec


def _raw_host_ram_bytes(hw: HardwareProfile) -> int:
    """vLLM's CPU worker sizes `gpu_memory_utilization` against the raw,
    cgroup-blind `psutil.virtual_memory().total` — reading the same value here
    keeps our recommended fraction faithful to what vLLM will actually reserve.
    Falls back to `hw.ram_bytes` (itself possibly cgroup-clamped) only if
    psutil is unavailable."""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        logger.debug("preflight: psutil total-RAM probe failed; using cgroup-aware fallback", exc_info=True)
        return hw.ram_bytes


def _load_model_config_json(model_path: str) -> dict | None:
    """Read the standard transformers-layout `config.json` from a model
    directory. Works for any model saved via `save_pretrained()`, regardless
    of whether it originated from HF Hub, a local fine-tune, or any other
    pipeline that follows the same on-disk layout."""
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            return json.load(f)
    except Exception:
        logger.debug("preflight: failed to parse %s", cfg_path, exc_info=True)
        return None


def _resolve_text_config(model_cfg: dict) -> dict:
    """Multimodal models nest the language-model geometry inside a sub-config.
    If we don't find `num_hidden_layers` at the top level, try the common
    nesting paths used by HF VLMs (Gemma 3+, LLaVA, Idefics, Qwen2-VL, etc.)."""
    if model_cfg.get("num_hidden_layers") or model_cfg.get("num_layers"):
        return model_cfg
    for key in ("text_config", "language_config", "llm_config", "language_model_config"):
        sub = model_cfg.get(key)
        if isinstance(sub, dict) and (sub.get("num_hidden_layers") or sub.get("num_layers")):
            return sub
    return model_cfg


def _kv_bytes_per_token(text_cfg: dict, model_cfg: dict, config: ModelshipModelConfig) -> tuple[int | None, int | None]:
    """Return (bytes-per-token-across-all-TP-ranks, max_position_embeddings).
    Reads geometry from `text_cfg`; falls back to `model_cfg` for dtype/limits
    that often stay at the top level even when geometry is nested."""
    num_layers = text_cfg.get("num_hidden_layers") or text_cfg.get("num_layers")
    num_attention_heads = text_cfg.get("num_attention_heads")
    num_kv_heads = text_cfg.get("num_key_value_heads") or num_attention_heads
    hidden_size = text_cfg.get("hidden_size")
    head_dim = text_cfg.get("head_dim")
    if head_dim is None and hidden_size and num_attention_heads:
        head_dim = hidden_size // num_attention_heads

    if not (num_layers and num_kv_heads and head_dim):
        return None, None

    kv_dtype_bytes = _resolve_kv_dtype_bytes(text_cfg, model_cfg, config)
    # Each token stores both K and V (factor of 2) for every layer.
    per_token = 2 * num_kv_heads * head_dim * kv_dtype_bytes * num_layers
    max_position_embeddings = text_cfg.get("max_position_embeddings") or model_cfg.get("max_position_embeddings")
    return int(per_token), int(max_position_embeddings) if max_position_embeddings else None


def _resolve_kv_dtype_bytes(text_cfg: dict, model_cfg: dict, config: ModelshipModelConfig) -> int:
    user_kv = (config.vllm_engine_kwargs.kv_cache_dtype or "auto").lower()
    if user_kv.startswith("fp8"):
        return 1
    return _resolve_compute_dtype_bytes(text_cfg, model_cfg)


def _resolve_compute_dtype_bytes(text_cfg: dict, model_cfg: dict) -> int:
    """The model's forward-pass dtype (bf16/fp16/fp32). Activations and CUDA-
    graph workspace use this, regardless of any kv_cache_dtype override. Some
    multimodal configs put torch_dtype only at the top level."""
    torch_dtype = (text_cfg.get("torch_dtype") or model_cfg.get("torch_dtype") or "float16").lower()
    return _DTYPE_BYTES.get(torch_dtype, 2)


def _recommended_mm_batched_tokens(mm_tokens_per_item: int | None) -> int:
    """Floor for `max_num_batched_tokens` on multimodal models — enough to fit
    one image/audio item in one batch with headroom for text tokens."""
    floor = _MULTIMODAL_BATCHED_TOKENS_FLOOR
    if mm_tokens_per_item is not None:
        floor = max(floor, mm_tokens_per_item * 2)
    return floor


def _estimate_cudagraph_bytes_per_gpu(
    text_cfg: dict,
    model_cfg: dict,
    config: ModelshipModelConfig,
    max_num_batched_tokens: int,
    tp_size: int,
    pp_size: int,
) -> int:
    """Estimate the VRAM vLLM's memory profiler reserves for CUDA graphs.

    vLLM 0.20+ profiles peak forward-pass memory by capturing a graph at the
    largest batch size, and that peak is roughly `(per-token activation) *
    max_num_batched_tokens`. Per-token activation is bounded by
    `hidden * num_layers * dtype_bytes` (each layer holds a `[tokens, hidden]`
    activation tensor). TP shards intra-layer activations, PP shards layers
    across stages, so we divide by `tp_size * pp_size`. Verified within ~10%
    against a measured Gemma-4 31B run (predicted 2.46 GiB, vLLM measured
    2.23 GiB).

    Returns 0 when `enforce_eager=True` (CUDA graphs disabled)."""
    if config.vllm_engine_kwargs.enforce_eager:
        return 0
    hidden = text_cfg.get("hidden_size")
    layers = text_cfg.get("num_hidden_layers") or text_cfg.get("num_layers")
    if not (hidden and layers):
        return 0
    dtype_bytes = _resolve_compute_dtype_bytes(text_cfg, model_cfg)
    divisor = max(tp_size, 1) * max(pp_size, 1)
    return int(hidden * layers * dtype_bytes * max_num_batched_tokens // divisor)


def _divide_kv_by_tp(kv_per_token: int, model_cfg: dict, tp_size: int) -> float:
    if tp_size <= 1:
        return float(kv_per_token)
    num_kv_heads = model_cfg.get("num_key_value_heads") or model_cfg.get("num_attention_heads") or 0
    if num_kv_heads and num_kv_heads % tp_size == 0:
        return kv_per_token / tp_size
    # GQA edge case: when num_kv_heads doesn't divide tp_size cleanly, vLLM
    # replicates KV heads across ranks, so per-GPU bytes don't shrink.
    return float(kv_per_token)


def _is_multimodal(model_cfg: dict) -> bool:
    """Heuristic: multimodal models carry a sub-config for the non-text modality
    (`vision_config`, `audio_config`) or advertise a conditional-generation
    architecture."""
    for key in ("vision_config", "audio_config", "video_config", "mm_processor_kwargs"):
        if model_cfg.get(key) is not None:
            return True
    architectures = model_cfg.get("architectures") or []
    arch_blob = " ".join(architectures).lower()
    return any(marker in arch_blob for marker in ("forconditionalgeneration", "vlm", "multimodal", "vision", "audio"))


def _estimate_mm_tokens_per_item(model_cfg: dict) -> int | None:
    """Best-effort lower-bound estimate of tokens generated per multimodal item
    (one image). Uses the vision encoder's patch grid: (image_size / patch_size)².
    Architecture-specific pooling/token-mergers (Qwen2-VL's 2x2 merger, etc.)
    are NOT accounted for — we over-estimate, which is the right direction for
    a safety floor."""
    vision = model_cfg.get("vision_config") or {}
    image_size = vision.get("image_size")
    patch_size = vision.get("patch_size")
    if not (image_size and patch_size):
        return None
    try:
        patches_per_side = int(image_size) // int(patch_size)
    except (TypeError, ValueError):
        return None
    if patches_per_side <= 0:
        return None
    return patches_per_side * patches_per_side


def _estimate_weight_footprint(model_path: str) -> int:
    """Estimate the on-disk weight footprint. Prefers safetensors, falling back
    to PyTorch `.bin`/`.pt` for models that haven't been converted (returns the
    first format found — models that ship both layouts would otherwise
    double-count).

    For safetensors, takes the max of the index's declared `total_size` and the
    summed size of every `*.safetensors` file actually present in the
    directory: some VLM checkpoints ship a vision tower/projector as a separate
    safetensors file that isn't referenced by `model.safetensors.index.json`
    (which only indexes the text-model shards), so trusting the index alone
    can silently drop real weight bytes."""
    try:
        names = os.listdir(model_path)
    except OSError:
        return 0

    safetensors_index = os.path.join(model_path, "model.safetensors.index.json")
    index_total = _read_index_total_size(safetensors_index) if os.path.isfile(safetensors_index) else 0
    directory_total = sum(os.path.getsize(os.path.join(model_path, n)) for n in names if n.endswith(".safetensors"))
    if index_total or directory_total:
        return max(index_total, directory_total)

    pytorch_index = os.path.join(model_path, "pytorch_model.bin.index.json")
    if os.path.isfile(pytorch_index):
        total = _read_index_total_size(pytorch_index)
        if total:
            return total

    return sum(os.path.getsize(os.path.join(model_path, n)) for n in names if n.endswith((".bin", ".pt")))


def _read_index_total_size(index_path: str) -> int:
    try:
        with open(index_path) as f:
            idx = json.load(f)
    except Exception:
        logger.debug("preflight: failed to read weight index %s", index_path, exc_info=True)
        return 0
    total = idx.get("metadata", {}).get("total_size")
    return int(total) if total else 0


class MambaStateInfo(NamedTuple):
    """Recurrent-state accounting for a hybrid/SSM model, from vLLM's own
    config-only APIs. `per_seq_state_bytes` is the mamba state one concurrent
    sequence slot occupies across all state layers (per worker; PP already
    folded in via the per-stage layer count)."""

    per_seq_state_bytes: int
    n_state_layers: int
    n_full_attention_layers: int
    n_total_layers: int
    default_max_num_seqs: int


@contextlib.contextmanager
def _quiet_vllm_logging():
    """vLLM's create_engine_config emits several INFO/WARNING lines about the
    throwaway config (dummy load format, enforce_eager). Silence them so
    preflight output stays clean; restore afterwards."""
    names = ("vllm", "vllm.config", "vllm.engine", "vllm.transformers_utils")
    prev = {n: logging.getLogger(n).level for n in names}
    for n in names:
        logging.getLogger(n).setLevel(logging.ERROR)
    try:
        yield
    finally:
        for n, lvl in prev.items():
            logging.getLogger(n).setLevel(lvl)


def _resolve_mamba_state(config: ModelshipModelConfig, model_path: str) -> MambaStateInfo | None:
    """Return recurrent-state accounting for hybrid/SSM models, else None.

    Builds a throwaway vLLM engine config offline (no weights, ~1s) and uses
    vLLM's authoritative primitives: `is_hybrid`/`is_attention_free` to detect
    recurrent state, the model class's `get_mamba_state_shape/dtype_from_config`
    for exact per-slot bytes, and `get_num_layers_by_block_type` for the layer
    split. Returns None for ordinary transformers and on any failure — the mamba
    term is simply skipped (graceful degrade, matching other preflight bails)."""
    tp = max(config.vllm_engine_kwargs.tensor_parallel_size, 1)
    pp = max(config.vllm_engine_kwargs.pipeline_parallel_size, 1)
    prev_offline = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    try:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.model_executor.models import ModelRegistry
        from vllm.model_executor.models.interfaces import is_attention_free, is_hybrid

        # The model is already local; force offline so config building never
        # reaches the network.
        for k in prev_offline:
            os.environ[k] = "1"

        with _quiet_vllm_logging():
            engine_args = EngineArgs(
                model=model_path,
                load_format="dummy",
                enforce_eager=True,
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                dtype=cast("Any", config.vllm_engine_kwargs.dtype or "auto"),
                trust_remote_code=config.vllm_engine_kwargs.trust_remote_code,
            )
            vllm_config = engine_args.create_engine_config()

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        archs = model_config.hf_config.architectures
        if not archs:
            return None
        cls, _arch = ModelRegistry.resolve_model_cls(archs, model_config)

        if not (is_hybrid(cls) or is_attention_free(cls)):
            return None

        # Per-slot bytes for one state layer, summed over conv + temporal caches.
        # `cls` is a resolved vLLM model class exposing these dynamic classmethods
        # (nn.Module's typing hides them from pyright).
        state_cls: Any = cls
        shapes = state_cls.get_mamba_state_shape_from_config(vllm_config)
        dtypes = state_cls.get_mamba_state_dtype_from_config(vllm_config)
        per_slot = sum(math.prod(shape) * dt.itemsize for shape, dt in zip(shapes, dtypes, strict=True))

        # Authoritative layer split (per PP stage). "Not attention" == "has
        # recurrent state"; over-counting exotic MLP-only layers errs safe.
        n_full_attention = model_config.get_num_layers_by_block_type(parallel_config, "attention")
        n_total = model_config.get_num_layers(parallel_config)
        n_state = n_total - n_full_attention
        if n_state <= 0:
            return None

        return MambaStateInfo(
            per_seq_state_bytes=int(per_slot * n_state),
            n_state_layers=n_state,
            n_full_attention_layers=n_full_attention,
            n_total_layers=n_total,
            default_max_num_seqs=int(vllm_config.scheduler_config.max_num_seqs),
        )
    except Exception:
        logger.debug("preflight '%s': mamba-state resolution failed; skipping term", config.name, exc_info=True)
        return None
    finally:
        for k, v in prev_offline.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _correct_kv_for_hybrid(kv_per_token: float, mamba: MambaStateInfo) -> float:
    """`_kv_bytes_per_token` counts all layers; on a hybrid only the
    full-attention layers hold a token-growing KV cache, so scale it down."""
    if mamba.n_total_layers <= 0:
        return kv_per_token
    return kv_per_token * mamba.n_full_attention_layers / mamba.n_total_layers


def _apply_hybrid_fit(
    config_name: str,
    kv_pool: float,
    per_seq_state: int,
    kv_per_token: float,
    target_len: int | None,
    user_max_num_seqs: int | None,
    default_max_num_seqs: int,
) -> dict[str, Any]:
    """Device-agnostic fit ladder for hybrid models. `kv_pool` is the bytes
    available for the KV cache AND mamba state together (the caller has already
    set aside weights/overhead/cudagraph and applied its utilization fraction).
    Protects max_model_len, uses max_num_seqs as the shock absorber. Returns a
    recommendation dict, or {} when even a minimal context won't fit."""

    def fit_len(budget: float) -> int:
        tokens = int(budget // kv_per_token) if kv_per_token > 0 else 0
        aligned = (tokens // _DEFAULT_BLOCK_SIZE) * _DEFAULT_BLOCK_SIZE
        if target_len:
            aligned = min(aligned, target_len)
        return aligned

    # User pinned max_num_seqs: honor it, size context around the resulting
    # mandatory state reservation.
    if user_max_num_seqs is not None:
        budget = kv_pool - per_seq_state * user_max_num_seqs
        suggested = fit_len(budget) if budget > 0 else 0
        if suggested < _DEFAULT_BLOCK_SIZE:
            logger.warning(
                "preflight '%s': hybrid state at max_num_seqs=%d (%.2f GiB) leaves no room for a "
                "minimum context in the %.2f GiB KV pool; deploy will be attempted anyway.",
                config_name,
                user_max_num_seqs,
                per_seq_state * user_max_num_seqs / 1024**3,
                kv_pool / 1024**3,
            )
            return {}
        logger.info(
            "preflight '%s': hybrid, user max_num_seqs=%d (state %.2f GiB) → max_model_len=%d",
            config_name,
            user_max_num_seqs,
            per_seq_state * user_max_num_seqs / 1024**3,
            suggested,
        )
        return {"max_model_len": suggested}

    # Auto: floor concurrency so state memory is minimal, then protect context.
    budget_at_floor = kv_pool - per_seq_state * _MIN_MAX_NUM_SEQS
    if budget_at_floor <= 0:
        logger.warning(
            "preflight '%s': hybrid state at the floor of %d seqs (%.2f GiB) exceeds the %.2f GiB "
            "KV pool; deploy will be attempted anyway.",
            config_name,
            _MIN_MAX_NUM_SEQS,
            per_seq_state * _MIN_MAX_NUM_SEQS / 1024**3,
            kv_pool / 1024**3,
        )
        return {}

    if target_len and budget_at_floor >= target_len * kv_per_token:
        chosen_len = target_len  # full capability preserved
    else:
        chosen_len = fit_len(budget_at_floor)
        if chosen_len < _DEFAULT_BLOCK_SIZE:
            logger.warning(
                "preflight '%s': hybrid budget yields max_model_len=%d (< block_size); "
                "deploy will be attempted anyway.",
                config_name,
                chosen_len,
            )
            return {}
        logger.info(
            "preflight '%s': hybrid, trimming context to max_model_len=%d to fit the %.2f GiB KV pool "
            "at floor concurrency (reduces the max-context contract).",
            config_name,
            chosen_len,
            kv_pool / 1024**3,
        )

    # Spend leftover budget on concurrency, capped at vLLM's own default.
    leftover = budget_at_floor - chosen_len * kv_per_token
    extra = int(leftover // per_seq_state) if per_seq_state > 0 else 0
    seqs = min(default_max_num_seqs, _MIN_MAX_NUM_SEQS + max(extra, 0))
    logger.info(
        "preflight '%s': hybrid → max_num_seqs=%d (floor %d + %d from %.2f GiB surplus; per-seq state %.1f MiB)",
        config_name,
        seqs,
        _MIN_MAX_NUM_SEQS,
        seqs - _MIN_MAX_NUM_SEQS,
        leftover / 1024**3,
        per_seq_state / 1024**2,
    )
    return {"max_model_len": chosen_len, "max_num_seqs": seqs}
