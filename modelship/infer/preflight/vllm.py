from __future__ import annotations

import json
import os
from typing import Any

from modelship.infer.infer_config import ModelshipModelConfig
from modelship.infer.preflight.base import HardwareProfile
from modelship.logging import get_logger

logger = get_logger("infer.preflight.vllm")

# vLLM default; KV cache is allocated in pages of `block_size` tokens.
_DEFAULT_BLOCK_SIZE = 16

# Conservative fixed overhead (NCCL buffers, Triton caches, encoder cache for
# MM models, fused_moe routing buffers for MoE, profiler scratch that's
# neither weights nor CUDA graphs). Calibrated against measured runs: ~0.9 GiB
# on dense Gemma-4 31B, ~2.3 GiB on the Gemma-4 26B-A4B MoE. Sized for the
# heavier case so MoE/MM deploys are safe.
_OVERHEAD_FIXED_BYTES = int(2.5 * 1024**3)

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


class VllmPreflight:
    def recommend(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
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
        gpu_util = config.vllm_engine_kwargs.gpu_memory_utilization
        budget = (
            gpu_available * gpu_util
            - weight_bytes_per_gpu
            - (_OVERHEAD_WEIGHT_FRACTION * weight_bytes_per_gpu + _OVERHEAD_FIXED_BYTES)
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

        logger.info(
            "preflight vllm '%s': gpu_free=%.2f GiB util=%.2f tp=%d pp=%d "
            "weights/GPU≈%.2f GiB cudagraph/GPU≈%.2f GiB kv/token=%d B "
            "→ suggested max_model_len=%d",
            config.name,
            gpu_available / 1024**3,
            gpu_util,
            tp_size,
            pp_size,
            weight_bytes_per_gpu / 1024**3,
            cudagraph_bytes_per_gpu / 1024**3,
            int(kv_per_token_per_gpu),
            suggested,
        )

        rec: dict[str, Any] = {"max_model_len": suggested}

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
    """Estimate the on-disk weight footprint. Prefers safetensors (index
    `total_size` if present, else summed file sizes) and falls back to PyTorch
    `.bin`/`.pt` for models that haven't been converted. Returns the first
    format found — models that ship both layouts would otherwise double-count."""
    safetensors_index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(safetensors_index):
        total = _read_index_total_size(safetensors_index)
        if total:
            return total

    try:
        names = os.listdir(model_path)
    except FileNotFoundError:
        return 0

    safetensors_total = sum(os.path.getsize(os.path.join(model_path, n)) for n in names if n.endswith(".safetensors"))
    if safetensors_total:
        return safetensors_total

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
