from __future__ import annotations

import os
from typing import Any

from modelship.infer.infer_config import ModelshipModelConfig
from modelship.logging import get_logger
from modelship.preflight.base import HardwareProfile

logger = get_logger("preflight.llama_cpp")

# Fraction of total system RAM the preflight will allocate. No equivalent of
# vLLM's `gpu_memory_utilization` exists for this loader; 0.8 leaves room
# for the OS, page cache, and other actors on the same node.
_RAM_UTILIZATION = 0.8

# Fixed overhead for llama.cpp runtime state (compute buffers, sampler stacks,
# tokenizer vocab). Much smaller than vLLM since there's no CUDA-graph capture,
# no NCCL, no fused kernels' workspace.
_OVERHEAD_FIXED_BYTES = 512 * 1024**2

# Round the recommended n_ctx to this alignment. llama.cpp doesn't require any
# specific alignment, but powers of 256 are the convention and keep numbers
# readable in logs.
_NCTX_ALIGNMENT = 256

# Minimum n_ctx we're willing to recommend. Below this the deployment is
# unusable for chat anyway; better to skip and let the user see the OOM than
# silently ship a 256-token context.
_MIN_NCTX = 512

# Safety cap when the GGUF doesn't declare `{arch}.context_length`. Without it,
# a high-RAM host could end up recommending an n_ctx far beyond what the model
# was actually trained for (older quant repos and custom conversions sometimes
# omit the field). 32k is the largest "native" context shipped by most
# instruct-tuned models without RoPE extension at the time of writing — picking
# higher risks gibberish and severe attention-cost blowup at inference time.
_UNKNOWN_CONTEXT_LENGTH_CAP = 32768

# Default KV-cache element size when neither `type_k` nor `type_v` is set in
# `model_kwargs`. llama.cpp defaults to fp16 (2 bytes).
_DEFAULT_KV_DTYPE_BYTES = 2


class LlamaCppPreflight:
    def recommend(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        if config.num_gpus > 0:
            logger.info(
                "preflight '%s': skipping — GPU offload requested; n_ctx left to user config",
                config.name,
            )
            return {}

        if hw.ram_bytes <= 0:
            logger.info("preflight '%s': skipping — system RAM not discoverable", config.name)
            return {}

        model_path = config._resolved_path
        if not model_path or not os.path.isfile(model_path):
            logger.info("preflight '%s': skipping — resolved path is not a GGUF file: %s", config.name, model_path)
            return {}

        meta = _read_gguf_metadata(model_path)
        if meta is None:
            logger.info("preflight '%s': skipping — GGUF metadata unreadable at %s", config.name, model_path)
            return {}

        kv_per_token = _kv_bytes_per_token(meta)
        if kv_per_token is None:
            logger.warning(
                "preflight '%s': skipping — GGUF metadata missing KV-cache geometry "
                "(block_count/head_count_kv/head_dim)",
                config.name,
            )
            return {}

        weight_bytes = _weight_bytes(model_path)

        ram_basis = hw.sizing_ram_bytes
        fallback = " [total fallback]" if not hw.available_ram_bytes else ""
        budget = ram_basis * _RAM_UTILIZATION - weight_bytes - _OVERHEAD_FIXED_BYTES
        if budget <= 0:
            logger.warning(
                "preflight '%s': no n_ctx budget (ram_avail=%.2f GiB%s, util=%.2f, weights=%.2f GiB, "
                "overhead=%.2f GiB). Model likely won't fit; deploy will be attempted anyway.",
                config.name,
                ram_basis / 1024**3,
                fallback,
                _RAM_UTILIZATION,
                weight_bytes / 1024**3,
                _OVERHEAD_FIXED_BYTES / 1024**3,
            )
            return {}

        max_tokens = int(budget // kv_per_token)
        suggested = (max_tokens // _NCTX_ALIGNMENT) * _NCTX_ALIGNMENT
        # Cap to the model's declared training context when available;
        # otherwise apply a conservative safety ceiling so a high-RAM host
        # doesn't recommend hundreds of thousands of tokens on a GGUF that
        # just happens to omit the field.
        cap = meta.context_length if meta.context_length else _UNKNOWN_CONTEXT_LENGTH_CAP
        if not meta.context_length:
            logger.info(
                "preflight '%s': GGUF metadata missing context_length; capping n_ctx at %d",
                config.name,
                _UNKNOWN_CONTEXT_LENGTH_CAP,
            )
        suggested = min(suggested, cap)
        if suggested < _MIN_NCTX:
            logger.warning(
                "preflight '%s': budget yields n_ctx=%d (< %d); skipping recommendation",
                config.name,
                suggested,
                _MIN_NCTX,
            )
            return {}

        logger.info(
            "preflight llama_cpp '%s': ram_avail=%.2f GiB%s util=%.2f weights=%.2f GiB kv/token=%d B "
            "→ suggested n_ctx=%d",
            config.name,
            ram_basis / 1024**3,
            fallback,
            _RAM_UTILIZATION,
            weight_bytes / 1024**3,
            int(kv_per_token),
            suggested,
        )

        return {"n_ctx": suggested}


class LlamaServerPreflight:
    """Reuses `LlamaCppPreflight`'s GGUF/RAM-budget math for the `llama_server`
    loader. That math sizes a single context to the RAM budget; llama-server
    instead splits its total context (`-c`) across `parallel` slots, so the
    per-slot `n_ctx` LlamaServerConfig expects is the total budget divided by
    the slot count (the loader's launch command re-multiplies by `parallel`
    to reconstruct the RAM-safe total)."""

    def recommend(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        rec = LlamaCppPreflight().recommend(config, hw)
        if "n_ctx" not in rec:
            return rec

        server_config = config.llama_server_config
        parallel = server_config.parallel if server_config else 1
        if parallel <= 1:
            return rec

        per_slot = (rec["n_ctx"] // parallel // _NCTX_ALIGNMENT) * _NCTX_ALIGNMENT
        if per_slot < _MIN_NCTX:
            logger.warning(
                "preflight '%s': RAM budget yields n_ctx=%d across %d parallel slots (< %d per slot); "
                "skipping recommendation",
                config.name,
                per_slot,
                parallel,
                _MIN_NCTX,
            )
            return {}
        logger.info(
            "preflight llama_server '%s': dividing total n_ctx budget %d across parallel=%d -> n_ctx=%d",
            config.name,
            rec["n_ctx"],
            parallel,
            per_slot,
        )
        return {"n_ctx": per_slot}


class _GGUFMeta:
    __slots__ = ("block_count", "context_length", "head_count_kv", "head_dim")

    def __init__(
        self,
        block_count: int,
        head_count_kv: int,
        head_dim: int,
        context_length: int | None,
    ) -> None:
        self.block_count = block_count
        self.head_count_kv = head_count_kv
        self.head_dim = head_dim
        self.context_length = context_length


def _weight_bytes(path: str) -> int:
    """On-disk size of the GGUF file. Wrapped as a helper so tests can mock
    a hypothetical weight footprint without writing real bytes to tmp."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _read_gguf_metadata(path: str) -> _GGUFMeta | None:
    """Read the architecture-relevant header fields from a GGUF file.

    `GGUFReader` mmaps the file and parses metadata only — tensor data is not
    loaded. Returns None on any parse failure; the caller treats that the same
    as a vLLM-side missing config.json (skip preflight, let the loader try)."""
    try:
        from gguf import GGUFReader
    except Exception:
        logger.debug("preflight: gguf package not available", exc_info=True)
        return None

    try:
        reader = GGUFReader(path)
    except Exception:
        logger.debug("preflight: GGUFReader failed to open %s", path, exc_info=True)
        return None

    arch = _read_string(reader, "general.architecture")
    if not arch:
        logger.debug("preflight: GGUF missing general.architecture at %s", path)
        return None

    block_count = _read_int(reader, f"{arch}.block_count")
    head_count = _read_int(reader, f"{arch}.attention.head_count")
    head_count_kv = _read_int(reader, f"{arch}.attention.head_count_kv") or head_count
    embedding_length = _read_int(reader, f"{arch}.embedding_length")
    key_length = _read_int(reader, f"{arch}.attention.key_length")

    # head_dim falls back to embedding_length / head_count when not stated
    # explicitly. Modern Llama/Qwen GGUFs include `key_length`; older ones rely
    # on the fallback.
    if key_length:
        head_dim = key_length
    elif embedding_length and head_count:
        head_dim = embedding_length // head_count
    else:
        head_dim = None

    context_length = _read_int(reader, f"{arch}.context_length")

    if not (block_count and head_count_kv and head_dim):
        return None

    return _GGUFMeta(
        block_count=int(block_count),
        head_count_kv=int(head_count_kv),
        head_dim=int(head_dim),
        context_length=int(context_length) if context_length else None,
    )


def _read_field_value(reader: Any, key: str) -> Any:
    field = reader.get_field(key)
    if field is None:
        return None
    # Modern gguf (>=0.10) exposes `.contents()` returning a Python primitive.
    contents = getattr(field, "contents", None)
    if callable(contents):
        try:
            return contents()
        except Exception:
            logger.debug("preflight: field.contents() raised for %s", key, exc_info=True)
    # Fallback: pull the first part out of the raw numpy data array.
    try:
        if field.data and field.parts:
            return field.parts[field.data[0]][0]
    except (IndexError, TypeError, AttributeError):
        pass
    return None


def _unwrap_scalar(val: Any) -> Any:
    """Extract a scalar from whatever shape gguf hands back.

    `ReaderField.contents()` returns numpy arrays for some field types and
    Python sequences for others; numpy `>=0`-dim scalars are also possible
    on older gguf releases. Pull out the first element when the value is
    array-like, leave true scalars alone."""
    if val is None:
        return None
    # numpy array (1-d or higher): take the first element.
    if hasattr(val, "ndim") and getattr(val, "ndim", 0) > 0:
        try:
            return val.item(0) if val.size else None
        except (AttributeError, IndexError, ValueError):
            return None
    # Python sequence.
    if isinstance(val, list | tuple):
        return val[0] if val else None
    return val


def _read_int(reader: Any, key: str) -> int | None:
    val = _unwrap_scalar(_read_field_value(reader, key))
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _read_string(reader: Any, key: str) -> str | None:
    val = _unwrap_scalar(_read_field_value(reader, key))
    if val is None:
        return None
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _kv_bytes_per_token(meta: _GGUFMeta) -> int | None:
    """Bytes of KV cache stored per token across all layers.

    `2 *` accounts for both K and V tensors. KV element size defaults to fp16
    (2 bytes) — `llama_server`'s config surface has no `type_k`/`type_v`
    override, so preflight always assumes it."""
    return 2 * meta.block_count * meta.head_count_kv * meta.head_dim * _DEFAULT_KV_DTYPE_BYTES
