from __future__ import annotations

import os
import re
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

# GPU-offload constants (num_gpus >= 1).
_VRAM_UTILIZATION = 0.9
# CUDA context + compute buffers, per GPU.
_GPU_OVERHEAD_FIXED_BYTES = 1 * 1024**3
# GGUF loads near-verbatim (no repack like AWQ/Marlin), so the runtime
# footprint tracks on-disk size closely; still leave a small margin.
_GGUF_WEIGHT_OVERHEAD_FRACTION = 0.05
# llama.cpp counts the output layer as one extra offloadable "layer" beyond
# the transformer blocks — full offload means block_count + 1.
_NON_BLOCK_LAYER_EQUIV = 1
# Default context to size partial-offload ngl against when the user hasn't
# pinned n_ctx themselves.
_PARTIAL_OFFLOAD_NCTX_TARGET = 8192

# Sharded GGUF filenames (e.g. model-00001-of-00003.gguf). The resolver only
# keeps the first shard's path; llama.cpp auto-loads the rest at load time,
# but the weight-footprint estimate needs every shard's size summed.
_SHARD_SUFFIX_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


class LlamaServerPreflight:
    """Sizes the `llama_server` loader's launch args to the hardware an actor
    lands on. Branches on `config.num_gpus` (the reservation is the intent
    signal), never on hardware discoverability — the pynvml node-level
    fallback in `discover_hardware()` can report GPUs Ray didn't assign to a
    `num_gpus=0` deploy."""

    def recommend(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        # Thread alignment is independent of context/offload sizing — recommend
        # it even when the GGUF-based math below declines.
        threads_rec = _recommend_threads(config)

        model_path = config._resolved_path
        if not model_path or not os.path.isfile(model_path):
            logger.info("preflight '%s': skipping — resolved path is not a GGUF file: %s", config.name, model_path)
            return threads_rec

        meta = _read_gguf_metadata(model_path)
        if meta is None:
            logger.info("preflight '%s': skipping — GGUF metadata unreadable at %s", config.name, model_path)
            return threads_rec

        kv_per_token = _kv_bytes_per_token(meta)
        if kv_per_token is None:
            logger.warning(
                "preflight '%s': skipping — GGUF metadata missing KV-cache geometry "
                "(block_count/head_count_kv/head_dim)",
                config.name,
            )
            return threads_rec

        weight_bytes = _weight_bytes(model_path)

        if config.num_gpus > 0:
            rec = self._recommend_gpu(config, hw, meta, kv_per_token, weight_bytes)
        else:
            rec = self._recommend_cpu(config, hw, meta, kv_per_token, weight_bytes)

        rec = self._apply_parallel_division(config, rec)
        return {**threads_rec, **rec}

    def _recommend_cpu(
        self, config: ModelshipModelConfig, hw: HardwareProfile, meta: _GGUFMeta, kv_per_token: int, weight_bytes: int
    ) -> dict[str, Any]:
        if hw.ram_bytes <= 0:
            logger.info("preflight '%s': skipping — system RAM not discoverable", config.name)
            return {}

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
            "preflight llama_server cpu '%s': ram_avail=%.2f GiB%s util=%.2f weights=%.2f GiB kv/token=%d B "
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

    def _recommend_gpu(
        self, config: ModelshipModelConfig, hw: HardwareProfile, meta: _GGUFMeta, kv_per_token: int, weight_bytes: int
    ) -> dict[str, Any]:
        if not hw.gpus:
            logger.info(
                "preflight '%s': skipping — GPU offload requested but no GPUs discoverable on this node",
                config.name,
            )
            return {}

        num_gpus = int(config.num_gpus)
        # llama.cpp's default --split-mode layer splits proportionally to free
        # memory, so summed free VRAM across the assigned GPUs is the real
        # capacity. Take the num_gpus smallest-free GPUs from the node-level
        # view, keeping this a lower bound when that view shows GPUs Ray
        # didn't actually assign to this deploy.
        picked = sorted(hw.gpus, key=lambda g: g.available_bytes)[:num_gpus]
        if len(picked) < num_gpus:
            logger.info(
                "preflight '%s': skipping — %d GPU(s) requested but only %d discoverable",
                config.name,
                num_gpus,
                len(picked),
            )
            return {}

        total_layers = meta.block_count + _NON_BLOCK_LAYER_EQUIV
        layer_bytes = weight_bytes * (1 + _GGUF_WEIGHT_OVERHEAD_FRACTION) / total_layers
        kv_per_layer = kv_per_token / meta.block_count
        ctx_cap = meta.context_length or _UNKNOWN_CONTEXT_LENGTH_CAP
        vram_budget = (
            sum(g.available_bytes for g in picked) * _VRAM_UTILIZATION - len(picked) * _GPU_OVERHEAD_FIXED_BYTES
        )

        ctx_full = int((vram_budget - layer_bytes * total_layers) // kv_per_token)
        if ctx_full >= _MIN_NCTX:
            suggested = min(ctx_full, ctx_cap)
            suggested = (suggested // _NCTX_ALIGNMENT) * _NCTX_ALIGNMENT
            if suggested >= _MIN_NCTX:
                logger.info(
                    "preflight llama_server gpu '%s': vram_budget=%.2f GiB across %d GPU(s), full offload "
                    "→ n_ctx=%d n_gpu_layers=%d",
                    config.name,
                    vram_budget / 1024**3,
                    len(picked),
                    suggested,
                    total_layers,
                )
                return {"n_ctx": suggested, "n_gpu_layers": total_layers}

        return self._recommend_gpu_partial(
            config, hw, meta, kv_per_layer, layer_bytes, vram_budget, total_layers, ctx_cap
        )

    def _recommend_gpu_partial(
        self,
        config: ModelshipModelConfig,
        hw: HardwareProfile,
        meta: _GGUFMeta,
        kv_per_layer: float,
        layer_bytes: float,
        vram_budget: float,
        total_layers: int,
        ctx_cap: int,
    ) -> dict[str, Any]:
        server_config = config.llama_server_config
        if server_config is not None and "n_ctx" in server_config.model_fields_set:
            target_ctx = server_config.n_ctx * server_config.parallel
        else:
            target_ctx = min(ctx_cap, _PARTIAL_OFFLOAD_NCTX_TARGET)

        def fit_ngl(ctx: int) -> int:
            denom = layer_bytes + kv_per_layer * ctx
            if denom <= 0:
                return total_layers
            return max(0, min(total_layers, int(vram_budget // denom)))

        ngl = fit_ngl(target_ctx)
        cpu_layers = meta.block_count - min(ngl, meta.block_count)

        if cpu_layers > 0:
            ram_budget = hw.sizing_ram_bytes * _RAM_UTILIZATION - _OVERHEAD_FIXED_BYTES
            kv_ram_per_ctx = kv_per_layer * cpu_layers
            ctx_ram = int((ram_budget - layer_bytes * cpu_layers) // kv_ram_per_ctx) if kv_ram_per_ctx > 0 else 0
            if ctx_ram < target_ctx:
                target_ctx = ctx_ram
                if target_ctx < _MIN_NCTX:
                    logger.warning(
                        "preflight '%s': RAM budget for %d CPU-resident layers yields n_ctx=%d "
                        "(< %d); skipping recommendation",
                        config.name,
                        cpu_layers,
                        target_ctx,
                        _MIN_NCTX,
                    )
                    return {}
                # Refit once against the shrunk context — a smaller context
                # needs less VRAM per layer, so more layers may now fit.
                ngl = fit_ngl(target_ctx)

        suggested = (target_ctx // _NCTX_ALIGNMENT) * _NCTX_ALIGNMENT
        if suggested < _MIN_NCTX:
            logger.warning(
                "preflight '%s': partial-offload budget yields n_ctx=%d (< %d); skipping recommendation",
                config.name,
                suggested,
                _MIN_NCTX,
            )
            return {}

        logger.info(
            "preflight llama_server gpu '%s': vram_budget=%.2f GiB, partial offload → n_ctx=%d n_gpu_layers=%d/%d",
            config.name,
            vram_budget / 1024**3,
            suggested,
            ngl,
            total_layers,
        )
        return {"n_ctx": suggested, "n_gpu_layers": ngl}

    def _apply_parallel_division(self, config: ModelshipModelConfig, rec: dict[str, Any]) -> dict[str, Any]:
        """llama-server splits its total context (`-c`) across `parallel`
        slots, so the per-slot `n_ctx` LlamaServerConfig expects is the total
        RAM/VRAM-budgeted context divided by the slot count (the loader's
        launch command re-multiplies by `parallel` to reconstruct the
        RAM/VRAM-safe total). `n_gpu_layers` is per-process, not per-slot, so
        it survives the division untouched."""
        if "n_ctx" not in rec:
            return rec

        server_config = config.llama_server_config
        parallel = server_config.parallel if server_config else 1
        if parallel <= 1:
            return rec

        per_slot = (rec["n_ctx"] // parallel // _NCTX_ALIGNMENT) * _NCTX_ALIGNMENT
        if per_slot < _MIN_NCTX:
            logger.warning(
                "preflight '%s': RAM/VRAM budget yields n_ctx=%d across %d parallel slots (< %d per slot); "
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
        return {**rec, "n_ctx": per_slot}


def _recommend_threads(config: ModelshipModelConfig) -> dict[str, Any]:
    """Align llama-server's compute threads with the actor's Ray CPU
    reservation so a subprocess on a shared node doesn't grab every core.
    `num_cpus` defaults to 0.1 (a fractional share), so only >= 1 (necessarily
    an explicit config value) is treated as a real thread budget."""
    if config.num_cpus < 1:
        return {}
    threads = int(config.num_cpus)
    logger.info("preflight '%s': aligning llama-server threads to num_cpus=%d", config.name, threads)
    return {"threads": threads}


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
    """On-disk size of the GGUF file, summed across shards for a sharded model
    (e.g. model-00001-of-00003.gguf) — the resolver only keeps the first
    shard's path, so a naive single-file size undercounts total weight bytes.
    Wrapped as a helper so tests can mock a hypothetical weight footprint
    without writing real bytes to tmp."""
    match = _SHARD_SUFFIX_RE.search(path)
    if match is None:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    prefix = path[: match.start()]
    total_shards = match.group(2)
    total = 0
    for shard_num in range(1, int(total_shards) + 1):
        shard_path = f"{prefix}-{shard_num:0{len(total_shards)}d}-of-{total_shards}.gguf"
        try:
            total += os.path.getsize(shard_path)
        except OSError:
            logger.debug("preflight: sharded GGUF sibling missing: %s", shard_path)
    return total


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
