import asyncio
import hashlib
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Literal

import ray
from fastapi import Request
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from ray.exceptions import RayActorError
from starlette.datastructures import Headers, State

from modelship.logging import get_logger

_logger = get_logger("config")

# Length (hex chars) of the per-deployment fingerprint suffix. 10 hex chars =
# 40 bits, collision-resistant for the realistic universe of model configs.
FINGERPRINT_LEN = 10

# Fields excluded from the fingerprint hash. `name` is the deployment prefix,
# not part of the fingerprint payload. `num_replicas` and `autoscaling_config`
# are excluded so changing replica count / scaling bounds doesn't force a full
# deployment replacement — Ray Serve updates these in place when serve.run() is
# re-bound with the same app name.
_FINGERPRINT_EXCLUDED_FIELDS = {"name", "num_replicas", "autoscaling_config"}

# vLLM's CPU backend repurposes gpu_memory_utilization to mean "fraction of
# HOST RAM to reserve for the KV cache" (not VRAM) — the GPU-oriented 0.9
# default asks to reserve 90% of node RAM and reliably raises at worker init
# on a real machine. Used only for num_gpus == 0 vllm deploys (see
# normalize_num_gpus_and_tp); an explicitly set value always wins.
_VLLM_CPU_DEFAULT_GPU_MEMORY_UTILIZATION = 0.4

ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]


class ModelUsecase(StrEnum):
    generate = "generate"
    embed = "embed"
    transcription = "transcription"
    translation = "translation"
    tts = "tts"
    image = "image"


class ModelLoader(StrEnum):
    vllm = "vllm"
    diffusers = "diffusers"
    llama_server = "llama_server"
    stable_diffusion_cpp = "stable_diffusion_cpp"
    custom = "custom"


class VllmEngineConfig(BaseModel):
    model: str = ""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: int | None = None
    dtype: str = "auto"
    tokenizer: str | None = None
    trust_remote_code: bool = False
    gpu_memory_utilization: float = 0.9  # overridden by num_gpus when num_gpus < 1 (incl. 0, CPU deploys)
    task: str = "auto"
    model_impl: str | None = None
    enable_log_requests: bool | None = False
    disable_log_stats: bool | None = False
    kv_cache_dtype: str | None = None
    quantization: str | None = None
    enable_auto_tool_choice: bool | None = None
    tool_call_parser: str | None = None
    enable_reasoning: bool | None = None
    reasoning_parser: str | None = None
    chat_template_content_format: ChatTemplateContentFormatOption = "auto"
    enforce_eager: bool | None = None
    max_num_batched_tokens: int | None = None
    # Cap on multimodal items per prompt (e.g. {"image": 4}). vLLM allows a
    # richer per-modality budget shape (dict of caps) so we mirror that.
    limit_mm_per_prompt: dict[str, int | dict[str, int]] | None = None
    # Per-model multimodal processor knobs (e.g. min_pixels / max_pixels for
    # Qwen2.5-VL). Forwarded verbatim to the HF processor.
    mm_processor_kwargs: dict[str, Any] | None = None


class DiffusersConfig(BaseModel):
    torch_dtype: str = "float16"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5


class LlamaServerConfig(BaseModel):
    """Tunables for the ``llama_server`` loader, which drives a `llama-server`
    subprocess over its native OpenAI-compatible HTTP API."""

    n_ctx: int = 2048
    n_batch: int = 512
    # Layers to offload when the deployment reserves GPUs (num_gpus > 0):
    # -1 auto-fits the offload to free VRAM, <= -2 offloads all layers.
    n_gpu_layers: int = -1
    # Concurrent request slots. llama-server splits its total context (`-c`)
    # across slots, so the process is launched with `n_ctx * parallel`.
    parallel: int = Field(default=1, ge=1)
    # Built-in template name (e.g. "chatml") or a path to a Jinja file;
    # None lets llama-server use the GGUF's embedded chat template.
    chat_template: str | None = None
    # Path to the multimodal projector file (e.g. clip-model-f16.gguf)
    mmproj: str | None = None
    # Escape hatch for launch flags not otherwise surfaced, appended verbatim.
    extra_args: list[str] = Field(default_factory=list)


class StableDiffusionCppConfig(BaseModel):
    """Tunables for the CPU-only `stable_diffusion_cpp` image loader
    (stable-diffusion.cpp via stable-diffusion-cpp-python). `sample_steps` and
    `cfg_scale` are the sd.cpp analogues of DiffusersConfig's
    `num_inference_steps` / `guidance_scale`."""

    sample_steps: int = 20
    cfg_scale: float = 7.0
    # "default" lets sd.cpp pick the sampler/scheduler per architecture
    # (euler_a for SD, euler for Flux/SD3).
    sample_method: str = "default"
    scheduler: str = "default"
    # On-the-fly weight quantization type ("default" auto-detects from the file;
    # e.g. "q4_0", "q8_0", "f16" when loading an unquantized checkpoint).
    wtype: str = "default"
    # -1 => half the CPU cores (stable-diffusion.cpp default).
    n_threads: int = -1
    # Tile the VAE decode to cut peak RAM on large images / low-memory hosts.
    vae_tiling: bool = False
    # Standalone component paths for split checkpoints (Flux / SD3.5). v1 resolves
    # only single-file models; these accept pre-placed local paths for advanced use.
    diffusion_model_path: str | None = None
    clip_l_path: str | None = None
    clip_g_path: str | None = None
    t5xxl_path: str | None = None
    vae_path: str | None = None
    # Forwarded verbatim to the StableDiffusion constructor for knobs not surfaced above.
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


class AutoscalingConfig(BaseModel):
    """Per-model Ray Serve autoscaling bounds.

    Maps to the subset of Ray Serve's ``autoscaling_config`` that's useful for
    inference replicas. When set on a model, replica count is governed by load
    between ``min_replicas`` and ``max_replicas`` instead of the fixed
    ``num_replicas`` (the two are mutually exclusive). ``min_replicas: 0`` enables
    scale-to-zero — the deployment idles with no replicas and cold-starts on the
    first request.
    """

    min_replicas: int = Field(default=1, ge=0)
    max_replicas: int = Field(default=1, ge=1)
    # Seed count on first deploy before the autoscaler has load signal. None ->
    # Serve starts at min_replicas.
    initial_replicas: int | None = Field(default=None, ge=0)
    # The autoscaler's setpoint: desired in-flight requests per replica. None ->
    # Serve's own default. Lower = scales out sooner.
    target_ongoing_requests: float | None = Field(default=None, gt=0)
    # Debounce windows (seconds) before acting on a sustained over/under-load
    # signal. None -> Serve defaults.
    upscale_delay_s: float | None = Field(default=None, ge=0)
    downscale_delay_s: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def check_bounds(self):
        if self.max_replicas < self.min_replicas:
            raise ValueError(
                f"autoscaling_config: max_replicas ({self.max_replicas}) must be >= min_replicas ({self.min_replicas})."
            )
        if self.initial_replicas is not None and not (self.min_replicas <= self.initial_replicas <= self.max_replicas):
            raise ValueError(
                f"autoscaling_config: initial_replicas ({self.initial_replicas}) must be "
                f"within [min_replicas={self.min_replicas}, max_replicas={self.max_replicas}]."
            )
        return self

    def to_serve_dict(self) -> dict[str, Any]:
        """The kwargs Ray Serve's ``.options(autoscaling_config=...)`` expects.
        Unset (None) tunables are omitted so Serve applies its own defaults."""
        out: dict[str, Any] = {"min_replicas": self.min_replicas, "max_replicas": self.max_replicas}
        for key in ("initial_replicas", "target_ongoing_requests", "upscale_delay_s", "downscale_delay_s"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


class ModelshipModelConfig(BaseModel):
    name: str
    model: str | None = None
    usecase: ModelUsecase
    loader: ModelLoader
    plugin: str | None = None  # only meaningful for loader='custom'
    num_gpus: float = 0
    num_cpus: float = 0.1
    num_replicas: int = 1
    # Load-driven replica scaling; mutually exclusive with the fixed num_replicas.
    autoscaling_config: AutoscalingConfig | None = None
    # Ray Serve's per-replica concurrency cap.
    max_ongoing_requests: int | None = None
    vllm_engine_kwargs: VllmEngineConfig = Field(default_factory=VllmEngineConfig)
    diffusers_config: DiffusersConfig | None = None
    llama_server_config: LlamaServerConfig | None = None
    stable_diffusion_cpp_config: StableDiffusionCppConfig | None = None
    plugin_config: dict[str, Any] | None = None  # plugin devs parse this themselves
    # Extra variables forwarded verbatim into the chat-template Jinja render on
    # every text loader (e.g. `enable_thinking: false` for Qwen3). Only does
    # something if the model's template branches on the key.
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)

    _resolved_path: str | None = PrivateAttr(default=None)
    _resolved_tool_call_parser: str | None = PrivateAttr(default=None)
    _resolved_reasoning_parser: str | None = PrivateAttr(default=None)
    _resolved_chat_template: str | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def default_diffusers_usecase(cls, data):
        # The image-only loaders (diffusers, stable_diffusion_cpp) leave `usecase`
        # implicit — let configs omit it. (An explicit non-image usecase is still
        # rejected below.)
        image_loaders = (ModelLoader.diffusers, ModelLoader.stable_diffusion_cpp)
        if isinstance(data, dict) and data.get("loader") in image_loaders and data.get("usecase") is None:
            data = {**data, "usecase": ModelUsecase.image}
        return data

    @model_validator(mode="after")
    def check_autoscaling_excludes_num_replicas(self):
        # num_replicas (fixed count) and autoscaling_config (load-driven range) are
        # two ways to set the same thing — Ray Serve rejects both at once. Catch an
        # explicit num_replicas alongside autoscaling_config here, with a clear
        # message, rather than letting it surface deep in serve.run(). An untouched
        # default num_replicas is fine (autoscaling simply takes over).
        if self.autoscaling_config is not None and "num_replicas" in self.model_fields_set:
            raise ValueError(
                f"model '{self.name}': set either num_replicas or autoscaling_config, not both. "
                f"num_replicas pins a fixed replica count; autoscaling_config scales between "
                f"min_replicas and max_replicas on load."
            )
        return self

    @model_validator(mode="after")
    def validate_llama_server_num_gpus(self):
        # llama.cpp has no VRAM-fraction knob, so a fractional GPU share can't be
        # honored — require whole GPUs (or 0 for CPU).
        if self.loader == ModelLoader.llama_server and self.num_gpus != int(self.num_gpus):
            raise ValueError(
                f"num_gpus={self.num_gpus!r} is not allowed for the {self.loader.value} loader: "
                f"use an integer number of whole GPUs, or 0 for CPU. Fractional GPU "
                f"sharing isn't supported (llama.cpp has no GPU-memory fraction control)."
            )
        return self

    @model_validator(mode="after")
    def check_custom_requires_plugin(self):
        if self.loader == ModelLoader.custom and self.plugin is None:
            raise ValueError("loader='custom' requires plugin to be set")
        if self.loader != ModelLoader.custom and not self.model:
            raise ValueError(f"`model:` is required for loader={self.loader!r}")
        if self.loader in (ModelLoader.diffusers, ModelLoader.stable_diffusion_cpp) and (
            self.usecase is not ModelUsecase.image
        ):
            raise ValueError(f"loader={self.loader.value!r} only supports usecase='image', got {self.usecase!r}")
        return self

    @model_validator(mode="after")
    def normalize_num_gpus_and_tp(self):
        """Enforce the num_gpus / tensor_parallel semantics for vLLM.

        - num_gpus < 1 (fractional): single GPU sharing. tp=pp=1 only — Ray
          cannot guarantee distinct physical-GPU placement for fractional
          placement-group bundles, so TP across shared GPUs is rejected.
        - num_gpus >= 1: must be an integer count of whole GPUs.
        - When tp x pp > 1, the GPU count is implied by tp x pp; if the user
          also set num_gpus, log a warning and use tp x pp (each slot owns a
          whole GPU).
        - When tp = pp = 1 and num_gpus >= 2 is set, auto-derive tp = num_gpus.
        - num_gpus == 0: a CPU deploy. Lower gpu_memory_utilization's default
          (see _VLLM_CPU_DEFAULT_GPU_MEMORY_UTILIZATION) — same rationale and
          "explicit value always wins" mechanism as the fractional-GPU case.
        """
        ng = self.num_gpus
        if self.loader != ModelLoader.vllm:
            return self

        if ng == 0:
            if "gpu_memory_utilization" not in self.vllm_engine_kwargs.model_fields_set:
                self.vllm_engine_kwargs.gpu_memory_utilization = _VLLM_CPU_DEFAULT_GPU_MEMORY_UTILIZATION
                self.vllm_engine_kwargs.model_fields_set.add("gpu_memory_utilization")
            return self
        if ng < 0:
            return self

        tp = self.vllm_engine_kwargs.tensor_parallel_size
        pp = self.vllm_engine_kwargs.pipeline_parallel_size
        world_size = tp * pp

        if 0 < ng < 1:
            if world_size > 1:
                raise ValueError(
                    f"num_gpus={ng!r} (fractional) is not compatible with "
                    f"tensor_parallel_size x pipeline_parallel_size > 1 "
                    f"(got {tp} x {pp}). Ray packs fractional placement-group "
                    f"bundles onto the same physical GPU, which breaks tensor "
                    f"parallelism. Use whole GPUs for multi-slot deploys "
                    f"(e.g. num_gpus={world_size}) or drop the parallelism "
                    f"settings to share a single GPU."
                )
            # A fractional GPU share caps the engine's VRAM to this fraction. Make
            # that the single source of truth on the config so EVERY reader agrees
            # — the engine, the preflight KV-cache sizer, logs — instead of leaving
            # gpu_memory_utilization at its 0.9 default for everyone but the loader.
            # An explicitly set utilization always wins.
            if "gpu_memory_utilization" not in self.vllm_engine_kwargs.model_fields_set:
                self.vllm_engine_kwargs.gpu_memory_utilization = ng
                self.vllm_engine_kwargs.model_fields_set.add("gpu_memory_utilization")
            return self

        # ng >= 1: integer-only.
        if ng != int(ng):
            raise ValueError(
                f"num_gpus={ng!r} is not allowed: values >= 1 must be integers. "
                f"Use a fractional value < 1 to share a single GPU, or an integer "
                f"to request that many whole GPUs."
            )
        ng_int = int(ng)

        if world_size > 1:
            if ng_int != world_size:
                raise ValueError(
                    f"num_gpus={ng_int} does not match tensor_parallel_size x "
                    f"pipeline_parallel_size={tp} x {pp}={world_size}. Either drop "
                    f"num_gpus (it's derived from tp x pp) or set num_gpus={world_size}."
                )
            if "num_gpus" in self.model_fields_set:
                _logger.warning(
                    "num_gpus=%d is redundant for model '%s': it matches "
                    "tensor_parallel_size x pipeline_parallel_size=%d, which "
                    "already determines the GPU count. Safe to drop.",
                    ng_int,
                    self.name,
                    world_size,
                )
        else:
            # tp=pp=1: auto-derive tp from num_gpus.
            self.vllm_engine_kwargs.tensor_parallel_size = ng_int

        self.num_gpus = 1.0
        return self

    def fingerprint(self, gateway_name: str = "") -> str:
        """Stable hash of the config fields that drive placement/runtime, used as
        the deployment-name suffix so reconcile detects drift by name comparison.
        `gateway_name` is mixed in (when given) so identical configs on different
        gateways get distinct app names in Serve's flat global namespace."""
        payload = self.model_dump_json(exclude=_FINGERPRINT_EXCLUDED_FIELDS)
        if gateway_name:
            payload = f"{gateway_name}\x00{payload}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:FINGERPRINT_LEN]

    def deployment_name(self, gateway_name: str) -> str:
        # Gateway folded into the fingerprint, not a visible prefix; ownership is
        # tracked in the coordinator registry, not parsed out of the name.
        return f"{self.name}-{self.fingerprint(gateway_name)}"


class ModelshipConfig(BaseModel):
    models: list[ModelshipModelConfig]

    @model_validator(mode="after")
    def check_unique_deployment_names(self):
        seen: dict[str, int] = {}
        for cfg in self.models:
            key = f"{cfg.name}-{cfg.fingerprint()}"
            seen[key] = seen.get(key, 0) + 1
        dupes = [name for name, count in seen.items() if count > 1]
        if dupes:
            raise ValueError(
                f"Duplicate model entries (same name + identical fingerprint): {dupes}. "
                f"Each model name must be unique; for multiple identical replicas use num_replicas."
            )
        return self


# How long a recorded disconnect lingers before the registry evicts it. The
# gateway no longer clears entries on request teardown — that clear ran in the
# same finally the disconnect itself triggered, racing (and usually beating) the
# model deployment's cross-process is_disconnected() poll, so the signal was
# dropped before it was read. This TTL is now what bounds the set. It only needs
# to outlast the deployment's poll interval, not the whole generation: the entry
# is added at disconnect time, by which point the deployment is already polling.
_DISCONNECT_TTL_SECONDS = 300.0


class _DisconnectStore:
    """Plain (non-actor) TTL set of disconnected request ids, factored out of
    DisconnectRegistry so the eviction logic is unit-testable without a Ray
    cluster. ``now`` is injectable for deterministic tests."""

    def __init__(self, ttl_seconds: float, now: Callable[[], float] = time.monotonic):
        self._ttl = ttl_seconds
        self._now = now
        # request_id -> monotonic deadline after which the entry is evicted.
        self._deadlines: dict[str, float] = {}

    def set(self, request_id: str) -> None:
        now = self._now()
        self._evict_expired(now)
        self._deadlines[request_id] = now + self._ttl

    def is_set(self, request_id: str) -> bool:
        deadline = self._deadlines.get(request_id)
        if deadline is None:
            return False
        if deadline <= self._now():
            del self._deadlines[request_id]
            return False
        return True

    def clear(self, request_id: str) -> None:
        self._deadlines.pop(request_id, None)

    def _evict_expired(self, now: float) -> None:
        for request_id in [rid for rid, deadline in self._deadlines.items() if deadline <= now]:
            del self._deadlines[request_id]


@ray.remote(num_cpus=0)
class DisconnectRegistry:
    """One cluster-wide actor tracking client-disconnect per request id, replacing
    the previous per-request DisconnectEvent actor. Async so concurrent polls don't
    head-of-line block on the single-threaded actor.

    Entries are TTL-evicted (``_DISCONNECT_TTL_SECONDS``) rather than cleared by the
    gateway — see ``_DISCONNECT_TTL_SECONDS`` for why."""

    def __init__(self, ttl_seconds: float = _DISCONNECT_TTL_SECONDS):
        self._store = _DisconnectStore(ttl_seconds)

    async def set(self, request_id: str) -> None:
        self._store.set(request_id)

    async def is_set(self, request_id: str) -> bool:
        return self._store.is_set(request_id)

    async def clear(self, request_id: str) -> None:
        self._store.clear(request_id)


_disconnect_registry = None


def get_disconnect_registry():
    """Get-or-create the single detached, named DisconnectRegistry shared by every
    gateway replica and model deployment. Cached to keep the lookup off the hot path."""
    global _disconnect_registry
    if _disconnect_registry is None:
        _disconnect_registry = DisconnectRegistry.options(
            name="modelship_disconnect_registry",
            get_if_exists=True,
            lifetime="detached",
            namespace="modelship",
        ).remote()
    return _disconnect_registry


def reset_disconnect_registry() -> None:
    """Drop the cached handle so the next get_disconnect_registry() re-resolves the
    named actor. Called after a RayActorError: the detached actor died (node
    preemption, GCS restart) and the cached handle is now stale. get_if_exists makes
    every process that re-resolves converge on the same recreated actor."""
    global _disconnect_registry
    _disconnect_registry = None


class RequestWatcher:
    """Watches a FastAPI Request for client disconnect and records it in the shared
    DisconnectRegistry, keyed by request id."""

    def __init__(self, raw_request: Request, request_id: str, model: str = "", endpoint: str = ""):
        self._request = raw_request
        self._registry = get_disconnect_registry()
        self._request_id = request_id
        self._model = model
        self._endpoint = endpoint
        self._task = asyncio.create_task(self._watch())

    async def _watch(self):
        from modelship.metrics import CLIENT_DISCONNECTS_TOTAL

        while True:
            if await self._request.is_disconnected():
                CLIENT_DISCONNECTS_TOTAL.inc(tags={"model": self._model, "endpoint": self._endpoint})
                await self._record_disconnect()
                break
            await asyncio.sleep(0.1)

    async def _record_disconnect(self) -> None:
        """Record the disconnect in the shared registry, re-resolving the actor and
        retrying once if it has died — otherwise a registry blip silently loses the
        signal and the deployment runs to completion."""
        try:
            await self._registry.set.remote(self._request_id)  # type: ignore[attr-defined]
        except RayActorError:
            reset_disconnect_registry()
            self._registry = get_disconnect_registry()
            try:
                await self._registry.set.remote(self._request_id)  # type: ignore[attr-defined]
            except RayActorError:
                _logger.warning("Disconnect registry unavailable; lost disconnect for %s", self._request_id)

    def stop(self):
        """Cancel the watch task. The disconnect entry (if any) is deliberately
        left for the DisconnectRegistry to TTL-evict: clearing it here ran in the
        same teardown the disconnect triggered and raced the model deployment's
        is_disconnected() poll, dropping the signal before it was read."""
        self._task.cancel()

    @property
    def registry(self):
        return self._registry


class RawRequestProxy:
    """
    Stands in for a FastAPI Request inside model deployment actors.

    The real FastAPI Request cannot cross Ray process boundaries — it holds a live
    TCP socket and ASGI callables that are not serializable. Instead, the gateway
    extracts the serializable parts (headers as a plain dict, disconnect signal via
    the shared DisconnectRegistry actor) and passes those to the model deployment.
    RawRequestProxy reconstructs them into the interface that vllm expects:

      - raw_request.headers.get(...)     → Starlette Headers built from the dict
      - await raw_request.is_disconnected() → polls the DisconnectRegistry by id

    Any additional attributes vllm reads from raw_request in future should be added here.
    """

    def __init__(self, registry, headers: dict, request_id: str | None = None):
        self._registry = registry
        self.headers = Headers(headers=headers)
        self.state = State()  # vllm writes per-request state here; lives in the actor process
        self.request_id = request_id

    async def is_disconnected(self) -> bool:
        if self._registry is None:
            # No real registry (e.g. an internal warmup request) — nothing to poll.
            return False
        try:
            return await self._registry.is_set.remote(self.request_id)
        except RayActorError:
            # The shared registry actor died (node preemption, GCS restart). Disconnect
            # propagation is best-effort, so degrade to "still connected" rather than
            # failing a healthy in-flight request, and re-resolve the (recreated, via
            # get_if_exists) actor so later polls in this request reconnect.
            _logger.warning("Disconnect registry unavailable; assuming client connected")
            reset_disconnect_registry()
            self._registry = get_disconnect_registry()
            return False
