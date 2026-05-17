import asyncio
import hashlib
from enum import StrEnum
from typing import Any, Literal

import ray
from fastapi import Request
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from starlette.datastructures import Headers, State

from modelship.logging import get_logger

_logger = get_logger("config")

# Length (hex chars) of the per-deployment fingerprint suffix. 10 hex chars =
# 40 bits, collision-resistant for the realistic universe of model configs.
FINGERPRINT_LEN = 10

# Fields excluded from the fingerprint hash. `name` is the deployment prefix,
# not part of the fingerprint payload. `num_replicas` is excluded so scaling
# replicas in/out doesn't force a full deployment replacement — Ray Serve
# updates replica count in place when serve.run() is re-bound with the same
# app name.
_FINGERPRINT_EXCLUDED_FIELDS = {"name", "num_replicas"}

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
    transformers = "transformers"
    diffusers = "diffusers"
    llama_cpp = "llama_cpp"
    custom = "custom"


class VllmEngineConfig(BaseModel):
    model: str = ""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: int | None = None
    dtype: str = "auto"
    tokenizer: str | None = None
    trust_remote_code: bool = False
    gpu_memory_utilization: float = 0.9  # overridden by num_gpus when num_gpus < 1
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


class TransformersConfig(BaseModel):
    device: str = "cpu"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    pipeline_kwargs: dict[str, Any] = Field(default_factory=dict)
    tool_call_parser: str | None = None
    # Explicit opt-out from auto-detected tool calling. None -> auto-detect; False -> disabled
    # even if the model's chat template advertises tools; True is a no-op (auto runs anyway).
    tool_calls_enabled: bool | None = None


class DiffusersConfig(BaseModel):
    torch_dtype: str = "float16"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5


class LlamaCppConfig(BaseModel):
    n_gpu_layers: int = -1
    n_ctx: int = 2048
    n_batch: int = 512
    chat_format: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    tool_calls_enabled: bool | None = None


class ModelshipModelConfig(BaseModel):
    name: str
    model: str | None = None
    usecase: ModelUsecase
    loader: ModelLoader
    plugin: str | None = None  # only meaningful for loader='custom'
    num_gpus: float = 0
    num_cpus: float = 0.1
    num_replicas: int = 1
    vllm_engine_kwargs: VllmEngineConfig = Field(default_factory=VllmEngineConfig)
    transformers_config: TransformersConfig | None = None
    diffusers_config: DiffusersConfig | None = None
    llama_cpp_config: LlamaCppConfig | None = None
    plugin_config: dict[str, Any] | None = None  # plugin devs parse this themselves

    _resolved_path: str | None = PrivateAttr(default=None)
    _resolved_tool_call_parser: str | None = PrivateAttr(default=None)
    _resolved_reasoning_parser: str | None = PrivateAttr(default=None)
    _resolved_chat_template: str | None = PrivateAttr(default=None)
    # Pinned at startup from the resolved tool-call parser's
    # ``markers_are_specials`` flag. Loaders that detokenize raw model
    # output (transformers' ``TextIteratorStreamer``) consult this to
    # decide whether to flip ``skip_special_tokens=False`` — required for
    # parsers like Mistral whose ``[TOOL_CALLS]`` marker is registered as a
    # special token in the tokenizer and would otherwise be stripped before
    # the parser sees it. ``None`` means the loader should keep its own
    # default.
    _resolved_skip_special_tokens: bool | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def check_custom_requires_plugin(self):
        if self.loader == ModelLoader.custom and self.plugin is None:
            raise ValueError("loader='custom' requires plugin to be set")
        if self.loader != ModelLoader.custom and not self.model:
            raise ValueError(f"`model:` is required for loader={self.loader!r}")
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
        """
        ng = self.num_gpus
        if ng <= 0 or self.loader != ModelLoader.vllm:
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

    def fingerprint(self) -> str:
        """Stable hash of the config fields that determine actor placement and
        runtime behavior. Used as the deployment-name suffix so reconcile can
        detect drift via a pure name comparison against `serve.status()`."""
        payload = self.model_dump_json(exclude=_FINGERPRINT_EXCLUDED_FIELDS)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:FINGERPRINT_LEN]

    def deployment_name(self) -> str:
        return f"{self.name}-{self.fingerprint()}"


class ModelshipConfig(BaseModel):
    models: list[ModelshipModelConfig]

    @model_validator(mode="after")
    def check_unique_deployment_names(self):
        seen: dict[str, int] = {}
        for cfg in self.models:
            key = cfg.deployment_name()
            seen[key] = seen.get(key, 0) + 1
        dupes = [name for name, count in seen.items() if count > 1]
        if dupes:
            raise ValueError(
                f"Duplicate model entries (same name + identical fingerprint): {dupes}. "
                f"Each model name must be unique; for multiple identical replicas use num_replicas."
            )
        return self


@ray.remote(num_cpus=0)
class DisconnectEvent:
    """Ray actor that holds a disconnect flag — shareable across process boundaries."""

    def __init__(self):
        self._set = False

    def set(self):
        self._set = True

    def is_set(self) -> bool:
        return self._set


class RequestWatcher:
    """Watches a FastAPI Request for client disconnect and signals via a Ray actor event."""

    def __init__(self, raw_request: Request, model: str = "", endpoint: str = ""):
        self._request = raw_request
        self._event = DisconnectEvent.remote()
        self._model = model
        self._endpoint = endpoint
        self._task = asyncio.create_task(self._watch())

    async def _watch(self):
        from modelship.metrics import CLIENT_DISCONNECTS_TOTAL

        while True:
            if await self._request.is_disconnected():
                CLIENT_DISCONNECTS_TOTAL.inc(tags={"model": self._model, "endpoint": self._endpoint})
                await self._event.set.remote()  # type: ignore[attr-defined]
                break
            await asyncio.sleep(0.1)

    def stop(self):
        """Cancel the watch task and kill the Ray actor. Call when the request is fully handled."""
        self._task.cancel()
        ray.kill(self._event)

    @property
    def event(self):
        return self._event


class RawRequestProxy:
    """
    Stands in for a FastAPI Request inside model deployment actors.

    The real FastAPI Request cannot cross Ray process boundaries — it holds a live
    TCP socket and ASGI callables that are not serializable. Instead, the gateway
    extracts the serializable parts (headers as a plain dict, disconnect signal via
    DisconnectEvent Ray actor) and passes those to the model deployment. RawRequestProxy
    reconstructs them into the interface that vllm expects from a raw_request:

      - raw_request.headers.get(...)     → Starlette Headers built from the dict
      - await raw_request.is_disconnected() → polls the DisconnectEvent Ray actor

    Any additional attributes vllm reads from raw_request in future should be added here.
    """

    def __init__(self, event, headers: dict, request_id: str | None = None):
        self._event = event
        self.headers = Headers(headers=headers)
        self.state = State()  # vllm writes per-request state here; initialized empty, lives in the actor process
        self.request_id = request_id

    async def is_disconnected(self) -> bool:
        return await self._event.is_set.remote()
