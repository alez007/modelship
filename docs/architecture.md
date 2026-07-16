# Architecture

## Overview

Modelship is built on [Ray Serve](https://docs.ray.io/en/latest/serve/) for deployment orchestration and a **FastAPI gateway** that exposes an OpenAI-compatible API. Multiple inference backends are supported:

- **[vLLM](https://github.com/vllm-project/vllm)** — high-throughput inference with continuous batching and PagedAttention, on GPU or CPU
- **llama-server** — a proxied `llama-server` subprocess for quantized GGUF chat, embeddings, and vision on CPU or GPU
- **[HuggingFace Diffusers](https://github.com/huggingface/diffusers)** — image generation via `AutoPipelineForText2Image`
- **Plugin system** — custom TTS and STT backends (Kokoro ONNX, Orpheus, whisper.cpp)

## Request Lifecycle

1. Client sends a request to the FastAPI gateway (e.g. `POST /v1/chat/completions`)
2. The gateway identifies the target model from the request body
3. A `RequestWatcher` begins monitoring the client connection for disconnects
4. The request is forwarded to the model's Ray Serve deployment via a `RawRequestProxy` (serializable headers + cancellation event)
5. The model deployment runs inference (vLLM, llama-server, diffusers, or plugin)
6. Response streams back as JSON or SSE
7. If the client disconnects mid-inference, the watcher fires the cancellation event, freeing GPU resources immediately

## Model Deployments

Each model in `models.yaml` becomes an isolated Ray Serve deployment (`ModelDeployment` actor). This gives:

- **Independent lifecycle** — one model crashing doesn't affect others
- **Per-model GPU budgeting** — `num_gpus` controls VRAM allocation (e.g. 0.70 for 70%)
- **Sequential startup** — models deploy one at a time to prevent memory spikes, ordered by tensor parallelism size (TP > 1 first)
- **Additive deploys** — by default, `mship_deploy.py` adds models to a running cluster without disrupting existing deployments, enabling incremental composition from multiple config files. Use `--reconcile` to instead make the cluster match a config exactly (add/remove/replace), without ever tearing down the cluster
- **Multi-deployment routing** — the same model name can appear multiple times with different configs (e.g. GPU + CPU). The gateway round-robins requests across all deployments sharing a name. Each deployment also supports `num_replicas` for scaling identical copies via Ray Serve's built-in load balancing
- **Multi-gateway support** — multiple independent gateways can run on the same cluster via `--gateway-name`, each managing its own set of models

### Inference Loaders

Each deployment uses one of the following loaders:

| Loader | Backend | Use cases | GPU required |
|--------|---------|-----------|--------------|
| `vllm` | vLLM engine | Chat/generation, embeddings, transcription, translation | No — installs on GPU or CPU |
| `llama_server` | llama-server subprocess | Chat/generation, embeddings, vision (GGUF models) | No — runs on CPU or GPU (GGUF offload) |
| `diffusers` | HuggingFace Diffusers | Image generation (any `AutoPipelineForText2Image` model) | Yes |
| `stable_diffusion_cpp` | stable-diffusion.cpp | Image generation (GGUF models: SD1.5/SDXL/SD-Turbo, all-in-one Flux) | No — currently CPU-only |
| `custom` | Plugin system | TTS backends (Kokoro ONNX, Orpheus), STT backends (whisper.cpp) | No |

The `llama_server` loader provides high-efficiency inference for quantized GGUF models on CPU or GPU (`n_gpu_layers` offload, whole GPUs only — fractional `num_gpus` is rejected) by proxying a `llama-server` subprocess's own OpenAI-compatible API. The `vllm` loader provides higher throughput with continuous batching and PagedAttention, on GPU or CPU.

## Responses API (`/v1/responses`)

`/v1/responses` is shaped natively per loader, not via a chat-completions round trip. `BaseInfer.create_response(request, raw_request)` is a hookable method (defaulting to "not supported," like every other unimplemented capability); `VllmInfer` and `LlamaServerInfer` are the loaders that implement it, building the Responses envelope directly from their own parsed `(reasoning, content, tool_calls)` output — the same `ParsedChatOutput` seam `/v1/chat/completions` uses — rather than baking a `ChatCompletionResponse` and translating that back. `ModelDeployment.respond()` mirrors the existing `generate()` dispatch; the gateway route does a fail-fast validation pass and then calls straight through to it.

- **Non-streaming** — `utils.responses.build_responses_items_from_parsed` maps the parsed tuple into `output[]` items (`reasoning` → reasoning item, content → message item, tool calls → `function_call` items), then `protocol/responses/adapter.build_response_object` builds the envelope and remaps usage.
- **Streaming** (`stream: true`) — `ResponsesStreamTranslator` is fed loader-native typed chunks directly (vLLM's `engine_ops.stream_chat_completion`, llama-server's own delta fields) and emits the Responses event protocol (`response.created` → `output_item.added` → `output_text.delta` / `reasoning_summary_text.delta` / `function_call_arguments.delta` → `output_item.done` → `response.completed`), tracking `output_index` / `sequence_number`. Output items are opened lazily on their first delta and closed at stream end, so the translation is independent of how a model interleaves reasoning, text, and tool calls.

**Supported:** text, reasoning (as a first-class `reasoning` output item), client-driven tool calling (`function_call` / `function_call_output` round-trip), and server-side conversation state (`store` / `previous_response_id`, `GET`/`DELETE /v1/responses/{id}`, `/input_items`), streaming and non-streaming — on the `vllm` and `llama_server` loaders.

**404s on other loaders** (`diffusers`, `custom`) — there is no generic fallback; a loader must implement `create_response` itself.

**Rejected with a clear 400** (rather than silently dropped): `background` and hosted built-in tools (e.g. `web_search`). Encrypted reasoning (`reasoning.encrypted_content`) is not implemented — server-side state supersedes it as the way to carry reasoning across turns.

### Conversation state

State lives **in the gateway**, not the loaders. `GET`/`DELETE` carry no model, so they could not be routed to a deployment at all; putting the write there too keeps one seam and leaves both loaders untouched. `api.py`'s routes delegate the state plumbing to `openai/utils/responses.py`: `resolve_history` prepends `previous_response_id`'s conversation into `input` *before* the Ray hop — so a store outage is a clean 503 before any GPU work — and `persist_response` tees `respond`'s output into the store on the way back, ahead of `_handle_response` so that stays generic. The loader therefore only ever sees a flat `input`, and echoes `store` / `previous_response_id` back on the envelope without acting on them.

`modelship/openai/state/responses.py` stores one **self-contained snapshot per response id**, keyed `responses/<identity>/<response_id>` via the generic `modelship.state` store: continuing is a single read rather than a walk down a pointer chain, and each turn's fresh id makes branching fall out for free. Identity scoping means another caller builds a different key and simply misses — isolation with no comparison logic. Reading the id back off the response (rather than minting one in the gateway) keeps the loader's ownership of it intact; injecting an id ahead of generation is what Phase E (background mode) will need.

The streaming write is the one asymmetry: the terminal `response.completed` event is re-parsed out of our own SSE to recover the response object. It is safe because `streaming._sse` is that format's only writer, and persisting *before* forwarding the event is what allows a store failure to downgrade the terminal event to `response.failed` — the response is uncontinuable, so reporting success would hand back an id that 404s next turn.

Any `vllm` or `llama_server` model works — no special `models.yaml` entry is needed:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="…")

# Non-streaming
resp = client.responses.create(model="reasoning-qwen", input="Which is larger, 9.11 or 9.9?")
print(resp.output_text)

# Streaming — named events; reasoning and tool-call argument deltas stream live
with client.responses.stream(model="reasoning-qwen", input="Explain why, briefly.") as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
```

## GPU Allocation

Ray automatically schedules model deployments across available GPUs based on the `num_gpus` fraction each model requests. For example, two models each requesting `num_gpus: 0.9` will be placed on separate GPUs.

## Plugin System

Custom backends are isolated `uv` workspace packages under `plugins/`. Each plugin:

- Implements `BasePlugin` and overrides the `create_*` method(s) matching its `usecase` (e.g. `create_speech` for TTS, `create_transcription` for STT)
- Has its own dependencies, isolated from the main project
- Is automatically loaded from wheels via Ray's `runtime_env` when referenced in `models.yaml`
- Returns raw, protocol-agnostic outputs; OpenAI-shape adaptation is handled by the serving wrappers in `modelship/infer/custom/openai/`

See [Plugin Development](plugins.md) for details.

## Key Files

| File | Purpose |
|------|---------|
| `mship_deploy.py` | Entry point — initializes Ray, deploys models additively (or reconciles with `--reconcile`) |
| `modelship/openai/api.py` | FastAPI gateway with OpenAI endpoints |
| `modelship/openai/protocol/responses/` | `/v1/responses` schemas + chat adapter (`adapter.py`) and streaming translator (`streaming.py`) |
| `modelship/state/` | Generic pluggable KV store (`memory://` via a detached Ray actor, `redis://`). Domain layers live with their callers: `openai/state/responses.py`, `deploy/effective_config.py` |
| `modelship/infer/model_deployment.py` | Ray Serve deployment actor |
| `modelship/infer/infer_config.py` | Pydantic config models and protocols |
| `modelship/infer/vllm/vllm_infer.py` | vLLM engine wrapper |
| `modelship/infer/llama_server/llama_server_infer.py` | llama-server subprocess proxy (GGUF chat/embed/vision) |
| `modelship/infer/diffusers/diffusers_infer.py` | Diffusers pipeline wrapper |
| `modelship/infer/stable_diffusion_cpp/stable_diffusion_cpp_infer.py` | stable-diffusion.cpp wrapper (CPU image gen) |
| `modelship/plugins/base_plugin.py` | Plugin base classes |
| `config/models.yaml` | Model configuration |
