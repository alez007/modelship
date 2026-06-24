# Model Configuration

Models are configured in a YAML file (default: `config/models.yaml`). Each entry defines one deployment.

## CLI Options

`mship_deploy.py` accepts the following arguments (env vars work as fallbacks):

| Argument | Env Var | Default | Description |
|---|---|---|---|
| `--config` | — | `config/models.yaml` | Path to models config file |
| `--model-stack` | `MSHIP_MODEL_STACK` | — | Auto-generate a config from a [profile](#profiles-mship_model_stack) (`chat`/`assistant`/`studio`/`everything`) sized to detected hardware |
| `--gateway-name` | `MSHIP_GATEWAY_NAME` | `modelship api` | Name for the API gateway app |
| `--use-existing-ray-cluster` | `MSHIP_USE_EXISTING_RAY_CLUSTER` | `false` | Connect to a Ray cluster you manage (must run on a cluster node) instead of starting one. Implies deploy-and-exit (no teardown) |
| `--prune-ray-sessions` | `MSHIP_PRUNE_RAY_SESSIONS` | `true` | When starting its own Ray head, delete stale `session_*` dirs left under the Ray temp root (default `/tmp/ray`) by previous, no-longer-running heads — Ray never cleans these up, so they fill the disk across restarts. A live head's session is always kept. Set `false` to keep them (e.g. for debugging). No effect with `--use-existing-ray-cluster` |
| `--reconcile` | — | `false` | Reconcile the cluster to the config: add new models, remove dropped ones, replace changed ones (vs. the default additive union). With no `--config`, reconciles to this gateway's persisted effective config (self-heal) |
| `--replace-strategy` | — | `blue_green` | How to replace a changed model: `blue_green` (deploy new before dropping old, no request loss) or `stop_start` (drop old first, brief unavailability) |
| `--cache-dir` | `MSHIP_CACHE_DIR` | `/.cache` | Base cache directory |
| `--state-store` | `MSHIP_STATE_STORE` | `memory://` | State-store connection URI for the effective config + deploy coordinator (see [State store](#state-store-mship_state_store)) |
| — | `MSHIP_LOG_LEVEL` | `INFO` | Log level (env-var-only: must be set before `import ray` so library loggers latch the right level) |
| `--log-format` | `MSHIP_LOG_FORMAT` | `text` | Log format (`text` or `json`) |
| `--log-target` | `MSHIP_LOG_TARGET` | `console` | Log target: `console` or syslog URI (e.g. `syslog://host:514`, `syslog+tcp://host:514`) |
| `--otel-endpoint` | `OTEL_EXPORTER_OTLP_ENDPOINT` | — | OpenTelemetry OTLP endpoint (e.g. `http://collector:4317`) |
| `--no-metrics` | `MSHIP_METRICS` | enabled | Disable Prometheus metrics |
| `--api-keys` | `MSHIP_API_KEYS` | — | Comma-separated API keys |
| `--max-request-body-bytes` | `MSHIP_MAX_REQUEST_BODY_BYTES` | `52428800` | Max request body size in bytes |

### Cache Directory Structure

The base cache directory (`MSHIP_CACHE_DIR`, default: `/.cache`) is organized into the following subdirectories:

- `{base_cache}/huggingface`: HuggingFace models and tokenizers (via `HF_HOME`).
- `{base_cache}/vllm`: vLLM-specific compiled artifacts and caches (via `VLLM_CACHE_ROOT`).
- `{base_cache}/flashinfer`: FlashInfer kernels (via `FLASHINFER_CACHE_DIR`).
- `{base_cache}/plugins`: Downloaded weights and artifacts used by custom plugins.

### Additive Deploys

By default, `mship_deploy.py` adds models to a running cluster without disrupting existing deployments. This allows incremental composition:

```bash
# Deploy LLM models
python mship_deploy.py --config config/llm.yaml

# Later, add TTS without touching the running LLMs
python mship_deploy.py --config config/tts.yaml

# Add more models from another config
python mship_deploy.py --config config/embeddings.yaml
```

Use `--reconcile` to make the running cluster match a config exactly — new models
are added, dropped ones removed, and changed ones replaced (gracefully, draining
in-flight requests). Unlike additive mode it never tears down the Ray cluster:

```bash
python mship_deploy.py --config config/models.yaml --reconcile
```

Multiple gateways can run independently by using `--gateway-name`:

```bash
python mship_deploy.py --config config/llm.yaml --gateway-name "llm-api"
python mship_deploy.py --config config/tts.yaml --gateway-name "tts-api"
```

## Profiles (`MSHIP_MODEL_STACK`)

Don't want to pick models, write YAML, or hand-allocate GPU/CPU? Set a **profile** and modelship generates a config sized to your detected hardware:

```bash
MSHIP_MODEL_STACK=studio uv run mship_deploy.py
# equivalently:
uv run mship_deploy.py --model-stack studio
```

A profile is a set of capabilities to serve. Every model in the catalog is **ungated**, so the one-click path needs no `HF_TOKEN`.

| Profile | Capabilities |
|---|---|
| `chat` | generate + embed |
| `assistant` | generate + transcription + tts |
| `studio` | generate + image + embed |
| `everything` | generate + image + embed + transcription + tts |

### Models per tier

The `generate` and `image` models scale with a **tier** (small/medium/large) picked from your hardware — GPU VRAM if a GPU is present, otherwise RAM and core count. Bigger box → bigger tier. The other capabilities are the same on every box.

| Capability | small | medium | large |
|---|---|---|---|
| generate (GPU, vLLM/AWQ) | Qwen2.5-7B | Qwen2.5-14B | Qwen2.5-32B |
| generate (CPU, llama.cpp/GGUF) | Llama-3.2-3B | Qwen2.5-7B | Qwen2.5-14B |
| image (GPU, diffusers) | SD-Turbo | SDXL-Turbo | playground-v2.5 |
| image (CPU, stable-diffusion.cpp) | SD-Turbo | SDXL-Turbo | SDXL-base |
| embed (CPU, llama.cpp) | nomic-embed-text-v1.5 | ← | ← |
| tts (CPU, `kokoroonnx`) | Kokoro-82M | ← | ← |
| transcription (CPU, `whispercpp`) | whisper `base` | whisper `small` | ← |

Selection is **all-or-nothing**: the highest tier whose *complete* stack fits is chosen. A capability is never dropped to squeeze something in — if even the smallest tier won't fit, the deploy refuses with a clear message instead of serving a partial stack. Resource requests (`num_cpus`/`num_gpus`) are filled in automatically; on a shared GPU the LLM gets the larger slice.

### File behavior and precedence

- The generated `config/models_stack_<profile>.yaml` is **regenerated from scratch on every start** while the profile is set — switch profiles just by changing the value, with no stale file to delete.
- Resolution order: an explicit `--config` always wins; otherwise `MSHIP_MODEL_STACK`/`--model-stack`; otherwise the default `config/models.yaml`.
- The file is normal, editable YAML, but **edits are overwritten** on the next profile-driven start. To keep changes, copy it to `config/models.yaml` (or pass `--config <file>`) and unset `MSHIP_MODEL_STACK`.

## Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Model identifier used in API requests |
| `model` | string | HuggingFace repo ID, local path, or `repo:filename` (see [Model source](#model-source)). Required for built-in loaders; optional for `loader: custom` |
| `usecase` | string | `generate`, `embed`, `transcription`, `translation`, `tts`, or `image` |
| `loader` | string | `vllm`, `transformers`, `diffusers`, `llama_cpp`, `stable_diffusion_cpp`, or `custom` |
| `plugin` | string | Plugin module name (required when `loader: custom`); automatically loaded from wheels when referenced |
| `num_gpus` | float \| int | GPU allocation. Fractional `< 1` shares one GPU (also sets vLLM `gpu_memory_utilization`); integer `≥ 1` requests that many whole GPUs (for `vllm`, this auto-sets `tensor_parallel_size = num_gpus` unless tp/pp is already specified). |
| `num_cpus` | float | CPU units to allocate (default `0.1`) |
| `num_replicas` | int | Fixed number of identical Ray Serve replicas for this deployment (default `1`). Mutually exclusive with `autoscaling_config`. |
| `autoscaling_config` | object | Autoscale replicas with load instead of a fixed `num_replicas` (see [Autoscaling](#autoscaling)). Mutually exclusive with `num_replicas`. |
| `max_ongoing_requests` | int | Per-replica Ray Serve concurrency cap (default: Ray Serve's own default of `100`). Streaming requests hold a slot for the whole generation, so a low cap throttles upstream of the engine; raise it for high-concurrency models. Omit to inherit the default. |
| `vllm_engine_kwargs` | object | Passed directly to the vLLM engine (see below) |
| `transformers_config` | object | Transformers loader options (see below) |
| `diffusers_config` | object | Diffusers pipeline options (see below) |
| `llama_cpp_config` | object | llama.cpp loader options (see below) |
| `stable_diffusion_cpp_config` | object | stable-diffusion.cpp loader options (see below) |
| `plugin_config` | object | Plugin-specific options passed through to the plugin |
| `chat_template_kwargs` | object | Extra variables forwarded into the chat-template render on text loaders (`vllm`, `transformers`, `llama_cpp`) — e.g. `enable_thinking: false` for Qwen3. Only has an effect if the model's template branches on the key; ignored on paths that bypass the template (llama.cpp's native chat-handler fallback when `chat_format` is set or no template resolves). A per-request `chat_template_kwargs` overrides the model default on `vllm`. |

## Model source

The `model:` field accepts three forms. For built-in loaders, Modelship resolves
the source on the **driver** before any Ray actor spins up — so auth failures,
missing repos, and missing files surface immediately at startup instead of inside
a stuck deployment.

| Form | Example | When to use |
|---|---|---|
| HuggingFace repo ID | `Qwen/Qwen3-7B` | Standard HF model. Modelship runs `snapshot_download` with a universal filter (prefers `*.safetensors`, skips `*.bin` when both exist). |
| Local path | `/mnt/nfs/models/qwen-7b` | A directory of HF-format files (or a single file for llama.cpp / vllm GGUF). |
| `repo:filename` | `lmstudio-community/Qwen2.5-7B-Instruct-GGUF:*Q4_K_M.gguf` | Pick a specific file inside an HF repo. The selector is a glob; it must match exactly one file (or a single sharded set, e.g. `*-of-*.gguf`). |

The `:filename` selector is also supported against a **local directory**: if `model:` points at a directory and the value contains `:`, the selector is matched against files inside that directory. The full path to the matched file is what the loader receives.

### Multi-node clusters

When Ray runs across multiple nodes, the resolver downloads to the driver's
`HF_HOME`. **Worker nodes must see the same path** — the simplest setup is to
mount `MSHIP_CACHE_DIR` (which contains `HF_HOME`) on shared storage (NFS, EFS,
or similar) so every node reads from one cache. Without shared storage the
worker can't open the file.

### Multi-variant GGUF repos

If `model:` points at an HF repo containing more than one `.gguf` file and no
`:filename` selector is given, Modelship raises at startup with the list of
variants and an example fix:

```
HF repo 'lmstudio-community/Qwen2.5-7B-Instruct-GGUF' contains 5 GGUF variants — pick one with the `:filename` syntax (glob supported, must match exactly one file):
  - Qwen2.5-7B-Instruct-Q2_K.gguf
  - Qwen2.5-7B-Instruct-Q4_K_M.gguf
  - Qwen2.5-7B-Instruct-Q5_K_M.gguf
  - Qwen2.5-7B-Instruct-Q8_0.gguf
  - Qwen2.5-7B-Instruct-fp16.gguf
Example: model: lmstudio-community/Qwen2.5-7B-Instruct-GGUF:*Q4_K_M.gguf
```

### Plugins (`loader: custom`)

Plugins manage their own model files; Modelship does not pre-resolve `model:` for
them. The field is optional for custom loaders and acts as a label only —
plugins are free to ignore it and use `plugin_config` instead.

## vLLM Loader

The `vllm` loader supports chat/generation, embeddings, transcription, and translation. Configuration is passed via `vllm_engine_kwargs`:

| Field | Type | Default | Description |
|---|---|---|---|
| `tensor_parallel_size` | int | `1` | Number of GPUs for tensor parallelism |
| `max_model_len` | int | auto | Maximum sequence length |
| `dtype` | string | `auto` | Model dtype (`auto`, `float16`, `bfloat16`) |
| `tokenizer` | string | model default | Custom tokenizer path |
| `trust_remote_code` | bool | `false` | Allow remote code execution |
| `gpu_memory_utilization` | float | `0.9` | VRAM fraction (overridden by `num_gpus` when `num_gpus < 1`) |
| `quantization` | string | — | Quantization method (e.g. `awq`, `gptq`) |
| `enable_auto_tool_choice` | bool | — | Enable automatic tool/function calling |
| `tool_call_parser` | string | — | Tool call parser (e.g. `llama3_json`, `hermes`) |
| `enforce_eager` | bool | — | Disable CUDA graph capture |
| `kv_cache_dtype` | string | — | KV cache dtype (e.g. `fp8`) |

### Chat / Text Generation

```yaml
models:
  - name: qwen
    model: Qwen/Qwen3-0.6B
    usecase: generate
    loader: vllm
    num_gpus: 0.30
    vllm_engine_kwargs:
      max_model_len: 8192
```

### LLM with Tool Calling

```yaml
models:
  - name: llama
    model: meta-llama/Llama-3.1-8B-Instruct
    usecase: generate
    loader: vllm
    num_gpus: 0.70
    vllm_engine_kwargs:
      enable_auto_tool_choice: true
      tool_call_parser: llama3_json
```

### Multi-GPU with Tensor Parallelism

`num_gpus: 2` is shorthand for "use 2 whole GPUs" — tensor parallelism is
auto-derived (`tensor_parallel_size: 2`). Setting both is redundant; setting
only `tensor_parallel_size` (and/or `pipeline_parallel_size`) is fine too.
Each slot always owns one whole GPU.

```yaml
models:
  - name: llama-70b
    model: meta-llama/Llama-3.1-70B-Instruct
    usecase: generate
    loader: vllm
    num_gpus: 2
```

Multi-slot deploys always use vLLM's ray distributed executor: each TP/PP
slot runs as its own Ray worker actor inside a Ray Serve placement group
(STRICT_PACK, one whole-GPU bundle per slot, all on one node for NVLink).

> **Note:** Fractional `num_gpus` (`< 1`) is **single-GPU only**. Combining
> `num_gpus < 1` with `tensor_parallel_size > 1` or `pipeline_parallel_size > 1`
> is rejected at config time, because Ray packs fractional placement-group
> bundles onto the same physical GPU — which breaks tensor parallelism. To
> share GPUs use `num_gpus: 0.x` with `tp: 1`; to do TP use whole-GPU
> integer `num_gpus`.

### Embeddings

```yaml
models:
  - name: nomic-embed
    model: nomic-ai/nomic-embed-text-v1.5
    usecase: embed
    loader: vllm
    num_gpus: 0.15
    vllm_engine_kwargs:
      trust_remote_code: true
```

### Speech-to-Text (Whisper)

```yaml
models:
  - name: whisper
    model: openai/whisper-small
    usecase: transcription
    loader: vllm
    num_gpus: 0.15
    vllm_engine_kwargs:
      trust_remote_code: true
```

## Transformers Loader

The `transformers` loader uses PyTorch with HuggingFace Transformers. Supports chat/generation, embeddings, transcription, translation, and TTS. Unlike the vLLM loader, it can run entirely on CPU — making it ideal for smaller models, development, or environments without a GPU.

| Field | Type | Default | Description |
|---|---|---|---|
| `device` | string | `cpu` | Device to run on (`cpu`, `cuda`, `cuda:0`, etc.) |
| `torch_dtype` | string | `auto` | Model dtype (`auto`, `float16`, `bfloat16`, `float32`) |
| `trust_remote_code` | bool | `false` | Allow remote code execution |
| `model_kwargs` | object | `{}` | Extra keyword arguments passed to the model constructor |
| `pipeline_kwargs` | object | `{}` | Extra keyword arguments passed to the pipeline at inference time |
| `tool_call_parser` | string | auto | Parser used to turn raw model output into OpenAI `tool_calls`. Currently supported: `hermes` (Hermes-2-Pro / Qwen2.5-Instruct / many community fine-tunes that emit `<tool_call>{...}</tool_call>` markers), `qwen3_coder` (Qwen3-Coder family — same `<tool_call>` envelope but an XML body of `<function=name><parameter=key>value</parameter></function>`; parameter values are returned as strings), `mistral` (Mistral 7B Instruct v0.3+ / Mistral Small/Large that emit `[TOOL_CALLS][...]`), `llama3_json` (Llama-3.1 / Llama-3.2 Instruct emitting bare `{"name": "...", "parameters": {...}}`). Auto-detected from the chat template when omitted. |

### Chat / Text Generation (CPU)

```yaml
models:
  - name: qwen
    model: Qwen/Qwen3-0.6B
    usecase: generate
    loader: transformers
    num_gpus: 0
    transformers_config:
      device: "cpu"
```

### Chat with Tool Calling (CPU)

The transformers loader renders `tools` into the prompt via the model's chat
template and parses the output back into OpenAI `tool_calls`. The model must
have been trained on a Hermes-style tool format (Qwen2.5-Instruct, Hermes-2,
many community fine-tunes); the parser is selected via `tool_call_parser`.

```yaml
models:
  - name: qwen-tools
    model: Qwen/Qwen2.5-0.5B-Instruct
    usecase: generate
    loader: transformers
    num_cpus: 2
    transformers_config:
      device: "cpu"
      tool_call_parser: hermes  # this is the default; shown for clarity
```

### Speech-to-Text (CPU)

Audio is automatically decoded and resampled to the model's expected sample rate (e.g. 16kHz for Whisper).

```yaml
models:
  - name: whisper
    model: openai/whisper-small
    usecase: transcription
    loader: transformers
    num_gpus: 0
    transformers_config:
      device: "cpu"
```

### Embeddings (CPU)

Uses `sentence-transformers` under the hood.

```yaml
models:
  - name: embeddings
    model: sentence-transformers/all-MiniLM-L6-v2
    usecase: embed
    loader: transformers
    num_gpus: 0
    transformers_config:
      device: "cpu"
```

### TTS (GPU)

```yaml
models:
  - name: my-tts
    model: some-org/some-tts-model
    usecase: tts
    loader: transformers
    num_gpus: 0.20
    transformers_config:
      device: "cuda:0"
```

## Diffusers Loader

The `diffusers` loader uses HuggingFace Diffusers for image generation. Any model supported by `AutoPipelineForText2Image` works out of the box.

| Field | Type | Default | Description |
|---|---|---|---|
| `torch_dtype` | string | `float16` | Torch dtype (`float16`, `bfloat16`, `float32`) |
| `num_inference_steps` | int | `30` | Default denoising steps (can be overridden per request) |
| `guidance_scale` | float | `7.5` | Default classifier-free guidance scale (can be overridden per request) |

```yaml
models:
  - name: sdxl-turbo
    model: stabilityai/sdxl-turbo
    usecase: image
    loader: diffusers
    num_gpus: 0.35
    diffusers_config:
      torch_dtype: "float16"
      num_inference_steps: 4
      guidance_scale: 0.0
```

## llama.cpp Loader

The `llama_cpp` loader uses [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) to run GGUF models. It currently supports **CPU-only inference** — any `num_gpus` or `n_gpu_layers` configuration is ignored (a warning is logged and `n_gpu_layers` is forced to `0`). This loader is ideal for running quantized models efficiently on hardware without dedicated GPUs.

| Field | Type | Default | Description |
|---|---|---|---|
| `n_ctx` | int | `2048` | Maximum sequence length |
| `n_batch` | int | `512` | Batch size for prompt processing |
| `n_gpu_layers` | int | `0` | Currently ignored — forced to `0` (CPU-only) |
| `chat_format` | string | — | Chat template format (e.g. `llama-3`) |
| `model_kwargs` | object | `{}` | Extra keyword arguments passed to the `Llama` constructor |
| `constrain_tool_calls` | bool | `false` | Constrain tool-call decoding with a GBNF grammar built from the request's `tools` (see below) |

> **Note:** Setting `MSHIP_LOG_LEVEL` to `TRACE` will enable `verbose` mode in the underlying llama.cpp engine.

#### Constrained tool calling (`constrain_tool_calls`)

When enabled, requests that carry `tools` are decoded under a [GBNF](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) grammar compiled from those tool schemas. The grammar's top level allows **either** a free-text answer **or** a bounded sequence (max two) of tool calls, each forced into the parser's envelope with `name` pinned to a real tool name and `arguments` constrained to that tool's JSON schema. This prevents malformed envelopes, invented fields, and runaway repetition while still letting the model answer in plain text.

Caveats:

- It enforces *structure*, not *choice* — the grammar cannot make the model pick the correct field or value, only that whatever it emits is well-typed and well-formed.
- JSON-schema numeric bounds (e.g. `minimum`/`maximum`) are not expressible as ranges in GBNF; numbers are constrained to digit shape only.
- The free-text branch cannot contain a literal `<`.
- Only the parser-driven path honors this. It requires a resolvable `tool_call_parser` from a JSON family (currently `hermes`); other families are left unconstrained (logged once). It also takes precedence over a `response_format` grammar on the same request — the two cannot be combined.

GGUF variants in a HuggingFace repo are picked via the `:filename` syntax on the
`model:` field (see [Model source](#model-source)). The selector is a glob and
must match exactly one file.

### Chat / Text Generation (GGUF)

```yaml
models:
  - name: "qwen-gguf-hf"
    model: "lmstudio-community/Qwen2.5-7B-Instruct-GGUF:*Q4_K_M.gguf"
    usecase: "generate"
    loader: "llama_cpp"
    num_cpus: 3
```

### Embeddings (GGUF)

```yaml
models:
  - name: nomic-embed
    model: "nomic-ai/nomic-embed-text-v1.5-GGUF:nomic-embed-text-v1.5.Q4_K_M.gguf"
    usecase: embed
    loader: llama_cpp
```

## stable-diffusion.cpp Loader

The `stable_diffusion_cpp` loader uses [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (via [stable-diffusion-cpp-python](https://github.com/william-murray1204/stable-diffusion-cpp-python)) for **CPU-only image generation** — the image counterpart to the `llama_cpp` text loader. It runs GGUF-quantized single-file diffusion checkpoints (SD1.5, SDXL, SD-Turbo, all-in-one Flux) in a few GB of RAM, with no GPU. Any `num_gpus` is ignored (a warning is logged and the actor is allocated `num_gpus: 0`). `usecase` is always `image` (defaulted if omitted) and it serves `/v1/images/generations`, `/v1/images/edits`, and `/v1/images/variations`.

| Field | Type | Default | Description |
|---|---|---|---|
| `sample_steps` | int | `20` | Denoising steps (sd.cpp analogue of `num_inference_steps`) |
| `cfg_scale` | float | `7.0` | Classifier-free guidance scale (analogue of `guidance_scale`) |
| `sample_method` | string | `default` | Sampler; `default` lets sd.cpp pick per architecture |
| `scheduler` | string | `default` | Denoiser sigma scheduler |
| `wtype` | string | `default` | On-the-fly weight quantization type (e.g. `q4_0`, `q8_0`, `f16`); `default` auto-detects |
| `n_threads` | int | `-1` | CPU threads; `-1` uses half the cores |
| `vae_tiling` | bool | `false` | Tile the VAE decode to cut peak RAM (auto-recommended by preflight on low-RAM hosts) |
| `diffusion_model_path` / `clip_l_path` / `clip_g_path` / `t5xxl_path` / `vae_path` | string | — | Standalone component paths for split checkpoints (accepted as pre-placed local paths; single-file models are the v1 focus) |
| `model_kwargs` | object | `{}` | Extra keyword arguments passed to the `StableDiffusion` constructor |

> **Note:** Setting `MSHIP_LOG_LEVEL` to `TRACE` enables `verbose` mode in the underlying stable-diffusion.cpp engine.

GGUF variants in a HuggingFace repo are picked via the `:filename` syntax on the `model:` field (see [Model source](#model-source)), exactly like the llama.cpp loader.

### Image Generation (GGUF, CPU)

```yaml
models:
  - name: sdxl-turbo
    model: "second-state/stable-diffusion-xl-turbo-GGUF:*Q4_0.gguf"
    usecase: image
    loader: stable_diffusion_cpp
    num_cpus: 4
    stable_diffusion_cpp_config:
      sample_steps: 4
      cfg_scale: 1.0
```

## Custom Loader (Plugins)

The `custom` loader delegates to a plugin module. The `plugin` field is required and must match an installed plugin package. Plugin-specific options are passed via `plugin_config`.

See each plugin's README for configuration details:
- [Kokoro ONNX TTS](../plugins/kokoroonnx/README.md)
- [Bark TTS](../plugins/bark/README.md)
- [Orpheus TTS](../plugins/orpheus/README.md)
- [whisper.cpp STT](../plugins/whispercpp/README.md)

For writing your own plugin, see [Plugin Development](plugins.md).

## Multi-Deployment Routing

You can run the same model on different hardware (e.g. GPU and CPU) by repeating the same `name` with different settings. The API exposes the model once under `/v1/models`, and round-robins requests across all deployments sharing that name.

Use `num_replicas` to scale identical copies of a single deployment (Ray Serve handles load balancing between replicas automatically).

```yaml
models:
  # GPU instance with 2 replicas
  - name: "kokoro"
    model: "hexgrad/Kokoro-82M"
    usecase: "tts"
    loader: "custom"
    plugin: "kokoroonnx"
    num_gpus: 0.07
    num_replicas: 2
    plugin_config:
      onnx_provider: "CUDAExecutionProvider"

  # CPU fallback
  - name: "kokoro"
    model: "hexgrad/Kokoro-82M"
    usecase: "tts"
    loader: "custom"
    plugin: "kokoroonnx"
    num_gpus: 0
    plugin_config:
      onnx_provider: "CPUExecutionProvider"
```

In this example, requests to model `kokoro` are distributed across three backends: two GPU replicas and one CPU instance.

## Autoscaling

Instead of a fixed `num_replicas`, set `autoscaling_config` to let Ray Serve grow
and shrink a deployment's replica count with load. The two are mutually exclusive
— setting both is a config error.

```yaml
models:
  - name: "bursty-llm"
    model: "Qwen/Qwen3-0.6B"
    usecase: "generate"
    loader: "vllm"
    num_gpus: 0.3
    autoscaling_config:
      min_replicas: 1            # floor; 0 enables scale-to-zero (cold-start on first request)
      max_replicas: 4            # ceiling
      target_ongoing_requests: 8 # autoscaler setpoint: in-flight requests per replica (lower = scales out sooner)
      initial_replicas: 1        # seed count on first deploy, before load signal (default: min_replicas)
      upscale_delay_s: 10        # debounce before scaling out
      downscale_delay_s: 300     # debounce before scaling in (longer avoids thrashing GPU warm-up)
```

| Field | Type | Description |
|---|---|---|
| `min_replicas` | int | Lower bound (default `1`). `0` enables scale-to-zero: the deployment idles with no replicas and cold-starts on the first request. |
| `max_replicas` | int | Upper bound (default `1`). Must be `≥ min_replicas`. |
| `initial_replicas` | int | Seed count on first deploy before the autoscaler has a load signal (default: `min_replicas`). |
| `target_ongoing_requests` | float | Desired in-flight requests per replica — the autoscaler's setpoint. Lower scales out sooner (default: Ray Serve's own default). |
| `upscale_delay_s` | float | Seconds of sustained over-load before adding replicas (default: Ray Serve default). |
| `downscale_delay_s` | float | Seconds of sustained under-load before removing replicas (default: Ray Serve default). Raise it to avoid thrashing on models with slow GPU warm-up. |

Autoscaling is changed in place on `mship_deploy --reconcile` (it's excluded from
the config fingerprint), so tuning these bounds doesn't tear down and rebuild the
deployment.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace access token | — |
| `MSHIP_MODEL_STACK` | [Profile](#profiles-mship_model_stack) to auto-generate a hardware-sized config from (`chat`/`assistant`/`studio`/`everything`) | — |
| `MSHIP_CACHE_DIR` | Model cache directory (HuggingFace + plugins) | `/.cache` |
| `MSHIP_STATE_STORE` | State-store connection URI for the effective config + deploy coordinator (see [State store](#state-store-mship_state_store)) | `memory://` |
| `MSHIP_STATE_DIR` | Default directory for a `file://` state store with no path | `<cache-dir>/state` |
| `MSHIP_GATEWAY_NAME` | Name for the API gateway app | `modelship api` |
| `MSHIP_GATEWAY_REPLICAS` | Number of API gateway replicas (routing/ingress HA; replicas sync routing via the deploy coordinator) | `1` |
| `MSHIP_MAX_REQUEST_BODY_BYTES` | Maximum allowed request body size in bytes | `52428800` (50 MB) |
| `MSHIP_LOG_TARGET` | Log target: `console` or syslog URI (e.g. `syslog://host:514`, `syslog+tcp://host:514`) | `console` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry OTLP endpoint for log export (e.g. `http://collector:4317`). Requires `uv sync --extra otel`. | — |
| `CUDA_DEVICE_ORDER` | GPU enumeration order; set to `PCI_BUS_ID` for deterministic ordering in multi-GPU systems | `PCI_BUS_ID` |
| `RAY_DASHBOARD_PORT` | Ray dashboard port | `8265` |
| `RAY_HEAD_CPU_NUM` | Optional override: CPUs allocated to Ray head | — |
| `RAY_HEAD_GPU_NUM` | Optional override: GPUs allocated to Ray head | — |
| `RAY_OBJECT_STORE_SHM_SIZE` | Shared memory for Ray object store | `8g` |
| `VLLM_USE_V1` | Use vLLM v1 API | `1` |
| `ONNX_PROVIDER` | ONNX Runtime execution provider | `CUDAExecutionProvider` |
| `NVIDIA_CUDA_VERSION` | CUDA toolkit version | `12.8.1` |

### State store (`MSHIP_STATE_STORE`)

Two pieces of durable state share one pluggable store: this gateway's **effective
config** (its desired model set, replayed by `--reconcile` with no `--config` to
self-heal after a cluster loss) and the **deploy coordinator's** routing registry
(which gateway owns which model + the expected set). The store is chosen by a single
connection URI — the scheme picks the backend, the rest carries its connection:

| URI | Backend | Durability |
|---|---|---|
| `memory://` (default) | in-process dict | none — lost on restart (fine self-hosted: a restart replaces the config anyway) |
| `file:///.cache/state` | one JSON file per key under the path (empty path → `MSHIP_STATE_DIR`) | survives cluster death on a mounted volume / PVC |
| `redis://[:pw@]host:6379/0` (`rediss://` = TLS) | one JSON value per key in Redis | survives head/coordinator death; the password is parsed from the URL by `redis.from_url` |

In Kubernetes the Helm chart sets this for you: `file://` on the cache PVC by
default, or `redis://…` when `redis.enabled=true` (the same Redis then also backs
Ray GCS fault tolerance — see the chart's **Head-node HA** section). A `redis://`
store is what lets the gateway self-heal its routing after a head restart instead of
needing a redeploy.
