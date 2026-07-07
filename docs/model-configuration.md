# Model Configuration

Models are configured in a YAML file (default: `config/models.yaml`). Each entry defines one deployment.

## CLI Options

`mship_deploy.py` accepts the following arguments (env vars work as fallbacks):

| Argument | Env Var | Default | Description |
|---|---|---|---|
| `--config` | — | `config/models.yaml` | Path to models config file |
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
| `--no-preflight` | `MSHIP_PREFLIGHT` | enabled | Disable preflight hardware auto-sizing; models run on loader/library defaults plus explicit config. Useful for benchmarking |
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

## Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Model identifier used in API requests |
| `model` | string | HuggingFace repo ID, local path, or `repo:filename` (see [Model source](#model-source)). Required for built-in loaders; optional for `loader: custom` |
| `usecase` | string | `generate`, `embed`, `transcription`, `translation`, `tts`, or `image` |
| `loader` | string | `vllm`, `diffusers`, `llama_server`, `stable_diffusion_cpp`, or `custom` |
| `plugin` | string | Plugin module name (required when `loader: custom`); automatically loaded from wheels when referenced |
| `num_gpus` | float \| int | GPU allocation. Fractional `< 1` shares one GPU (also sets vLLM `gpu_memory_utilization`); integer `≥ 1` requests that many whole GPUs (for `vllm`, this auto-sets `tensor_parallel_size = num_gpus` unless tp/pp is already specified). |
| `num_cpus` | float | CPU units to allocate (default `0.1`) |
| `num_replicas` | int | Fixed number of identical Ray Serve replicas for this deployment (default `1`). Mutually exclusive with `autoscaling_config`. |
| `autoscaling_config` | object | Autoscale replicas with load instead of a fixed `num_replicas` (see [Autoscaling](#autoscaling)). Mutually exclusive with `num_replicas`. |
| `max_ongoing_requests` | int | Per-replica Ray Serve concurrency cap (default: Ray Serve's own default of `100`). Streaming requests hold a slot for the whole generation, so a low cap throttles upstream of the engine; raise it for high-concurrency models. Omit to inherit the default. |
| `vllm_engine_kwargs` | object | Passed directly to the vLLM engine (see below) |
| `diffusers_config` | object | Diffusers pipeline options (see below) |
| `llama_server_config` | object | llama-server loader options (see below) |
| `stable_diffusion_cpp_config` | object | stable-diffusion.cpp loader options (see below) |
| `plugin_config` | object | Plugin-specific options passed through to the plugin |
| `chat_template_kwargs` | object | Extra variables forwarded into the chat-template render on text loaders (`vllm`) — e.g. `enable_thinking: false` for Qwen3. Only has an effect if the model's template branches on the key. A per-request `chat_template_kwargs` overrides the model default on `vllm`. |

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
| `max_model_len` | int | auto (preflight) | Maximum sequence length. Preflight sizes this to the hardware an actor lands on — GPU VRAM or, on `num_gpus: 0`, host RAM — falling back to vLLM's own default when it declines (missing `config.json`, unreadable KV-cache geometry, etc.) |
| `dtype` | string | `auto` | Model dtype (`auto`, `float16`, `bfloat16`) |
| `tokenizer` | string | model default | Custom tokenizer path |
| `trust_remote_code` | bool | `false` | Allow remote code execution |
| `gpu_memory_utilization` | float | `0.9` (`0.4` on CPU deploys) | VRAM fraction on GPU; on CPU it means *host RAM* fraction reserved for the KV cache instead (see [CPU (no GPU required)](#cpu-no-gpu-required) below). Overridden by `num_gpus` when `num_gpus < 1`, including `num_gpus: 0`; on a `num_gpus: 0` deploy, preflight may also recommend a tighter value than the `0.4` fallback — an explicit value always wins over both. |
| `quantization` | string | — | Quantization method (e.g. `awq`, `gptq`) |
| `enable_auto_tool_choice` | bool | — | Enable automatic tool/function calling |
| `tool_call_parser` | string | — | Tool call parser (e.g. `llama3_json`, `hermes`) |
| `enforce_eager` | bool | — | Disable CUDA graph capture |
| `kv_cache_dtype` | string | — | KV cache dtype (e.g. `fp8`) |

> **GGUF is not supported on the `vllm` loader.** vLLM 0.24 dropped in-tree GGUF, so
> pointing the vllm loader at a `.gguf` is rejected at startup. Use `loader: llama_server`
> for GGUF models; the vllm loader takes safetensors checkpoints or AWQ/GPTQ/FP8 quants.
> This is unconditional regardless of GPU vs. CPU — see below.

### CPU (no GPU required)

The `vllm` loader also installs on the `cpu` extra (`num_gpus: 0`), for quantized chat
without a GPU. The GGUF rejection above applies here too, so you need a non-GGUF
checkpoint (safetensors, or an AWQ/GPTQ/compressed-tensors quant — the CPU backend
supports AWQ/GPTQ on x86 plus INT8 W8A8).

`gpu_memory_utilization` means something different on CPU: vLLM repurposes it as the
fraction of **host RAM** to reserve for the KV cache, not VRAM. modelship lowers its
default to `0.4` for `num_gpus: 0` deploys — the GPU-oriented `0.9` default would try to
reserve 90% of node RAM and fail at worker init on a real machine. Preflight goes a step
further: it reads the actual RAM available on the actor's node and the model's weight
footprint, and recommends both `max_model_len` and a tighter `gpu_memory_utilization` than
the `0.4` fallback whenever it can (the fallback only applies when preflight declines —
e.g. an unreadable `config.json`). Set either explicitly and it always wins over both the
preflight recommendation and the fallback. For finer control than a RAM fraction, vLLM
also reads `VLLM_CPU_KVCACHE_SPACE` (a fixed GiB budget) and `VLLM_CPU_OMP_THREADS_BIND`
(CPU thread pinning) directly from the process environment; these are vLLM-native env
vars, not modelship config.

The fool-proof minimum — preflight fills in everything else:

```yaml
models:
  - name: qwen-cpu
    model: Qwen/Qwen2.5-7B-Instruct-AWQ
    usecase: generate
    loader: vllm
    num_gpus: 0
```

See `config/examples/vllm-cpu.yaml` for a complete example with tool calling enabled.

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

## llama_server Loader

The `llama_server` loader runs GGUF models by launching a [`llama-server`](https://github.com/ggml-org/llama.cpp) subprocess and proxying its native OpenAI-compatible HTTP API. Chat templating, tool-call parsing, and reasoning parsing are all llama-server's own (`--jinja --reasoning-format auto`), not modelship's. `--parallel` request slots let concurrent requests actually overlap instead of serializing behind a single lock. It requires the `llama-server` binary to be discoverable via `MSHIP_LLAMA_SERVER_BIN` (see [development.md](development.md#llama-server-binary-llama_server-loader)); the Docker images ship a pinned build at `/opt/llama.cpp`.

`num_gpus` must be `0` (CPU-only) or a whole integer number of GPUs — fractional is rejected at config time, since llama.cpp has no VRAM-fraction knob.

| Field | Type | Default | Description |
|---|---|---|---|
| `n_ctx` | int | auto (preflight); `2048` when preflight declines | Per-slot context length. The launch command multiplies this by `parallel` for llama-server's total `-c` (it splits one context budget across slots). Preflight sizes it from GGUF metadata and the actor's hardware: RAM on `num_gpus: 0`, VRAM (and RAM for any CPU-resident layers) on `num_gpus >= 1` |
| `n_batch` | int | `512` | Batch size for prompt processing |
| `n_gpu_layers` | int | auto (preflight); `-1` when preflight declines | Layers to offload to GPU when `num_gpus >= 1`; forced to `0` when `num_gpus` is `0`. Preflight always recommends a concrete count (full or partial offload, sized to free VRAM) when GGUF metadata is readable. The `-1` fallback hits llama-server's own auto-fit-to-free-memory behavior (any negative value does — verified against the pinned b9859 binary) |
| `threads` | int | `None` (llama-server's own default: all cores) | Compute thread count (`--threads`). Preflight recommends `num_cpus` when the deploy reserves one or more whole CPUs, so the subprocess doesn't grab every core on a shared node |
| `parallel` | int | `1` | Concurrent request slots (`--parallel`). When `max_ongoing_requests` isn't set explicitly, it defaults to this value so overflow queues in Ray Serve rather than inside llama-server |
| `chat_template` | string | — | Built-in template name (e.g. `chatml`) or a path to a Jinja file. Omit to use the GGUF's embedded chat template |
| `mmproj` | string | — | Multimodal projector file/repo ref (e.g. a CLIP model) for vision models — see [Vision](#vision-gguf) below |
| `extra_args` | list[string] | `[]` | Escape hatch: extra flags appended verbatim to the `llama-server` launch command |

> **Note:** there is no persistent on-disk prompt cache. llama-server manages request-level prefix caching internally and automatically within its slots, but has no equivalent to modelship's restart-persistent disk cache.

The fool-proof minimum — preflight fills in `n_ctx`, `n_gpu_layers`, and `threads`:

```yaml
models:
  - name: "qwen-llama-server"
    model: "lmstudio-community/Qwen2.5-7B-Instruct-GGUF:*Q4_K_M.gguf"
    usecase: "generate"
    loader: "llama_server"
    num_gpus: 1
```

### Tool calling and reasoning gaps vs. the OpenAI spec

llama-server auto-detects both the tool-call and reasoning parser from the model's chat template — there is no modelship-level override. Two gaps are real and per-model-family, not per-loader, so test against the specific model in use before relying on either:

- **Named-function forcing is unsupported.** `tool_choice: {"type": "function", "function": {"name": "X"}}` silently falls back to `auto` (llama-server logs a warning; modelship does not surface it as an error).
- **`tool_choice: required` enforcement depends on the model's chat template family.** It's grammar-enforced for harmony-style templates (e.g. gpt-oss) but a silent no-op for hermes-style templates (e.g. Qwen3) — the model may still answer in free text with no error.
- **Bare `response_format: {"type": "json_object"}` (no `schema` key) is not enforced**, despite llama-server's own docs describing it as supported "plain JSON output" — verified directly against the b9859 binary. The model can answer in free text with no error. This doesn't affect `type: json_schema` requests (which modelship sends whenever a schema is given, e.g. structured outputs) — those carry a `schema` and llama-server does constrain them correctly.

`response_format`/`json_schema` can be combined with reasoning in the same request, and `logprobs`/`top_logprobs` are forwarded and returned.

### Vision (GGUF)

Set `mmproj` to a multimodal projector file (local path or `repo:filename`, resolved the same way as `model:`) to enable image input; requests with `image_url`/`input_image` content parts are rejected at the gateway when `mmproj` isn't configured.

```yaml
models:
  - name: "llava-llama-server"
    model: "second-state/Llava-v1.5-7B-GGUF:llava-v1.5-7b-Q4_K_M.gguf"
    usecase: "generate"
    loader: "llama_server"
    llama_server_config:
      mmproj: "second-state/Llava-v1.5-7B-GGUF:llava-v1.5-7b-mmproj-model-f16.gguf"
```

### Embeddings (GGUF)

```yaml
models:
  - name: nomic-embed-server
    model: "nomic-ai/nomic-embed-text-v1.5-GGUF:nomic-embed-text-v1.5.Q4_K_M.gguf"
    usecase: embed
    loader: llama_server
```

## stable-diffusion.cpp Loader

The `stable_diffusion_cpp` loader uses [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (via [stable-diffusion-cpp-python](https://github.com/william-murray1204/stable-diffusion-cpp-python)) for **CPU-only image generation**. It runs GGUF-quantized single-file diffusion checkpoints (SD1.5, SDXL, SD-Turbo, all-in-one Flux) in a few GB of RAM, with no GPU. Any `num_gpus` is ignored (a warning is logged and the actor is allocated `num_gpus: 0`). `usecase` is always `image` (defaulted if omitted) and it serves `/v1/images/generations`, `/v1/images/edits`, and `/v1/images/variations`.

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
