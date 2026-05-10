# Model Configuration

Models are configured in a YAML file (default: `config/models.yaml`). Each entry defines one deployment.

## CLI Options

`mship_deploy.py` accepts the following arguments (env vars work as fallbacks):

| Argument | Env Var | Default | Description |
|---|---|---|---|
| `--config` | — | `config/models.yaml` | Path to models config file |
| `--gateway-name` | `MSHIP_GATEWAY_NAME` | `modelship api` | Name for the API gateway app |
| `--ray-cluster-address` | `RAY_CLUSTER_ADDRESS` | — | Ray cluster address |
| `--ray-redis-port` | `RAY_REDIS_PORT` | — | Ray Redis port |
| `--use-existing-ray-cluster` | `MSHIP_USE_EXISTING_RAY_CLUSTER` | `false` | Connect to an existing Ray cluster |
| `--redeploy` | — | `false` | Tear down all existing deployments before deploying |
| `--cache-dir` | `MSHIP_CACHE_DIR` | `/.cache` | Base cache directory |
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

Use `--redeploy` to tear down everything and start fresh:

```bash
python mship_deploy.py --config config/models.yaml --redeploy
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
| `loader` | string | `vllm`, `transformers`, `diffusers`, `llama_cpp`, or `custom` |
| `plugin` | string | Plugin module name (required when `loader: custom`); automatically loaded from wheels when referenced |
| `num_gpus` | float | Fraction of a GPU to allocate (0.0-1.0); also sets vLLM `gpu_memory_utilization` |
| `num_cpus` | float | CPU units to allocate (default `0.1`) |
| `num_replicas` | int | Number of identical Ray Serve replicas for this deployment (default `1`) |
| `vllm_engine_kwargs` | object | Passed directly to the vLLM engine (see below) |
| `transformers_config` | object | Transformers loader options (see below) |
| `diffusers_config` | object | Diffusers pipeline options (see below) |
| `llama_cpp_config` | object | llama.cpp loader options (see below) |
| `plugin_config` | object | Plugin-specific options passed through to the plugin |

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
| `gpu_memory_utilization` | float | `0.9` | VRAM fraction (overridden by `num_gpus` when set) |
| `distributed_executor_backend` | string | auto | `ray` or `mp` for multi-GPU |
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

```yaml
models:
  - name: llama-70b
    model: meta-llama/Llama-3.1-70B-Instruct
    usecase: generate
    loader: vllm
    vllm_engine_kwargs:
      tensor_parallel_size: 2
      distributed_executor_backend: ray
```

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
| `tool_call_parser` | string | auto | Parser used to turn raw model output into OpenAI `tool_calls`. Currently supported: `hermes` (Hermes-2-Pro / Qwen2.5-Instruct / many community fine-tunes that emit `<tool_call>{...}</tool_call>` markers), `mistral` (Mistral 7B Instruct v0.3+ / Mistral Small/Large that emit `[TOOL_CALLS][...]`), `llama3_json` (Llama-3.1 / Llama-3.2 Instruct emitting bare `{"name": "...", "parameters": {...}}`). Auto-detected from the chat template when omitted. |

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

> **Note:** Setting `MSHIP_LOG_LEVEL` to `TRACE` will enable `verbose` mode in the underlying llama.cpp engine.

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

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace access token | — |
| `MSHIP_CACHE_DIR` | Model cache directory (HuggingFace + plugins) | `/.cache` |
| `MSHIP_GATEWAY_NAME` | Name for the API gateway app | `modelship api` |
| `MSHIP_MAX_REQUEST_BODY_BYTES` | Maximum allowed request body size in bytes | `52428800` (50 MB) |
| `MSHIP_LOG_TARGET` | Log target: `console` or syslog URI (e.g. `syslog://host:514`, `syslog+tcp://host:514`) | `console` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry OTLP endpoint for log export (e.g. `http://collector:4317`). Requires `uv sync --extra otel`. | — |
| `CUDA_DEVICE_ORDER` | GPU enumeration order; set to `PCI_BUS_ID` for deterministic ordering in multi-GPU systems | `PCI_BUS_ID` |
| `RAY_REDIS_PORT` | Ray GCS server port | `6379` |
| `RAY_DASHBOARD_PORT` | Ray dashboard port | `8265` |
| `RAY_HEAD_CPU_NUM` | Optional override: CPUs allocated to Ray head | — |
| `RAY_HEAD_GPU_NUM` | Optional override: GPUs allocated to Ray head | — |
| `RAY_OBJECT_STORE_SHM_SIZE` | Shared memory for Ray object store | `8g` |
| `VLLM_USE_V1` | Use vLLM v1 API | `1` |
| `ONNX_PROVIDER` | ONNX Runtime execution provider | `CUDAExecutionProvider` |
| `NVIDIA_CUDA_VERSION` | CUDA toolkit version | `12.8.1` |
