# Modelship

[![CI](https://github.com/alez007/modelship/actions/workflows/ci.yml/badge.svg)](https://github.com/alez007/modelship/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Self-hosted, OpenAI-compatible inference for the agentic era. Modelship runs your whole AI stack — reasoning LLMs with universal tool calling and the Responses API, plus embeddings, speech-to-text, text-to-speech, and image generation — as many models sharing your GPUs (or CPU) behind a single gateway. Built on [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) with pluggable inference backends: [vLLM](https://github.com/vllm-project/vllm) for high-throughput GPU or CPU inference, [llama.cpp](https://github.com/ggml-org/llama.cpp) for high-efficiency GGUF models on CPU or GPU, [Diffusers](https://github.com/huggingface/diffusers) for image generation, and a plugin system for custom backends.

## Why Modelship?

Most self-hosted inference tools focus on running a single model. Modelship is for when you need **multiple models running simultaneously** — an LLM, a TTS engine, a speech-to-text model, an embedding model, and an image generator — all behind a single OpenAI-compatible API, with fine-grained control over GPU memory allocation across them.

- **One server, many models** — run a full AI stack (chat + TTS + STT + embeddings + image gen) on a single machine instead of juggling separate services
- **GPU memory control** — allocate exact GPU fractions per model (e.g. 70% for the LLM, 5% for TTS) so everything fits on your hardware
- **Mix and match backends** — use vLLM for high-throughput GPU inference, vLLM or llama.cpp for CPU-only workloads, Diffusers for images, and plugins for custom backends — in the same deployment
- **Agentic-ready** — reasoning (`<think>` → `reasoning_content`), universal tool/function calling, and the `/v1/responses` API work across the vLLM and llama.cpp (`llama_server`) loaders — the OpenAI-spec surface agents already target
- **Drop-in OpenAI replacement** — any OpenAI SDK client works out of the box, making it easy to integrate with existing apps and tools like [Home Assistant](docs/home-assistant.md)

## Architecture

```mermaid
graph TD
    Client["Client (OpenAI SDK / curl)"]
    API["FastAPI Gateway<br/>OpenAI-compatible API<br/>:8000"]

    Client -->|HTTP| API
    API -->|round-robin| LLM_GPU
    API -->|round-robin| LLM_CPU
    API -->|round-robin| TTS
    API -->|round-robin| STT
    API -->|round-robin| EMB
    API -->|round-robin| IMG

    subgraph GPU0["GPU 0 — vLLM"]
        LLM_GPU["LLM Deployment<br/>e.g. Llama 3.1 8B<br/>70% GPU"]
        TTS["TTS Deployment<br/>e.g. Kokoro 82M<br/>5% GPU"]
    end

    subgraph GPU1["GPU 1 — Mixed backends"]
        STT["STT Deployment (vLLM)<br/>e.g. Whisper Large<br/>50% GPU"]
        EMB["Embedding Deployment<br/>e.g. Nomic Embed<br/>50% GPU"]
    end

    subgraph CPU["CPU — vLLM / llama-server"]
        LLM_CPU["LLM Deployment<br/>e.g. Qwen3-0.6B<br/>CPU-only"]
        STT_CPU["STT Deployment<br/>e.g. Whisper Small<br/>CPU-only"]
    end

    subgraph GPU2["GPU 2 — Diffusers"]
        IMG["Image Generation<br/>e.g. SDXL Turbo<br/>35% GPU"]
    end
```

Each model runs as an isolated [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) deployment with its own lifecycle, health checks, and resource budget. Four inference backends are available:

| Backend | Best for | GPU required |
|---|---|---|
| **vLLM** | High-throughput chat, embeddings, transcription | No — installs on GPU or CPU |
| **llama.cpp** (`llama_server`) | High-efficiency quantized GGUF models (chat, embeddings, vision) | No |
| **Diffusers** | Image generation | Yes |
| **Custom (plugins)** | TTS backends (Kokoro ONNX, Orpheus), STT backends (whisper.cpp) | No |

Models can be deployed across multiple GPUs, run on CPU-only, or both — multiple deployments of the same model (e.g. one on GPU via vLLM, one on CPU via vLLM or llama.cpp) are load-balanced with round-robin routing. Each deployment can also scale horizontally with `num_replicas`.
...

## Requirements

- **Docker** (or Python 3.12+ with `uv` for local development)
- **NVIDIA GPU** (optional) — 16 GB+ VRAM recommended for a full stack (LLM + TTS + STT + embeddings) via vLLM; 8 GB is sufficient for lighter setups. Not required when using the vLLM or llama.cpp backends on CPU
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)** — required only when running GPU models in Docker
- **HuggingFace token** for gated models

## Features

- **Multi-model, multi-GPU** — run chat, embedding, STT, TTS, and image generation models simultaneously across one or more GPUs with tunable per-model GPU memory allocation
- **CPU-only support** — run models without a GPU using the vLLM or llama.cpp (`llama_server`) backends (chat, embeddings, transcription, vision). Useful for development, testing, or small models that don't need GPU acceleration
- **Multiple inference backends** — vLLM for high-throughput GPU or CPU inference, llama.cpp for efficient quantized GGUF models on CPU or GPU, Diffusers for image generation, and a plugin system for custom backends
- **Zero-downtime hot-reloads** — modify your `models.yaml` and run a cluster reconcile; changes are applied incrementally without interrupting the API gateway or unchanged models
- **Advanced agentic capabilities** — native support for DeepSeek-style reasoning (`<think>` blocks parsed into `reasoning_content`) and universal tool/function calling across the vLLM and GGUF (`llama_server`) backends
- **Per-model isolated deployments** — each model runs in its own Ray Serve deployment with independent lifecycle, health checks, failure isolation, and configurable replica count
- **OpenAI-compatible API** — drop-in replacement for any OpenAI SDK client
- **Streaming** — SSE streaming for chat completions and TTS audio
- **Plugin system** — opt-in TTS and STT backends installed as isolated uv workspace packages
- **Multi-GPU & hybrid routing** — assign models to specific GPUs or run them on CPU-only; deploy the same model on both GPU and CPU and requests are load-balanced via round-robin; full tensor parallelism support for large models spanning multiple GPUs
- **Client disconnect detection** — cancels in-flight inference when the client disconnects, freeing GPU resources immediately
- **Built-in observability** — Prometheus metrics, custom `modelship:*` metrics, vLLM engine stats, Ray cluster metrics, structured JSON logging, and OpenTelemetry log export; pre-built Grafana dashboard and alerting rules included

## Supported OpenAI Endpoints

| Endpoint | Usecase |
|---|---|
| `POST /v1/chat/completions` | Chat / text generation (streaming and non-streaming) |
| `POST /v1/responses` | Responses API — text, reasoning and client-driven tool calls (streaming and non-streaming) |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/audio/transcriptions` | Speech-to-text |
| `POST /v1/audio/translations` | Audio translation |
| `POST /v1/audio/speech` | Text-to-speech (SSE streaming or single-response) |
| `POST /v1/images/generations` | Image generation |
| `GET /v1/models` | List available models |

## Quick Start

The fastest way to try Modelship: run a tiny reasoning model on a laptop — no GPU required. Copy-paste this block and you'll have an OpenAI-compatible API on `http://localhost:8000` in a few minutes.

```bash
mkdir -p models-cache && cat > models.yaml <<'EOF'
models:
  - name: reasoning-qwen
    model: "lmstudio-community/Qwen3-0.6B-GGUF:*Q4_K_M.gguf"
    usecase: generate
    loader: llama_server
    num_cpus: 3
    llama_server_config:
      n_ctx: 4096  # Give reasoning space to think
EOF

docker run --rm --shm-size=8g \
  -v ./models.yaml:/modelship/config/models.yaml \
  -v ./models-cache:/.cache \
  -p 8000:8000 \
  ghcr.io/alez007/modelship:latest-cpu
```

Images are multi-arch (amd64 + arm64), so this works on Apple Silicon and ARM Linux hosts too.

Once the server is up (look for `Deployed app 'modelship api' successfully`), call it and watch the model think:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "reasoning-qwen",
    "messages": [{"role": "user", "content": "Which is larger, 9.11 or 9.9?"}]
  }'
```

### GPU (vLLM, Diffusers)

For high-throughput GPU inference, use the standard image and add `--gpus all`. You'll also need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and an `HF_TOKEN` for gated models. Example `models.yaml` entries for vLLM, Diffusers, and multi-GPU setups live in [docs/model-configuration.md](docs/model-configuration.md); ready-to-run configs are in [config/examples/](config/examples/).

```bash
docker run --rm --shm-size=8g --gpus all \
  -e HF_TOKEN=your_token_here \
  -v ./models.yaml:/modelship/config/models.yaml \
  -v ./models-cache:/.cache \
  -p 8000:8000 \
  ghcr.io/alez007/modelship:latest
```

> [!TIP]
> Always set `--shm-size=8g` (or higher) when running the docker container to prevent PyTorch from hitting shared memory limits during multi-process operations.

Hitting an error? Check [docs/troubleshooting.md](docs/troubleshooting.md).

## Plugin Support

Modelship's TTS and STT systems are built around a plugin architecture — each backend is an opt-in package with its own isolated dependencies. Plugins ship inside this repo (`plugins/`) or can be installed from PyPI.

Built-in plugins:

- [Kokoro ONNX](plugins/kokoroonnx/README.md) — lightweight TTS via ONNX Runtime (CPU or GPU)
- [Orpheus](plugins/orpheus/README.md) — expressive TTS
- [whisper.cpp](plugins/whispercpp/README.md) — CPU-only STT via `pywhispercpp`

To enable plugins for local development, pass them as extras at sync time:

```bash
uv sync --extra kokoroonnx
uv sync --extra kokoroonnx --extra whispercpp  # multiple plugins
```

For deployment, plugins are automatically loaded from standalone Python wheels via Ray's `runtime_env` when referenced in `models.yaml`. This ensures that complex backend dependencies don't pollute the main API gateway or other deployments.

For a full guide on writing your own plugin, see [Plugin Development](docs/plugins.md).

## Documentation

- [Development](docs/development.md) — dev environment setup, building, and running locally
- [Model Configuration](docs/model-configuration.md) — full `models.yaml` reference, GPU pinning, environment variables
- [Architecture](docs/architecture.md) — system design, request lifecycle, plugin loading
- [Plugin Development](docs/plugins.md) — writing custom TTS/STT backends
- [Home Assistant Integration](docs/home-assistant.md) — Wyoming protocol setup for voice automation
- [Monitoring & Logging](docs/monitoring.md) — Prometheus metrics, Grafana dashboard, structured logging, health checks
- [Troubleshooting](docs/troubleshooting.md) — common first-run errors and fixes

## Monitoring

Modelship exposes Prometheus metrics (Ray cluster, Ray Serve, vLLM, and custom `modelship:*` metrics) through a single scrape endpoint on port 8079. Metrics are **enabled by default** — set `MSHIP_METRICS=false` to disable. A pre-built [Grafana dashboard](docs/grafana-dashboard.json) and [Prometheus alerting rules](docs/prometheus-alerts.yml) are included in the repository.

Logging supports structured JSON output (`MSHIP_LOG_FORMAT=json`) and request ID correlation across Ray actor boundaries. Logs can be shipped to a remote syslog server (`--log-target syslog://host:514`) or an OpenTelemetry collector (`--otel-endpoint http://collector:4317`). Set `MSHIP_LOG_LEVEL` to `TRACE` for full request/response payloads, or `DEBUG` for detailed diagnostics without payloads.

See [Monitoring & Logging](docs/monitoring.md) for full details.

## Production Readiness

Modelship is actively used and designed for stability in multi-tenant setups. Key guarantees include:

- **Mutex-backed deployments:** A cluster-wide deploy coordinator prevents VRAM exhaustion by ensuring models are never loaded concurrently if resources are tight.
- **Comprehensive HTTP-level tests:** The `tests/test_integration.py` suite validates chat, reasoning, tool-calling, and streaming across all loaders using real (small) models.
- **Payload & concurrency limits:** Built-in safeguards against large payloads (`MSHIP_MAX_REQUEST_BODY_BYTES`) and configurable limits per backend.
- **Observability:** Deep integration with Prometheus, OpenTelemetry, and structured logging.

We are currently hardening the Kubernetes/KubeRay path (a Helm chart ships in [`helm/`](helm/modelship/); GPU-aware probes and gateway-level rate-limiting are next). See the full [Production Readiness Plan](docs/production-readiness.md) for the scorecard and roadmap.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up the dev environment, code style, and submitting pull requests.
