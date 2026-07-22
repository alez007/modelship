# Installation

## Requirements

- **Docker** (or Python 3.12+ with `uv` for local development)
- **NVIDIA GPU** (optional) — 16 GB+ VRAM recommended for a full stack (LLM +
  TTS + STT + embeddings) via vLLM; 8 GB is sufficient for lighter setups.
  Not required when using the vLLM or llama.cpp backends on CPU
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)**
  — required only when running GPU models in Docker
- **HuggingFace token** for gated models

## Image variants

Three images are published from a single `Dockerfile`, all under
`ghcr.io/alez007/modelship`:

| Variant | Tag suffix | Platforms | Purpose |
|---|---|---|---|
| Thin | *(none)* — bare tag | amd64, arm64 | Control/coordinator image, no torch/vllm. Runs the driver/head role only — cannot serve models by itself |
| CUDA | `-cuda` | amd64 | GPU node image (vLLM, Diffusers, llama.cpp GPU offload) |
| CPU | `-cpu` | amd64, arm64 | CPU node image (vLLM CPU backend, llama.cpp, Diffusers CPU) |

Floating tags (`:latest`, `:latest-cuda`, `:latest-cpu`) are single-node
only — multi-node deployments must pin every node to the same `X.Y.Z`
(`-cuda`/`-cpu`) tag, or Ray refuses to form the cluster across mismatched
versions.

## Running a single node

CPU, no GPU required:

```bash
docker run --rm --shm-size=8g \
  -v ./models.yaml:/modelship/config/models.yaml \
  -v ./models-cache:/.cache \
  -p 8000:8000 \
  ghcr.io/alez007/modelship:latest-cpu
```

GPU, with the NVIDIA Container Toolkit installed:

```bash
docker run --rm --shm-size=8g --gpus all \
  -e HF_TOKEN=your_token_here \
  -v ./models.yaml:/modelship/config/models.yaml \
  -v ./models-cache:/.cache \
  -p 8000:8000 \
  ghcr.io/alez007/modelship:latest-cuda
```

See [Quickstart](quickstart.md) for a full copy-pasteable `models.yaml` and
walkthrough.

!!! tip
    Always set `--shm-size=8g` (or higher) — Ray falls back to slower
    disk-backed storage instead of `/dev/shm` if the container's shared
    memory is too small for the object store.

## Local development

For building from source, running inside the dev container, or a manual
`uv sync` + `mship_deploy.py` workflow (including the full CLI/env var
reference and port list), see [Development Setup](development.md).

## Scaling beyond one node

To join multiple hosts into one Ray cluster with plain `docker run` (no
Kubernetes), see [Multi-node without Kubernetes](multi-node-docker.md). For a
Kubernetes/KubeRay deployment, see the
[Helm chart](https://github.com/alez007/modelship/tree/main/helm/modelship).
