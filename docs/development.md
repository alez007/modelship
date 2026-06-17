# Development

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (optional — only required for GPU development)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

## Quick start (Dev Container)

The recommended way to develop Modelship is with VS Code Dev Containers. The configuration in `.devcontainer/` builds the dev image, mounts the repo, forwards ports, and installs all required extensions automatically.

1. Set required environment variables on your host:

   ```bash
   export HF_TOKEN=your_token_here
   ```

2. Open the repo in VS Code and run **Dev Containers: Reopen in Container** from the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`).

3. Once inside the container, sync dependencies:

   ```bash
   # Sync project deps (add --extra <plugin> for each plugin you need)
   uv sync --extra dev
   ```

4. Start the server. It starts its own Ray head, auto-detecting CPUs/GPUs unless `RAY_HEAD_CPU_NUM` / `RAY_HEAD_GPU_NUM` are set:

   ```bash
   uv run mship_deploy.py
   ```

> **Why the extra steps?** The Dev Container overrides the image's default `CMD` (`uv run --no-sync mship_deploy.py`). Inside a Dev Container you sync deps and start it manually.

The Dev Container automatically:
- Builds the dev image from `Dockerfile` (target: `dev`)
- Bind-mounts the repo to `/modelship` for live editing
- Forwards ports `8000` (API) and `8265` (Ray Dashboard)
- Installs extensions: Ruff, Python, Pyright, and Claude Code
- Configures the Python interpreter and linting to use the container's venv at `/.venv`

### Environment variables

The following environment variables are set in the dev image with sensible defaults. Override them in your host shell or in `.devcontainer/devcontainer.json` under `remoteEnv` as needed:

| Variable | Default | Description |
|---|---|---|
| `RAY_HEAD_CPU_NUM` | *(unset)* | **Optional override:** CPUs allocated to Ray head. If unset, Ray auto-detects. |
| `RAY_HEAD_GPU_NUM` | *(unset)* | **Optional override:** GPUs allocated to Ray head. If unset, Ray auto-detects. |
| `MSHIP_CACHE_DIR` | `/.cache` | Model cache directory |
| `MSHIP_STATE_DIR` | `<cache-dir>/state` | Durable effective-config store (this gateway's desired model set). Lives on the cache PVC in k8s so it survives a full cluster loss; `mship_deploy --reconcile` (no `--config`) replays it to self-heal. |
| `MSHIP_USE_EXISTING_RAY_CLUSTER` | `false` | Set to `true` to connect to a Ray cluster you manage (must run on a cluster node) instead of starting one; implies deploy-and-exit |
| `MSHIP_GATEWAY_REPLICAS` | `1` | Number of API gateway replicas. Raise for routing/ingress HA and to spread request-proxying load under high concurrency; replicas keep routing tables in sync via the deploy coordinator's watch loop. |
| `MSHIP_GATEWAY_MAX_ONGOING` | `1024` | Per-replica Ray Serve concurrency cap for the gateway. The gateway holds a slot for the whole lifetime of each streamed response, so a low cap throttles before the engine does. |

### Installing plugin dependencies for IntelliSense

Plugin packages (e.g. `kokoro_onnx`, `onnxruntime`) are only installed when their extra is enabled. If you're working on a plugin and want full IntelliSense, sync the extras inside the container:

```bash
uv sync --extra kokoroonnx --extra dev
```

## Manual setup (without Dev Containers)

If you prefer not to use Dev Containers, you can build and run the dev image directly.

### Building the dev image

**GPU (Standard):**
```bash
docker build -t modelship_dev --target dev .
```

**CPU (Lightweight):**
```bash
docker build -t modelship_dev_cpu --target dev --build-arg MSHIP_VARIANT=cpu .
```

### Running with live source mounting

The dev image does not bake in source files. Mount the repo root so changes take effect without rebuilding:

**GPU:**
```bash
docker run -it --rm --shm-size=8g --gpus all \
  -e HF_TOKEN=your_token_here \
  --mount type=bind,src=./,dst=/modelship \
  -v ./models-cache:/.cache \
  -p 8000:8000 modelship_dev
```

**CPU:**
```bash
docker run -it --rm --shm-size=8g \
  -e HF_TOKEN=your_token_here \
  --mount type=bind,src=./,dst=/modelship \
  -v ./models-cache:/.cache \
  -p 8000:8000 modelship_dev_cpu
```

The dev image drops into a shell. Start the server — it starts its own Ray head, auto-detecting resources:

```bash
uv run mship_deploy.py
```

## Production Builds

To build the production images locally:

**GPU:**
```bash
docker build -t modelship:latest --target prod .
```

**CPU:**
```bash
docker build -t modelship:latest-cpu --target prod --build-arg MSHIP_VARIANT=cpu .
```

## Ports

| Port | Service |
|---|---|
| `8000` | API |
| `8265` | Ray dashboard |
