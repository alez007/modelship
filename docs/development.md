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

3. Once inside the container, sync dependencies and start the Ray head node. By default, Ray will auto-detect the available CPUs and GPUs. You can restrict its usage by passing `--num-cpus` and `--num-gpus`.

   ```bash
   # Sync project deps (add --extra <plugin> for each plugin you need)
   uv sync --extra dev

   # Start the Ray head node (auto-detects resources by default)
   ray start --head \
     --port=${RAY_REDIS_PORT} \
     --dashboard-host=0.0.0.0 \
     --disable-usage-stats
   ```

4. Start the server:

   ```bash
   uv run mship_deploy.py
   ```

> **Why the extra steps?** The Dev Container overrides the image's default `CMD` (which normally runs `start.sh` to sync deps and start Ray). Inside a Dev Container you need to run these steps manually.

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
| `RAY_REDIS_PORT` | `6379` | Ray GCS port |
| `RAY_CLUSTER_ADDRESS` | `ray://0.0.0.0` | Ray cluster address |
| `RAY_HEAD_CPU_NUM` | *(unset)* | **Optional override:** CPUs allocated to Ray head. If unset, Ray auto-detects. |
| `RAY_HEAD_GPU_NUM` | *(unset)* | **Optional override:** GPUs allocated to Ray head. If unset, Ray auto-detects. |
| `MSHIP_CACHE_DIR` | `/.cache` | Model cache directory |
| `MSHIP_USE_EXISTING_RAY_CLUSTER` | `false` | Set to `true` to skip starting a Ray head node |
| `MSHIP_GATEWAY_REPLICAS` | `1` | Number of API gateway replicas. Raise to spread request-proxying load across processes under high concurrency. |
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

The container's entrypoint (`start.sh`) automatically starts the Ray head node (auto-detecting resources) and drops into a shell. Then start the server:

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
