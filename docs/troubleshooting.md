# Troubleshooting

Common issues hit during first-run and deployment.

## `HF_TOKEN` not set / 401 on gated models

Some HuggingFace models (Llama 3, Gemma, Mistral variants) require accepting a license and authenticating. Get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), accept the model license on its HF page, then pass the token in:

```bash
docker run ... -e HF_TOKEN=hf_xxx ghcr.io/alez007/modelship:latest-cpu
```

Ungated models (e.g. `lmstudio-community/Qwen2.5-7B-Instruct-GGUF`) don't need a token.

## Permission denied on `/.cache`

The container runs as a non-root user (since v0.1.23). If you're mounting a host directory to `/.cache`, make sure it's writable by UID 1000, or let Docker create it fresh:

```bash
mkdir -p models-cache && chmod 777 models-cache
```

If you previously used the old `/root/.cache/huggingface` mount path, switch to `/.cache` and move any cached weights across — the container no longer looks at the old location.

## `shm-size` too small / Ray crashes at startup

Ray's object store needs shared memory. Always pass `--shm-size=8g` (or larger for big models). Without it, you'll see Ray worker crashes or silent hangs during deployment.

## `n_gpu_layers` ignored with llama_cpp loader

The `llama_cpp` loader currently runs CPU-only in this build — any `n_gpu_layers` value is forced to `0` and a warning is logged. For GPU inference, use the `vllm` loader instead. Tracked for a future release.

## arm64 vs amd64 image selection

Both `ghcr.io/alez007/modelship:latest` and `:latest-cpu` are multi-arch since v0.1.25. Docker picks the right one automatically for your host. If you need to force an arch (e.g. cross-building), use `--platform linux/arm64` or `linux/amd64`.

Note that the GPU (`latest`) image on arm64 will only be useful on arm64 hosts with NVIDIA GPUs (e.g. Jetson, GH200) — not Apple Silicon. For Apple Silicon, use `latest-cpu`.

## Port 8000 already in use

Another service is bound to `8000`. Either free it up or remap:

```bash
docker run ... -p 8001:8000 ...   # exposed on host:8001
```

## Model download is slow / stalls

Weights are cached to `/.cache/huggingface` inside the container. Mount a persistent host directory (`-v ./models-cache:/.cache`) so subsequent runs reuse them. For large models, set a longer `docker run` timeout or pre-pull with `huggingface-cli download`.

## `CUDA out of memory` with vLLM

vLLM reserves VRAM based on `num_gpus` (fraction of one GPU). If a single model uses more than its budget, lower `num_gpus` for other deployments, or set `vllm_engine_kwargs.max_model_len` to cap KV cache size.

## Can't reach the server from another host

The API binds to `0.0.0.0:8000` by default, but if you're on a remote machine, make sure the port is reachable through your firewall and you're using the host's IP, not `localhost`.

## Getting more diagnostic detail

- Set `MSHIP_LOG_LEVEL=DEBUG` for verbose logs.
- Set `MSHIP_LOG_LEVEL=TRACE` to log full request/response payloads (and enable llama.cpp `verbose` mode).
- The Ray dashboard is **disabled by default** to save host RAM. Start the container with `MSHIP_RAY_DASHBOARD=true` (and publish port `8265`) to get per-actor logs and resource usage in the UI. Prometheus metrics on `8079` are exported regardless.
