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

## arm64 vs amd64 image selection

`ghcr.io/alez007/modelship:latest` (thin) and `:latest-cpu` are multi-arch (amd64 + arm64). Docker picks the right one automatically for your host. If you need to force an arch (e.g. cross-building), use `--platform linux/arm64` or `linux/amd64`.

`:latest-cuda` is **amd64-only** — the Dockerfile hard-wires the x86_64 CUDA apt repo and torch's CUDA wheels aren't guaranteed for arm64 at this pin. arm64+CUDA hosts (Jetson, GH200) aren't supported by this image; use `:latest-cpu` there, or build a custom image. Apple Silicon should always use `:latest-cpu` (no CUDA path applies).

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
- The Ray dashboard is **always on**, publish port `8265` to reach it. It binds to `127.0.0.1` inside the container by default — set `MSHIP_RAY_DASHBOARD=0.0.0.0` (or a specific interface) to expose it beyond the container. This is the exposure vector behind ShadowRay/CVE-2023-48022, so only do this on a trusted/private network. Prometheus metrics on `8079` are exported regardless.
- Ray cluster authentication is **off by default**. Pass `--ray-auth=token` when modelship starts its own head to require a bearer token for the dashboard and cluster-internal RPC — the dashboard UI will then ask for one on first load; retrieve it with `docker exec <container> cat /home/modelship/.ray/auth_token` and paste it in once. The OpenAI API on `8000` and Prometheus metrics on `8079` are never gated by this either way.
