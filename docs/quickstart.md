# Quick Start (GPU)

Zero-config launcher for Modelship on a single NVIDIA GPU. Detects your VRAM,
picks a matching preset, and starts the stack with Docker Compose.

## Requirements

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose v2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- One NVIDIA GPU with **8 GB, 16 GB, or 24 GB** of VRAM

## One command

```bash
./easy-run.sh
```

That's it. The script:

1. Reads your GPU's VRAM via `nvidia-smi`.
2. Copies the matching preset from `config/examples/gpu-{8,16,24}gb.yaml` to `config/models.yaml`.
3. Runs `docker compose up -d`, which auto-pulls the image, installs the Kokoro TTS plugin, and starts the API gateway.

First run downloads model weights into `./models-cache` (can take several minutes).

Once you see `Deployed app 'modelship api' successfully` in the logs:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llm", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## What each preset runs

| Preset | Chat | STT | TTS | Embeddings |
|---|---|---|---|---|
| **8 GB**  | Qwen 2.5 3B AWQ | —                | Kokoro 82M | Nomic Embed v1.5 |
| **16 GB** | Qwen 2.5 7B AWQ | Whisper small    | Kokoro 82M | Nomic Embed v1.5 |
| **24 GB** | Qwen 2.5 7B AWQ | Whisper large-v3 | Kokoro 82M | Nomic Embed v1.5 |

The GPU budget (`num_gpus` fractions) in each preset is tuned to leave headroom for the vLLM KV cache and CUDA context overhead (each process reserves ~500 MiB outside its fractional budget). Edit `config/models.yaml` afterwards to swap models, add image generation, or adjust budgets.

## Monitoring startup

First run downloads tens of GB of weights and compiles vLLM CUDA kernels on first engine init — expect 10–30 minutes depending on preset and connection. Subsequent runs reuse `./models-cache` and are typically 1–2 min.

Watch readiness + per-model load times without tailing raw logs:

```bash
./easy-test.sh --timeout 1800
```

`easy-test.sh` polls `GET /status` — which returns 503 with a JSON body listing loaded/pending models while loading, and 200 with `time_to_ready_s` + `model_load_times_s` once ready. It also runs smoke tests against every configured usecase (chat streaming, embeddings, TTS round-tripped into STT).

Useful flags:

```bash
./easy-test.sh --only health,models          # just readiness + /v1/models
./easy-test.sh --skip stt,image              # skip suites that need long-running models
./easy-test.sh --verbose                     # echo curl request/response bodies
```

## Useful commands

```bash
./easy-run.sh --preset 16gb     # skip detection, force a preset
./easy-run.sh --force           # overwrite config/models.yaml
./easy-run.sh --dry-run         # show plan without starting anything

docker compose logs -f modelship   # tail logs
docker compose down                # stop everything
docker compose restart modelship   # restart after editing config/models.yaml
```

Ports exposed: `8000` (OpenAI API), `8079` (Prometheus metrics), `8265` (Ray dashboard).

## Gated models

Most presets use open models and don't need a token. If you swap in a gated one (e.g. Llama), set `HF_TOKEN` before running:

```bash
export HF_TOKEN=hf_...
./easy-run.sh
```

## Troubleshooting

- **`nvidia-smi not found`** — install the NVIDIA driver and Container Toolkit, or pass `--preset <size>` to skip detection.
- **VRAM < 8 GB** — use the CPU-only example instead: `config/examples/mini-pc.yaml` (see [main README](../README.md#quick-start)).
- **OOM on startup** — the presets leave headroom, but heavy context use can still OOM. Lower `max_model_len` or `num_gpus` in `config/models.yaml` and `docker compose restart modelship`.
- **Other errors** — see [troubleshooting.md](troubleshooting.md).
