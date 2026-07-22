# Quickstart

The fastest way to try Modelship: run a tiny reasoning model on a laptop — no
GPU required. Copy-paste this block and you'll have an OpenAI-compatible API
on `http://localhost:8000` in a few minutes.

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

Images are multi-arch (amd64 + arm64), so this works on Apple Silicon and ARM
Linux hosts too.

Once the server is up (look for `Deployed app 'modelship api' successfully`),
call the **Responses API** and watch the model think:

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "reasoning-qwen",
    "input": "Which is larger, 9.11 or 9.9?"
  }'
```

The response includes both `output_text` and a first-class `reasoning` output
item — the same server-side conversation state (`previous_response_id`) and
tool-calling support work here as they do on GPU-backed models.
`/v1/chat/completions` remains available too, if that's what your client
speaks.

## GPU (vLLM, Diffusers)

For high-throughput GPU inference, use the `-cuda` image and add `--gpus
all`. You'll also need the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
and an `HF_TOKEN` for gated models. Example `models.yaml` entries for vLLM,
Diffusers, and multi-GPU setups live in
[Model Configuration](model-configuration.md); ready-to-run configs are in
[`config/examples/`](https://github.com/alez007/modelship/tree/main/config/examples).

```bash
docker run --rm --shm-size=8g --gpus all \
  -e HF_TOKEN=your_token_here \
  -v ./models.yaml:/modelship/config/models.yaml \
  -v ./models-cache:/.cache \
  -p 8000:8000 \
  ghcr.io/alez007/modelship:latest-cuda
```

!!! note
    `ghcr.io/alez007/modelship:latest` (bare tag, no suffix) is the **thin**
    control/coordinator image — no torch/vllm, for a driver/head role only.
    It cannot serve models by itself; always use `-cuda` or `-cpu` to
    actually run inference. See [Installation](installation.md) for the full
    three-image breakdown.

!!! tip
    Always set `--shm-size=8g` (or higher) when running the docker container
    to prevent PyTorch from hitting shared memory limits during
    multi-process operations.

Hitting an error? Check [Troubleshooting](troubleshooting.md).

## Next up

Point an OpenAI SDK client at it — see [Integrations](integrations/index.md)
— or scale it out across a cluster with
[Multi-node without Kubernetes](multi-node-docker.md).
