# Example configs

Ready-to-run `models.yaml` configs for common scenarios. Mount one into the container at `/modelship/config/models.yaml` to use it.

| File | What it runs | Hardware |
|---|---|---|
| [llama-server.yaml](llama-server.yaml) | Quantized GGUF chat (concurrent), vision, embeddings via a llama-server subprocess | CPU or NVIDIA GPU |
| [vllm-cpu.yaml](vllm-cpu.yaml) | Quantized (AWQ/GPTQ) chat via vLLM's CPU backend — the path for gemma tool calling, which llama-server's parsers can't handle | CPU |
| [vllm.yaml](vllm.yaml) | High-throughput chat with tool calling, embeddings, Whisper | NVIDIA GPU |
| [diffusers.yaml](diffusers.yaml) | SDXL Turbo image generation | NVIDIA GPU |
| [kokoro-tts.yaml](kokoro-tts.yaml) | Kokoro ONNX TTS with GPU + CPU fallback replicas | Mixed |
| [full-stack.yaml](full-stack.yaml) | LLM + TTS + STT + embeddings on one GPU | NVIDIA GPU |
| [mini-pc.yaml](mini-pc.yaml) | Low-resource stack: llama-server chat + Kokoro ONNX TTS + whisper.cpp STT | CPU (e.g. Intel N100) |

Example:

```bash
docker run --rm --shm-size=8g \
  -v ./config/examples/llama-server.yaml:/modelship/config/models.yaml \
  -v ./models-cache:/.cache \
  -p 8000:8000 \
  ghcr.io/alez007/modelship:latest-cpu
```

See [../../docs/model-configuration.md](../../docs/model-configuration.md) for the full field reference.
