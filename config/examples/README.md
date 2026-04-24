# Example configs

Ready-to-run `models.yaml` configs for common scenarios. Mount one into the container at `/modelship/config/models.yaml` to use it.

The three `gpu-*.yaml` presets are what `easy-run.sh` installs automatically based on detected VRAM — see [docs/quickstart.md](../../docs/quickstart.md).

| File | What it runs | Hardware |
|---|---|---|
| [gpu-8gb.yaml](gpu-8gb.yaml) | Qwen 2.5 3B AWQ + Kokoro TTS + Nomic embed | NVIDIA GPU, 8 GB VRAM |
| [gpu-16gb.yaml](gpu-16gb.yaml) | Qwen 2.5 7B AWQ + Whisper small + Kokoro TTS + Nomic embed | NVIDIA GPU, 16 GB VRAM |
| [gpu-24gb.yaml](gpu-24gb.yaml) | Qwen 2.5 14B AWQ + Whisper large-v3 + Kokoro TTS + Nomic embed + SDXL Turbo | NVIDIA GPU, 24 GB VRAM |
| [llama-cpp.yaml](llama-cpp.yaml) | Quantized GGUF chat + embeddings | CPU (any arch) |
| [transformers-cpu.yaml](transformers-cpu.yaml) | Llama 3.2 1B + Nomic embed + Whisper + MMS-TTS | CPU |
| [vllm.yaml](vllm.yaml) | High-throughput chat with tool calling, embeddings, Whisper | NVIDIA GPU |
| [diffusers.yaml](diffusers.yaml) | SDXL Turbo image generation | NVIDIA GPU |
| [kokoro-tts.yaml](kokoro-tts.yaml) | Kokoro ONNX TTS with GPU + CPU fallback replicas | Mixed |
| [full-stack.yaml](full-stack.yaml) | LLM + TTS + STT + embeddings on one GPU | NVIDIA GPU |
| [mini-pc.yaml](mini-pc.yaml) | Low-resource stack: llama.cpp chat + Kokoro ONNX TTS + whisper.cpp STT | CPU (e.g. Intel N100) |

Example:

```bash
docker run --rm --shm-size=8g \
  -v ./config/examples/llama-cpp.yaml:/modelship/config/models.yaml \
  -v ./models-cache:/.cache \
  -p 8000:8000 \
  ghcr.io/alez007/modelship:latest-cpu
```

See [../../docs/model-configuration.md](../../docs/model-configuration.md) for the full field reference.
