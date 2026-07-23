# Modelship

Modelship runs the AI stack your agents call — chat, the **Responses API** with
server-side conversation state (durable with Redis), universal **tool calling**,
and **reasoning**, alongside embeddings, speech, and image generation — behind
one OpenAI-compatible endpoint on your own GPUs (or CPU).

Built on [Ray Serve](https://docs.ray.io/en/latest/serve/index.html): state is
shared across gateway replicas, deploys are declarative, and everything is
observable. Point the OpenAI SDK at it and your agent runs unchanged — private,
with no per-token bill.

[Get started :material-arrow-right:](quickstart.md){ .md-button .md-button--primary }
[View on GitHub :fontawesome-brands-github:](https://github.com/alez007/modelship){ .md-button }

## Why Modelship?

- **Agent state that isn't siloed per replica** — the `/v1/responses` API with
  reasoning, universal tool/function calling, and server-side conversation
  state (`previous_response_id`) live in one pluggable store shared by every
  gateway replica — in-memory by default, or Redis for durability across
  restarts and node failure. Works across both the vLLM and llama.cpp
  (`llama_server`) loaders.
- **Everything an agent app calls, one endpoint** — chat, embeddings for RAG,
  speech-to-text, text-to-speech, and image generation, all behind a single
  OpenAI-compatible `/v1` surface. No juggling separate services for each
  modality.
- **Drop-in OpenAI, on your hardware** — any OpenAI SDK client works out of
  the box. Point it at Modelship instead of the OpenAI API and your agent
  code doesn't change — it just runs privately, on infrastructure you
  control.
- **GPU memory control** — allocate exact GPU fractions per model (e.g. 70%
  for the LLM, 5% for TTS) so a full stack fits on hardware you already own.
- **Mix and match backends** — vLLM for high-throughput GPU or CPU inference,
  llama.cpp for efficient quantized GGUF models, Diffusers for images, and a
  plugin system for custom backends — in the same deployment.

## Architecture

![Modelship architecture: an agent app calls the Modelship gateway's OpenAI-compatible API, which exposes chat, embeddings, audio, and image endpoints plus a Responses API backed by a shared conversation-state store, routing round-robin to Ray Serve deployments across GPU and CPU cluster nodes.](assets/architecture-light.svg#only-light)
![Modelship architecture: an agent app calls the Modelship gateway's OpenAI-compatible API, which exposes chat, embeddings, audio, and image endpoints plus a Responses API backed by a shared conversation-state store, routing round-robin to Ray Serve deployments across GPU and CPU cluster nodes.](assets/architecture-dark.svg#only-dark)

Each model runs as an isolated [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)
deployment with its own lifecycle, health checks, and resource budget.

| Backend | Best for | GPU required |
|---|---|---|
| **vLLM** | High-throughput chat, embeddings, transcription | No — installs on GPU or CPU |
| **llama.cpp** (`llama_server`) | High-efficiency quantized GGUF models (chat, embeddings, vision) | No |
| **Diffusers** | Image generation | Yes |
| **Custom (plugins)** | TTS backends (Kokoro ONNX, Orpheus), STT backends (whisper.cpp) | No |

Models can be deployed across multiple GPUs, run on CPU-only, or both —
multiple deployments of the same model (e.g. one on GPU via vLLM, one on CPU
via vLLM or llama.cpp) are load-balanced with round-robin routing. Each
deployment can also scale horizontally with `num_replicas`, and the gateway
itself scales with `--gateway-replicas`. See [Architecture](architecture.md)
for the full request lifecycle and design.

## Supported OpenAI Endpoints

| Endpoint | Usecase |
|---|---|
| `POST /v1/chat/completions` | Chat / text generation (streaming and non-streaming) |
| `POST /v1/responses` | Responses API — text, reasoning, client-driven tool calls, and stored conversations (streaming and non-streaming) |
| `GET`/`DELETE /v1/responses/{id}` | Fetch or drop a stored response (`/input_items` lists its input) |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/audio/transcriptions` | Speech-to-text |
| `POST /v1/audio/translations` | Audio translation |
| `POST /v1/audio/speech` | Text-to-speech (SSE streaming or single-response) |
| `POST /v1/images/generations` | Image generation |
| `GET /v1/models` | List available models |

## Next steps

- [Quickstart](quickstart.md) — a tiny reasoning model running in a few minutes, no GPU required
- [Installation](installation.md) — requirements, image variants, and running with a GPU
- [Model Configuration](model-configuration.md) — the full `models.yaml` reference
- [Integrations](integrations/index.md) — connecting the OpenAI SDK, Open WebUI, Dify, n8n, and Responses-speaking agents
