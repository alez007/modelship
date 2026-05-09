# Roadmap

High-level view of where Modelship is headed. If something here interests you, open an issue or jump into a discussion — contributions are welcome.

## Recently Shipped

- **Reasoning content support** (v0.1.35) — Native `<think>` block extraction and OpenAI-compatible `reasoning_content` field for vLLM, llama_cpp, and transformers loaders.
- **Tool calling support** (v0.1.35) — Cross-loader tool-call parsing and auto-detection for transformers and llama_cpp (GGUF) models.
- **Integration testing suite** (v0.1.35) — Comprehensive HTTP-level tests for chat, reasoning, tool-calling, embeddings, and streaming across all major loaders using real (small) models.
- **Cluster-wide deploy coordinator** (v0.1.35) — Mutex-backed deployment to prevent VRAM races and `--reconcile` support for zero-downtime model hot-reloads.
- **Centralized model resolution** (v0.1.35) — Driver-side resolution of Hugging Face models and GGUF files to simplify worker environment setup.
- **Dynamic wheel-based plugins** (v0.1.32) — plugin packages build into standalone wheels and are injected into Ray workers at deployment via `runtime_env`.

## Up Next

### Core Improvements
- **Detailed health checks** — `/health` should verify model state, GPU status, and Ray cluster connectivity per model
- **Docker Compose** — for simpler non-K8s deployments
- **Helm chart** — Kubernetes manifests with proper GPU scheduling, probes, and resource limits
- **Multi-node Ray cluster setup** — head + workers, networking, failure handling

### Plugin Ecosystem
- **More TTS / STT backends** — (community-contributed)
- **Plugin template / scaffolding CLI** — for faster plugin development
- **Plugin sandboxing** — signature verification or restricted execution environments

### Documentation
- **OpenAPI/Swagger spec** — formal API reference
- **Performance tuning guide** — vLLM engine kwargs, batch sizes, KV cache
- **Capacity planning guide** — model co-location recommendations per GPU size

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started. The **Testing** and **Plugin Ecosystem** sections above are great places to make an impact — most items are self-contained and well-defined.
