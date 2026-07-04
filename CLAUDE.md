# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` is the canonical operational guide — read it first for toolchain, commands, gotchas, release flow, and plugin authoring. This file summarizes the points most often needed mid-task.

## Toolchain essentials

- Python is pinned exactly to `3.12.10` (not `>=`). Dependency manager is **uv** with a workspace; `plugins/*` are workspace members. Never use `pip install`.
- `gpu` and `cpu` extras are **mutually exclusive** (declared in `[tool.uv] conflicts`) — `torch`/`torchvision` come from different indexes per extra.
- Line length is **120**, not 88. Ruff owns formatting (`E501` disabled); don't hand-sort imports (isort via `I` rule handles it). `plugins/*` are third-party to isort; `modelship` is first-party.
- Pyright runs in `basic` mode, scoped to `modelship`, `plugins`, `mship_deploy.py`. Pre-commit only runs ruff — don't rely on it to catch type errors.

## Common commands

```bash
# Install (choose gpu XOR cpu, plus dev, plus optional plugin extras)
uv sync --extra dev --extra gpu                        # what CI uses
uv sync --extra dev --extra cpu --extra kokoroonnx     # CPU + a plugin

make lint        # ruff check + ruff format --check + pyright — all three MUST pass
make lint-fix    # auto-fix ruff issues
make test        # uv run pytest tests/ -v

# Single test
uv run pytest tests/test_config.py::TestLlamaServerConfig::test_defaults -v
```

CI mirrors `make lint` + `pytest tests/ -v`. Match it locally before pushing.

`make lint` requires the `gpu` extra — pyright fails with `reportMissingImports` for `gguf`, `diffusers`, and `psutil` under the cpu sync. (`vllm` is now importable under both extras — Stage E0 wired a CPU wheel index — but that alone doesn't unblock a cpu-only lint.) Tests pass on either extra.

When running tests on your own initiative, skip the slow integration suite: `uv run pytest tests/ -v -m "not integration"`. Only run full `make test` when explicitly requested.

## Running the server

`mship_deploy.py` is the entry point (not a console script, not `python -m`). It:

1. Reads `config/models.yaml` (gitignored — copy from `config/examples/`; `mship_deploy.py` errors out pointing there if missing).
2. Starts its **own** Ray head by default and tears it down on exit. With `--use-existing-ray-cluster` it instead connects to a cluster you manage via `ray.init(address="auto")` and deploys-and-exits without teardown — the driver must run **on** a cluster node (it can't attach from off-cluster; k8s does this via a KubeRay RayJob).
3. Deploys models **additively** by default (each gets a random suffix like `qwen-a3f9k`). Use `--reconcile` to instead make the cluster match the config exactly (add/remove/replace) — it never tears the cluster down.
4. Starts a FastAPI Ray Serve app named `modelship api` on port 8000. Override via `--gateway-name` (multiple gateways can coexist on one cluster).

Docker's `CMD` is `uv run --no-sync mship_deploy.py` (auto-detecting CPUs/GPUs unless `RAY_HEAD_CPU_NUM`/`RAY_HEAD_GPU_NUM` set), against the prebuilt venv (extras chosen by `--build-arg MSHIP_VARIANT=gpu|cpu`). Plugin wheels in `MSHIP_PLUGIN_WHEEL_DIR` ship to Ray workers per-deployment via `runtime_env`, resolved from `models.yaml`. The Dev Container overrides this — inside it you must `uv sync` and run `mship_deploy.py` manually.

## Architecture map

- `mship_deploy.py` — Ray init + deploy loop. `build_deployment_options` (in `modelship/deploy/actor_options.py`) handles GPU allocation: multi-slot vLLM deploys (`tp*pp > 1`) always build a Ray Serve placement group (one whole-GPU bundle per slot, STRICT_PACK) that vLLM's ray executor inherits via `get_current_placement_group()`. Single-slot deploys use a scalar `num_gpus` on the outer actor (fractional sharing supported). Fractional `num_gpus` with `tp*pp > 1` is rejected at config time — Ray packs fractional PG bundles onto the same physical GPU. `llama_server` loader supports whole-GPU offload (`num_gpus` must be `0` or an integer — fractional is rejected, llama.cpp has no VRAM-fraction knob); `stable_diffusion_cpp` remains CPU-only, forced to `num_gpus: 0`.
- `modelship/openai/api.py` — FastAPI gateway. Uses `RequestWatcher` + a single shared `DisconnectRegistry` Ray actor (keyed by request id) to propagate client disconnects across process boundaries and cancel in-flight inference.
- `modelship/infer/model_deployment.py` — the single `@serve.deployment` actor class; lazily imports the right backend from `config.loader`.
- `modelship/infer/infer_config.py` — pydantic config schemas plus `RawRequestProxy` / `DisconnectRegistry`. `RawRequestProxy` exists because FastAPI `Request` can't cross Ray process boundaries. **Any new attribute vLLM reads from `raw_request` must be added there.**
- `modelship/infer/{vllm,diffusers,custom}/` — one subdir per loader, each with an `*_infer.py` and (for non-custom) an `openai/` adapter subpackage. `modelship/infer/llama_server/llama_server_infer.py` is the exception: a single flat file (no `openai/` subpackage) — it proxies a `llama-server` subprocess's own OpenAI-compatible HTTP API rather than running modelship's parsers in-process.
- `modelship/plugins/base_plugin.py` — `BasePlugin` ABC that each plugin package subclasses as `ModelPlugin`.
- `plugins/*` — workspace packages, each opt-in via a matching root extra. The plugin module name and extra name **must match** (`ensure_plugin()` does `importlib.import_module(config.plugin)`).

Multiple deployments with the same model name are round-robin load-balanced by the gateway. Each deployment can also scale with `num_replicas` via Ray Serve.

## Tests

Under `tests/`, `pytest-asyncio` for async. Tests **mock out Ray Serve** — they don't spin up a real cluster. Pattern: access the wrapped class via `ModelshipAPI.func_or_class` to bypass the `@serve.deployment` wrapper (see `tests/test_api.py`). There are no GPU/real-model integration tests; keep it that way unless added behind an opt-in marker.

## Releases

`make release-{patch,minor,major}` is the only supported path — refuses off `main` or dirty tree, bumps `pyproject.toml`, runs `uv lock`, generates CHANGELOG from Conventional Commits, commits, tags `vX.Y.Z`, pushes. `release.yml` then publishes Docker + PyPI. **Don't bump versions by hand.** Commit message prefixes (`feat:`, `fix:`, `refactor:|perf:|docs:|chore:|build:|ci:|style:|test:`) are parsed into CHANGELOG sections, so use them.

## Sharp edges

- `vllm==0.24.0` is pinned. Don't bump casually — TP scheduling in `mship_deploy.py:build_deployment_options` defaults to the Ray V2 executor, and the loader binds to vLLM-internal `entrypoints.*` module paths that upstream restructures between minors (the `vllm_infer.py` imports moved in 0.22/0.23).
- **GGUF is unsupported on the `vllm` loader.** 0.24 moved GGUF out of tree, and the only external `vllm-gguf-plugin` (`0.0.2`) has a stale `override_quantization_method` signature that breaks *every* quantized model on 0.24 — so it's deliberately not installed. `resolve_all_model_sources` (in `deploy/config.py`) rejects a `.gguf` on the vllm loader at driver preflight with a pointer to `llama_server`. Use `loader: llama_server` for GGUF; the vllm loader takes safetensors or AWQ/GPTQ/FP8 quants. **This is unconditional on GPU vs. CPU** — the CPU wheel (below) doesn't relax it either; a GGUF gemma still needs `llama_server`, or a non-GGUF checkpoint to run on `vllm`.
- `vllm==0.24.0+cpu` is installable on the `cpu` extra via an explicit `vllm-cpu` index (`wheels.vllm.ai/0.24.0/cpu`, scoped through `tool.uv.sources`) — the URL embeds the vLLM version, so a future bump must update it. On vLLM's CPU backend, `gpu_memory_utilization` is repurposed to mean *fraction of host RAM* reserved for KV cache, not VRAM; `normalize_num_gpus_and_tp` (`infer_config.py`) lowers its default to `0.4` for `num_gpus: 0` vllm deploys so a naive CPU config doesn't ask to reserve 90% of node RAM and crash at worker init — an explicit value always overrides it. See `config/examples/vllm-cpu.yaml`.
- Metrics live on port **8079** (not 8000). `MSHIP_METRICS=false` or `--no-metrics` disables. When `mship_deploy` starts its own head (no `--use-existing-ray-cluster`), `connect_ray` pins that port via `ray.init(_metrics_export_port=…)` — a **private** Ray kwarg (accepted through `**kwargs`). A `TestConnectRay` test guards it so a Ray bump that drops it fails loudly.
- `TRACE` is a custom log level below `DEBUG`; it logs full request/response payloads.
- Docker CPU image uses the unified `Dockerfile` with `--build-arg MSHIP_VARIANT=cpu` and has a `:latest-cpu` tag suffix.
- `llama_server` loader launches a `llama-server` subprocess (found via `MSHIP_LLAMA_SERVER_BIN`, pinned in the Docker images at `/opt/llama.cpp`) and proxies its native OpenAI API — concurrency comes from `--parallel` slots instead of a single `asyncio.Lock`. `num_gpus` must be `0` or a whole integer (fractional is rejected, llama.cpp has no VRAM-fraction knob). Tool-call/reasoning parsing is llama-server's own, auto-detected per chat template: named-function `tool_choice` forcing is globally unsupported (silently falls back to `auto`), and `tool_choice: required` is grammar-enforced for harmony-style templates but a silent no-op for hermes-style ones (e.g. Qwen3); bare `response_format: {"type": "json_object"}` (no `schema` key) is also unenforced despite llama-server's own docs claiming support (verified against the b9859 binary directly) — `type: json_schema` requests, which is what modelship actually sends whenever a schema is given, are unaffected and correctly constrained. See `docs/model-configuration.md`'s llama_server section. No persistent on-disk prompt cache.

## Further reading

- `AGENTS.md` — the full operational guide
- `docs/architecture.md` — request lifecycle, loaders, plugin system
- `docs/development.md` — dev-container + manual-Docker setup, env vars
- `docs/model-configuration.md` — `models.yaml` reference
- `docs/plugins.md` — plugin authoring
- `config/examples/` — working `models.yaml` files for each backend
