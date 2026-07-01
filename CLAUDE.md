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
uv run pytest tests/test_config.py::TestLlamaCppConfig::test_defaults -v
```

CI mirrors `make lint` + `pytest tests/ -v`. Match it locally before pushing.

`make lint` requires the `gpu` extra — pyright fails with `reportMissingImports` for `vllm`, `gguf`, `diffusers`, and `psutil` under the cpu sync. Tests pass on either extra.

When running tests on your own initiative, skip the slow integration suite: `uv run pytest tests/ -v -m "not integration"`. Only run full `make test` when explicitly requested.

## Running the server

`mship_deploy.py` is the entry point (not a console script, not `python -m`). It:

1. Reads `config/models.yaml` (gitignored — copy from `config/examples/`; `mship_deploy.py` errors out pointing there if missing).
2. Starts its **own** Ray head by default and tears it down on exit. With `--use-existing-ray-cluster` it instead connects to a cluster you manage via `ray.init(address="auto")` and deploys-and-exits without teardown — the driver must run **on** a cluster node (it can't attach from off-cluster; k8s does this via a KubeRay RayJob).
3. Deploys models **additively** by default (each gets a random suffix like `qwen-a3f9k`). Use `--reconcile` to instead make the cluster match the config exactly (add/remove/replace) — it never tears the cluster down.
4. Starts a FastAPI Ray Serve app named `modelship api` on port 8000. Override via `--gateway-name` (multiple gateways can coexist on one cluster).

Docker's `CMD` is `uv run --no-sync mship_deploy.py` (auto-detecting CPUs/GPUs unless `RAY_HEAD_CPU_NUM`/`RAY_HEAD_GPU_NUM` set), against the prebuilt venv (extras chosen by `--build-arg MSHIP_VARIANT=gpu|cpu`). Plugin wheels in `MSHIP_PLUGIN_WHEEL_DIR` ship to Ray workers per-deployment via `runtime_env`, resolved from `models.yaml`. The Dev Container overrides this — inside it you must `uv sync` and run `mship_deploy.py` manually.

## Architecture map

- `mship_deploy.py` — Ray init + deploy loop. `build_deployment_options` (in `modelship/deploy/actor_options.py`) handles GPU allocation: multi-slot vLLM deploys (`tp*pp > 1`) always build a Ray Serve placement group (one whole-GPU bundle per slot, STRICT_PACK) that vLLM's ray executor inherits via `get_current_placement_group()`. Single-slot deploys use a scalar `num_gpus` on the outer actor (fractional sharing supported). Fractional `num_gpus` with `tp*pp > 1` is rejected at config time — Ray packs fractional PG bundles onto the same physical GPU. `llama_cpp` loader supports whole-GPU offload (`num_gpus` must be `0` or an integer — fractional is rejected, llama.cpp has no VRAM-fraction knob); `stable_diffusion_cpp` remains CPU-only, forced to `num_gpus: 0`.
- `modelship/openai/api.py` — FastAPI gateway. Uses `RequestWatcher` + a single shared `DisconnectRegistry` Ray actor (keyed by request id) to propagate client disconnects across process boundaries and cancel in-flight inference.
- `modelship/infer/model_deployment.py` — the single `@serve.deployment` actor class; lazily imports the right backend from `config.loader`.
- `modelship/infer/infer_config.py` — pydantic config schemas plus `RawRequestProxy` / `DisconnectRegistry`. `RawRequestProxy` exists because FastAPI `Request` can't cross Ray process boundaries. **Any new attribute vLLM reads from `raw_request` must be added there.**
- `modelship/infer/{vllm,transformers,diffusers,llama_cpp,custom}/` — one subdir per loader, each with an `*_infer.py` and (for non-custom) an `openai/` adapter subpackage.
- `modelship/plugins/base_plugin.py` — `BasePlugin` ABC that each plugin package subclasses as `ModelPlugin`.
- `plugins/*` — workspace packages, each opt-in via a matching root extra. The plugin module name and extra name **must match** (`ensure_plugin()` does `importlib.import_module(config.plugin)`).

Multiple deployments with the same model name are round-robin load-balanced by the gateway. Each deployment can also scale with `num_replicas` via Ray Serve.

## Tests

Under `tests/`, `pytest-asyncio` for async. Tests **mock out Ray Serve** — they don't spin up a real cluster. Pattern: access the wrapped class via `ModelshipAPI.func_or_class` to bypass the `@serve.deployment` wrapper (see `tests/test_api.py`). There are no GPU/real-model integration tests; keep it that way unless added behind an opt-in marker.

## Releases

`make release-{patch,minor,major}` is the only supported path — refuses off `main` or dirty tree, bumps `pyproject.toml`, runs `uv lock`, generates CHANGELOG from Conventional Commits, commits, tags `vX.Y.Z`, pushes. `release.yml` then publishes Docker + PyPI. **Don't bump versions by hand.** Commit message prefixes (`feat:`, `fix:`, `refactor:|perf:|docs:|chore:|build:|ci:|style:|test:`) are parsed into CHANGELOG sections, so use them.

## Sharp edges

- `vllm==0.24.0` is pinned. Don't bump casually — TP scheduling in `mship_deploy.py:build_deployment_options` defaults to the Ray V2 executor, and the loader binds to vLLM-internal `entrypoints.*` module paths that upstream restructures between minors (the `vllm_infer.py` imports moved in 0.22/0.23).
- **GGUF is unsupported on the `vllm` loader.** 0.24 moved GGUF out of tree, and the only external `vllm-gguf-plugin` (`0.0.2`) has a stale `override_quantization_method` signature that breaks *every* quantized model on 0.24 — so it's deliberately not installed. `resolve_all_model_sources` (in `deploy/config.py`) rejects a `.gguf` on the vllm loader at driver preflight with a pointer to `llama_cpp`. Use `loader: llama_cpp` for GGUF; the vllm loader takes safetensors or AWQ/GPTQ/FP8 quants.
- `llama_cpp` GPU support relies on the prebuilt CUDA wheel from the `llama-cpp-cu130` index (`abetlen.github.io/llama-cpp-python/whl/cu130`), scoped to the `gpu` extra via `tool.uv.sources`. The `cpu` extra has no source entry, so it resolves the plain PyPI sdist (built from source) as before. Both extras pin the same `llama-cpp-python` floor so they stay in lockstep.
- The cu130 wheel's `libllama.so` statically links `libcuda.so.1` (the real NVIDIA driver library), unlike torch/vllm which resolve CUDA lazily — so merely `import llama_cpp` crashes on any driverless machine, including CI's `ubuntu-latest` runner. `ci.yml`'s `test` job installs NVIDIA's `cuda-driver-dev` stub `libcuda.so` (via a pinned `.deb`, extracted with `dpkg-deb`, no apt repo added) and symlinks it to `libcuda.so.1` on `LD_LIBRARY_PATH` — its `cuInit()` returns `CUDA_ERROR_STUB_LIBRARY` instead of crashing, which is enough for the import to succeed. If that pinned `.deb` URL ever 404s, grab a current `cuda-driver-dev-13-*` build from `developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/`.
- Metrics live on port **8079** (not 8000). `MSHIP_METRICS=false` or `--no-metrics` disables. When `mship_deploy` starts its own head (no `--use-existing-ray-cluster`), `connect_ray` pins that port via `ray.init(_metrics_export_port=…)` — a **private** Ray kwarg (accepted through `**kwargs`). A `TestConnectRay` test guards it so a Ray bump that drops it fails loudly.
- `TRACE` is a custom log level below `DEBUG`; it logs full request/response payloads.
- Docker CPU image uses the unified `Dockerfile` with `--build-arg MSHIP_VARIANT=cpu` and has a `:latest-cpu` tag suffix.

## Further reading

- `AGENTS.md` — the full operational guide
- `docs/architecture.md` — request lifecycle, loaders, plugin system
- `docs/development.md` — dev-container + manual-Docker setup, env vars
- `docs/model-configuration.md` — `models.yaml` reference
- `docs/plugins.md` — plugin authoring
- `config/examples/` — working `models.yaml` files for each backend
