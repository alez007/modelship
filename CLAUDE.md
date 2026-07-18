# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` is the canonical operational guide — read it first for toolchain, commands, gotchas, release flow, and plugin authoring. This file summarizes the points most often needed mid-task.

## Toolchain essentials

- Python is pinned exactly to `3.12.10` (not `>=`). Dependency manager is **uv** with a workspace; `plugins/*` are workspace members. Never use `pip install`.
- `cuda` and `cpu` extras are **mutually exclusive** (declared in `[tool.uv] conflicts`) — `torch`/`torchvision` come from different indexes per extra. A third extra, `thin`, is empty — no torch/vllm.
- Line length is **120**, not 88. Ruff owns formatting (`E501` disabled); don't hand-sort imports (isort via `I` rule handles it). `plugins/*` are third-party to isort; `modelship` is first-party.
- Pyright runs in `basic` mode, scoped to `modelship`, `plugins`, `mship_deploy.py`. Pre-commit only runs ruff — don't rely on it to catch type errors.

## Common commands

```bash
# Install (choose cuda XOR cpu, plus dev, plus optional plugin extras)
uv sync --extra dev --extra cuda                       # what CI uses
uv sync --extra dev --extra cpu --extra kokoroonnx     # CPU + a plugin

make lint        # ruff check + ruff format --check + pyright — all three MUST pass
make lint-fix    # auto-fix ruff issues
make test        # uv run pytest tests/ -v

# Single test
uv run pytest tests/test_config.py::TestLlamaServerConfig::test_defaults -v
```

CI mirrors `make lint` + `pytest tests/ -v`. Match it locally before pushing.

`make lint` requires the `cuda` extra — pyright fails with `reportMissingImports` for `gguf`, `diffusers`, and `psutil` under the cpu sync. (`vllm` is now importable under both extras — Stage E0 wired a CPU wheel index — but that alone doesn't unblock a cpu-only lint.) Tests pass on either extra.

When running tests on your own initiative, skip the slow integration suite: `uv run pytest tests/ -v -m "not integration"`. Only run full `make test` when explicitly requested.

## Running the server

`mship_deploy.py` is the entry point (not a console script, not `python -m`). It:

1. Reads `config/models.yaml` (gitignored — copy from `config/examples/`; `mship_deploy.py` errors out pointing there if missing).
2. Starts its **own** Ray head by default and tears it down on exit. With `--use-existing-ray-cluster` it instead connects to a cluster you manage via `ray.init(address="auto")` and deploys-and-exits without teardown — the driver must run **on** a cluster node (it can't attach from off-cluster; k8s does this via a KubeRay RayJob).
3. Deploys models **additively** by default (each gets a random suffix like `qwen-a3f9k`). Use `--reconcile` to instead make the cluster match the config exactly (add/remove/replace) — it never tears the cluster down.
4. Starts a FastAPI Ray Serve app named `modelship api` on port 8000. Override via `--gateway-name` (multiple gateways can coexist on one cluster).

Docker's `CMD` is `uv run --no-sync mship_deploy.py` (auto-detecting CPUs/GPUs unless `MSHIP_NODE_NUM_CPUS`/`MSHIP_NODE_NUM_GPUS` set), against the prebuilt venv (extras chosen by `--build-arg MSHIP_VARIANT=thin|cpu|cuda`). Plugin wheels in `MSHIP_PLUGIN_WHEEL_DIR` ship to Ray workers per-deployment via `runtime_env`, resolved from `models.yaml`. The Dev Container overrides this — inside it you must `uv sync` and run `mship_deploy.py` manually. Right after connecting, the driver logs the cluster's observed node/GPU/CPU totals (total vs. currently schedulable) — the quickest way to tell a legitimately-waiting head apart from a misconfigured one.

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
- **GGUF is unsupported on the `vllm` loader.** 0.24 moved GGUF out of tree, and the only external `vllm-gguf-plugin` (`0.0.2`) has a stale `override_quantization_method` signature that breaks *every* quantized model on 0.24 — so it's deliberately not installed. `resolve_all_model_sources` (in `deploy/config.py`) rejects a `.gguf` on the vllm loader at driver preflight with a pointer to `llama_server`. Use `loader: llama_server` for GGUF; the vllm loader takes safetensors or AWQ/GPTQ/FP8 quants. **This is unconditional on GPU vs. CPU** — the CPU wheel (below) doesn't relax it either.
- `vllm==0.24.0+cpu` is installable on the `cpu` extra via an explicit `vllm-cpu` index (`wheels.vllm.ai/0.24.0/cpu`, scoped through `tool.uv.sources`) — the URL embeds the vLLM version, so a future bump must update it. On vLLM's CPU backend, `gpu_memory_utilization` is repurposed to mean *fraction of host RAM* reserved for KV cache, not VRAM, so vLLM's own 0.9 default asks to reserve 90% of node RAM and crashes at worker init on a naive CPU config. `VllmEngineConfig.gpu_memory_utilization` defaults to `None` (not 0.9) precisely so "unset" is never confused with a real value; `infer_config.default_gpu_memory_utilization()` resolves the loader-appropriate fallback (0.9 GPU / 0.4 CPU) lazily, applied via `setdefault` only after `VllmPreflight._recommend_cpu` (`preflight/vllm.py`) has had a chance to recommend a tighter value sized to actual free RAM — an explicit user value always wins over both. See `config/examples/vllm-cpu.yaml`.
- **Preflight (`modelship/preflight/`) branches on `config.num_gpus`, never on hardware discoverability.** `discover_hardware()`'s pynvml node-level fallback can report GPUs Ray didn't actually assign to a `num_gpus: 0` deploy (a PG-coordinator actor, or a shared node), so both `VllmPreflight.recommend` and `LlamaServerPreflight.recommend` dispatch on the reservation, then treat `hw.gpus` as "what's actually usable" within that branch. `vllm_engine_kwargs.gpu_memory_utilization` precedence is: explicit user value > preflight recommendation > `default_gpu_memory_utilization()`'s loader-appropriate fallback — the field itself stays `None` until one of those three actually resolves it, so there's no auto-injected placeholder value that could masquerade as a user override.
- Metrics live on port **8079** (not 8000). `MSHIP_METRICS=false` or `--no-metrics` disables. When `mship_deploy` starts its own head (no `--use-existing-ray-cluster`), `connect_ray` pins that port via `ray.init(_metrics_export_port=…)` — a **private** Ray kwarg (accepted through `**kwargs`). A `TestConnectRay` test guards it so a Ray bump that drops it fails loudly.
- The Ray dashboard always starts on the own-head path; `MSHIP_RAY_DASHBOARD` sets its bind host (default `127.0.0.1`, not on/off). Ray cluster auth (`RAY_AUTH_MODE=token`) is off by default; opt in via `--ray-auth=token`/`MSHIP_RAY_AUTH=token` — Ray generates and reads its own token at `~/.ray/auth_token`, modelship never writes it. `connect_ray` only honors this opt-in when `_ray_auth_is_safe()` holds (a token already exists, or no local cluster is running yet to attach to instead of start) — otherwise it raises `RuntimeError` before calling `ray.init` rather than silently attaching unauthenticated or letting Ray crash at connect time (Ray only auto-generates a token on the start-a-new-cluster path, never on attach). Both the dashboard and auth settings are scoped to `connect_ray`'s own-head branch only; the existing-cluster branch (`MSHIP_USE_EXISTING_RAY_CLUSTER=true`, what KubeRay's RayJob always uses) never touches either.
- `TRACE` is a custom log level below `DEBUG`; it logs full request/response payloads.
- Three images are published from the unified `Dockerfile` (`--build-arg MSHIP_VARIANT=thin|cpu|cuda`), all under `ghcr.io/alez007/modelship`: `thin` (bare tag, `:X.Y.Z`/`:latest`, amd64+arm64) is the control/coordinator image — no torch/vllm, bakes `MSHIP_NODE_NUM_CPUS=0`/`MSHIP_NODE_NUM_GPUS=0` so it never advertises capacity it can't serve; `cuda` (`-cuda` suffix, amd64 only) is the GPU node image; `cpu` (`-cpu` suffix, amd64+arm64) is the CPU node image. Floating tags (`:latest*`) are single-node only — multi-node deployments must pin every node to the same `X.Y.Z`(-suffix) tag or Ray refuses to form the cluster across mismatched versions.
- `llama_server` loader launches a `llama-server` subprocess (found via `MSHIP_LLAMA_SERVER_BIN`, pinned in the Docker images at `/opt/llama.cpp`) and proxies its native OpenAI API — concurrency comes from `--parallel` slots instead of a single `asyncio.Lock`. `num_gpus` must be `0` or a whole integer (fractional is rejected, llama.cpp has no VRAM-fraction knob). Tool-call/reasoning parsing is llama-server's own, auto-detected per chat template: named-function `tool_choice` forcing is globally unsupported (silently falls back to `auto`), and `tool_choice: required` is grammar-enforced for harmony-style templates but a silent no-op for hermes-style ones (e.g. Qwen3); bare `response_format: {"type": "json_object"}` (no `schema` key) is also unenforced despite llama-server's own docs claiming support (verified against the b9859 binary directly) — `type: json_schema` requests, which is what modelship actually sends whenever a schema is given, are unaffected and correctly constrained. See `docs/model-configuration.md`'s llama_server section. No persistent on-disk prompt cache.
- `LlamaServerPreflight` (`modelship/preflight/llama_cpp.py`) sizes `n_ctx` + `n_gpu_layers` from GGUF metadata and hardware — RAM budget on `num_gpus: 0`, VRAM (with a CPU-RAM fallback check for any partially-offloaded layers) on `num_gpus >= 1` — and always emits an explicit `n_gpu_layers` rather than trusting `-1`. Verified against the pinned b9859 `llama-server` binary: any negative `n_gpu_layers` (not just `-1`) hits the same `params.n_gpu_layers < 0` auto-fit-to-free-memory code path (`--help` documents `'auto'`/`'all'` string tokens instead, which this int-typed config field can't send) — the old "`<= -2` offloads all layers" convention from earlier llama.cpp releases is **not** how this binary behaves. It also recommends `threads` from `num_cpus` when the deploy reserves one or more whole CPUs — but declines if that would undercut `parallel` (Ray's `num_cpus` is a scheduling hint, not an enforced cap like `num_gpus`, so a low-`num_cpus`/high-`parallel` deploy shouldn't have its concurrent slots starved of compute; a prior version of this without the guard broke `test_concurrent_requests_are_not_serialized`).

## Further reading

- `AGENTS.md` — the full operational guide
- `docs/architecture.md` — request lifecycle, loaders, plugin system
- `docs/development.md` — dev-container + manual-Docker setup, env vars
- `docs/model-configuration.md` — `models.yaml` reference
- `docs/plugins.md` — plugin authoring
- `config/examples/` — working `models.yaml` files for each backend
