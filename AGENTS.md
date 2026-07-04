# AGENTS.md

Operational notes for agents working in this repo. Read before making changes.

## Toolchain

- Python is pinned exactly to `3.12.10` (`requires-python = "==3.12.10"`). Not `>=3.12`.
- Dependency manager is **uv** with a workspace. Plugins under `plugins/*` are workspace members.
- Never run `pip install`; always use `uv sync` / `uv run` / `uv lock`.
- `gpu` and `cpu` extras are mutually exclusive (declared in `[tool.uv] conflicts`). `torch` / `torchvision` come from different indexes per extra (`pytorch-cu128` vs `pytorch-cpu`).

## Commands you'd otherwise guess wrong

```bash
# Install deps for development (choose gpu OR cpu, plus dev, plus any plugin extras)
uv sync --extra dev --extra gpu                    # what CI uses
uv sync --extra dev --extra cpu                    # CPU-only dev
uv sync --extra dev --extra cpu --extra kokoroonnx # with a plugin

# The canonical dev loop (mirrored in CI and Makefile)
make lint        # ruff check + ruff format --check + pyright  (all three MUST pass)
make lint-fix    # ruff check --fix + ruff format
make test        # uv run pytest tests/ -v

# Run a single test
uv run pytest tests/test_config.py::TestLlamaCppConfig::test_defaults -v
```

CI (`.github/workflows/ci.yml`) runs `uv sync --extra dev --extra gpu` on Linux, then `ruff check`, `ruff format --check`, `pyright`, and `pytest tests/ -v`. Match that locally before pushing.

`make lint` requires `--extra gpu` to be installed. Pyright resolves imports against the active venv, and `gguf`, `diffusers`, and `psutil` only ship under the gpu extra, so lint on a cpu-only sync fails with `reportMissingImports`. (`vllm` is importable under both extras as of the Stage E0 CPU wheel wiring — it's no longer gpu-only, just not enough on its own to make lint pass cpu-only.) Tests run fine on either extra (the gpu extra is a superset).

Agents: when running tests on your own initiative (sanity-checking a change, verifying a bump), skip the slow `integration`-marked suite by default — `uv run pytest tests/ -v -m "not integration"`. Only run the full `make test` (which includes integration) when explicitly requested.

Pre-commit only runs ruff; it does **not** run pyright or tests, so don't rely on the hook to catch type errors.

## OpenAI protocol fidelity

`modelship/openai/protocol.py` is the request/response surface clients see. When adding or changing models there:

- **Follow the official OpenAI API specification strictly.** Field names, types, defaults, optionality, and shape of nested objects must match what `platform.openai.com/docs` documents for the corresponding route.
- Do not invent fields to expose loader-specific knobs (Diffusers `strength`, vLLM `stop_reason`, etc.). Carry loader-specific defaults via the per-model `*_config` in `infer_config.py` instead.
- Missing optional OpenAI fields are fine when a feature is genuinely unsupported. Adding fields that aren't in OpenAI's spec is not — it locks clients into a modelship-specific dialect and breaks the drop-in-replacement guarantee.
- When OpenAI's spec evolves (new fields, new response_format values, new routes), update the protocol shapes before wiring the backend.

When in doubt, check OpenAI's reference for the exact route. Existing deviations are documented and tracked separately; do not add new ones.

## Lint / format / typecheck rules

- Line length **120** (not 88). Ruff handles formatting; `E501` is disabled because the formatter owns line length.
- Ruff rule set: `E, W, F, I, N, UP, B, SIM, RUF`. `I` means isort runs — don't hand-sort imports.
- `known-first-party = ["modelship"]` — the `plugins/*` packages are treated as third-party by isort.
- Pyright `typeCheckingMode = "basic"`, scoped to `modelship`, `plugins`, `mship_deploy.py`. Don't add `# type: ignore` without checking pyright actually complains in basic mode.

## Running the server

Entry point is `mship_deploy.py` (not a console script, not `python -m`). It:

1. Reads `config/models.yaml` (gitignored — copy one from `config/examples/`).
2. Starts its **own** Ray head by default (sized from `RAY_HEAD_CPU_NUM`/`RAY_HEAD_GPU_NUM`, auto-detected if unset; metrics on `RAY_METRICS_EXPORT_PORT`) and tears it down on exit. With `--use-existing-ray-cluster` it instead connects to a cluster you manage via `ray.init(address="auto")` and deploys-and-exits without teardown — the driver must run **on** a cluster node (Docker co-located / k8s RayJob / bare-metal node); it cannot attach from off-cluster.
3. Deploys models **additively** by default (new deployments get a random suffix, e.g. `qwen-a3f9k`). Pass `--reconcile` to instead make the cluster match the config exactly (add/remove/replace) — it never tears the cluster down.
4. Starts a FastAPI gateway Ray Serve app named `modelship api` (override with `--gateway-name`), listening on port `8000`.

The Docker image's `CMD` is `uv run --no-sync mship_deploy.py` (against the venv baked at build time; extras selected by `--build-arg MSHIP_VARIANT=gpu|cpu`), which starts its own Ray head and runs the deploy loop. Plugin wheels under `MSHIP_PLUGIN_WHEEL_DIR` are injected per-deployment via Ray `runtime_env`, resolved automatically from `models.yaml`. The Dev Container overrides this `CMD`, so inside a Dev Container you run `mship_deploy.py` manually (see `docs/development.md`).

## Architecture quick map

- `mship_deploy.py` — Ray init + deploy loop. `build_deployment_options` (in `modelship/deploy/actor_options.py`) handles GPU allocation: multi-slot vLLM deploys (`tp*pp > 1`) always build a Ray Serve placement group (one whole-GPU bundle per slot, STRICT_PACK) that vLLM's ray executor inherits via `get_current_placement_group()`. Single-slot deploys use a scalar `num_gpus` on the outer actor. Fractional `num_gpus` (`<1`) is single-GPU only — combining it with TP/PP is rejected at config time (Ray packs fractional PG bundles onto the same physical GPU).
- `modelship/openai/api.py` — FastAPI gateway. Uses `RequestWatcher` + a single shared `DisconnectRegistry` Ray actor (keyed by request id) to propagate client disconnects across process boundaries.
- `modelship/infer/model_deployment.py` — the single `@serve.deployment` actor class; lazily imports the right backend based on `config.loader`.
- `modelship/infer/infer_config.py` — pydantic config schemas **and** `RawRequestProxy` / `DisconnectRegistry`. `RawRequestProxy` exists because FastAPI `Request` cannot cross Ray process boundaries; any new attribute vLLM reads from `raw_request` must be added there.
- `modelship/infer/{vllm,transformers,diffusers,llama_cpp,custom}/` — one subdir per loader. Each has an `*_infer.py` and (for non-custom) an `openai/` adapter subpackage. `modelship/infer/llama_server/llama_server_infer.py` is a flat file with no `openai/` subpackage — it proxies a `llama-server` subprocess's own OpenAI-compatible HTTP API rather than parsing output in-process.
- `modelship/plugins/base_plugin.py` — `BasePlugin` ABC that plugin packages subclass as `ModelPlugin`.
- `plugins/*` — workspace packages, each opt-in via a root extra. The plugin module name and the extra name must match (`ensure_plugin()` calls `importlib.import_module(config.plugin)` and the error message says `uv sync --extra <plugin>`).

## Adding a plugin (checklist that's easy to miss)

1. Create `plugins/<name>/` with its own `pyproject.toml` (module-name = `<name>`, depends on `modelship` via `{ workspace = true }`).
2. Export `ModelPlugin` from `plugins/<name>/<name>/__init__.py`.
3. In root `pyproject.toml`: add `<name> = ["<name>"]` under `[project.optional-dependencies]` **and** `<name> = { workspace = true }` under `[tool.uv.sources]`. Both are required.
4. Run `uv lock` to refresh `uv.lock`.
5. Add a `README.md` inside the plugin (required — see `docs/plugins.md`).

## Tests

- Under `tests/`. Use `pytest-asyncio` for async tests.
- Tests mock out Ray Serve; they do **not** spin up a real cluster. Pattern: access the wrapped class via `ModelshipAPI.func_or_class` to bypass the `@serve.deployment` wrapper (see `tests/test_api.py`).
- There are no integration tests that require a GPU or real models. Keep it that way unless you add them behind an opt-in marker.

## Releases

`make release-{patch,minor,major}` is the only supported path. It refuses to run off `main` or with a dirty tree, bumps `pyproject.toml`, runs `uv lock`, generates a CHANGELOG entry from conventional commits (`feat:`, `fix:`, `refactor:|perf:|docs:|chore:|build:|ci:|style:|test:`), commits, tags `vX.Y.Z`, and pushes. The `release.yml` workflow publishes the Docker images and PyPI package. Do not bump the version by hand.

Commit messages matter: use Conventional Commits prefixes so the changelog generator picks them up.

## Working with git

- **The maintainer pushes; agents don't.** Create branches and commits locally, but leave `git push` to the human (this environment has no `ssh`, and the remote is SSH anyway). Hand back the branch name and let them push and open the PR.
- **Never amend; always add a new commit.** Don't `git commit --amend` (or rebase/squash) unless explicitly asked. Follow-up work — review feedback, refactors, even bug fixes to a just-made commit — goes in its own commit stacked on top of the original, so history stays reviewable.

## Gotchas

- `config/models.yaml` is gitignored; `mship_deploy.py` errors out with a pointer to `config/examples/` if missing.
- vLLM version is pinned (`vllm==0.24.0`). Do not bump casually — the TP scheduling logic in `mship_deploy.py:build_deployment_options` defaults to the Ray V2 executor, and the loader imports vLLM-internal `entrypoints.*` module paths that upstream restructures between minors.
- **GGUF is not supported on the `vllm` loader.** 0.24 moved GGUF out of tree; the only external `vllm-gguf-plugin` (`0.0.2`) has a stale `override_quantization_method` signature incompatible with 0.24's quantization API (it breaks *all* quantized models, not just GGUF), so it is deliberately not installed. `resolve_all_model_sources` rejects a `.gguf` on the vllm loader at driver preflight and points to `llama_cpp`. For GGUF use `loader: llama_cpp`; feed the vllm loader safetensors or an AWQ/GPTQ/FP8 quant.
- `llama_cpp` loader supports CPU or whole-GPU offload. The gpu extra installs a prebuilt CUDA (cu130) wheel from `abetlen.github.io/llama-cpp-python/whl/cu130`; the cpu extra keeps the plain PyPI CPU-only wheel. `num_gpus` must be `0` or a whole integer for llama_cpp — fractional is rejected at config time (llama.cpp has no VRAM-fraction knob). `num_gpus >= 1` honors `n_gpu_layers`; `stable_diffusion_cpp` remains forced to `num_gpus: 0` in `mship_deploy.py:build_deployment_options`.
- `llama_server` loader (GGUF, same whole-GPU-only `num_gpus` rule as `llama_cpp`) launches a `llama-server` subprocess — found via `MSHIP_LLAMA_SERVER_BIN`, pinned in the Docker images at `/opt/llama.cpp/llama-server.sh` — and proxies its native OpenAI API instead of parsing output in-process. `--parallel` slots give real request concurrency, unlike `llama_cpp`'s single `asyncio.Lock`. Tool-call/reasoning parsing is llama-server's own, auto-detected per chat template: named-function `tool_choice` forcing is unsupported globally (silently falls back to `auto`), and `tool_choice: required` is grammar-enforced for harmony-style templates but a silent no-op for hermes-style ones (e.g. Qwen3); bare `response_format: {"type": "json_object"}` (no `schema` key) is also unenforced despite llama-server's own docs claiming support — `type: json_schema` requests (what modelship sends whenever a schema is given) are unaffected. No persistent on-disk prompt cache. See `docs/model-configuration.md`'s llama_server section for the full field table and examples.
- Metrics are on by default on port **8079** (not 8000). Disable with `--no-metrics` or `MSHIP_METRICS=false`.
- Log level `TRACE` (below `DEBUG`) is a custom level and logs full request/response payloads.
- Docker images are multi-arch (amd64 + arm64). The CPU image uses the unified `Dockerfile` with `--build-arg MSHIP_VARIANT=cpu` and has a different tag suffix (`:latest-cpu`).

## Further reading

Prefer these over re-reading source when orienting:

- `docs/architecture.md` — request lifecycle, loaders, plugin system
- `docs/development.md` — full dev-container + manual-Docker setup, env vars
- `docs/model-configuration.md` — `models.yaml` reference
- `docs/plugins.md` — plugin authoring
- `config/examples/` — working `models.yaml` files for each backend
