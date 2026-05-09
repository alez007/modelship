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

`make lint` requires `--extra gpu` to be installed. Pyright resolves imports against the active venv, and `vllm`, `gguf`, `diffusers`, and `psutil` only ship under the gpu extra, so lint on a cpu-only sync fails with `reportMissingImports`. Tests run fine on either extra (the gpu extra is a superset).

Agents: when running tests on your own initiative (sanity-checking a change, verifying a bump), skip the slow `integration`-marked suite by default — `uv run pytest tests/ -v -m "not integration"`. Only run the full `make test` (which includes integration) when explicitly requested.

Pre-commit only runs ruff; it does **not** run pyright or tests, so don't rely on the hook to catch type errors.

## Lint / format / typecheck rules

- Line length **120** (not 88). Ruff handles formatting; `E501` is disabled because the formatter owns line length.
- Ruff rule set: `E, W, F, I, N, UP, B, SIM, RUF`. `I` means isort runs — don't hand-sort imports.
- `known-first-party = ["modelship"]` — the `plugins/*` packages are treated as third-party by isort.
- Pyright `typeCheckingMode = "basic"`, scoped to `modelship`, `plugins`, `mship_deploy.py`. Don't add `# type: ignore` without checking pyright actually complains in basic mode.

## Running the server

Entry point is `mship_deploy.py` (not a console script, not `python -m`). It:

1. Reads `config/models.yaml` (gitignored — copy one from `config/examples/`).
2. Connects to a **running** Ray cluster (`ray.init(address="auto")` unless `MSHIP_USE_EXISTING_RAY_CLUSTER=true`). You must `ray start --head ...` first when running outside Docker.
3. Deploys models **additively** by default (new deployments get a random suffix, e.g. `qwen-a3f9k`). Pass `--redeploy` to tear everything down first.
4. Starts a FastAPI gateway Ray Serve app named `modelship api` (override with `--gateway-name`), listening on port `8000`.

The Docker image's `CMD` (`scripts/start.sh`) starts the Ray head (auto-detecting CPUs/GPUs unless `RAY_HEAD_CPU_NUM` / `RAY_HEAD_GPU_NUM` are set) and runs `uv run --no-sync mship_deploy.py` against the venv baked at build time (extras selected by `--build-arg MSHIP_VARIANT=gpu|cpu`). Plugin wheels under `MSHIP_PLUGIN_WHEEL_DIR` are injected per-deployment via Ray `runtime_env`, resolved automatically from `models.yaml`. The Dev Container overrides this `CMD`, so inside a Dev Container you must run those steps manually (see `docs/development.md`).

## Architecture quick map

- `mship_deploy.py` — Ray init + deploy loop. Contains non-obvious GPU allocation logic in `build_actor_options` for vLLM `tensor_parallel_size > 1` (mp vs ray backend behave differently; `num_gpus` is sometimes ignored and replaced by `VLLM_RAY_PER_WORKER_GPUS`).
- `modelship/openai/api.py` — FastAPI gateway. Uses `RequestWatcher` + `DisconnectEvent` Ray actor to propagate client disconnects across process boundaries.
- `modelship/infer/model_deployment.py` — the single `@serve.deployment` actor class; lazily imports the right backend based on `config.loader`.
- `modelship/infer/infer_config.py` — pydantic config schemas **and** `RawRequestProxy` / `DisconnectEvent`. `RawRequestProxy` exists because FastAPI `Request` cannot cross Ray process boundaries; any new attribute vLLM reads from `raw_request` must be added there.
- `modelship/infer/{vllm,transformers,diffusers,llama_cpp,custom}/` — one subdir per loader. Each has an `*_infer.py` and (for non-custom) an `openai/` adapter subpackage.
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

## Gotchas

- `config/models.yaml` is gitignored; `mship_deploy.py` errors out with a pointer to `config/examples/` if missing.
- vLLM version is pinned (`vllm==0.20.1`). Do not bump casually — the TP scheduling logic in `mship_deploy.py:build_actor_options` defaults to the Ray V2 executor.
- `llama_cpp` loader is CPU-only today; `num_gpus > 0` is silently warned and forced to 0 in `mship_deploy.py:build_actor_options`.
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
