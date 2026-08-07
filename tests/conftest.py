"""Shared test fixtures, including the integration suite's session-scoped cluster
infrastructure (`mship_cluster`, `model_deployer`, `client`, `MODEL_CONFIGS`), so
every `@pytest.mark.integration` file shares one live cluster per session."""

import os
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml

import modelship.infer.infer_config as infer_config
from openai import OpenAI


@pytest.fixture(autouse=True)
def neutralize_request_watcher():
    """Stub the disconnect registry and watch loop so route-handler tests don't spin
    up a real Ray cluster. A test module that exercises the real watcher/registry can
    override this by defining a same-named autouse fixture of its own."""

    async def _noop_watch(self):
        return

    with (
        patch.object(infer_config, "get_disconnect_registry", return_value=MagicMock()),
        patch.object(infer_config.RequestWatcher, "_watch", _noop_watch),
    ):
        yield


# ---------------------------------------------------------------------------
# Integration suite: real Ray cluster + real models, `@pytest.mark.integration`.
# ---------------------------------------------------------------------------

OPENAI_API_BASE = "http://localhost:8000/v1"
HEALTH_URL = "http://localhost:8000/health"

# Per-model configs. Each `Deployer.deploy(*names)` call writes one of these
# (or a subset) into a one-shot models.yaml and runs `mship_deploy.py
# --reconcile` to swap the currently-deployed set in-place.
MODEL_CONFIGS: dict[str, dict] = {
    "chat-capable": {
        "name": "chat-capable",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "usecase": "generate",
        "loader": "vllm",
        # num_gpus is also wired into vllm's gpu_memory_utilization; 0.1 leaves no room for KV cache
        "num_gpus": 0.5,
        "vllm_engine_kwargs": {
            "max_model_len": 2048,
            "enforce_eager": True,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
        },
    },
    "chat-reasoning": {
        "name": "chat-reasoning",
        # Qwen3-0.6B is the smallest reasoning-capable model in the Qwen3
        # family; it natively emits `<think>...</think>`. Reasoning chains
        # need headroom so `max_model_len` is bumped.
        "model": "Qwen/Qwen3-0.6B",
        "usecase": "generate",
        "loader": "vllm",
        "num_gpus": 0.5,
        "vllm_engine_kwargs": {
            "max_model_len": 4096,
            "enforce_eager": True,
            "enable_reasoning": True,
            "reasoning_parser": "deepseek_r1",
        },
    },
    "chat-vlm": {
        "name": "chat-vlm",
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "usecase": "generate",
        "loader": "vllm",
        "num_gpus": 1,
        "vllm_engine_kwargs": {
            "max_model_len": 8192,
            "enforce_eager": True,
            "limit_mm_per_prompt": {"image": 2},
            "mm_processor_kwargs": {"min_pixels": 50176, "max_pixels": 200704},
        },
    },
    "autoscale-llama": {
        "name": "autoscale-llama",
        # Tiny CPU GGUF so the host can hold several replicas (1 cpu each, up to
        # max_replicas) at once. autoscaling_config replaces num_replicas:
        # target_ongoing_requests=1 makes a handful of concurrent requests
        # exceed one replica's setpoint and drive scale-out; the short delays
        # keep the test's poll windows tractable.
        "model": "lmstudio-community/Qwen3-0.6B-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_cpus": 1,
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 3,
            "target_ongoing_requests": 1,
            "upscale_delay_s": 2,
            "downscale_delay_s": 10,
        },
    },
    "chat-llama-server": {
        "name": "chat-llama-server",
        # Qwen3-0.6B GGUF through the llama_server loader: a llama-server
        # subprocess doing its own chat templating, tool-call, and reasoning
        # parsing (`--jinja --reasoning-format auto`). `parallel: 4` exercises
        # the loader's headline capability: true multi-slot concurrency
        # instead of a single asyncio.Lock serializing every request. n_ctx is per-slot (the
        # loader launches with `-c n_ctx*parallel`), bumped for reasoning
        # headroom.
        "model": "lmstudio-community/Qwen3-0.6B-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_cpus": 2,
        "llama_server_config": {
            "n_ctx": 4096,
            "parallel": 4,
        },
    },
    "chat-llama-server-plain": {
        "name": "chat-llama-server-plain",
        # Same non-reasoning Qwen2.5-0.5B GGUF as chat-llama-server, through
        # the llama_server loader. Used for the response_format tests, which
        # need a model that doesn't emit a `<think>...</think>` preamble.
        "model": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_cpus": 1,
    },
    "chat-llama-server-gpu": {
        "name": "chat-llama-server-gpu",
        # Same GGUF as chat-llama-server-plain, on a whole GPU — exercises the
        # llama_server loader's offload path (actor GPU allocation, -ngl
        # honored).
        "model": "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF:*Q4_K_M.gguf",
        "usecase": "generate",
        "loader": "llama_server",
        "num_gpus": 1,
        "num_cpus": 1,
    },
    "embed-model-llama-server": {
        "name": "embed-model-llama-server",
        # Real embeddings through a live llama-server subprocess (`--embedding`)
        # — the existing `test_embeddings` integration test only exercises the
        # vllm loader; llama_server's B4 embeddings support was otherwise
        # only unit-tested against a mocked httpx transport.
        "model": "nomic-ai/nomic-embed-text-v1.5-GGUF:nomic-embed-text-v1.5.Q4_K_M.gguf",
        "usecase": "embed",
        "loader": "llama_server",
        "num_cpus": 1,
    },
    "embed-model": {
        "name": "embed-model",
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "usecase": "embed",
        "loader": "vllm",
        "num_gpus": 0.15,
        "vllm_engine_kwargs": {
            "trust_remote_code": True,
        },
    },
    "stt-model": {
        "name": "stt-model",
        "model": "openai/whisper-tiny",
        "usecase": "transcription",
        "loader": "vllm",
        "num_gpus": 0.15,
        "vllm_engine_kwargs": {
            "trust_remote_code": True,
        },
    },
    "stt-cpp-model": {
        "name": "stt-cpp-model",
        "model": "tiny.en",
        "usecase": "transcription",
        "loader": "whispercpp",
        "num_cpus": 1,
    },
    "stt-cpp-multilingual": {
        "name": "stt-cpp-multilingual",
        # Multilingual counterpart of stt-cpp-model: exercises auto-detection and translate.
        "model": "tiny",
        "usecase": "transcription",
        "loader": "whispercpp",
        "num_cpus": 1,
    },
    "tts-model": {
        "name": "tts-model",
        "model": "hexgrad/Kokoro-82M",
        "usecase": "tts",
        "loader": "custom",
        "plugin": "kokoroonnx",
        "num_cpus": 1,
        "plugin_config": {"onnx_provider": "CPUExecutionProvider"},
    },
    "image-model": {
        "name": "image-model",
        "model": "stabilityai/sdxl-turbo",
        "usecase": "image",
        "loader": "diffusers",
        "num_gpus": 1,
        "diffusers_config": {"num_inference_steps": 2, "guidance_scale": 0.0},
    },
    "image-cpu-model": {
        "name": "image-cpu-model",
        "model": "gpustack/stable-diffusion-v2-1-turbo-GGUF:*Q4_1.gguf",
        "usecase": "image",
        "loader": "stable_diffusion_cpp",
        "num_cpus": 4,
        "stable_diffusion_cpp_config": {"sample_steps": 4, "cfg_scale": 1.0},
    },
}


class _Deployer:
    """Owns the per-test reconcile cycle: writes a one-shot models.yaml with
    the requested set and runs `mship_deploy.py --reconcile` synchronously
    against the already-running gateway. Re-deploying the same set is a no-op."""

    def __init__(self, tmp_dir: Path) -> None:
        self._tmp = tmp_dir
        self._current: frozenset[str] = frozenset()

    def deploy(self, *model_names: str) -> None:
        wanted = frozenset(model_names)
        if wanted == self._current:
            return

        slug = "+".join(sorted(wanted)) or "empty"
        config_path = self._tmp / f"models-{slug}.yaml"
        log_path = self._tmp / f"reconcile-{slug}.log"
        with open(config_path, "w") as f:
            yaml.dump({"models": [MODEL_CONFIGS[n] for n in sorted(wanted)]}, f)

        with open(log_path, "w") as log_file:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "mship_deploy.py",
                    "--config",
                    str(config_path),
                    "--reconcile",
                    "--replace-strategy",
                    "stop_start",
                    "--prune-ray-sessions",
                    "false",
                    # Attach to mship_cluster's already-running head instead of starting a second one.
                    "--use-existing-ray-cluster",
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=900,
            )
        if result.returncode != 0:
            tail = log_path.read_text()[-4000:]
            pytest.fail(
                f"mship_deploy --reconcile failed for {slug} (exit {result.returncode}).\n"
                f"Log file: {log_path}\nLast 4KB:\n{tail}"
            )
        self._current = wanted

    def deploy_raw(self, models: list[dict], *, replace_strategy: str = "stop_start") -> None:
        """Like deploy(), but takes raw model dicts directly (e.g. a MODEL_CONFIGS
        entry with fields overridden to force a new fingerprint under the same
        name) and lets the caller pick --replace-strategy. Resets self._current
        to force the next deploy() call to always re-apply its own set, since
        this bypasses the by-name cache."""
        slug = "+".join(sorted(m["name"] for m in models)) or "empty"
        config_path = self._tmp / f"models-raw-{slug}-{replace_strategy}.yaml"
        log_path = self._tmp / f"reconcile-raw-{slug}-{replace_strategy}.log"
        with open(config_path, "w") as f:
            yaml.dump({"models": models}, f)

        with open(log_path, "w") as log_file:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "mship_deploy.py",
                    "--config",
                    str(config_path),
                    "--reconcile",
                    "--replace-strategy",
                    replace_strategy,
                    "--prune-ray-sessions",
                    "false",
                    "--use-existing-ray-cluster",
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=900,
            )
        self._current = frozenset()
        if result.returncode != 0:
            tail = log_path.read_text()[-4000:]
            pytest.fail(
                f"mship_deploy --reconcile failed for {slug} ({replace_strategy}, exit {result.returncode}).\n"
                f"Log file: {log_path}\nLast 4KB:\n{tail}"
            )


@pytest.fixture(scope="session")
def mship_cluster(tmp_path_factory):
    """Start a Ray cluster and a long-lived `mship_deploy` operator process
    bound to an empty models.yaml — owns the gateway via the fresh-install
    path and `signal.pause()`s for the rest of the session. Per-test code
    deploys models additively via `_Deployer.deploy(...)`."""
    tmp_dir = tmp_path_factory.mktemp("mship_integration")
    empty_config = tmp_dir / "empty-models.yaml"
    log_path = tmp_dir / "mship_deploy.log"

    # Clear any stale cluster, but don't pre-start a head: mship_deploy.py starts
    # its own by default, and a pre-started head forces its ray.init() into
    # attach-mode, which rejects the explicit num_gpus/num_cpus it passes on a GPU host.
    subprocess.run(["ray", "stop", "--force"], check=False)

    with open(empty_config, "w") as f:
        yaml.dump({"models": []}, f)

    log_file = open(log_path, "w")  # noqa: SIM115 — kept open for subprocess lifetime, closed in cleanup
    proc = subprocess.Popen(
        # 2 gateway replicas for the whole session so TestGatewayReplicaConsistency
        # can verify the coordinator watch loop converges every replica (and all
        # other tests get free multi-replica coverage). Gateway replicas are
        # num_cpus=0, so this is cheap.
        [
            "uv",
            "run",
            "mship_deploy.py",
            "--config",
            str(empty_config),
            "--gateway-replicas",
            "2",
            "--prune-ray-sessions",
            "false",
        ],
        # Non-blocking if absent, so safe to enable session-wide; lets tests
        # simulate distinct identities via extra_headers.
        env={**os.environ, "MSHIP_TRUSTED_IDENTITY_HEADER": "X-Mship-Test-Identity"},
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def cleanup():
        log_file.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        subprocess.run(["ray", "stop", "--force"], check=False)

    try:
        deadline = time.time() + 120
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                if httpx.get(HEALTH_URL).status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(2)

        if not ready:
            tail = log_path.read_text()[-4000:] if log_path.exists() else "<no log>"
            cleanup()
            pytest.fail(f"Gateway failed to become ready within timeout.\nLog file: {log_path}\nLast 4KB:\n{tail}")

        yield tmp_dir
    finally:
        cleanup()


@pytest.fixture(scope="session")
def model_deployer(mship_cluster) -> _Deployer:
    return _Deployer(mship_cluster)


@pytest.fixture(scope="session")
def client(mship_cluster) -> OpenAI:
    return OpenAI(base_url=OPENAI_API_BASE, api_key="not-needed")
