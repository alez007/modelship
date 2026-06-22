"""Integration tests for one-click profiles (`--model-stack`).

Each test starts its OWN Ray head at a chosen `(num_cpus, num_gpus)` stage, then
runs `mship_deploy.py --model-stack <profile>`. Selection reads the deploy budget
from `ray.cluster_resources()`, so the cpu/gpu counts we pass to `ray start` drive
which models the knapsack picks. We then assert:

  * `/v1/models` lists exactly the profile's capability set (deployment names are
    the usecases — `generate`, `embed`, `image`, `transcription`, `tts`), proving
    the all-or-nothing stack actually loaded and serves; and
  * the generated `config/models_stack_<profile>.yaml` selected the model we expect
    for that stage (e.g. the 1.5B at 2 cores, something larger at 8).

Host RAM can't be fenced per cluster, so model-size assertions stay machine-robust
(the small box gets the smallest rung; a roomier box gets *something larger*),
never pinning an absolute pick that depends on free RAM.

NOTE: this module fully owns the Ray cluster and port 8000. Do NOT run it in the
same pytest invocation as `test_integration.py` (which holds a session-scoped
cluster on the same port). Select it on its own with `-m profiles`.
"""

from __future__ import annotations

import contextlib
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import yaml

from modelship.deploy.profiles.catalog import PROFILES
from modelship.infer.infer_config import ModelUsecase
from openai import OpenAI

OPENAI_API_BASE = "http://localhost:8000/v1"
HEALTH_URL = "http://localhost:8000/health"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "config"

# Generous: a CPU profile pulls 1-2 GiB of GGUF weights on first run, then loads.
_READY_TIMEOUT_S = 600


def _expected_usecase_ids(profile: str) -> set[str]:
    return {uc.value for uc in PROFILES[profile]}


@dataclass
class _Deployment:
    client: OpenAI
    config_path: Path  # the generated models_stack_<profile>.yaml

    def model_ids(self) -> set[str]:
        return {m.id for m in self.client.models.list().data}

    def selected_model(self, usecase: ModelUsecase) -> str:
        """The `model:` string the generator chose for a usecase, read from the
        generated yaml (the /v1/models id is just the usecase name)."""
        doc = yaml.safe_load(self.config_path.read_text())
        entry = next(m for m in doc["models"] if m["usecase"] == usecase.value)
        return entry["model"]


def _ray_start(num_cpus: int, num_gpus: int) -> None:
    subprocess.run(["ray", "stop", "--force"], check=False)
    subprocess.run(
        [
            "ray",
            "start",
            "--head",
            f"--num-cpus={num_cpus}",
            f"--num-gpus={num_gpus}",
            "--dashboard-host=0.0.0.0",
            "--disable-usage-stats",
        ],
        check=True,
    )


def _deploy_cmd(profile: str) -> list[str]:
    return ["uv", "run", "mship_deploy.py", "--model-stack", profile, "--use-existing-ray-cluster"]


def _wait_for_gateway(log_path: Path) -> None:
    """Poll /health until the just-deployed gateway serves (the deploy already
    completed, so this is a short confirmation, not a long wait)."""
    deadline = time.time() + 120
    while time.time() < deadline:
        with contextlib.suppress(Exception):
            if httpx.get(HEALTH_URL, timeout=5).status_code == 200:
                return
        time.sleep(2)
    tail = log_path.read_text()[-4000:] if log_path.exists() else "<no log>"
    pytest.fail(f"gateway not serving after a successful deploy.\nLast 4KB:\n{tail}")


@contextlib.contextmanager
def _profile_cluster(profile: str, num_cpus: int, num_gpus: int, tmp_path: Path) -> Iterator[_Deployment]:
    """Start a Ray head at `(num_cpus, num_gpus)`, deploy `profile` onto it, and
    yield the live `_Deployment`. The deploy runs to completion (models loaded)
    before we assert, then `ray stop` tears the whole thing down."""
    log_path = tmp_path / f"mship_deploy_{profile}.log"
    config_path = _CONFIG_DIR / f"models_stack_{profile}.yaml"
    config_path.unlink(missing_ok=True)

    _ray_start(num_cpus, num_gpus)
    try:
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                _deploy_cmd(profile), stdout=log_file, stderr=subprocess.STDOUT, text=True, timeout=_READY_TIMEOUT_S
            )
        if result.returncode != 0:
            tail = log_path.read_text()[-4000:]
            pytest.fail(f"profile {profile!r} ({num_cpus}cpu/{num_gpus}gpu) deploy exited {result.returncode}.\n{tail}")
        _wait_for_gateway(log_path)
        yield _Deployment(OpenAI(base_url=OPENAI_API_BASE, api_key="not-needed"), config_path)
    finally:
        subprocess.run(["ray", "stop", "--force"], check=False)
        config_path.unlink(missing_ok=True)


def _run_deploy_expecting_failure(profile: str, num_cpus: int, num_gpus: int, tmp_path: Path) -> str:
    """Deploy a profile that should be refused; return the captured log. Asserts the
    deploy exited non-zero (a clean refusal, no gateway)."""
    log_path = tmp_path / f"mship_deploy_{profile}_refused.log"
    _ray_start(num_cpus, num_gpus)
    try:
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                _deploy_cmd(profile), stdout=log_file, stderr=subprocess.STDOUT, text=True, timeout=120
            )
        assert result.returncode != 0, "expected mship_deploy to refuse the profile and exit non-zero"
        return log_path.read_text()
    finally:
        subprocess.run(["ray", "stop", "--force"], check=False)


# --- CPU stages ---------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.profiles
@pytest.mark.llama_cpp
def test_chat_on_2_cores_deploys_smallest_generate(tmp_path):
    # 2 cores: chat (generate + embed) can only afford the 1.5B generate rung
    # alongside embed — the smallest box gets the smallest model.
    with _profile_cluster("chat", num_cpus=2, num_gpus=0, tmp_path=tmp_path) as dep:
        assert dep.model_ids() == _expected_usecase_ids("chat")
        assert "1.5B" in dep.selected_model(ModelUsecase.generate)


@pytest.mark.integration
@pytest.mark.profiles
@pytest.mark.llama_cpp
def test_chat_on_8_cores_scales_generate_up(tmp_path):
    # 8 cores: the knapsack must pick a larger generate than the 2-core box did
    # (at least the 3B), proving cpu headroom scales the selection up.
    with _profile_cluster("chat", num_cpus=8, num_gpus=0, tmp_path=tmp_path) as dep:
        assert dep.model_ids() == _expected_usecase_ids("chat")
        assert "1.5B" not in dep.selected_model(ModelUsecase.generate)


@pytest.mark.integration
@pytest.mark.profiles
@pytest.mark.llama_cpp
def test_assistant_deploys_full_capability_set(tmp_path):
    # assistant = generate + transcription + tts. Needs the whispercpp + kokoroonnx
    # plugin wheels available (MSHIP_PLUGIN_WHEEL_DIR / installed extras).
    with _profile_cluster("assistant", num_cpus=6, num_gpus=0, tmp_path=tmp_path) as dep:
        assert dep.model_ids() == _expected_usecase_ids("assistant")


@pytest.mark.integration
@pytest.mark.profiles
def test_everything_refused_on_2_cores_writes_no_gateway(tmp_path):
    # 5 models can't each get their minimum cores on a 2-core box → clean refusal
    # before any model loads (cheap: no downloads).
    log = _run_deploy_expecting_failure("everything", num_cpus=2, num_gpus=0, tmp_path=tmp_path)
    assert "does not fit" in log


# --- GPU stage ----------------------------------------------------------------


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.profiles
@pytest.mark.vllm
@pytest.mark.diffusers
@pytest.mark.skipif(not _has_cuda(), reason="GPU profile stage requires a CUDA device")
def test_studio_on_one_gpu_uses_gpu_loaders(tmp_path):
    # A single GPU: studio (generate + image + embed) must route generate→vllm and
    # image→diffusers (embed stays CPU), and serve all three.
    with _profile_cluster("studio", num_cpus=8, num_gpus=1, tmp_path=tmp_path) as dep:
        assert dep.model_ids() == _expected_usecase_ids("studio")
        doc = yaml.safe_load(dep.config_path.read_text())
        by_uc = {m["usecase"]: m for m in doc["models"]}
        assert by_uc["generate"]["loader"] == "vllm"
        assert by_uc["image"]["loader"] == "diffusers"
        assert by_uc["embed"]["loader"] == "llama_cpp"
