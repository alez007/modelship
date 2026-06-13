"""Tests for the profile generator's deploy-budget reader.

CPU/GPU counts come from Ray's ledger; RAM and per-GPU VRAM from physical
detection. See modelship/deploy/profiles/budget.py."""

from __future__ import annotations

from unittest.mock import patch

from modelship.deploy.profiles.budget import DeployBudget, read_deploy_budget
from modelship.preflight import GPUInfo

_GiB = 1024**3


def _read(*, ledger: dict, ram: int, gpus: list[GPUInfo]) -> DeployBudget:
    with (
        patch("ray.cluster_resources", return_value=ledger),
        patch("modelship.deploy.profiles.budget.detect_ram_bytes", return_value=ram),
        patch("modelship.deploy.profiles.budget.detect_gpus", return_value=gpus),
    ):
        return read_deploy_budget()


def test_cpu_only_box():
    b = _read(ledger={"CPU": 8.0}, ram=16 * _GiB, gpus=[])
    assert b.cpu_units == 8.0
    assert b.gpu_count == 0
    assert b.ram_bytes == 16 * _GiB
    assert b.vram_bytes_per_gpu == 0
    assert b.has_gpu is False


def test_gpu_box_reports_vram():
    gpus = [GPUInfo(0, 24 * _GiB, "L4"), GPUInfo(1, 24 * _GiB, "L4")]
    b = _read(ledger={"CPU": 16.0, "GPU": 2.0}, ram=64 * _GiB, gpus=gpus)
    assert b.gpu_count == 2
    assert b.vram_bytes_per_gpu == 24 * _GiB
    assert b.has_gpu is True


def test_heterogeneous_gpus_size_to_smallest():
    gpus = [GPUInfo(0, 16 * _GiB, "A4000"), GPUInfo(1, 24 * _GiB, "L4")]
    b = _read(ledger={"CPU": 16.0, "GPU": 2.0}, ram=64 * _GiB, gpus=gpus)
    assert b.vram_bytes_per_gpu == 16 * _GiB  # conservative: the smaller card


def test_ledger_fences_gpus_to_zero_forces_cpu_bundle():
    # Driver physically sees a GPU, but RAY_HEAD_GPU_NUM=0 → ledger has no GPU.
    gpus = [GPUInfo(0, 24 * _GiB, "L4")]
    b = _read(ledger={"CPU": 8.0, "GPU": 0.0}, ram=32 * _GiB, gpus=gpus)
    assert b.gpu_count == 0
    assert b.vram_bytes_per_gpu == 0
    assert b.has_gpu is False


def test_ledger_has_gpu_but_no_vram_detected_degrades_to_cpu():
    # Ray ledgers a GPU but the driver couldn't read VRAM (no CUDA ctx / pynvml).
    b = _read(ledger={"CPU": 16.0, "GPU": 1.0}, ram=64 * _GiB, gpus=[])
    assert b.gpu_count == 1
    assert b.vram_bytes_per_gpu == 0
    assert b.has_gpu is False


def test_missing_ledger_keys_default_to_zero():
    b = _read(ledger={}, ram=8 * _GiB, gpus=[])
    assert b.cpu_units == 0.0
    assert b.gpu_count == 0
    assert b.has_gpu is False
