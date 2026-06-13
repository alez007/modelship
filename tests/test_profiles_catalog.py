"""Tests for the profile catalog + tier classifier.

Pure data/functions — no cluster, no model downloads. Source strings are
verified live against HF separately; here we assert structure and tiering."""

from __future__ import annotations

import pytest

from modelship.deploy.profiles.budget import DeployBudget
from modelship.deploy.profiles.catalog import (
    LADDER_USECASES,
    PROFILES,
    generate_ladder,
    image_ladder,
    satellite,
)
from modelship.deploy.profiles.tiers import Accelerator, Tier, classify
from modelship.infer.infer_config import ModelLoader, ModelUsecase

_GiB = 1024**3


# --- tier classifier ----------------------------------------------------------


def _cpu_budget(ram_gib: float) -> DeployBudget:
    return DeployBudget(cpu_units=8.0, gpu_count=0, ram_bytes=int(ram_gib * _GiB), vram_bytes_per_gpu=0)


def _gpu_budget(vram_gib: float) -> DeployBudget:
    return DeployBudget(cpu_units=16.0, gpu_count=1, ram_bytes=64 * _GiB, vram_bytes_per_gpu=int(vram_gib * _GiB))


@pytest.mark.parametrize(
    "ram_gib,expected",
    [
        (4, Tier.small),
        (8, Tier.small),
        (15.9, Tier.small),
        (16, Tier.medium),
        (31, Tier.medium),
        (32, Tier.large),
        (128, Tier.large),
    ],
)
def test_cpu_tiers(ram_gib, expected):
    accel, tier = classify(_cpu_budget(ram_gib))
    assert accel == Accelerator.cpu
    assert tier == expected


@pytest.mark.parametrize(
    "vram_gib,expected",
    [
        # Thresholds sit 1 GiB below nominal card size (we measure free VRAM):
        # < 15 small, 15-22 medium, >= 23 large — so 8/16/24 GiB cards land S/M/L.
        (8, Tier.small),
        (14, Tier.small),
        (14.9, Tier.small),
        (15, Tier.medium),
        (15.3, Tier.medium),  # a real "16 GiB" card's free VRAM
        (16, Tier.medium),
        (22.9, Tier.medium),
        (23, Tier.large),
        (24, Tier.large),
        (80, Tier.large),
    ],
)
def test_gpu_tiers(vram_gib, expected):
    accel, tier = classify(_gpu_budget(vram_gib))
    assert accel == Accelerator.gpu
    assert tier == expected


@pytest.mark.parametrize(
    "cores,ram_gib,expected",
    [
        (4, 64, Tier.small),  # cores cap a roomy box down to S
        (6, 64, Tier.medium),
        (8, 64, Tier.large),
        (16, 64, Tier.large),
        (8, 8, Tier.small),  # RAM is the binding constraint here
        (4, 32, Tier.small),  # the user's box: 4 cores caps 32 GiB to S
    ],
)
def test_cpu_tier_is_min_of_ram_and_cores(cores, ram_gib, expected):
    b = DeployBudget(cpu_units=cores, gpu_count=0, ram_bytes=int(ram_gib * _GiB), vram_bytes_per_gpu=0)
    accel, tier = classify(b)
    assert accel == Accelerator.cpu
    assert tier == expected


def test_gpu_tier_ignores_host_cores():
    # Few host cores must NOT cap the GPU tier — the GPU does the compute.
    b = DeployBudget(cpu_units=4, gpu_count=1, ram_bytes=64 * _GiB, vram_bytes_per_gpu=24 * _GiB)
    accel, tier = classify(b)
    assert accel == Accelerator.gpu and tier == Tier.large


def test_gpu_presence_beats_ram_for_accelerator_choice():
    # Lots of RAM but a GPU is present → GPU bundle.
    b = DeployBudget(cpu_units=16.0, gpu_count=1, ram_bytes=256 * _GiB, vram_bytes_per_gpu=24 * _GiB)
    accel, tier = classify(b)
    assert accel == Accelerator.gpu and tier == Tier.large


# --- catalog structure --------------------------------------------------------


def test_profiles_are_the_locked_four():
    assert set(PROFILES) == {"chat", "assistant", "studio", "everything"}


def test_ladders_have_three_rungs_ascending_footprint():
    for accel in (Accelerator.cpu, Accelerator.gpu):
        for ladder in (generate_ladder(accel), image_ladder(accel)):
            assert len(ladder) == 3
            assert [s.footprint_bytes for s in ladder] == sorted(s.footprint_bytes for s in ladder)


def test_cpu_ladders_use_cpu_loaders_gpu_ladders_use_gpu_loaders():
    for s in generate_ladder(Accelerator.cpu) + image_ladder(Accelerator.cpu):
        assert s.draws_from_vram is False
    for s in generate_ladder(Accelerator.gpu):
        assert s.loader == ModelLoader.vllm and s.draws_from_vram is True
    for s in image_ladder(Accelerator.gpu):
        assert s.loader == ModelLoader.diffusers and s.draws_from_vram is True


def test_satellites_are_constant_and_cpu_pinned():
    # embed + tts identical regardless of tier; both CPU loaders (RAM-drawing).
    for tier in Tier:
        assert satellite(ModelUsecase.embed, tier).model == "nomic-ai/nomic-embed-text-v1.5-GGUF:*f16.gguf"
        assert satellite(ModelUsecase.tts, tier).plugin == "kokoroonnx"
        assert satellite(ModelUsecase.embed, tier).draws_from_vram is False
        assert satellite(ModelUsecase.tts, tier).draws_from_vram is False


def test_transcription_steps_base_on_smallest_cpu_tier_else_small():
    assert satellite(ModelUsecase.transcription, Tier.small).model == "base"
    assert satellite(ModelUsecase.transcription, Tier.medium).model == "small"
    assert satellite(ModelUsecase.transcription, Tier.large).model == "small"


def test_satellite_rejects_ladder_usecases():
    for uc in LADDER_USECASES:
        with pytest.raises(ValueError):
            satellite(uc, Tier.small)


def test_image_turbo_rungs_use_few_steps():
    cpu_img = image_ladder(Accelerator.cpu)
    # SD-Turbo / SDXL-Turbo: ~4 steps; SDXL-base: ~30.
    assert cpu_img[0].loader_config["sample_steps"] == 4
    assert cpu_img[1].loader_config["sample_steps"] == 4
    assert cpu_img[2].loader_config["sample_steps"] == 30
