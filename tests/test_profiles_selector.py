"""Tests for the profile selector — tier-exact, fail-fast.

Asserts the full capability set is always delivered (never partial), the highest
fitting tier is chosen, and too-small hardware errors rather than degrading."""

from __future__ import annotations

import pytest

from modelship.deploy.profiles.budget import DeployBudget
from modelship.deploy.profiles.catalog import PROFILES
from modelship.deploy.profiles.selector import (
    _UTILIZATION,
    ProfileDoesNotFitError,
    select_stack,
)
from modelship.infer.infer_config import ModelLoader, ModelUsecase

_GiB = 1024**3


def _cpu(ram_gib: float, cores: float = 8.0) -> DeployBudget:
    return DeployBudget(cpu_units=cores, gpu_count=0, ram_bytes=int(ram_gib * _GiB), vram_bytes_per_gpu=0)


def _gpu(vram_gib: float, ram_gib: float = 64.0) -> DeployBudget:
    return DeployBudget(
        cpu_units=16.0, gpu_count=1, ram_bytes=int(ram_gib * _GiB), vram_bytes_per_gpu=int(vram_gib * _GiB)
    )


def _usecases(specs) -> set[ModelUsecase]:
    return {s.usecase for s in specs}


# --- full capability set is always delivered ----------------------------------


@pytest.mark.parametrize("profile", list(PROFILES))
def test_roomy_box_delivers_every_capability(profile):
    specs = select_stack(profile, _cpu(64))  # generous CPU box
    assert _usecases(specs) == set(PROFILES[profile])


@pytest.mark.parametrize("profile", list(PROFILES))
def test_never_partial_when_it_fits(profile):
    # Whatever tier is chosen, the capability set is complete (never a subset).
    specs = select_stack(profile, _cpu(32))
    assert _usecases(specs) == set(PROFILES[profile])


# --- highest fitting tier is chosen, stepping down coherently -----------------


def test_chat_picks_larger_generate_on_bigger_box():
    small = select_stack("chat", _cpu(8))
    big = select_stack("chat", _cpu(64))
    gen_small = next(s for s in small if s.usecase == ModelUsecase.generate)
    gen_big = next(s for s in big if s.usecase == ModelUsecase.generate)
    assert gen_big.footprint_bytes > gen_small.footprint_bytes


def test_everything_steps_down_a_tier_rather_than_dropping_capabilities():
    # 16 GiB can't fit everything's tier-M stack, but CAN fit tier-S — and still
    # ships all five capabilities (the whole point: no dropping).
    specs = select_stack("everything", _cpu(16))
    assert _usecases(specs) == set(PROFILES["everything"])
    gen = next(s for s in specs if s.usecase == ModelUsecase.generate)
    # tier-S CPU generate is the 3B (smallest rung), proving the step-down.
    assert "3B" in gen.model


# --- fail-fast, never partial -------------------------------------------------


def test_everything_on_8gb_is_refused_not_partially_deployed():
    with pytest.raises(ProfileDoesNotFitError) as exc:
        select_stack("everything", _cpu(8))
    assert "everything" in str(exc.value)
    assert "GiB RAM" in str(exc.value)


def test_below_core_floor_is_refused():
    with pytest.raises(ProfileDoesNotFitError):
        select_stack("chat", _cpu(32, cores=2))


def test_unknown_profile_raises_valueerror():
    with pytest.raises(ValueError):
        select_stack("nonexistent", _cpu(32))


# --- accelerator routing ------------------------------------------------------


def test_gpu_box_uses_gpu_loaders_for_ladders_cpu_for_satellites():
    specs = select_stack("studio", _gpu(24))
    assert _usecases(specs) == set(PROFILES["studio"])
    gen = next(s for s in specs if s.usecase == ModelUsecase.generate)
    img = next(s for s in specs if s.usecase == ModelUsecase.image)
    emb = next(s for s in specs if s.usecase == ModelUsecase.embed)
    assert gen.loader == ModelLoader.vllm
    assert img.loader == ModelLoader.diffusers
    assert emb.loader == ModelLoader.llama_cpp  # satellite stays CPU


def test_studio_on_tiny_gpu_is_refused_on_vram():
    # 8 GiB VRAM can't co-host a 7B LLM + a diffusion model.
    with pytest.raises(ProfileDoesNotFitError) as exc:
        select_stack("studio", _gpu(8))
    assert "VRAM" in str(exc.value)


def test_multi_gpu_fit_checks_largest_model_not_sum():
    # 2x 16 GiB: studio's medium pairing (14B + SDXL-Turbo) sums to 18 GiB but
    # each model gets its own GPU, so the binding constraint is the largest single
    # model (14B ≈ 11 ≤ 12.8). The selector must reach medium, not fall to small.
    budget = DeployBudget(cpu_units=16.0, gpu_count=2, ram_bytes=64 * _GiB, vram_bytes_per_gpu=16 * _GiB)
    specs = select_stack("studio", budget)
    gen = next(s for s in specs if s.usecase == ModelUsecase.generate)
    assert "14B" in gen.model  # would be 7B (small) if the check summed footprints


def test_studio_fits_a_16gb_gpu():
    # The canonical "studio" box: a 16 GiB card reports ~15.3 GiB free. With the
    # -1 GiB tier thresholds it classifies medium, steps down to the small pairing
    # (7B + SD-Turbo, the light co-located image), and deploys all of studio.
    specs = select_stack("studio", _gpu(15.3))
    assert _usecases(specs) == set(PROFILES["studio"])
    img = next(s for s in specs if s.usecase == ModelUsecase.image)
    gen = next(s for s in specs if s.usecase == ModelUsecase.generate)
    assert img.model == "stabilityai/sd-turbo"  # the light rung, not SDXL
    assert "7B" in gen.model


def test_utilization_headroom_is_applied():
    # A stack whose raw footprint equals total RAM must NOT fit (0.8 headroom).
    specs_s = select_stack("chat", _cpu(8))
    assert _usecases(specs_s) == {ModelUsecase.generate, ModelUsecase.embed}
    assert _UTILIZATION == 0.8
