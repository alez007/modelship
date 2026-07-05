"""Tests for the profile selector — weighted knapsack, fail-fast.

Asserts the full capability set is always delivered (never partial), that sizing
runs against FREE RAM, that the weight lever prefers a comfortable smaller model
over a starved larger one, that generate/image deploy last, and that too-small
hardware errors rather than degrading."""

from __future__ import annotations

import pytest

from modelship.deploy.profiles.budget import DeployBudget
from modelship.deploy.profiles.catalog import PROFILES
from modelship.deploy.profiles.selector import (
    _DEPLOY_LAST,
    _UTILIZATION,
    ProfileDoesNotFitError,
    select_stack,
)
from modelship.infer.infer_config import ModelLoader, ModelUsecase

_GiB = 1024**3


def _cpu(ram_gib: float, cores: float = 8.0, avail_gib: float | None = None) -> DeployBudget:
    avail = ram_gib if avail_gib is None else avail_gib
    return DeployBudget(
        cpu_units=cores,
        gpu_count=0,
        ram_bytes=int(ram_gib * _GiB),
        vram_bytes_per_gpu=0,
        available_ram_bytes=int(avail * _GiB),
    )


def _gpu(vram_gib: float, ram_gib: float = 64.0, gpus: int = 1, cores: float = 16.0) -> DeployBudget:
    return DeployBudget(
        cpu_units=cores,
        gpu_count=gpus,
        ram_bytes=int(ram_gib * _GiB),
        vram_bytes_per_gpu=int(vram_gib * _GiB),
        available_ram_bytes=int(ram_gib * _GiB),
    )


def _usecases(specs) -> set[ModelUsecase]:
    return {s.usecase for s in specs}


def _gen(specs):
    return next(s for s in specs if s.usecase == ModelUsecase.generate)


# --- full capability set is always delivered ----------------------------------


@pytest.mark.parametrize("profile", list(PROFILES))
def test_roomy_box_delivers_every_capability(profile):
    specs = select_stack(profile, _cpu(64, cores=16))
    assert _usecases(specs) == set(PROFILES[profile])


@pytest.mark.parametrize("profile", list(PROFILES))
def test_never_partial_when_it_fits(profile):
    specs = select_stack(profile, _cpu(32, cores=16))
    assert _usecases(specs) == set(PROFILES[profile])


# --- deploy order: satellites first, generate/image last ----------------------


@pytest.mark.parametrize("profile", list(PROFILES))
def test_generate_and_image_deploy_last(profile):
    specs = select_stack(profile, _cpu(64, cores=16))
    adaptive_seen = False
    for s in specs:
        if s.usecase in _DEPLOY_LAST:
            adaptive_seen = True
        else:
            # No satellite may appear after a generate/image model.
            assert not adaptive_seen, f"{s.usecase} deployed after a generate/image model"
    # The very last model is always an adaptive one (every profile has generate).
    assert specs[-1].usecase in _DEPLOY_LAST


# --- weighted picks: bigger box, comfortable-over-starved lever ----------------


def test_bigger_box_picks_a_larger_generate():
    small = _gen(select_stack("chat", _cpu(8)))
    big = _gen(select_stack("chat", _cpu(64, cores=16)))
    assert big.footprint_bytes > small.footprint_bytes


def test_prefers_comfortable_smaller_over_starved_larger():
    # ~14 GiB free + plenty of cores: the 14B's *minimum* fits (≈10.5 GiB) and so
    # does the 7B's *recommended* (≈8 GiB). The weights make 7B@rec outscore
    # 14B@min, so the selector takes the comfortable 7B rather than a starved 14B.
    specs = select_stack("chat", _cpu(64, cores=16, avail_gib=14))
    assert "7B" in _gen(specs).model


# --- sizing runs against FREE RAM, not total ----------------------------------


def test_available_ram_caps_the_stack_below_total():
    # Huge total RAM but only 8 GiB free → must size like an 8 GiB box (3B), not
    # like a 64 GiB one. This is the core fix: co-resident models ate the RAM.
    roomy = _gen(select_stack("chat", _cpu(64, cores=16)))
    busy = _gen(select_stack("chat", _cpu(64, cores=16, avail_gib=8)))
    assert "3B" in busy.model
    assert busy.footprint_bytes < roomy.footprint_bytes


def test_zero_available_falls_back_to_total():
    # available_ram_bytes == 0 (probe read nothing) must behave like the old total
    # basis, not refuse everything.
    budget = DeployBudget(cpu_units=16.0, gpu_count=0, ram_bytes=64 * _GiB, vram_bytes_per_gpu=0, available_ram_bytes=0)
    specs = select_stack("chat", budget)
    assert _usecases(specs) == set(PROFILES["chat"])


# --- fail-fast, never partial -------------------------------------------------


def test_everything_on_tiny_box_is_refused_not_partially_deployed():
    with pytest.raises(ProfileDoesNotFitError) as exc:
        select_stack("everything", _cpu(6))
    assert "everything" in str(exc.value)
    assert "RAM" in str(exc.value)


def test_below_core_floor_is_refused():
    with pytest.raises(ProfileDoesNotFitError):
        select_stack("chat", _cpu(32, cores=1))


def test_unknown_profile_raises_valueerror():
    with pytest.raises(ValueError):
        select_stack("nonexistent", _cpu(32))


def test_capability_with_no_catalog_models_refuses_cleanly(monkeypatch):
    # A profile naming a usecase the catalog has no models for must raise a clean
    # ProfileDoesNotFitError, not crash on min() over an empty pool.
    from modelship.deploy.profiles import selector as sel

    monkeypatch.setitem(sel.PROFILES, "translate-only", (ModelUsecase.translation,))
    with pytest.raises(ProfileDoesNotFitError) as exc:
        select_stack("translate-only", _cpu(32))
    assert "translation" in str(exc.value)


# --- accelerator routing + VRAM placement -------------------------------------


def test_gpu_box_uses_gpu_loaders_for_generate_image_cpu_for_satellites():
    specs = select_stack("studio", _gpu(24))
    assert _usecases(specs) == set(PROFILES["studio"])
    gen = _gen(specs)
    img = next(s for s in specs if s.usecase == ModelUsecase.image)
    emb = next(s for s in specs if s.usecase == ModelUsecase.embed)
    assert gen.loader == ModelLoader.vllm
    assert img.loader == ModelLoader.diffusers
    assert emb.loader == ModelLoader.llama_server  # satellite stays CPU


def test_studio_on_tiny_gpu_is_refused_on_vram():
    # 8 GiB VRAM can't co-host even the smallest LLM + image model on one card.
    with pytest.raises(ProfileDoesNotFitError) as exc:
        select_stack("studio", _gpu(8))
    assert "VRAM" in str(exc.value)


def test_multi_gpu_fit_checks_largest_model_not_sum():
    # 2x 16 GiB: a 14B (11) + an image model (≤8) sum to >16 but each gets its own
    # card, so the binding constraint is the largest single model (11 ≤ 16). The
    # selector must reach the 14B, not fall back to the 7B as a summed check would.
    specs = select_stack("studio", _gpu(16, gpus=2))
    assert "14B" in _gen(specs).model


def test_gpu_models_host_ram_counts_against_the_ram_cap():
    # Ample VRAM but a starved host: studio's GPU generate + image each need a few
    # GiB of *host* RAM (weights/KV/CUDA ctx), which must count. With only ~4 GiB
    # free the stack can't fit even though every model fits VRAM — the old code
    # excluded GPU host RAM and would have wrongly accepted it.
    starved = DeployBudget(
        cpu_units=16.0, gpu_count=2, ram_bytes=64 * _GiB, vram_bytes_per_gpu=24 * _GiB, available_ram_bytes=4 * _GiB
    )
    with pytest.raises(ProfileDoesNotFitError):
        select_stack("studio", starved)


def test_utilization_headroom_is_applied():
    assert _UTILIZATION == 0.8
    with pytest.raises(ProfileDoesNotFitError):
        select_stack("chat", _cpu(2.5, cores=8))
