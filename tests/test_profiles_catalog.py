"""Tests for the profile catalog.

Pure data/functions — no cluster, no model downloads. Source strings are verified
live against HF separately; here we assert structure, resource sets, and weights."""

from __future__ import annotations

import pytest

from modelship.deploy.profiles.catalog import (
    PROFILES,
    ModelSpec,
    candidates,
)
from modelship.deploy.profiles.tiers import Accelerator
from modelship.infer.infer_config import ModelLoader, ModelUsecase

_ALL_USECASES = sorted({uc for caps in PROFILES.values() for uc in caps}, key=lambda u: u.value)


def _every_spec() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for uc in _ALL_USECASES:
        for accel in (Accelerator.cpu, Accelerator.gpu):
            specs.extend(candidates(uc, accel))
    return specs


# --- profiles -----------------------------------------------------------------


def test_profiles_are_the_locked_four():
    assert set(PROFILES) == {"chat", "assistant", "studio", "everything"}


# --- candidate pools ----------------------------------------------------------


@pytest.mark.parametrize("uc", _ALL_USECASES)
def test_every_required_usecase_has_candidates_on_both_accelerators(uc):
    assert candidates(uc, Accelerator.cpu), f"{uc} has no CPU candidates"
    assert candidates(uc, Accelerator.gpu), f"{uc} has no GPU candidates"


def test_generate_and_image_split_loaders_by_accelerator():
    for s in candidates(ModelUsecase.generate, Accelerator.cpu) + candidates(ModelUsecase.image, Accelerator.cpu):
        assert s.draws_from_vram is False
    for s in candidates(ModelUsecase.generate, Accelerator.gpu):
        assert s.loader == ModelLoader.vllm and s.draws_from_vram is True
    for s in candidates(ModelUsecase.image, Accelerator.gpu):
        assert s.loader == ModelLoader.diffusers and s.draws_from_vram is True


def test_satellites_are_the_same_cpu_pinned_pool_on_both_accelerators():
    for uc in (ModelUsecase.embed, ModelUsecase.tts, ModelUsecase.transcription):
        cpu_pool = candidates(uc, Accelerator.cpu)
        assert candidates(uc, Accelerator.gpu) == cpu_pool  # identical tuple
        assert all(s.draws_from_vram is False for s in cpu_pool)


def test_transcription_pool_offers_tiny_base_and_small():
    models = {s.model for s in candidates(ModelUsecase.transcription, Accelerator.cpu)}
    assert models == {"tiny", "base", "small"}


def test_unknown_usecase_returns_empty_pool():
    # ModelUsecase has members not in any profile (e.g. rerank); pool is empty, not an error.
    missing = [uc for uc in ModelUsecase if uc not in _ALL_USECASES]
    for uc in missing:
        assert candidates(uc, Accelerator.cpu) == ()


# --- resource sets + weights --------------------------------------------------


def test_every_spec_has_coherent_min_and_rec_reqs():
    for s in _every_spec():
        assert s.req_min.cpu <= s.req_rec.cpu, s.model
        assert s.req_min.ram_bytes <= s.req_rec.ram_bytes, s.model
        # A comfortable run is worth at least as much as the same model starved.
        assert s.req_min.weight <= s.req_rec.weight, s.model
        assert s.req_min.weight > 0, s.model


@pytest.mark.parametrize("accel", [Accelerator.cpu, Accelerator.gpu])
def test_generate_and_image_pools_ascend_in_footprint_and_weight(accel):
    for uc in (ModelUsecase.generate, ModelUsecase.image):
        pool = candidates(uc, accel)
        assert [s.footprint_bytes for s in pool] == sorted(s.footprint_bytes for s in pool)
        # Bigger rungs are worth more at the same (recommended) mode.
        assert [s.req_rec.weight for s in pool] == sorted(s.req_rec.weight for s in pool)


def test_recommended_of_a_rung_outscores_minimum_of_the_next():
    # The tuning lever: a smaller model running comfortably should beat the next
    # rung up running starved, so the selector prefers the comfortable fit.
    from itertools import pairwise

    gen = candidates(ModelUsecase.generate, Accelerator.cpu)
    for smaller, bigger in pairwise(gen):
        assert smaller.req_rec.weight > bigger.req_min.weight


def test_image_turbo_rungs_use_few_steps():
    cpu_img = candidates(ModelUsecase.image, Accelerator.cpu)
    # SD-Turbo / SDXL-Turbo: ~4 steps; SDXL-base: ~30.
    assert cpu_img[0].loader_config["sample_steps"] == 4
    assert cpu_img[1].loader_config["sample_steps"] == 4
    assert cpu_img[2].loader_config["sample_steps"] == 30
