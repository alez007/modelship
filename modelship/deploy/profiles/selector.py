"""Select the concrete model stack for a profile on a given budget.

Policy (locked with the user): **weighted, fail-fast.** A profile is all-or-nothing
on capabilities — we never drop one. For each capability we pick exactly one model
from its candidate pool, at either its *minimum* or *recommended* resource set, and
choose the combination that **maximises total quality weight** while fitting the
box's free cpu / RAM (and per-GPU VRAM). Weights are hand-set in the catalog so a
smaller model at `recommended` can outscore a bigger one at `min` — the tuning
lever. If no combination fits even at minimum, we raise `ProfileDoesNotFitError`
rather than ship a partial stack.

The search is a Multiple-Choice Multi-Dimensional Knapsack, but the instances are
tiny (≤ ~1k combos for `everything`, dozens for the rest) so we brute-force every
one-pick-per-capability combination — optimal, no DP.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from modelship.deploy.profiles.budget import DeployBudget
from modelship.deploy.profiles.catalog import (
    PROFILES,
    ModelSpec,
    ModeReq,
    candidates,
)
from modelship.deploy.profiles.tiers import Accelerator, accelerator_for
from modelship.infer.infer_config import ModelUsecase
from modelship.logging import get_logger

logger = get_logger("deploy.profiles.selector")

# Fraction of free RAM a stack may occupy — headroom for the OS, page cache, and
# KV-cache growth. Matches the llama_cpp preflight's RAM util.
_UTILIZATION = 0.8

# Absolute floor: below this we refuse outright (the user's "don't even try").
_MIN_CPU_UNITS = 2

# Capabilities whose models adapt their own footprint at runtime (llama.cpp n_ctx,
# diffusers tiling) and so must deploy LAST — after the fixed satellites are resident
# — so their preflight sizes against the RAM that's actually left.
_DEPLOY_LAST = (ModelUsecase.generate, ModelUsecase.image)

Mode = Literal["min", "rec"]


class ProfileDoesNotFitError(RuntimeError):
    """The chosen profile cannot be delivered in full on this hardware."""


@dataclass(frozen=True)
class _Candidate:
    """One (model, resource-mode) option the knapsack may pick for a capability.
    `req` carries that mode's cpu/ram demand and its quality weight."""

    spec: ModelSpec
    mode: Mode
    req: ModeReq


def select_stack(profile: str, budget: DeployBudget) -> list[ModelSpec]:
    """Return the model specs for `profile` — the highest-weight combination that
    fits this box, ordered satellites-first / generate+image-last (deploy order).

    Raises `ProfileDoesNotFitError` if no combination fits even at minimum or the box
    is below the absolute core floor; `ValueError` for an unknown profile."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose one of {sorted(PROFILES)}")
    caps = PROFILES[profile]

    if budget.cpu_units < _MIN_CPU_UNITS:
        raise ProfileDoesNotFitError(
            f"profile {profile!r} needs at least {_MIN_CPU_UNITS} CPU cores; Ray reports "
            f"{budget.cpu_units:.0f}. Free up cores, raise RAY_HEAD_CPU_NUM, or write "
            f"config/models.yaml by hand."
        )

    accel = accelerator_for(budget)
    cpu_cap = budget.cpu_units
    ram_cap = int((budget.available_ram_bytes or budget.ram_bytes) * _UTILIZATION)
    vram_cap = budget.vram_bytes_per_gpu

    # Per-capability candidate lists (each model in min + rec mode), admission-filtered:
    # a candidate whose own demand already exceeds a single host cap can never be in
    # a feasible combo, so drop it before enumerating. An empty group means we can't
    # place that capability at all → fail fast with a specific message.
    groups: list[list[_Candidate]] = []
    for uc in caps:
        admitted = [
            cand for spec in candidates(uc, accel) for cand in _modes(spec) if _admits(cand, cpu_cap, ram_cap, vram_cap)
        ]
        if not admitted:
            raise ProfileDoesNotFitError(_no_candidate_message(profile, uc, accel, budget))
        groups.append(admitted)

    best_combo: tuple[_Candidate, ...] | None = None
    best_key = (float("-inf"), 0)
    for combo in itertools.product(*groups):
        cpu_sum = sum(c.req.cpu for c in combo)
        ram_sum = sum(c.req.ram_bytes for c in combo if not c.spec.draws_from_vram)
        if cpu_sum > cpu_cap or ram_sum > ram_cap:
            continue
        if not _vram_fits([c.spec for c in combo], budget.gpu_count, vram_cap):
            continue
        weight = sum(c.req.weight for c in combo)
        # Maximise weight; tie-break toward more RAM headroom (lower ram_sum).
        key = (weight, -ram_sum)
        if key > best_key:
            best_key, best_combo = key, combo

    if best_combo is None:
        raise ProfileDoesNotFitError(_does_not_fit_message(profile, accel, budget, groups))

    logger.info(
        "profiles: %r -> %s stack (%d models, weight=%.0f): %s",
        profile,
        accel.value,
        len(best_combo),
        best_key[0],
        ", ".join(f"{c.spec.usecase.value}={_short(c.spec.model)}@{c.mode}" for c in best_combo),
    )

    specs = [c.spec for c in best_combo]
    # Deploy order == list order: satellites first, the adaptive generate/image last.
    specs.sort(key=lambda s: s.usecase in _DEPLOY_LAST)
    return specs


def _modes(spec: ModelSpec) -> Iterator[_Candidate]:
    """The two resource modes (recommended, minimum) of a model as candidates."""
    yield _Candidate(spec, "rec", spec.req_rec)
    yield _Candidate(spec, "min", spec.req_min)


def _admits(cand: _Candidate, cpu_cap: float, ram_cap: int, vram_cap: int) -> bool:
    """True if this single candidate's own demand fits the host caps. GPU models are
    gated on VRAM (their coarse footprint); CPU models on RAM. cpu applies to both."""
    if cand.req.cpu > cpu_cap:
        return False
    if cand.spec.draws_from_vram:
        return cand.spec.footprint_bytes <= vram_cap
    return cand.req.ram_bytes <= ram_cap


def _vram_fits(specs: list[ModelSpec], gpu_count: int, vram_cap: int) -> bool:
    """Whether the VRAM-drawing models fit a single GPU's budget. Mirrors how the
    generator places them: with at least as many GPUs as GPU models each gets its
    own card (constraint = the largest single model), otherwise they share one card
    (constraint = their sum)."""
    gpu_specs = [s for s in specs if s.draws_from_vram]
    if not gpu_specs:
        return True
    if gpu_count >= len(gpu_specs):
        need = max(s.footprint_bytes for s in gpu_specs)  # one model per GPU
    else:
        need = sum(s.footprint_bytes for s in gpu_specs)  # shared on one GPU
    return need <= vram_cap


def _short(model: str) -> str:
    """Trailing path component of a model id, for compact log lines."""
    return model.split("/")[-1]


def _no_candidate_message(profile: str, uc: ModelUsecase, accel: Accelerator, budget: DeployBudget) -> str:
    """Message for a capability with no admissible model — even its smallest option
    is too big for a single host dimension."""
    pool = candidates(uc, accel)
    lighter = ", ".join(p for p in PROFILES if p != profile)
    if accel == Accelerator.gpu and any(s.draws_from_vram for s in pool):
        smallest = min(s.footprint_bytes for s in pool if s.draws_from_vram)
        return (
            f"profile {profile!r} can't place its {uc.value} model: the smallest option needs "
            f"~{smallest / 1024**3:.0f} GiB VRAM, but only {budget.vram_bytes_per_gpu / 1024**3:.0f} GiB/GPU is "
            f"free. Choose a lighter profile ({lighter}), add VRAM, or write config/models.yaml by hand."
        )
    ram_avail = budget.available_ram_bytes or budget.ram_bytes
    smallest_ram = min(s.req_min.ram_bytes for s in pool)
    smallest_cpu = min(s.req_min.cpu for s in pool)
    return (
        f"profile {profile!r} can't place its {uc.value} model: the smallest option needs "
        f"~{smallest_ram / 1024**3:.1f} GiB RAM and {smallest_cpu:.0f} cores, but only "
        f"{ram_avail / 1024**3 * _UTILIZATION:.1f} GiB usable RAM and {budget.cpu_units:.0f} cores are free. "
        f"Choose a lighter profile ({lighter}), free up RAM, or write config/models.yaml by hand."
    )


def _does_not_fit_message(
    profile: str, accel: Accelerator, budget: DeployBudget, groups: list[list[_Candidate]]
) -> str:
    """Message when each capability has options but no combination fits together —
    reports the lightest possible combined demand against the caps."""
    lighter = ", ".join(p for p in PROFILES if p != profile)
    cpu_need = sum(min(c.req.cpu for c in g) for g in groups)
    if accel == Accelerator.gpu:
        gpu_lightest = [min((c.spec.footprint_bytes for c in g if c.spec.draws_from_vram), default=0) for g in groups]
        gpu_lightest = [x for x in gpu_lightest if x]
        vram_need = max(gpu_lightest, default=0) if budget.gpu_count >= len(gpu_lightest) else sum(gpu_lightest)
        return (
            f"profile {profile!r} does not fit: its lightest stack needs ~{vram_need / 1024**3:.0f} GiB VRAM "
            f"and {cpu_need:.0f} cores, but only {budget.vram_bytes_per_gpu / 1024**3:.0f} GiB/GPU and "
            f"{budget.cpu_units:.0f} cores are free. Choose a lighter profile ({lighter}), add VRAM, or write "
            f"config/models.yaml by hand."
        )
    ram_need = sum(min((c.req.ram_bytes for c in g if not c.spec.draws_from_vram), default=0) for g in groups)
    ram_avail = budget.available_ram_bytes or budget.ram_bytes
    return (
        f"profile {profile!r} does not fit: its lightest stack needs ~{ram_need / 1024**3:.0f} GiB RAM and "
        f"{cpu_need:.0f} cores, but only {ram_avail / 1024**3 * _UTILIZATION:.0f} GiB usable RAM and "
        f"{budget.cpu_units:.0f} cores are free. Choose a lighter profile ({lighter}), free up RAM, or write "
        f"config/models.yaml by hand."
    )
