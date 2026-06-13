"""Select the concrete model stack for a profile on a given budget.

Policy (locked with the user): **tier-exact, fail-fast.** A profile is
all-or-nothing on capabilities — we never drop one, and we never piecemeal-shrink
an individual model. We pick the HIGHEST tier whose *complete* tier-exact stack
fits the budget, stepping down through coherent lower-tier bundles. If even the
smallest tier's stack doesn't fit (or the box is below the absolute floor), we
raise `ProfileDoesNotFitError` rather than deploy a partial/degraded stack — the
user picked the profile to get the whole set.
"""

from __future__ import annotations

from modelship.deploy.profiles.budget import DeployBudget
from modelship.deploy.profiles.catalog import (
    PROFILES,
    ModelSpec,
    generate_at,
    image_at,
    satellite,
)
from modelship.deploy.profiles.tiers import Accelerator, Tier, classify
from modelship.infer.infer_config import ModelUsecase
from modelship.logging import get_logger

logger = get_logger("deploy.profiles.selector")

# Fraction of RAM / VRAM a stack may occupy — headroom for the OS, page cache,
# CUDA context, and KV-cache growth. Matches the llama_cpp preflight's RAM util.
_UTILIZATION = 0.8

# Absolute floor: below this we refuse outright (the user's "don't even try").
_MIN_CPU_UNITS = 4


class ProfileDoesNotFitError(RuntimeError):
    """The chosen profile cannot be delivered in full on this hardware."""


def select_stack(profile: str, budget: DeployBudget) -> list[ModelSpec]:
    """Return the model specs for `profile` at the highest tier that fits.

    Raises `ProfileDoesNotFitError` if even the smallest tier doesn't fit or the
    box is below the absolute core floor; `ValueError` for an unknown profile."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose one of {sorted(PROFILES)}")
    caps = PROFILES[profile]
    # The classified tier is the CEILING: a 16 GiB box is "tier M", so `chat`
    # gets the 7B even though a 14B's weights would technically squeeze in (it'd
    # leave no KV headroom). Heavy profiles step DOWN from here; we never exceed
    # the tier the hardware classifies as.
    accel, max_tier = classify(budget)

    if budget.cpu_units < _MIN_CPU_UNITS:
        raise ProfileDoesNotFitError(
            f"profile {profile!r} needs at least {_MIN_CPU_UNITS} CPU cores; Ray reports "
            f"{budget.cpu_units:.0f}. Free up cores, raise RAY_HEAD_CPU_NUM, or write "
            f"config/models.yaml by hand."
        )

    ram_avail = int(budget.ram_bytes * _UTILIZATION)
    vram_avail = int(budget.vram_bytes_per_gpu * _UTILIZATION)

    # Start at the classified tier (the ceiling); step down to a coherent
    # lower-tier bundle for heavy profiles that don't fit there.
    for tier in (Tier(t) for t in range(max_tier, Tier.small - 1, -1)):
        specs = _build_stack(caps, accel, tier)
        ram_need, vram_need = _footprints(specs)
        if ram_need <= ram_avail and vram_need <= vram_avail:
            logger.info(
                "profiles: %r -> %s tier %s (%d models; ram %.1f/%.1f GiB, vram %.1f/%.1f GiB)",
                profile,
                accel.value,
                tier.name,
                len(specs),
                ram_need / 1024**3,
                ram_avail / 1024**3,
                vram_need / 1024**3,
                vram_avail / 1024**3,
            )
            return specs

    # Even the smallest tier didn't fit — refuse with the requirement spelled out.
    ram_need, vram_need = _footprints(_build_stack(caps, accel, Tier.small))
    raise ProfileDoesNotFitError(_too_small_message(profile, accel, budget, ram_need, vram_need))


def _build_stack(caps: tuple[ModelUsecase, ...], accel: Accelerator, tier: Tier) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for uc in caps:
        if uc == ModelUsecase.generate:
            specs.append(generate_at(accel, tier))
        elif uc == ModelUsecase.image:
            specs.append(image_at(accel, tier))
        else:
            specs.append(satellite(uc, tier))
    return specs


def _footprints(specs: list[ModelSpec]) -> tuple[int, int]:
    """`(ram_bytes, vram_bytes)` the stack needs. Satellites and CPU loaders draw
    RAM; vllm/diffusers draw VRAM."""
    ram = sum(s.footprint_bytes for s in specs if not s.draws_from_vram)
    vram = sum(s.footprint_bytes for s in specs if s.draws_from_vram)
    return ram, vram


def _too_small_message(profile: str, accel: Accelerator, budget: DeployBudget, ram_need: int, vram_need: int) -> str:
    lighter = ", ".join(p for p in PROFILES if p != profile)
    if accel == Accelerator.gpu:
        need_gib = vram_need / 1024**3 / _UTILIZATION
        have_gib = budget.vram_bytes_per_gpu / 1024**3
        unit = "VRAM"
    else:
        need_gib = ram_need / 1024**3 / _UTILIZATION
        have_gib = budget.ram_bytes / 1024**3
        unit = "RAM"
    return (
        f"profile {profile!r} does not fit this hardware: its smallest tier needs "
        f"~{need_gib:.0f} GiB {unit}, but only {have_gib:.0f} GiB is available. Choose a "
        f"lighter profile ({lighter}), add {unit}, or write config/models.yaml by hand."
    )
