"""Map a `DeployBudget` to an accelerator + tier bucket.

The accelerator split (GPU present vs not) chooses which loader family the
catalog draws from; the tier index (0/1/2) picks the rung of the generate/image
ladders. Satellites (embed/tts/transcription) are constant across tiers.

Thresholds (locked with the user):
  CPU, on container-aware RAM:   S < 16 GiB,  M 16-31,  L >= 32
  CPU, also capped by cores:     S 4-5,       M 6-7,    L >= 8
  GPU, on per-GPU free VRAM:     S < 15 GiB,  M 15-22,  L >= 23

The GPU thresholds sit 1 GiB below the nominal card sizes (8/16/24) because we
measure *free* VRAM, not the device's total: a "16 GiB" card only ever reports
~15.3 GiB free (driver + CUDA context overhead), so a >= 16 check would miss it.
Dropping each rung by 1 GiB lets 8/16/24 GiB cards land in S/M/L as expected.

RAM picks the size, but cores gate it too: CPU inference speed scales with cores,
so the final CPU tier is the LOWER of the RAM tier and the core tier — a 4-core /
32 GiB box runs a tier-S model, not a sluggish tier-L one. GPU tiers are
VRAM-only (the GPU does the compute; host cores don't gate it).

Boxes below the smallest bucket still classify as tier 0; the budget-aware
selector handles "too small" by stepping down / refusing.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

from modelship.deploy.profiles.budget import DeployBudget

_GiB = 1024**3


class Accelerator(StrEnum):
    cpu = "cpu"
    gpu = "gpu"


class Tier(IntEnum):
    """Ladder rung. S/M/L for CPU; 8G/16G/24G for GPU — same ordinal."""

    small = 0
    medium = 1
    large = 2


# (lower-bound, Tier) rungs, checked high→low. The smallest rung has no lower
# bound (anything below the next threshold lands here). GiB for RAM/VRAM, raw
# core counts for the CPU core cap.
_CPU_TIERS = ((32, Tier.large), (16, Tier.medium))
# 1 GiB below nominal card sizes (24/16) — we measure free VRAM, not total.
_GPU_TIERS = ((23, Tier.large), (15, Tier.medium))
_CPU_CORE_TIERS = ((8, Tier.large), (6, Tier.medium))


def _bucket(value: float, rungs: tuple[tuple[int, Tier], ...]) -> Tier:
    for lower, tier in rungs:
        if value >= lower:
            return tier
    return Tier.small


def classify(budget: DeployBudget) -> tuple[Accelerator, Tier]:
    """Pick the accelerator family and tier rung for this box."""
    if budget.has_gpu:
        return Accelerator.gpu, _bucket(budget.vram_bytes_per_gpu / _GiB, _GPU_TIERS)
    # CPU: RAM picks the size, cores cap it — take the lower of the two.
    ram_tier = _bucket(budget.ram_bytes / _GiB, _CPU_TIERS)
    core_tier = _bucket(budget.cpu_units, _CPU_CORE_TIERS)
    return Accelerator.cpu, min(ram_tier, core_tier)
