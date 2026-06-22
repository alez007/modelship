"""Accelerator family for a `DeployBudget`.

The accelerator split (GPU present vs not) chooses which loader family the catalog
draws from. Model *sizing* within a family is no longer a tier bucket — the
selector runs a weighted knapsack over each capability's candidate pool against the
box's free cpu/RAM (and per-GPU VRAM). See `selector.py`.
"""

from __future__ import annotations

from enum import StrEnum

from modelship.deploy.profiles.budget import DeployBudget


class Accelerator(StrEnum):
    cpu = "cpu"
    gpu = "gpu"


def accelerator_for(budget: DeployBudget) -> Accelerator:
    """GPU bundle when Ray will schedule GPUs *and* we measured their VRAM,
    else the CPU bundle."""
    return Accelerator.gpu if budget.has_gpu else Accelerator.cpu
