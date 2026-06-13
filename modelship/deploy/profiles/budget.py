"""The resource budget the profile generator sizes models against.

Split-sourced, per the detection model we locked:

- **CPU / GPU counts** come from Ray's ledger (`ray.cluster_resources()`), which
  already reflects `RAY_HEAD_CPU_NUM` / `RAY_HEAD_GPU_NUM` (or Ray's auto-detected
  value, including the container CFS-quota). These are what a generated
  deployment may *request* of the scheduler, so they bound `num_cpus`/`num_gpus`
  and gate which capabilities we offer.
- **RAM** comes from `detect_ram_bytes()` (cgroup-aware physical RAM) and
  **per-GPU VRAM** from `detect_gpus()` — both standalone utilities rather than
  the whole `discover_hardware()`, since we don't need its CPU-count or the
  HardwareProfile wrapper here. Ray doesn't ledger usable RAM/VRAM bytes (its
  `memory` resource under-reports, reserving object-store space), so these come
  from physical/container detection.

This runs on the driver (outside any actor) — the single-node / homogeneous-pool
assumption applies (see the profiles plan in memory). Heterogeneous multi-node
clusters write `models.yaml` by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelship.logging import get_logger
from modelship.preflight import detect_gpus, detect_ram_bytes

logger = get_logger("deploy.profiles.budget")


@dataclass(frozen=True)
class DeployBudget:
    """What the generator may allocate on this box.

    `cpu_units` / `gpu_count` are Ray-ledger counts (schedulable). `ram_bytes` is
    the container-aware physical RAM. `vram_bytes_per_gpu` is the conservative
    per-GPU VRAM across the pool (0 when no usable GPU)."""

    cpu_units: float
    gpu_count: int
    ram_bytes: int
    vram_bytes_per_gpu: int

    @property
    def has_gpu(self) -> bool:
        """True when Ray will schedule GPUs *and* we measured their VRAM — the
        signal to pick the GPU bundle over the CPU bundle."""
        return self.gpu_count > 0 and self.vram_bytes_per_gpu > 0


def read_deploy_budget() -> DeployBudget:
    """Read the deploy budget from Ray's ledger + physical detection.

    Must be called after `ray.init()` (the deploy driver is already connected).
    """
    import ray

    ledger = ray.cluster_resources()
    cpu_units = float(ledger.get("CPU", 0.0))
    gpu_count = int(ledger.get("GPU", 0))

    ram_bytes = detect_ram_bytes()

    # Per-GPU VRAM for tiering: take the smallest across the pool so a
    # homogeneous fleet sizes to its real per-card budget and a (mis)matched one
    # degrades conservatively rather than over-promising.
    vram_per_gpu = min((g.available_bytes for g in detect_gpus()), default=0)

    if gpu_count == 0:
        # Ray won't schedule GPUs here (no GPU, or fenced to 0) — force the CPU
        # path even if the driver physically sees cards.
        vram_per_gpu = 0
    elif vram_per_gpu == 0:
        # Ray ledgers GPUs but the driver couldn't read their VRAM (no CUDA
        # context / pynvml). Degrade to the CPU bundle rather than guess a tier.
        logger.warning(
            "profiles: Ray ledger reports %d GPU(s) but no VRAM was detected on the driver; "
            "falling back to the CPU model bundle.",
            gpu_count,
        )

    budget = DeployBudget(
        cpu_units=cpu_units,
        gpu_count=gpu_count,
        ram_bytes=ram_bytes,
        vram_bytes_per_gpu=vram_per_gpu,
    )
    logger.info(
        "profiles: deploy budget — cpu_units=%.1f gpu_count=%d ram=%.1f GiB vram/gpu=%.1f GiB (has_gpu=%s)",
        budget.cpu_units,
        budget.gpu_count,
        budget.ram_bytes / 1024**3,
        budget.vram_bytes_per_gpu / 1024**3,
        budget.has_gpu,
    )
    return budget
