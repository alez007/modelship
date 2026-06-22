from __future__ import annotations

from typing import Any

from modelship.infer.infer_config import ModelshipModelConfig
from modelship.logging import get_logger
from modelship.preflight.base import HardwareProfile

logger = get_logger("preflight.stable_diffusion_cpp")

# Below this much *free* system RAM, default to tiled VAE decode. The VAE decode is
# the memory high-water mark for SD/SDXL generation, so tiling it trades a little
# speed for a markedly lower peak — the right default on small NAS-class hosts.
# Measured against free (not total) RAM because the image model deploys after the
# generate model + satellites, so what's left is what matters.
_VAE_TILING_RAM_THRESHOLD_BYTES = 8 * 1024**3


class StableDiffusionCppPreflight:
    """Hardware-aware defaults for the stable_diffusion_cpp loader. Conservative
    in v1: the only recommendation is enabling VAE tiling on low-RAM hosts. Sizing
    by model footprint (like the llama_cpp preflight does for n_ctx) is a
    follow-up once split-file resolution lands."""

    def recommend(self, config: ModelshipModelConfig, hw: HardwareProfile) -> dict[str, Any]:
        if hw.ram_bytes <= 0:
            logger.info("preflight '%s': skipping — system RAM not discoverable", config.name)
            return {}

        ram_basis = hw.sizing_ram_bytes
        fallback = " [total fallback]" if not hw.available_ram_bytes else ""
        rec: dict[str, Any] = {}
        if ram_basis < _VAE_TILING_RAM_THRESHOLD_BYTES:
            rec["vae_tiling"] = True
            logger.info(
                "preflight stable_diffusion_cpp '%s': ram_avail=%.2f GiB%s < %.0f GiB → recommend vae_tiling=True",
                config.name,
                ram_basis / 1024**3,
                fallback,
                _VAE_TILING_RAM_THRESHOLD_BYTES / 1024**3,
            )
        return rec
