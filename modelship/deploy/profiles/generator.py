"""Generate a transparent, editable `models.yaml` from a profile.

`MSHIP_MODEL_STACK=<profile>` → read the hardware budget → select the stack →
allocate per-deployment resources → write a normal, commented `models.yaml`. From
there the existing deploy path runs unchanged; the file is the user's to edit.

Resource allocation (the request budgeting we locked):
- **CPU cores** (`num_cpus`) are split from Ray's ledger so the sum never exceeds
  it (or Ray would never schedule): the generate anchor gets the bulk, satellites
  ~1 core each, image ~2.
- **GPU** (`num_gpus`) is shared fractionally when a profile puts both generate
  and image on one GPU (`studio`/`everything`): each gets `gpu_count / n_gpu_models`.
  `num_gpus` is the *single* sharing knob — the loader derives each process's VRAM
  cap from it (`base_infer._get_memory_fraction` feeds it to vLLM's
  `gpu_memory_utilization` and diffusers' per-process fraction), so we deliberately
  do NOT also write `gpu_memory_utilization` here (it would be overridden anyway).
"""

from __future__ import annotations

import yaml

from modelship.deploy.profiles.budget import DeployBudget, read_deploy_budget
from modelship.deploy.profiles.catalog import ModelSpec
from modelship.deploy.profiles.selector import select_stack
from modelship.infer.infer_config import ModelLoader, ModelUsecase
from modelship.logging import get_logger

logger = get_logger("deploy.profiles.generator")

# Base CPU-core reservation per role before the generate anchor absorbs leftover.
_CORES_GENERATE_BASE = 2.0
_CORES_IMAGE = 2.0
_CORES_SATELLITE = 1.0

# Maps a built-in loader to the models.yaml field its inner config lives under.
_LOADER_CONFIG_FIELD = {
    ModelLoader.llama_cpp: "llama_cpp_config",
    ModelLoader.stable_diffusion_cpp: "stable_diffusion_cpp_config",
    ModelLoader.diffusers: "diffusers_config",
    ModelLoader.vllm: "vllm_engine_kwargs",
}


def generate_models_yaml(profile: str, path: str) -> None:
    """Select a stack for `profile` on the detected hardware and write it to
    `path`. Raises `ProfileDoesNotFitError` (from the selector) without writing
    anything if the profile can't be delivered in full."""
    budget = read_deploy_budget()
    specs = select_stack(profile, budget)  # raises before any file is written
    entries = _to_entries(specs, budget)
    text = _render(profile, budget, entries)
    with open(path, "w") as f:
        f.write(text)
    logger.info("profiles: wrote %d-model '%s' stack to %s", len(entries), profile, path)


def _to_entries(specs: list[ModelSpec], budget: DeployBudget) -> list[dict]:
    cpus = _cpu_allocation(specs, budget.cpu_units)
    gpu_share = _gpu_allocation(specs, budget.gpu_count)

    entries: list[dict] = []
    for spec, num_cpus in zip(specs, cpus, strict=True):
        entry: dict = {
            "name": spec.usecase.value,
            "model": spec.model,
            "usecase": spec.usecase.value,
            "loader": spec.loader.value,
            "num_cpus": num_cpus,
            "num_gpus": gpu_share.get(id(spec), 0),
        }
        if spec.plugin:
            entry["plugin"] = spec.plugin
        if spec.plugin_config:
            entry["plugin_config"] = dict(spec.plugin_config)

        # No gpu_memory_utilization here: for a fractional num_gpus the loader
        # derives the VRAM cap from num_gpus itself (base_infer._get_memory_fraction),
        # and that override would clobber anything we wrote anyway.
        loader_cfg = dict(spec.loader_config) if spec.loader_config else {}
        if loader_cfg and spec.loader in _LOADER_CONFIG_FIELD:
            entry[_LOADER_CONFIG_FIELD[spec.loader]] = loader_cfg

        entries.append(entry)
    return entries


def _gpu_allocation(specs: list[ModelSpec], gpu_count: int) -> dict[int, float]:
    """Per-deployment `num_gpus` for the VRAM-drawing models, keyed by `id(spec)`.

    A lone GPU model takes the whole allocation (`num_gpus == gpu_count`, no
    fractional sharing). When generate + image co-locate on one GPU, the share is
    split **by footprint** — the bigger model (the LLM) gets the larger slice, so
    a small co-resident image model doesn't wall off VRAM the LLM could spend on
    KV cache. (`num_gpus` is the single VRAM knob: it drives gpu_memory_utilization
    for vLLM and the per-process cap for diffusers.) Shares sum to exactly
    `gpu_count` — the last model absorbs the rounding — so Ray can pack them all.
    """
    gpu_models = [s for s in specs if s.draws_from_vram]
    if not gpu_models:
        return {}
    if len(gpu_models) == 1:
        return {id(gpu_models[0]): float(gpu_count)}
    total_fp = sum(s.footprint_bytes for s in gpu_models) or 1
    shares = [round(gpu_count * s.footprint_bytes / total_fp, 3) for s in gpu_models[:-1]]
    shares.append(round(gpu_count - sum(shares), 3))
    return {id(s): share for s, share in zip(gpu_models, shares, strict=True)}


def _cpu_allocation(specs: list[ModelSpec], cpu_units: float) -> list[float]:
    """Per-deployment `num_cpus`, summing to <= `cpu_units` so Ray can always
    schedule. Generate gets the leftover; if even the bases don't fit, scale
    everything down proportionally."""
    bases = [
        _CORES_GENERATE_BASE
        if s.usecase == ModelUsecase.generate
        else _CORES_IMAGE
        if s.usecase == ModelUsecase.image
        else _CORES_SATELLITE
        for s in specs
    ]
    total = sum(bases)
    if total <= cpu_units:
        leftover = cpu_units - total
        return [
            round(b + leftover, 2) if s.usecase == ModelUsecase.generate else b
            for s, b in zip(specs, bases, strict=True)
        ]
    scale = cpu_units / total
    return [round(b * scale, 2) for b in bases]


def _render(profile: str, budget: DeployBudget, entries: list[dict]) -> str:
    accel = "GPU" if budget.has_gpu else "CPU"
    detected = f"{budget.ram_bytes / 1024**3:.0f} GiB RAM, {budget.cpu_units:.0f} cores"
    if budget.has_gpu:
        detected += f", {budget.gpu_count}x {budget.vram_bytes_per_gpu / 1024**3:.0f} GiB GPU"
    header = (
        f"# Auto-generated by MSHIP_MODEL_STACK={profile} ({accel} stack).\n"
        f"# Detected hardware: {detected}.\n"
        f"#\n"
        f"# REGENERATED FROM SCRATCH on every start while MSHIP_MODEL_STACK={profile} is set\n"
        f"# (or --model-stack {profile} is passed) — edits to this file are overwritten.\n"
        f"# To customize and keep your changes, copy it to config/models.yaml (or pass\n"
        f"# --config <file>) and unset MSHIP_MODEL_STACK.\n\n"
    )
    body = yaml.safe_dump({"models": entries}, sort_keys=False, default_flow_style=False)
    return header + body
