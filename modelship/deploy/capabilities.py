"""Node/deployment capability resources: `mship_<loader>` alongside Ray's native
`GPU`, so a deploy only schedules onto a node with that loader installed.
`loader: custom` requests neither — plugins install post-scheduling via runtime_env.

Must stay ray-free at import time: modelship/launcher.py imports LOADER_MODULES
from here before resolve_ray_auth_env runs, so the ModelshipModelConfig annotation
below is deferred behind `if TYPE_CHECKING` rather than imported for real.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelship.infer.infer_config import ModelshipModelConfig

RESOURCE_PREFIX = "mship_"

# loader name -> the module find_spec() probes for it. llama_server is handled
# separately below (a subprocess binary, not an importable module).
LOADER_MODULES = {
    "vllm": "vllm",
    "diffusers": "diffusers",
    "stable_diffusion_cpp": "stable_diffusion_cpp",
    "whispercpp": "pywhispercpp",
}

_LLAMA_SERVER_LOADER = "llama_server"

# Every loader that gets a capability resource, including llama_server (probed via a
# binary check below, not find_spec like the LOADER_MODULES entries).
ALL_CAPABILITY_LOADERS = frozenset({*LOADER_MODULES, _LLAMA_SERVER_LOADER})

# Matches a wrapper script's `exec <target> ...` line, quoted or not (Dockerfile's
# llama-server.sh emits it unquoted; launcher._write_wrapper quotes it).
_WRAPPER_EXEC_RE = re.compile(r'exec\s+"?([^"\s]+)"?')


def node_capability_resources() -> dict[str, float]:
    """{"mship_vllm": 1, ...} for every loader this node can run. MSHIP_NODE_CAPABILITIES
    (a JSON object) overrides the probe wholesale."""
    override = os.environ.get("MSHIP_NODE_CAPABILITIES")
    if override:
        return {str(name): float(qty) for name, qty in json.loads(override).items()}

    resources: dict[str, float] = {}
    for loader, module in LOADER_MODULES.items():
        if importlib.util.find_spec(module) is not None:
            resources[f"{RESOURCE_PREFIX}{loader}"] = 1
    if _llama_server_available():
        resources[f"{RESOURCE_PREFIX}{_LLAMA_SERVER_LOADER}"] = 1
    return resources


def deployment_capability_resources(config: ModelshipModelConfig) -> dict[str, float]:
    """{"mship_vllm": 0.001} for the config's loader; empty for loader='custom'."""
    loader = str(config.loader)
    if loader == "custom":
        return {}
    return {f"{RESOURCE_PREFIX}{loader}": 0.001}


def _llama_server_available() -> bool:
    """`thin` bakes the wrapper script unconditionally but ships no binary behind
    it, so a bare existence check on MSHIP_LLAMA_SERVER_BIN can't tell `thin`
    apart from `cpu`/`cuda`/`metal` — resolve the wrapper's own exec target."""
    bin_path = os.environ.get("MSHIP_LLAMA_SERVER_BIN")
    if not bin_path or not os.path.isfile(bin_path):
        return False
    target = _wrapper_exec_target(bin_path)
    if target is None:
        return True  # not a recognized wrapper — the existence check above suffices
    return os.path.isfile(target)


def _wrapper_exec_target(path: str) -> str | None:
    try:
        with open(path) as f:
            content = f.read()
    except OSError:
        return None
    match = _WRAPPER_EXEC_RE.search(content)
    return match.group(1) if match else None
