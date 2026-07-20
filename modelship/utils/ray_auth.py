"""Ray cluster-auth env resolution, deliberately free of any Ray import."""

from __future__ import annotations

import os


def resolve_ray_auth_env() -> None:
    """Translate modelship's auth/join env vars into Ray's own
    RAY_AUTH_MODE/RAY_AUTH_TOKEN. Must run after argv is folded into the MSHIP_*
    vars (apply_args_to_env) and before the first `import ray` — Ray's
    RAY_AUTH_MODE check latches at import time, so setting it later has no
    effect on this process's own ray.init()/Node() calls."""
    join_address = os.environ.get("MSHIP_ADDRESS")
    use_existing = os.environ.get("MSHIP_USE_EXISTING_RAY_CLUSTER", "false").lower() == "true"
    own_head_token = (
        not use_existing and not join_address and os.environ.get("MSHIP_RAY_AUTH", "none").lower() == "token"
    )
    join_token = bool(join_address) and bool(os.environ.get("MSHIP_RAY_AUTH_TOKEN"))

    if own_head_token or join_token:
        os.environ.setdefault("RAY_AUTH_MODE", "token")

    if token := os.environ.get("MSHIP_RAY_AUTH_TOKEN"):
        os.environ.setdefault("RAY_AUTH_TOKEN", token)
