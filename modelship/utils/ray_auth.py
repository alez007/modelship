"""Ray cluster-auth env resolution, deliberately free of any Ray import.

Ray latches ``RAY_AUTH_MODE`` into a C++ singleton at ``import ray`` time, so a
later ``os.environ`` write is invisible to this process's own ``ray.init()`` /
``Node()`` calls. mship_deploy therefore resolves the auth env vars here, from
argv-derived ``MSHIP_*`` vars, *before* it imports ray — which rules out importing
anything under ``ray.*`` (that would trigger the very latch we're front-running).
``get_ray_temp_dir()`` is reimplemented for the same reason; it mirrors Ray's own
``get_user_temp_dir()`` + ``/ray`` (verified against the installed Ray)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ray_current_cluster_marker() -> Path:
    """Path to Ray's local `ray_current_cluster` discovery marker, computed
    without importing ray. Mirrors ray._common.utils.get_ray_temp_dir()."""
    if "RAY_TMPDIR" in os.environ:
        temp_root = os.environ["RAY_TMPDIR"]
    elif sys.platform.startswith("linux") and "TMPDIR" in os.environ:
        temp_root = os.environ["TMPDIR"]
    else:
        temp_root = os.path.join(os.sep, "tmp")
    return Path(temp_root) / "ray" / "ray_current_cluster"


def ray_auth_is_safe() -> bool:
    """False only when attaching to an already-running local cluster with no
    auth token — Ray only generates a token when starting a new cluster, so
    enabling token auth against a running unauthenticated cluster would silently
    attach without one instead of failing loudly."""
    try:
        if (Path.home() / ".ray" / "auth_token").exists():
            return True
        return not _ray_current_cluster_marker().exists()
    except (OSError, RuntimeError):
        # Path.home() raises RuntimeError when no home dir is resolvable.
        return False


def resolve_ray_auth_env() -> None:
    """Translate modelship's auth/join env vars into Ray's own
    RAY_AUTH_MODE/RAY_AUTH_TOKEN. Must run after argv is folded into the MSHIP_*
    vars (apply_args_to_env) and before the first `import ray`.

    Own-head token auth is gated on ray_auth_is_safe(); when it can't confirm
    safety it leaves RAY_AUTH_MODE unset, and connect_ray re-checks and raises a
    clear error at its normal point. Join token auth needs no such guard — the
    local node is freshly created, so a bad/missing token surfaces as a connect
    error rather than a silent unauthenticated attach."""
    join_address = os.environ.get("MSHIP_ADDRESS")
    use_existing = os.environ.get("MSHIP_USE_EXISTING_RAY_CLUSTER", "false").lower() == "true"
    own_head_token = (
        not use_existing and not join_address and os.environ.get("MSHIP_RAY_AUTH", "none").lower() == "token"
    )
    join_token = bool(join_address) and bool(os.environ.get("MSHIP_RAY_AUTH_TOKEN"))

    if own_head_token:
        if ray_auth_is_safe():
            os.environ.setdefault("RAY_AUTH_MODE", "token")
    elif join_token:
        os.environ.setdefault("RAY_AUTH_MODE", "token")

    if token := os.environ.get("MSHIP_RAY_AUTH_TOKEN"):
        os.environ.setdefault("RAY_AUTH_TOKEN", token)
