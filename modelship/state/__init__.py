"""Generic, pluggable state stores. See base.StateStore."""

import os
from pathlib import Path

from modelship.state.base import JsonValue, StateStore
from modelship.state.file import FileStateStore

__all__ = ["FileStateStore", "JsonValue", "StateStore", "get_state_store"]


def get_state_store() -> StateStore:
    """The configured default StateStore.

    Today always a file store under ``$MSHIP_STATE_DIR`` (falling back to
    ``$MSHIP_CACHE_DIR/state``, default ``/.cache/state``) — the one location
    durable in every environment (Docker volume, k8s PVC, local). A future
    ``MSHIP_STATE_BACKEND`` switch can select other backends here without callers
    changing.
    """
    base = os.environ.get("MSHIP_STATE_DIR")
    if not base:
        cache = os.environ.get("MSHIP_CACHE_DIR", "/.cache")
        base = os.path.join(cache, "state")
    return FileStateStore(Path(base))
