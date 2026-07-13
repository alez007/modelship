"""In-memory StateStore — a process-local dict.

The default backend. Holds nothing across process death, so it suits self-hosted
/ Docker deployments where a restart replaces the config anyway, and any actor
that wants the StateStore interface without durable infra. Selected by the
``memory://`` URI scheme.
"""

from __future__ import annotations

import copy
import time

from modelship.state.base import JsonValue, StateStore, normalize_prefix


class MemoryStateStore(StateStore):
    def __init__(self) -> None:
        # key -> (value, expires_at epoch | None). Expiry is enforced lazily on read.
        self._data: dict[str, tuple[JsonValue, float | None]] = {}

    def get(self, key: str) -> JsonValue | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.time() >= expires_at:
            self._data.pop(key, None)
            return None
        # Deep-copy on the way out so callers can't mutate stored state in place.
        return copy.deepcopy(value)

    def set(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds is not None else None
        self._data[key] = (copy.deepcopy(value), expires_at)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def list(self, prefix: str) -> list[str]:
        prefix = normalize_prefix(prefix)
        now = time.time()
        return [
            k
            for k, (_, expires_at) in list(self._data.items())
            if (not prefix or k == prefix or k.startswith(f"{prefix}/")) and (expires_at is None or now < expires_at)
        ]

    # In-process: no thread needed, so skip the base's to_thread hop.
    async def get_async(self, key: str) -> JsonValue | None:
        return self.get(key)

    async def set_async(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        self.set(key, value, ttl_seconds=ttl_seconds)

    async def delete_async(self, key: str) -> None:
        self.delete(key)

    async def list_async(self, prefix: str) -> list[str]:
        return self.list(prefix)
