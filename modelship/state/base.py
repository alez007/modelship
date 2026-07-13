"""Generic durable state store.

A pluggable key→value store shared across the codebase: the deploy driver uses it
for the per-gateway *effective config* (durable desired state for self-heal), and
actors can use it for their own state (e.g. ``/v1/responses``). Keys are
``/``-separated namespace paths; values are JSON/YAML-serializable (``dict`` or
``list``).

Backends differ in durability, so each caller picks the one its use needs: the
default file backend survives cluster death (required for the effective config),
while an in-memory / Ray-actor backend would suit ephemeral actor state.

Sync ``get``/``set``/``delete``/``list`` are the primitive each backend must
implement; the ``*_async`` variants default to running the sync method in a thread
(non-blocking on an event loop) and a backend overrides them where a native/direct
path is better (redis: ``redis.asyncio``; memory: no thread at all).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

# JSON/YAML-serializable value. Kept deliberately narrow so every backend (file,
# Ray-actor, ConfigMap, Redis) can store it without custom encoders.
JsonValue = dict | list


class StateStoreUnavailableError(Exception):
    """The backend is unreachable/errored — distinct from a key being absent.

    ``get``/``list`` raise this on genuine I/O failure so a caller can tell a real
    outage (surface a 503) apart from a missing key (``None`` / empty). Never raised
    for merely-corrupt stored data, which is logged and treated as missing.
    """


class StateStore(ABC):
    """Key→value store. Keys are ``/``-separated namespace paths."""

    @abstractmethod
    def get(self, key: str) -> JsonValue | None:
        """Return the value for *key*, or ``None`` if absent/expired. Raise
        ``StateStoreUnavailableError`` if the backend can't be reached."""

    @abstractmethod
    def set(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        """Persist *value* under *key*, replacing any existing value. With
        *ttl_seconds* the entry expires after that many seconds."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove *key* if present (no error if absent)."""

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """Return the keys under *prefix*. Best-effort on expiry (may transiently
        include just-expired keys — ``get`` is authoritative). Raise
        ``StateStoreUnavailableError`` if the backend can't be reached."""

    # Async variants: default to offloading the sync method to a thread so a caller
    # on an event loop never blocks. Backends override where they can do better.
    async def get_async(self, key: str) -> JsonValue | None:
        return await asyncio.to_thread(self.get, key)

    async def set_async(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        await asyncio.to_thread(self.set, key, value, ttl_seconds=ttl_seconds)

    async def delete_async(self, key: str) -> None:
        await asyncio.to_thread(self.delete, key)

    async def list_async(self, prefix: str) -> list[str]:
        return await asyncio.to_thread(self.list, prefix)
