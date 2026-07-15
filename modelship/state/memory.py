"""In-memory StateStore — a dict shared cluster-wide through a detached Ray actor.

The default backend. Every process and gateway replica in the cluster shares one
``MemoryStoreActor``, so writes from one process (e.g. the deploy driver) are
visible to another (e.g. a re-run of the driver, or a gateway replica) — unlike a
plain process-local dict. It survives the actor being restarted (``max_restarts``)
but NOT the cluster dying, which is what distinguishes it from the durable
``redis://`` backend. Selected by the ``memory://`` URI scheme.
"""

from __future__ import annotations

import copy
import time

import ray
from ray import exceptions as ray_exceptions

from modelship.state.base import JsonValue, StateStore, StateStoreUnavailableError, normalize_prefix

# Detached-actor identity: same namespace as the other cluster-wide coordinators
# (modelship.infer.deploy_coordinator.COORDINATOR_NAMESPACE / replica_coordinator).
# Not imported from there — modelship.state is the generic lower layer and infer
# depends on it, not the reverse.
_ACTOR_NAME = "modelship-memory-store"
_ACTOR_NAMESPACE = "modelship"


@ray.remote(num_cpus=0)
class MemoryStoreActor(StateStore):
    """Holds the dict. One actor for the whole cluster — memory:// targets
    small-traffic single-node deployments, so a single actor is the design point,
    not a stopgap. A restart returns an empty store: this fails safe for both
    current callers (the replica coordinator's in-RAM registry is untouched and
    write-through repopulates it; an empty effective config makes the deploy
    driver's reconcile remove nothing rather than remove wrongly)."""

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

    # In-process (this runs as the actor body itself, not over RPC from within):
    # no thread needed, so skip the base's to_thread hop.
    async def get_async(self, key: str) -> JsonValue | None:
        return self.get(key)

    async def set_async(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        self.set(key, value, ttl_seconds=ttl_seconds)

    async def delete_async(self, key: str) -> None:
        self.delete(key)

    async def list_async(self, prefix: str) -> list[str]:
        return self.list(prefix)


def get_or_create_memory_store_actor():
    """Return the cluster-wide memory-store actor handle, creating it if absent.
    Mirrors deploy_coordinator.get_or_create_coordinator's race-safe pattern."""
    try:
        return ray.get_actor(_ACTOR_NAME, namespace=_ACTOR_NAMESPACE)
    except ValueError:
        pass
    try:
        return MemoryStoreActor.options(
            name=_ACTOR_NAME,
            namespace=_ACTOR_NAMESPACE,
            lifetime="detached",
            num_cpus=0,
            max_restarts=-1,
        ).remote()
    except ValueError:
        return ray.get_actor(_ACTOR_NAME, namespace=_ACTOR_NAMESPACE)


class MemoryStateStore(StateStore):
    """Client for the cluster-wide MemoryStoreActor. Construction is inert (no Ray
    call) so it can be built before/without a cluster, matching RedisStateStore's
    lazy-client pattern; the actor handle is resolved on first use."""

    def __init__(self) -> None:
        self._handle = None

    def _actor(self):
        if self._handle is None:
            if not ray.is_initialized():
                raise StateStoreUnavailableError(
                    "memory:// requires an initialized Ray cluster (no ray.init() has run in this process)."
                )
            self._handle = get_or_create_memory_store_actor()
        return self._handle

    def _call(self, method: str, *args, **kwargs):
        try:
            return ray.get(getattr(self._actor(), method).remote(*args, **kwargs))
        except ray_exceptions.RayActorError as exc:
            # The actor died and won't come back reachable via this stale handle
            # (max_restarts keeps the same actor id alive, but a fresh
            # ray.get_actor() re-resolves it) — drop the cache and let the next
            # call re-resolve.
            self._handle = None
            raise StateStoreUnavailableError(f"memory:// store actor unreachable: {exc}") from exc

    async def _acall(self, method: str, *args, **kwargs):
        try:
            return await getattr(self._actor(), method).remote(*args, **kwargs)
        except ray_exceptions.RayActorError as exc:
            self._handle = None
            raise StateStoreUnavailableError(f"memory:// store actor unreachable: {exc}") from exc

    def get(self, key: str) -> JsonValue | None:
        return self._call("get", key)

    def set(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        self._call("set", key, value, ttl_seconds=ttl_seconds)

    def delete(self, key: str) -> None:
        self._call("delete", key)

    def list(self, prefix: str) -> list[str]:
        return self._call("list", prefix)

    async def get_async(self, key: str) -> JsonValue | None:
        return await self._acall("get", key)

    async def set_async(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        await self._acall("set", key, value, ttl_seconds=ttl_seconds)

    async def delete_async(self, key: str) -> None:
        await self._acall("delete", key)

    async def list_async(self, prefix: str) -> list[str]:
        return await self._acall("list", prefix)
