"""Cluster-wide routing registry shared by every gateway replica.

`ReplicaCoordinator` is a detached, named Ray actor holding the durable mapping of
model deployments each gateway owns. `mship_deploy.py` writes to it as models are
(un)deployed; every gateway replica long-polls `wait_for_change` and reconciles its
own routing table from `get_routing` — the driver never pushes to individual
replicas.

The registry is persisted through `get_state_store()` (file/redis for multi-node
HA) so a resurrected actor reloads live ownership instead of starting empty. The
per-gateway generation counter and its wakeup `asyncio.Event` are ephemeral: on
restart the generation resets to 0, which `wait_for_change` already treats as
"changed" so replicas re-pull and reconcile from the reloaded registry.
"""

import asyncio
import contextlib

import ray

from modelship.infer.deploy_coordinator import COORDINATOR_NAMESPACE
from modelship.logging import get_logger
from modelship.metrics import COORDINATOR_GENERATION
from modelship.state import MemoryStateStore, get_state_store

logger = get_logger("replica_coordinator")

REPLICA_COORDINATOR_ACTOR_NAME = "modelship-replica-coordinator"

_STATE_KEY = "coordinator/state"

# How long a gateway replica's wait_for_change blocks before returning the
# current generation unchanged. Bounds how long a missed wake / coordinator
# restart can leave a replica un-reconciled (it re-pulls on every return).
_WATCH_TIMEOUT_S = 30.0


@ray.remote(num_cpus=0)
class ReplicaCoordinator:
    """Durable per-gateway routing registry with long-poll change notification."""

    def __init__(self):
        # Durable ownership registry: gateway_name -> {deployment_name -> model_name}.
        # The driver writes it on (un)deploy; gateway replicas reconcile their
        # routing tables from it (see get_routing / wait_for_change), so the driver
        # never pushes to individual replicas.
        # _registry and _expected are durable (loaded below, written through on
        # every change); _generation/_change are ephemeral wakeup state. On a
        # resurrected coordinator the generation restarts at 0, which the gateway's
        # wait_for_change already treats as "changed" so replicas re-pull and
        # reconcile from the reloaded registry.
        self._store = get_state_store()
        if isinstance(getattr(self._store, "inner", self._store), MemoryStateStore):
            # A memory store dies with the actor: a restarted coordinator reloads an
            # empty registry, and the next deploy (gen advances) re-enables removals
            # against it — dropping still-healthy models from gateway routing. Fine
            # single-node; for multi-node/HA set MSHIP_STATE_STORE to file:// or redis://.
            logger.warning(
                "Replica coordinator is backed by a non-durable memory state store; its routing "
                "registry will be lost on coordinator restart. Set MSHIP_STATE_STORE to file:// "
                "or redis:// for multi-node/HA."
            )
        saved = self._store.get(_STATE_KEY)
        saved = saved if isinstance(saved, dict) else {}
        self._registry: dict[str, dict[str, str]] = saved.get("registry") or {}
        # Per-gateway change notification driving the gateway watch loop: a
        # monotonic generation bumped on every routing/expected change, plus an
        # asyncio.Event woken on each bump so a long-polling replica returns at
        # once. _expected is the desired model set used for gateway readiness.
        self._generation: dict[str, int] = {}
        self._expected: dict[str, list[str]] = saved.get("expected") or {}
        self._change: dict[str, asyncio.Event] = {}

    # These are async so every registry / generation / Event mutation runs on the
    # actor's single event loop, serialised with wait_for_change and race-free.

    def _bump(self, gateway_name: str) -> None:
        """Advance the gateway's generation and wake any current waiters. The old
        Event is set (releasing replicas blocked on it) then replaced with a fresh
        unset Event for the next round."""
        self._generation[gateway_name] = self._generation.get(gateway_name, 0) + 1
        COORDINATOR_GENERATION.set(self._generation[gateway_name], tags={"gateway": gateway_name})
        old = self._change.get(gateway_name)
        if old is not None:
            old.set()
        self._change[gateway_name] = asyncio.Event()

    def _persist(self) -> None:
        """Write the durable routing state through the StateStore. A no-op for the
        memory store; for redis/file this is what survives coordinator death."""
        self._store.set(_STATE_KEY, {"registry": self._registry, "expected": self._expected})

    async def register_deployment(self, gateway_name: str, deployment_name: str, model_name: str) -> None:
        self._registry.setdefault(gateway_name, {})[deployment_name] = model_name
        self._persist()
        self._bump(gateway_name)

    async def unregister_deployment(self, gateway_name: str, deployment_name: str) -> None:
        gw = self._registry.get(gateway_name)
        if gw is not None:
            gw.pop(deployment_name, None)
            if not gw:
                del self._registry[gateway_name]
        self._persist()
        self._bump(gateway_name)

    async def set_expected(self, gateway_name: str, names: list[str]) -> None:
        """Record the desired model set for readiness; bumps so replicas adopt it."""
        self._expected[gateway_name] = list(names)
        self._persist()
        self._bump(gateway_name)

    async def get_routing(self, gateway_name: str) -> dict:
        """Snapshot a replica pulls after a change: the app->model map, the
        expected-model set, and the current generation."""
        return {
            "models": dict(self._registry.get(gateway_name, {})),
            "expected": list(self._expected.get(gateway_name, [])),
            "generation": self._generation.get(gateway_name, 0),
        }

    async def wait_for_change(self, gateway_name: str, since_gen: int, timeout: float = _WATCH_TIMEOUT_S) -> int:
        """Long-poll for a routing change. Returns the current generation at once
        if it differs from since_gen (covers both a forward bump and a coordinator
        restart that reset it to 0); otherwise waits for the next bump up to
        timeout, then returns the current generation regardless."""
        current = self._generation.get(gateway_name, 0)
        if current != since_gen:
            return current
        event = self._change.setdefault(gateway_name, asyncio.Event())
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(event.wait(), timeout)
        return self._generation.get(gateway_name, 0)


def get_or_create_replica_coordinator():
    """Return the cluster-wide replica-routing coordinator handle, creating it if absent."""
    try:
        return ray.get_actor(REPLICA_COORDINATOR_ACTOR_NAME, namespace=COORDINATOR_NAMESPACE)
    except ValueError:
        pass
    try:
        return ReplicaCoordinator.options(
            name=REPLICA_COORDINATOR_ACTOR_NAME,
            namespace=COORDINATOR_NAMESPACE,
            lifetime="detached",
            num_cpus=0,
            max_restarts=-1,
        ).remote()
    except ValueError:
        return ray.get_actor(REPLICA_COORDINATOR_ACTOR_NAME, namespace=COORDINATOR_NAMESPACE)
