"""Cluster-wide coordinator for serialising model deploys across operators.

A `mship_deploy.py` driver ("operator") cannot safely assume it is the only process
deploying models to the Ray cluster. Two operators both checking
`ray.available_resources()`, both seeing "GPU free", and both calling
`serve.run()` concurrently can trigger simultaneous VRAM loads on the same
device and OOM. This module provides a cluster-level mutex that combines
"is the lock free?" with "can this request actually be placed?" into one
atomic check, so operators never race.

Design:

- `ModelshipDeployCoordinator` is a detached, named Ray actor. The first
  operator to start creates it; subsequent operators look it up by name.
- Operators reserve via `try_reserve(operator_id, probe, num_gpus, num_cpus)`.
  Granted only when the lock is unheld AND the cluster has the requested
  resources available right now.
- The operator passes a handle to a small owned actor (`OperatorProbe`) when
  reserving. The coordinator polls that handle via `__ray_ready__` to detect
  ungraceful operator death (SIGKILL, host crash, partition). Because the
  probe is owned by the operator driver, Ray tears it down when the driver
  dies — the coordinator sees `RayActorError` and force-releases the lock.
- Graceful shutdown uses `release(operator_id)` from the operator's
  try/finally, cancelling the liveness watcher cleanly.
"""

import asyncio
import contextlib
import time

import ray
from ray import exceptions as ray_exceptions

from modelship.logging import get_logger
from modelship.metrics import (
    COORDINATOR_GENERATION,
    DEPLOY_LOCK_HELD,
    DEPLOY_RESERVATIONS_TOTAL,
    OPERATOR_FORCE_RELEASE_TOTAL,
)
from modelship.state import MemoryStateStore, get_state_store

logger = get_logger("deploy_coordinator")

COORDINATOR_ACTOR_NAME = "modelship-deploy-coordinator"
COORDINATOR_NAMESPACE = "modelship"

_STATE_KEY = "coordinator/state"

_LIVENESS_POLL_INTERVAL_S = 5.0
_LIVENESS_CALL_TIMEOUT_S = 3.0
_LIVENESS_TIMEOUT_STRIKES = 3

# How long a gateway replica's wait_for_change blocks before returning the
# current generation unchanged. Bounds how long a missed wake / coordinator
# restart can leave a replica un-reconciled (it re-pulls on every return).
_WATCH_TIMEOUT_S = 30.0


@ray.remote(num_cpus=0)
class OperatorProbe:
    """Empty actor whose only purpose is to be owned by the operator driver.

    Ray destroys owned actors when the owning process dies. The coordinator
    uses `__ray_ready__` on this handle as a liveness signal — if the call
    starts raising `RayActorError`, the operator is gone and the lock can be
    force-released.
    """

    def ping(self) -> str:
        return "alive"


@ray.remote(num_cpus=0)
class ModelshipDeployCoordinator:
    """Cluster-wide mutex + resource-aware admission gate for model deploys."""

    def __init__(self):
        self._held_by: str | None = None
        self._held_deployment: str | None = None
        self._held_since: float = 0.0
        self._watcher_task: asyncio.Task | None = None
        self._fatal_errors: dict[str, str] = {}
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
                "Deploy coordinator is backed by a non-durable memory state store; its routing "
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

    def report_fatal_error(self, deployment_name: str, reason: str) -> None:
        self._fatal_errors[deployment_name] = reason

    def pop_fatal_error(self, deployment_name: str) -> str | None:
        return self._fatal_errors.pop(deployment_name, None)

    # --- Routing registry + change notification (drives the gateway watch loop) ---
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

    async def try_reserve(
        self,
        operator_id: str,
        deployment_name: str,
        num_gpus: float,
        num_cpus: float,
        probe_handle,
    ) -> tuple[bool, str]:
        if self._held_by is not None:
            DEPLOY_RESERVATIONS_TOTAL.inc(tags={"result": "locked"})
            return False, f"locked_by:{self._held_by}:{self._held_deployment}"

        avail = ray.available_resources()
        eps = 1e-6
        if float(num_gpus or 0) > avail.get("GPU", 0) + eps:
            DEPLOY_RESERVATIONS_TOTAL.inc(tags={"result": "insufficient_gpu"})
            return False, "insufficient_gpu"
        if float(num_cpus or 0) > avail.get("CPU", 0) + eps:
            DEPLOY_RESERVATIONS_TOTAL.inc(tags={"result": "insufficient_cpu"})
            return False, "insufficient_cpu"

        self._held_by = operator_id
        self._held_deployment = deployment_name
        self._held_since = time.time()
        DEPLOY_RESERVATIONS_TOTAL.inc(tags={"result": "granted"})
        DEPLOY_LOCK_HELD.set(1)
        self._watcher_task = asyncio.create_task(self._watch_operator_liveness(operator_id, probe_handle))
        logger.info(
            "Reserved for operator=%s deployment=%s (num_gpus=%s, num_cpus=%s)",
            operator_id,
            deployment_name,
            num_gpus,
            num_cpus,
        )
        return True, "ok"

    async def release(self, operator_id: str) -> bool:
        if self._held_by != operator_id:
            logger.warning(
                "Stale release from %s (current holder: %s) — ignoring",
                operator_id,
                self._held_by,
            )
            return False
        self._clear_hold()
        logger.info("Released by operator=%s", operator_id)
        return True

    async def status(self) -> dict:
        return {
            "held_by": self._held_by,
            "held_deployment": self._held_deployment,
            "held_for_seconds": (time.time() - self._held_since) if self._held_by else 0.0,
        }

    def _clear_hold(self):
        self._held_by = None
        self._held_deployment = None
        self._held_since = 0.0
        DEPLOY_LOCK_HELD.set(0)
        if self._watcher_task is not None and not self._watcher_task.done():
            self._watcher_task.cancel()
        self._watcher_task = None

    async def _watch_operator_liveness(self, operator_id: str, probe_handle):
        timeout_strikes = 0
        while True:
            try:
                await asyncio.sleep(_LIVENESS_POLL_INTERVAL_S)
            except asyncio.CancelledError:
                return

            if self._held_by != operator_id:
                return

            try:
                await asyncio.wait_for(
                    probe_handle.__ray_ready__.remote(),
                    timeout=_LIVENESS_CALL_TIMEOUT_S,
                )
                timeout_strikes = 0
            except ray_exceptions.RayActorError:
                logger.warning(
                    "Probe for operator=%s is gone — force-releasing lock (deployment=%s)",
                    operator_id,
                    self._held_deployment,
                )
                OPERATOR_FORCE_RELEASE_TOTAL.inc(tags={"reason": "probe_gone"})
                self._clear_hold()
                return
            except TimeoutError:
                timeout_strikes += 1
                if timeout_strikes >= _LIVENESS_TIMEOUT_STRIKES:
                    logger.warning(
                        "Probe for operator=%s unresponsive for %ds — force-releasing lock",
                        operator_id,
                        timeout_strikes * _LIVENESS_POLL_INTERVAL_S,
                    )
                    OPERATOR_FORCE_RELEASE_TOTAL.inc(tags={"reason": "unresponsive"})
                    self._clear_hold()
                    return


def get_or_create_coordinator():
    """Return the cluster-wide coordinator handle, creating it if absent."""
    try:
        return ray.get_actor(COORDINATOR_ACTOR_NAME, namespace=COORDINATOR_NAMESPACE)
    except ValueError:
        pass
    try:
        return ModelshipDeployCoordinator.options(
            name=COORDINATOR_ACTOR_NAME,
            namespace=COORDINATOR_NAMESPACE,
            lifetime="detached",
            num_cpus=0,
            max_restarts=-1,
        ).remote()
    except ValueError:
        return ray.get_actor(COORDINATOR_ACTOR_NAME, namespace=COORDINATOR_NAMESPACE)
