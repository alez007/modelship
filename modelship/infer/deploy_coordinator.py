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
import time

import ray
from ray import exceptions as ray_exceptions

from modelship.logging import get_logger

logger = get_logger("deploy_coordinator")

COORDINATOR_ACTOR_NAME = "modelship-deploy-coordinator"
COORDINATOR_NAMESPACE = "modelship"

_LIVENESS_POLL_INTERVAL_S = 5.0
_LIVENESS_CALL_TIMEOUT_S = 3.0
_LIVENESS_TIMEOUT_STRIKES = 3


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
        # The driver writes it on (un)deploy; a restarted gateway reads its own
        # entry to rebuild its routing table without the driver being present.
        self._registry: dict[str, dict[str, str]] = {}

    def report_fatal_error(self, deployment_name: str, reason: str) -> None:
        self._fatal_errors[deployment_name] = reason

    def pop_fatal_error(self, deployment_name: str) -> str | None:
        return self._fatal_errors.pop(deployment_name, None)

    def register_deployment(self, gateway_name: str, deployment_name: str, model_name: str) -> None:
        self._registry.setdefault(gateway_name, {})[deployment_name] = model_name

    def unregister_deployment(self, gateway_name: str, deployment_name: str) -> None:
        gw = self._registry.get(gateway_name)
        if gw is not None:
            gw.pop(deployment_name, None)
            if not gw:
                del self._registry[gateway_name]

    def get_deployments(self, gateway_name: str) -> dict[str, str]:
        """`{deployment_name -> model_name}` recorded for this gateway."""
        return dict(self._registry.get(gateway_name, {}))

    async def try_reserve(
        self,
        operator_id: str,
        deployment_name: str,
        num_gpus: float,
        num_cpus: float,
        probe_handle,
    ) -> tuple[bool, str]:
        if self._held_by is not None:
            return False, f"locked_by:{self._held_by}:{self._held_deployment}"

        avail = ray.available_resources()
        eps = 1e-6
        if float(num_gpus or 0) > avail.get("GPU", 0) + eps:
            return False, "insufficient_gpu"
        if float(num_cpus or 0) > avail.get("CPU", 0) + eps:
            return False, "insufficient_cpu"

        self._held_by = operator_id
        self._held_deployment = deployment_name
        self._held_since = time.time()
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
        ).remote()
    except ValueError:
        return ray.get_actor(COORDINATOR_ACTOR_NAME, namespace=COORDINATOR_NAMESPACE)
