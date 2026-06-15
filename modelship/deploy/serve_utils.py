import logging
import os
import socket

import ray
from ray import serve
from ray.serve.config import HTTPOptions
from ray.serve.schema import LoggingConfig

from modelship.infer.infer_config import ModelshipConfig
from modelship.logging import get_logger
from modelship.openai.api import ModelshipAPI
from modelship.utils import rand_suffix

logger = get_logger("startup")
_DEFAULT_OPENAI_API_PORT = 8000


def make_operator_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{rand_suffix(4)}"


def get_existing_apps() -> set[str]:
    """Return the set of currently deployed Serve app names."""
    try:
        return set(serve.status().applications.keys())
    except Exception:
        return set()


def shutdown_ray() -> None:
    """Shut down Ray Serve and Ray. Logs but swallows errors."""
    for label, fn in (("serve.shutdown()", serve.shutdown), ("ray.shutdown()", ray.shutdown)):
        try:
            fn()
        except Exception:
            logger.exception("%s failed", label)


def delete_apps_quietly(app_names) -> None:
    """Best-effort serve.delete for cleanup paths — never raises."""
    for name in app_names:
        try:
            logger.info("Deleting deployment: %s", name)
            serve.delete(name)
        except Exception:
            logger.exception("Failed to delete deployment: %s", name)


def remove_apps(gateway_handle, app_names: list[str], coordinator=None, gateway_name: str | None = None) -> None:
    """Unregister the given deployment apps from the gateway (so new requests stop
    routing), drop them from the coordinator's ownership registry, then delete
    them from Ray Serve (`serve.delete` drains in-flight requests first)."""
    if not app_names:
        return
    try:
        gateway_handle.remove_deployments.remote(app_names).result()
    except Exception:
        logger.exception("Failed to unregister deployments from gateway: %s", app_names)
    if coordinator is not None and gateway_name is not None:
        try:
            ray.get([coordinator.unregister_deployment.remote(gateway_name, a) for a in app_names])
        except Exception:
            logger.exception("Failed to drop deployments from registry: %s", app_names)
    delete_apps_quietly(app_names)


def connect_ray(lib_level: int) -> None:
    use_existing_cluster = os.environ.get("MSHIP_USE_EXISTING_RAY_CLUSTER", "false").lower() == "true"
    os.environ.setdefault("RAY_GCS_RPC_TIMEOUT_S", "30")

    if use_existing_cluster:
        ray_cluster_address = os.environ["RAY_CLUSTER_ADDRESS"]
        ray_redis_port = os.environ["RAY_REDIS_PORT"]
        address = f"{ray_cluster_address}:{ray_redis_port}"
    else:
        address = "auto"

    ray.init(address=address, ignore_reinit_error=True, logging_level=lib_level)
    # ray.init re-sets ray.* loggers, so re-pin them after init.
    logging.getLogger("ray").setLevel(lib_level)
    logging.getLogger("ray._private.worker").setLevel(lib_level)


def start_serve(serve_logging_config: LoggingConfig) -> None:
    port = int(os.environ.get("MSHIP_OPENAI_API_PORT", str(_DEFAULT_OPENAI_API_PORT)))
    serve.start(
        http_options=HTTPOptions(host="0.0.0.0", port=port),
        logging_config=serve_logging_config,
    )


def _positive_int_env(name: str, default: int) -> int:
    """Read an env var as an int >= 1, failing fast with a clear message.

    Ray Serve rejects num_replicas / max_ongoing_requests < 1 deep in
    deployment, so validate up front to surface the misconfiguration plainly.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def start_gateway(gateway_name: str, serve_logging_config: LoggingConfig) -> None:
    logger.info("Starting API gateway...")
    gateway_replicas = _positive_int_env("MSHIP_GATEWAY_REPLICAS", 1)
    gateway_max_ongoing = _positive_int_env("MSHIP_GATEWAY_MAX_ONGOING", 1024)
    serve.run(
        ModelshipAPI.options(
            name=gateway_name,
            num_replicas=gateway_replicas,
            max_ongoing_requests=gateway_max_ongoing,
            ray_actor_options={"num_cpus": 0},
            logging_config=serve_logging_config,
        ).bind(gateway_name),
        name=gateway_name,
        route_prefix="/",
    )
    logger.info(
        "Gateway up — /health and /readyz now serving. (replicas=%d, max_ongoing=%d)",
        gateway_replicas,
        gateway_max_ongoing,
    )


def seed_expected_models(gateway_handle, yml_conf: ModelshipConfig) -> None:
    # Pass the full desired set, not just models_to_add — already-deployed
    # models also count toward "ready".
    try:
        gateway_handle.set_expected_models.remote([c.name for c in yml_conf.models]).result()
    except Exception:
        logger.exception("Failed to seed expected model list on gateway (non-fatal).")
