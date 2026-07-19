import logging
import os
import re
import shutil
import socket
from pathlib import Path

import ray
from ray import serve
from ray._common.utils import get_ray_temp_dir
from ray.serve.config import HTTPOptions, ProxyLocation
from ray.serve.schema import LoggingConfig

from modelship.deploy.actor_options import build_passthrough_env_vars
from modelship.infer.infer_config import ModelshipConfig
from modelship.logging import get_logger
from modelship.openai.api import ModelshipAPI
from modelship.utils import rand_suffix

logger = get_logger("startup")
_DEFAULT_OPENAI_API_PORT = 8000
# Not 6379 (ray-start-head's default) — that collides with the docs-recommended
# same-host Redis state store (MSHIP_STATE_STORE=redis://) under --network=host.
_DEFAULT_RAY_GCS_PORT = 6380

# Ray names each head's session dir `session_<timestamp>_<pid>` under its temp
# root and never cleans them up; the trailing group captures the owning pid.
_RAY_SESSION_DIR_RE = re.compile(r"^session_.*_(\d+)$")


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


def remove_apps(app_names: list[str], replica_coordinator, gateway_name: str) -> None:
    """Drop the given deployment apps from the replica coordinator's ownership registry
    (which bumps the gateway generation so every replica's watch loop stops routing
    to them), then delete them from Ray Serve (`serve.delete` drains in-flight
    requests first)."""
    if not app_names:
        return
    try:
        ray.get([replica_coordinator.unregister_deployment.remote(gateway_name, a) for a in app_names])
    except Exception:
        logger.exception("Failed to drop deployments from registry: %s", app_names)
    delete_apps_quietly(app_names)


def _own_cluster_init_kwargs() -> dict[str, object]:
    """ray.init kwargs to start our own head. Dashboard always starts;
    MSHIP_RAY_DASHBOARD sets its bind host (default 127.0.0.1), not on/off.
    Resources auto-detect when MSHIP_NODE_NUM_* are unset."""
    kwargs: dict[str, object] = {
        "include_dashboard": True,
        "dashboard_host": os.environ.get("MSHIP_RAY_DASHBOARD", "127.0.0.1"),
    }
    if cpus := os.environ.get("MSHIP_NODE_NUM_CPUS"):
        kwargs["num_cpus"] = int(cpus)
    if gpus := os.environ.get("MSHIP_NODE_NUM_GPUS"):
        kwargs["num_gpus"] = int(gpus)
    if os.environ.get("MSHIP_METRICS", "true").lower() == "true":
        # _metrics_export_port is a private ray.init kwarg (accepted via **kwargs)
        # that pins Ray's metrics agent — and thus all serve_*/vllm:*/modelship
        # Prometheus metrics — to a stable port. Guarded by a connect_ray test.
        kwargs["_metrics_export_port"] = int(os.environ.get("RAY_METRICS_EXPORT_PORT", "8079"))
    return kwargs


def _pid_alive(pid: int) -> bool:
    """True if a process with *pid* currently exists. Used to avoid deleting a
    still-running head's session dir."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user — treat as alive (keep it).
        return True
    except OSError:
        return True
    return True


def prune_ray_sessions() -> None:
    """Delete stale `session_<timestamp>_<pid>` dirs Ray leaves under its temp
    root (default /tmp/ray). Ray never cleans them up, so they accumulate across
    container/process restarts and slowly fill the disk on long-lived self-hosted
    boxes.

    Called only when we start our OWN head (connect_ray's own-cluster branch),
    before ray.init creates this run's session — so this run's dir doesn't exist
    yet and can't be removed. A session whose owning pid is still alive is kept:
    on a single machine a second non---use-existing deploy joins this machine's
    running head (ray.init reads /tmp/ray/ray_current_cluster), so a live head may
    be present even on the own-cluster path. The `session_latest` symlink and
    non-session files (e.g. ray_current_cluster) never match and are left alone.

    Best-effort: pruning never aborts startup — any failure is logged as a
    warning and the deploy proceeds. Set MSHIP_PRUNE_RAY_SESSIONS=false to disable
    (e.g. to keep a crashed session's logs for debugging)."""
    if os.environ.get("MSHIP_PRUNE_RAY_SESSIONS", "true").lower() != "true":
        return
    try:
        temp_root = Path(get_ray_temp_dir())
        if not temp_root.is_dir():
            return
        removed = 0
        for entry in temp_root.iterdir():
            match = _RAY_SESSION_DIR_RE.match(entry.name)
            if not match or entry.is_symlink() or not entry.is_dir():
                continue
            if _pid_alive(int(match.group(1))):
                continue
            try:
                shutil.rmtree(entry)
                removed += 1
            except OSError:
                logger.warning("Failed to prune stale Ray session dir %s (continuing).", entry, exc_info=True)
        if removed:
            logger.info("Pruned %d stale Ray session dir(s) under %s", removed, temp_root)
    except Exception:
        # Cleanup is never worth failing a deploy over — warn and carry on.
        logger.warning("Ray session pruning failed; continuing without it.", exc_info=True)


def _ray_auth_is_safe() -> bool:
    """False only when attaching to an already-running local cluster with no
    auth token — Ray only generates one when starting a new cluster."""
    try:
        if (Path.home() / ".ray" / "auth_token").exists():
            return True
        return not (Path(get_ray_temp_dir()) / "ray_current_cluster").exists()
    except (OSError, RuntimeError):
        return False


def connect_ray(lib_level: int) -> None:
    use_existing_cluster = os.environ.get("MSHIP_USE_EXISTING_RAY_CLUSTER", "false").lower() == "true"
    os.environ.setdefault("RAY_GCS_RPC_TIMEOUT_S", "30")

    if use_existing_cluster:
        # We don't own the cluster: attach to the running one. The driver must run
        # ON a cluster node (Docker co-located / k8s RayJob / bare-metal node) —
        # "auto" finds the local raylet + GCS. A driver cannot attach via a remote
        # GCS address from off-cluster.
        ray.init(address="auto", ignore_reinit_error=True, logging_level=lib_level)
    else:
        # We own the cluster: start a local head sized from MSHIP_NODE_NUM_* (what
        # start_ray.sh used to do). mship_deploy stays alive as the operator and
        # tears it down on exit (owns_cluster in mship_deploy).
        os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
        # RAY_GCS_SERVER_PORT is the only hook ray.init() exposes for this — it's not a
        # kwarg. Left unset, Ray picks a random GCS port every head start, which breaks a
        # joiner's --address across restarts — so pin a stable default here rather than
        # requiring every operator to pass --ray-port (setdefault: an operator's explicit
        # RAY_GCS_SERVER_PORT always wins over both).
        ray_port = os.environ.get("MSHIP_RAY_PORT", str(_DEFAULT_RAY_GCS_PORT))
        os.environ.setdefault("RAY_GCS_SERVER_PORT", ray_port)
        if os.environ.get("MSHIP_RAY_AUTH", "none").lower() == "token":
            if not _ray_auth_is_safe():
                raise RuntimeError(
                    "MSHIP_RAY_AUTH=token requested, but a local Ray cluster with no auth "
                    "token is already running — attaching to it would start unauthenticated "
                    "instead of failing loudly. Stop that cluster first, or unset "
                    "MSHIP_RAY_AUTH/--ray-auth to attach to it without a token."
                )
            os.environ.setdefault("RAY_AUTH_MODE", "token")
        # Reclaim disk from prior runs' leftover session dirs before this run's
        # session is created. Skipped on the existing-cluster branch (KubeRay /
        # an operator we don't own manages its own temp root).
        prune_ray_sessions()
        ray.init(ignore_reinit_error=True, logging_level=lib_level, **_own_cluster_init_kwargs())
    # ray.init re-sets ray.* loggers, so re-pin them after init.
    logging.getLogger("ray").setLevel(lib_level)
    logging.getLogger("ray._private.worker").setLevel(lib_level)


def start_serve(serve_logging_config: LoggingConfig) -> None:
    port = int(os.environ.get("MSHIP_OPENAI_API_PORT", str(_DEFAULT_OPENAI_API_PORT)))
    serve.start(
        proxy_location=ProxyLocation.EveryNode,
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
    # Forward logging/metrics/gateway-name env vars so the gateway replica configures
    # logging at the driver's level even when it lands on a node whose env carries
    # different (or no) values. MSHIP_GATEWAY_NAME is pinned from the gateway_name arg
    # (not just os.environ) so metrics stamping stays correct regardless of driver env.
    env_vars = build_passthrough_env_vars()
    env_vars["MSHIP_GATEWAY_NAME"] = gateway_name
    serve.run(
        ModelshipAPI.options(
            name=gateway_name,
            num_replicas=gateway_replicas,
            max_ongoing_requests=gateway_max_ongoing,
            ray_actor_options={"num_cpus": 0, "runtime_env": {"env_vars": env_vars}},
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


def seed_expected_models(replica_coordinator, gateway_name: str, yml_conf: ModelshipConfig) -> None:
    # Record the full desired set on the replica coordinator (the gateway's
    # readiness baseline) — already-deployed models also count toward "ready".
    # Bumping the generation makes every replica adopt it via its watch loop.
    try:
        ray.get(replica_coordinator.set_expected.remote(gateway_name, [c.name for c in yml_conf.models]))
    except Exception:
        logger.exception("Failed to seed expected model list on coordinator (non-fatal).")
