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

from modelship.infer.infer_config import ModelshipConfig
from modelship.logging import get_logger
from modelship.openai.api import ModelshipAPI
from modelship.utils import rand_suffix

logger = get_logger("startup")
_DEFAULT_OPENAI_API_PORT = 8000

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


def remove_apps(app_names: list[str], coordinator, gateway_name: str) -> None:
    """Drop the given deployment apps from the coordinator's ownership registry
    (which bumps the gateway generation so every replica's watch loop stops routing
    to them), then delete them from Ray Serve (`serve.delete` drains in-flight
    requests first)."""
    if not app_names:
        return
    try:
        ray.get([coordinator.unregister_deployment.remote(gateway_name, a) for a in app_names])
    except Exception:
        logger.exception("Failed to drop deployments from registry: %s", app_names)
    delete_apps_quietly(app_names)


def _own_cluster_init_kwargs() -> dict[str, object]:
    """ray.init kwargs to start our own head, mirroring the flags start_ray.sh
    used to pass. Resources auto-detect when RAY_HEAD_* are unset."""
    kwargs: dict[str, object] = {}
    if os.environ.get("MSHIP_RAY_DASHBOARD", "false").lower() == "true":
        kwargs["include_dashboard"] = True
        kwargs["dashboard_host"] = "0.0.0.0"
    else:
        kwargs["include_dashboard"] = False
    if cpus := os.environ.get("RAY_HEAD_CPU_NUM"):
        kwargs["num_cpus"] = int(cpus)
    if gpus := os.environ.get("RAY_HEAD_GPU_NUM"):
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
        # We own the cluster: start a local head sized from RAY_HEAD_* (what
        # start_ray.sh used to do). mship_deploy stays alive as the operator and
        # tears it down on exit (owns_cluster in mship_deploy).
        os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
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


def seed_expected_models(coordinator, gateway_name: str, yml_conf: ModelshipConfig) -> None:
    # Record the full desired set on the coordinator (the gateway's readiness
    # baseline) — already-deployed models also count toward "ready". Bumping the
    # generation makes every replica adopt it via its watch loop.
    try:
        ray.get(coordinator.set_expected.remote(gateway_name, [c.name for c in yml_conf.models]))
    except Exception:
        logger.exception("Failed to seed expected model list on coordinator (non-fatal).")
