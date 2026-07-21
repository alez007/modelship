"""CLI argument parsing for the modelship deploy entry point."""

from __future__ import annotations

import argparse
import os

from modelship.utils import parse_memory_bytes

# Maps argparse attribute names to the env vars they override. CLI flags take
# precedence over env vars; downstream code (Ray init, logging, gateway start)
# reads exclusively from os.environ so a single source of truth is preserved.
_STRING_ARG_TO_ENV: dict[str, str] = {
    "cache_dir": "MSHIP_CACHE_DIR",
    "state_store": "MSHIP_STATE_STORE",
    "log_format": "MSHIP_LOG_FORMAT",
    "log_target": "MSHIP_LOG_TARGET",
    "otel_endpoint": "OTEL_EXPORTER_OTLP_ENDPOINT",
    "api_keys": "MSHIP_API_KEYS",
    "trusted_identity_header": "MSHIP_TRUSTED_IDENTITY_HEADER",
    "gateway_name": "MSHIP_GATEWAY_NAME",
    "address": "MSHIP_ADDRESS",
    "token": "MSHIP_RAY_AUTH_TOKEN",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modelship — serve LLMs with Ray Serve")
    parser.add_argument("--config", help="Path to models.yaml config file (default: config/models.yaml)")
    parser.add_argument("--cache-dir", help="Model cache directory (env: MSHIP_CACHE_DIR)")
    parser.add_argument(
        "--state-store",
        help=(
            "State-store connection URI for the effective config, deploy coordinator and "
            "/v1/responses conversations (env: MSHIP_STATE_STORE, default: memory://). Schemes: "
            "memory:// | redis://[:password@]host:port/db (rediss:// for TLS). memory:// is "
            "cluster-scoped but dies with the cluster; redis:// survives it and also enables "
            "GCS fault tolerance when modelship starts its own Ray head."
        ),
    )
    parser.add_argument(
        "--gateway-name",
        help="Name for the API gateway app (env: MSHIP_GATEWAY_NAME, default: modelship api)",
    )
    parser.add_argument(
        "--gateway-replicas",
        type=int,
        help="Number of API gateway replicas (env: MSHIP_GATEWAY_REPLICAS, default: 1)",
    )
    parser.add_argument(
        "--use-existing-ray-cluster",
        action="store_true",
        default=None,
        help="Connect to an existing Ray cluster (env: MSHIP_USE_EXISTING_RAY_CLUSTER)",
    )
    parser.add_argument(
        "--ray-auth",
        choices=["token", "none"],
        help=(
            "Ray cluster authentication, applied only when modelship starts its own head "
            "(env: MSHIP_RAY_AUTH, default: none). 'token' requires a bearer token, generated "
            "by Ray itself at ~/.ray/auth_token, for the dashboard and cluster-internal RPC."
        ),
    )
    parser.add_argument(
        "--address",
        help=(
            "Join an existing Ray cluster as an additional compute node, given the head's GCS "
            "server address as plain host:port, e.g. mship-docker-head:6380 (env: MSHIP_ADDRESS; "
            "matches --ray-port's default on the head). Not a ray:// Ray Client URI — this node "
            "becomes real cluster compute, not a remote driver. Mutually exclusive with "
            "--use-existing-ray-cluster (that attaches to a cluster this process deploys to and "
            "exits, without joining it as a node)."
        ),
    )
    parser.add_argument(
        "--token",
        help=(
            "Cluster auth token for joining a Ray cluster whose head runs --ray-auth=token "
            "(env: MSHIP_RAY_AUTH_TOKEN). Only meaningful together with --address; retrieve the "
            "head's token via `docker exec <head> cat ~/.ray/auth_token`."
        ),
    )
    parser.add_argument(
        "--ray-port",
        type=int,
        help=(
            "Port for Ray's GCS server, applied only when modelship starts its own head "
            "(env: MSHIP_RAY_PORT, default: 6380). Change this if 6380 is already taken on "
            "the host — e.g. avoid 6379, which the docs-recommended same-host Redis state "
            "store (MSHIP_STATE_STORE=redis://) may also want under --network=host."
        ),
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        help=(
            "Port for Ray's dashboard, applied only when modelship starts its own head "
            "(env: MSHIP_RAY_DASHBOARD_PORT, default: 8265, Ray's own default). Unlike "
            "--ray-port/--openai-api-port, Ray gives this no per-node override otherwise — "
            "set it when running multiple modelship heads on one host under --network=host, "
            "so each head's dashboard gets a distinct port."
        ),
    )
    parser.add_argument(
        "--node-num-cpus", type=int, help="CPUs this node reserves (env: MSHIP_NODE_NUM_CPUS, default: auto-detect)"
    )
    parser.add_argument(
        "--node-num-gpus", type=int, help="GPUs this node reserves (env: MSHIP_NODE_NUM_GPUS, default: auto-detect)"
    )
    parser.add_argument(
        "--node-memory",
        type=parse_memory_bytes,
        help=(
            "This node's total memory budget, e.g. '8Gi' (env: MSHIP_NODE_MEMORY, default: "
            "auto-detect). Split into Ray's object_store_memory (30%%) and schedulable 'memory' "
            "resource (70%%) the same way Ray splits an auto-detected total. Set this explicitly "
            "when co-locating multiple modelship node containers on one physical host without "
            "per-container cgroup memory limits — otherwise each node auto-detects the full "
            "host's RAM independently, and the cluster total double/triple-counts the same "
            "physical memory (mirrors --node-num-cpus/--node-num-gpus for the memory dimension). "
            "Under Docker, also pass `--shm-size` >= the derived object_store_memory (~30%% of "
            "this value) — Docker's 64MB default /dev/shm is far below that, and Ray silently "
            "falls back to slower disk-backed storage instead of erroring when it doesn't fit."
        ),
    )
    parser.add_argument("--log-format", choices=["text", "json"], help="Log format (env: MSHIP_LOG_FORMAT)")
    parser.add_argument(
        "--log-target",
        help="Log target: 'console' (default) or syslog URI e.g. syslog://host:514, syslog+tcp://host:514 (env: MSHIP_LOG_TARGET)",
    )
    parser.add_argument(
        "--otel-endpoint",
        help="OpenTelemetry OTLP endpoint e.g. http://collector:4317 (env: OTEL_EXPORTER_OTLP_ENDPOINT)",
    )
    parser.add_argument("--no-metrics", action="store_true", default=None, help="Disable metrics (env: MSHIP_METRICS)")
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        default=None,
        help=(
            "Disable preflight hardware-based auto-sizing; models run on loader/library "
            "defaults plus explicit config (env: MSHIP_PREFLIGHT). Useful for benchmarking."
        ),
    )
    parser.add_argument(
        "--prune-ray-sessions",
        choices=["true", "false"],
        default=None,
        help=(
            "Whether to delete stale dead-pid Ray session dirs under the temp root on "
            "own-cluster startup (env: MSHIP_PRUNE_RAY_SESSIONS, default: true)"
        ),
    )
    parser.add_argument("--api-keys", help="Comma-separated API keys (env: MSHIP_API_KEYS)")
    parser.add_argument(
        "--trusted-identity-header",
        help=(
            "Header name (e.g. X-Consumer-Id) whose value a fronting credentials layer has "
            "already resolved and authorized; modelship trusts it unconditionally for log "
            "correlation and future state-keying (env: MSHIP_TRUSTED_IDENTITY_HEADER). "
            "Only enable when modelship is reachable exclusively from that layer — see "
            "docs/model-configuration.md."
        ),
    )
    parser.add_argument(
        "--max-request-body-bytes", type=int, help="Max request body size in bytes (env: MSHIP_MAX_REQUEST_BODY_BYTES)"
    )
    parser.add_argument(
        "--openai-api-port",
        type=int,
        help="Port for the OpenAI-compatible API (env: MSHIP_OPENAI_API_PORT, default: 8000)",
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        default=False,
        help=(
            "Diff models.yaml against the cluster: add new models, remove dropped ones, "
            "replace those whose config changed (matched by name + fingerprint). "
            "With no --config, reconciles the live cluster to this gateway's persisted "
            "effective config only (self-heal after cluster loss)."
        ),
    )
    parser.add_argument(
        "--responses-ttl-s",
        type=float,
        help=(
            "TTL in seconds for stored /v1/responses conversation state; <=0 disables "
            "expiry (env: MSHIP_RESPONSES_TTL_S, default: 2592000 = 30 days)"
        ),
    )
    parser.add_argument(
        "--state-sweep-interval-s",
        type=float,
        help=(
            "Interval in seconds between expired-key sweeps in the in-memory state store "
            "(env: MSHIP_STATE_SWEEP_INTERVAL_S, default: 300)"
        ),
    )
    parser.add_argument(
        "--replace-strategy",
        choices=["blue_green", "stop_start"],
        default="blue_green",
        help=(
            "How to replace a model whose config changed. blue_green (default): deploy "
            "new alongside old, then drop old (no request loss, peak resource = old+new). "
            "stop_start: drop old first, then deploy new (brief unavailability, no overlap)."
        ),
    )
    return parser.parse_args(argv)


def apply_args_to_env(args: argparse.Namespace) -> None:
    """Write CLI args into os.environ. CLI takes precedence over pre-set env vars."""
    for attr, env_var in _STRING_ARG_TO_ENV.items():
        val = getattr(args, attr, None)
        if val is not None:
            os.environ[env_var] = val

    if args.use_existing_ray_cluster is True:
        os.environ["MSHIP_USE_EXISTING_RAY_CLUSTER"] = "true"
    if args.ray_auth is not None:
        os.environ["MSHIP_RAY_AUTH"] = args.ray_auth
    if args.ray_port is not None:
        os.environ["MSHIP_RAY_PORT"] = str(args.ray_port)
    if args.dashboard_port is not None:
        os.environ["MSHIP_RAY_DASHBOARD_PORT"] = str(args.dashboard_port)
    if args.node_num_cpus is not None:
        os.environ["MSHIP_NODE_NUM_CPUS"] = str(args.node_num_cpus)
    if args.node_num_gpus is not None:
        os.environ["MSHIP_NODE_NUM_GPUS"] = str(args.node_num_gpus)
    if args.node_memory is not None:
        os.environ["MSHIP_NODE_MEMORY"] = str(args.node_memory)
    if args.no_metrics is True:
        os.environ["MSHIP_METRICS"] = "false"
    if args.no_preflight is True:
        os.environ["MSHIP_PREFLIGHT"] = "false"
    if args.prune_ray_sessions is not None:
        os.environ["MSHIP_PRUNE_RAY_SESSIONS"] = args.prune_ray_sessions
    if args.max_request_body_bytes is not None:
        os.environ["MSHIP_MAX_REQUEST_BODY_BYTES"] = str(args.max_request_body_bytes)
    if args.gateway_replicas is not None:
        os.environ["MSHIP_GATEWAY_REPLICAS"] = str(args.gateway_replicas)
    if args.openai_api_port is not None:
        os.environ["MSHIP_OPENAI_API_PORT"] = str(args.openai_api_port)
    if args.responses_ttl_s is not None:
        os.environ["MSHIP_RESPONSES_TTL_S"] = str(args.responses_ttl_s)
    if args.state_sweep_interval_s is not None:
        os.environ["MSHIP_STATE_SWEEP_INTERVAL_S"] = str(args.state_sweep_interval_s)
