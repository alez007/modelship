"""CLI argument parsing for the modelship deploy entry point."""

from __future__ import annotations

import argparse
import os

# Maps argparse attribute names to the env vars they override. CLI flags take
# precedence over env vars; downstream code (Ray init, logging, gateway start)
# reads exclusively from os.environ so a single source of truth is preserved.
_STRING_ARG_TO_ENV: dict[str, str] = {
    "cache_dir": "MSHIP_CACHE_DIR",
    "state_dir": "MSHIP_STATE_DIR",
    "state_store": "MSHIP_STATE_STORE",
    "log_format": "MSHIP_LOG_FORMAT",
    "log_target": "MSHIP_LOG_TARGET",
    "otel_endpoint": "OTEL_EXPORTER_OTLP_ENDPOINT",
    "api_keys": "MSHIP_API_KEYS",
    "trusted_identity_header": "MSHIP_TRUSTED_IDENTITY_HEADER",
    "gateway_name": "MSHIP_GATEWAY_NAME",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modelship — serve LLMs with Ray Serve")
    parser.add_argument("--config", help="Path to models.yaml config file (default: config/models.yaml)")
    parser.add_argument("--cache-dir", help="Model cache directory (env: MSHIP_CACHE_DIR)")
    parser.add_argument(
        "--state-dir",
        help=(
            "Directory for the durable effective-config state store (env: MSHIP_STATE_DIR, default: <cache-dir>/state)"
        ),
    )
    parser.add_argument(
        "--state-store",
        help=(
            "State-store connection URI for the effective config and deploy coordinator "
            "(env: MSHIP_STATE_STORE, default: memory://). Schemes: memory:// | file:///path | "
            "redis://[:password@]host:port/db (rediss:// for TLS). A redis:// store also enables "
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
