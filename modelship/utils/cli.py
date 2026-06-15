"""CLI argument parsing for the modelship deploy entry point."""

from __future__ import annotations

import argparse
import os

# Maps argparse attribute names to the env vars they override. CLI flags take
# precedence over env vars; downstream code (Ray init, logging, gateway start)
# reads exclusively from os.environ so a single source of truth is preserved.
_STRING_ARG_TO_ENV: dict[str, str] = {
    "cache_dir": "MSHIP_CACHE_DIR",
    "log_format": "MSHIP_LOG_FORMAT",
    "log_target": "MSHIP_LOG_TARGET",
    "otel_endpoint": "OTEL_EXPORTER_OTLP_ENDPOINT",
    "api_keys": "MSHIP_API_KEYS",
    "gateway_name": "MSHIP_GATEWAY_NAME",
    "model_stack": "MSHIP_MODEL_STACK",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modelship — serve LLMs with Ray Serve")
    parser.add_argument("--config", help="Path to models.yaml config file (default: config/models.yaml)")
    parser.add_argument(
        "--model-stack",
        help=(
            "Profile to auto-generate a config from, e.g. chat/assistant/studio/everything "
            "(env: MSHIP_MODEL_STACK). Regenerates config/models_stack_<profile>.yaml sized to "
            "the detected hardware on every start. Ignored when --config is given."
        ),
    )
    parser.add_argument("--cache-dir", help="Model cache directory (env: MSHIP_CACHE_DIR)")
    parser.add_argument(
        "--gateway-name",
        help="Name for the API gateway app (env: MSHIP_GATEWAY_NAME, default: modelship api)",
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
    parser.add_argument("--api-keys", help="Comma-separated API keys (env: MSHIP_API_KEYS)")
    parser.add_argument(
        "--max-request-body-bytes", type=int, help="Max request body size in bytes (env: MSHIP_MAX_REQUEST_BODY_BYTES)"
    )
    parser.add_argument(
        "--openai-api-port",
        type=int,
        help="Port for the OpenAI-compatible API (env: MSHIP_OPENAI_API_PORT, default: 8000)",
    )
    parser.add_argument(
        "--redeploy",
        action="store_true",
        default=False,
        help="Tear down all existing deployments before deploying (default: additive)",
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        default=False,
        help=(
            "Diff models.yaml against the cluster: add new models, remove dropped ones, "
            "replace those whose config changed (matched by name + fingerprint). "
            "Mutually exclusive with --redeploy."
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
    args = parser.parse_args(argv)
    if args.redeploy and args.reconcile:
        parser.error("--redeploy and --reconcile are mutually exclusive")
    return args


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
    if args.max_request_body_bytes is not None:
        os.environ["MSHIP_MAX_REQUEST_BODY_BYTES"] = str(args.max_request_body_bytes)
    if args.openai_api_port is not None:
        os.environ["MSHIP_OPENAI_API_PORT"] = str(args.openai_api_port)
