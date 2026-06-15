import os
import signal
import sys

# Must be set BEFORE `import ray`: ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV
# is a module-level constant, evaluated at ray's import time. Leaving it on
# makes Ray auto-inject `py_executable="uv run --python X"` whenever the driver
# runs under `uv run`, which overrides the per-job virtualenv that
# runtime_env.pip creates for plugin wheels — breaking plugin imports in the
# worker. Keep the uv hook off so runtime_env.pip's py_executable survives.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "false")

# Set HF / vLLM / FlashInfer cache dirs BEFORE importing anything that
# transitively pulls in `huggingface_hub` — its `HF_HOME` constant is latched
# at import time, so setting the env later does nothing. Driver-side downloads
# (the model resolver) and Ray workers must agree on the cache path; workers
# get these via runtime_env.env_vars in actor_options.build_cache_env_vars.
_BASE_CACHE = os.environ.get("MSHIP_CACHE_DIR", "/.cache")
os.environ.setdefault("HF_HOME", f"{_BASE_CACHE}/huggingface")
os.environ.setdefault("VLLM_CACHE_ROOT", f"{_BASE_CACHE}/vllm")
os.environ.setdefault("FLASHINFER_CACHE_DIR", f"{_BASE_CACHE}/flashinfer")

# Set RAY_LOG_LEVEL/RAY_SERVE_LOG_LEVEL/VLLM_LOGGING_LEVEL/TRANSFORMERS_VERBOSITY
# from MSHIP_LOG_LEVEL BEFORE `import ray` — Ray's loggers latch the env value
# at import time, so configuring them later (in configure_logging) is too late
# for the driver process. The level is env-var-only (no CLI flag) since argv
# is parsed inside main(), well after `import ray`.
from modelship.logging import propagate_lib_log_env  # noqa: E402

propagate_lib_log_env()

import ray  # noqa: E402
from ray import serve  # noqa: E402
from ray.serve.schema import LoggingConfig  # noqa: E402

from modelship.deploy.config import (  # noqa: E402
    load_yaml_config,
    resolve_all_model_sources,
    resolve_all_plugin_wheels,
    resolve_all_reasoning_parsers,
    resolve_all_tool_parsers,
)
from modelship.deploy.serve_utils import (  # noqa: E402
    connect_ray,
    delete_apps_quietly,
    get_existing_apps,
    make_operator_id,
    remove_apps,
    seed_expected_models,
    shutdown_ray,
    start_gateway,
    start_serve,
)
from modelship.deploy.strategy import DeployContext, compute_deploy_plan, run_deploy_loop  # noqa: E402
from modelship.infer.deploy_coordinator import OperatorProbe, get_or_create_coordinator  # noqa: E402
from modelship.logging import configure_logging, get_lib_log_config, get_logger  # noqa: E402
from modelship.utils.cli import apply_args_to_env, parse_args  # noqa: E402

logger = get_logger("startup")
_DEFAULT_GATEWAY_NAME = "modelship api"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    apply_args_to_env(args)

    configure_logging()
    gateway_name = os.environ.get("MSHIP_GATEWAY_NAME", _DEFAULT_GATEWAY_NAME)
    # apply_args_to_env (above) has folded --use-existing-ray-cluster into this env
    # var, the same source connect_ray uses. With an external cluster (KubeRay)
    # this process is a one-shot deployer: exit after deploying, never tear down a
    # cluster it doesn't own. The gateway + deployments persist and the gateway
    # self-heals its routing table from the coordinator registry.
    owns_cluster = os.environ.get("MSHIP_USE_EXISTING_RAY_CLUSTER", "false").lower() != "true"
    # Library log level (one step above app level). Used to silence Ray Serve's
    # system actors (controller/proxy/replica access logs) and Ray's driver
    # logger, which both ignore Python-level setLevel from the parent process.
    lib_level, lib_level_name = get_lib_log_config()
    serve_logging_config = LoggingConfig(log_level=lib_level_name)

    if args.redeploy:
        logger.info("--redeploy: tearing down existing deployments...")
        shutdown_ray()

    connect_ray(lib_level)
    start_serve(serve_logging_config)

    existing_apps = set() if args.redeploy else get_existing_apps()
    fresh_install = gateway_name not in existing_apps
    if existing_apps:
        logger.info("Found existing deployments: %s", ", ".join(sorted(existing_apps)))
    if fresh_install and not args.redeploy:
        logger.info("No existing gateway found — treating as fresh install.")

    yml_conf = load_yaml_config(args.config)
    logger.info("Init modelship app with config: %s", yml_conf)

    plugin_wheels = resolve_all_plugin_wheels(yml_conf)

    # The detached coordinator holds the cross-operator deploy lock and the durable
    # ownership registry. Read this gateway's recorded deployments so the reconcile
    # diff is scoped to apps we own (never another gateway's).
    coordinator = get_or_create_coordinator()
    owned_deployments = set(ray.get(coordinator.get_deployments.remote(gateway_name)))
    plan = compute_deploy_plan(
        yml_conf,
        existing_apps,
        owned_deployments,
        gateway_name,
        fresh_install=fresh_install,
        reconcile=args.reconcile,
    )
    apps_to_remove = list(plan.apps_to_remove)

    # Track deployments created by this invocation: deployment_name -> model_name.
    # Shared with the SIGINT/SIGTERM cleanup handler below via closure.
    deployed_this_run: dict[str, str] = {}

    def _cleanup(sig, _frame) -> None:
        logger.info("Shutting down (signal %s), cleaning up deployments from this run...", sig)
        delete_apps_quietly(reversed(deployed_this_run))
        if fresh_install and owns_cluster:
            shutdown_ray()
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        # Start the gateway FIRST on fresh install so /health, /v1/models, and
        # /readyz are reachable while models are still loading (or downloading).
        # Models register with the gateway as they come up.
        if fresh_install:
            start_gateway(gateway_name, serve_logging_config)

        # Pre-flight: download/validate every built-in-loader model on the driver
        # before any model deployment spins up. Surfaces auth / missing-repo /
        # missing-file errors here instead of inside an UNHEALTHY replica. Runs
        # AFTER the gateway is up so /health and /readyz answer during downloads.
        resolve_all_model_sources(yml_conf)
        resolve_all_tool_parsers(yml_conf)
        resolve_all_reasoning_parsers(yml_conf)

        gateway_handle = serve.get_app_handle(gateway_name)
        seed_expected_models(gateway_handle, yml_conf)

        # stop_start: drop old deployments BEFORE deploying new ones, so the
        # freed resources are available for the deploy loop. Used when the
        # cluster can't fit old + new at the same time.
        if args.replace_strategy == "stop_start":
            remove_apps(gateway_handle, apps_to_remove, coordinator, gateway_name)
            apps_to_remove = []

        # The probe is driver-owned so Ray force-releases the coordinator lock if
        # this process dies ungracefully.
        operator_id = make_operator_id()
        probe = OperatorProbe.options(num_cpus=0).remote()
        logger.info("Operator id=%s; coordinator acquired.", operator_id)

        ctx = DeployContext(
            plugin_wheels=plugin_wheels,
            coordinator=coordinator,
            probe=probe,
            operator_id=operator_id,
            gateway_name=gateway_name,
            gateway_handle=gateway_handle,
            serve_logging_config=serve_logging_config,
            deployed_this_run=deployed_this_run,
        )
        pass_count, fatally_failed = run_deploy_loop(plan.models_to_add, ctx)

        logger.info(
            "Deploy complete. %d new deployment(s) from this run (over %d pass(es)).",
            len(deployed_this_run),
            pass_count,
        )

        # blue_green: drop old deployments AFTER new ones are live and
        # registered with the gateway. During the brief overlap the gateway
        # round-robins across both old and new handles for the same model,
        # so no requests are lost.
        if apps_to_remove:
            remove_apps(gateway_handle, apps_to_remove, coordinator, gateway_name)

        if fatally_failed:
            logger.error(
                "%d model(s) failed to deploy and were skipped — fix config and restart:",
                len(fatally_failed),
            )
            for name, reason in fatally_failed:
                logger.error("  - %s: %s", name, reason)

        if fresh_install and owns_cluster:
            # Standalone (we own Ray): stay alive as the operator process.
            # _cleanup gracefully deletes each deployment (letting actors run
            # __del__ and clean up child processes like vllm EngineCore) before
            # tearing down Ray. With an external cluster we instead exit here,
            # leaving the gateway + deployments running under KubeRay.
            signal.pause()

    except BaseException as e:
        if isinstance(e, SystemExit):
            raise
        logger.exception("Startup failed, cleaning up deployments from this run...")
        delete_apps_quietly(reversed(deployed_this_run))
        if fresh_install and owns_cluster:
            shutdown_ray()
        raise


if __name__ == "__main__":
    main()
