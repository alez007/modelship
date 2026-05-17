import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray
from ray import serve
from ray.serve.schema import LoggingConfig

from modelship.deploy.actor_options import build_deployment_options, total_cpu_reservation, total_gpu_reservation
from modelship.infer.infer_config import ModelshipConfig, ModelshipModelConfig
from modelship.infer.model_deployment import ModelDeployment
from modelship.logging import get_logger

logger = get_logger("startup")

_DEPLOY_RETRY_SLEEP_S = 2.0
_WAITING_LOG_EVERY_N_PASSES = 30  # with 2s sleep, log "still waiting" once per minute


@dataclass
class DeployPlan:
    """Result of diffing models.yaml against the cluster."""

    models_to_add: list[ModelshipModelConfig]
    apps_to_remove: list[str]


def compute_deploy_plan(
    yml_conf: ModelshipConfig,
    existing_apps: set[str],
    gateway_name: str,
    *,
    fresh_install: bool,
    reconcile: bool,
) -> DeployPlan:
    """Diff desired (models.yaml) against deployed (serve.status). Deployment
    names are `{model_name}-{fingerprint}`, so a pure set comparison detects
    both renames and config drift: a model whose num_gpus changed gets a new
    fingerprint -> new deployment_name -> shows up as both an add and a
    remove (handled as a replace by callers)."""

    # Schedule larger world-size models first so they claim whole GPU units before
    # fractional models consume the pool. World size = tp * pp.
    sorted_models = sorted(
        yml_conf.models,
        key=lambda c: (
            c.vllm_engine_kwargs.tensor_parallel_size * c.vllm_engine_kwargs.pipeline_parallel_size
            if c.vllm_engine_kwargs
            else 1
        ),
        reverse=True,
    )

    desired_names = {c.deployment_name() for c in sorted_models}
    # `existing_apps` includes the gateway app itself; exclude it so reconcile
    # never targets the gateway for removal.
    deployed_names = existing_apps - {gateway_name}

    if reconcile:
        if fresh_install:
            logger.info("--reconcile: no existing gateway — equivalent to a fresh deploy.")
        apps_to_remove = sorted(deployed_names - desired_names)
        if apps_to_remove:
            logger.info("--reconcile: %d deployment(s) to remove: %s", len(apps_to_remove), apps_to_remove)
    else:
        apps_to_remove = []

    # In all modes, skip configs already deployed under their fingerprint —
    # makes plain re-runs idempotent instead of double-deploying.
    models_to_add = [c for c in sorted_models if c.deployment_name() not in deployed_names]
    if models_to_add:
        logger.info(
            "%d deployment(s) to add: %s",
            len(models_to_add),
            [c.deployment_name() for c in models_to_add],
        )
    return DeployPlan(models_to_add=models_to_add, apps_to_remove=apps_to_remove)


@dataclass
class DeployContext:
    plugin_wheels: dict[str, Path]
    coordinator: Any
    probe: Any
    operator_id: str
    gateway_handle: Any
    serve_logging_config: LoggingConfig
    deployed_this_run: dict[str, str]


def try_reserve_and_deploy(config: ModelshipModelConfig, ctx: DeployContext) -> tuple[str, str | None]:
    """One attempt at deploying *config*. Returns (status, detail) where status is:
    "skipped" (no progress, retry), "deployed", "transient" (deploy raised; retry),
    "fatal" (deployment reported a permanent error; skip permanently)."""
    wheel = ctx.plugin_wheels.get(config.plugin) if config.plugin else None
    deploy_opts = build_deployment_options(config, plugin_wheel=wheel)
    deployment_name = config.deployment_name()

    reserved, _reason = ray.get(
        ctx.coordinator.try_reserve.remote(
            ctx.operator_id,
            deployment_name,
            total_gpu_reservation(deploy_opts),
            total_cpu_reservation(deploy_opts),
            ctx.probe,
        )
    )
    if not reserved:
        return "skipped", None

    try:
        logger.info("Deploying model: %s (deployment: %s)", config.name, deployment_name)
        ctx.deployed_this_run[deployment_name] = config.name
        serve.run(
            ModelDeployment.options(
                name=deployment_name,
                num_replicas=config.num_replicas,
                max_constructor_retry_count=1,
                logging_config=ctx.serve_logging_config,
                **deploy_opts,
            ).bind(config),
            name=deployment_name,
            route_prefix=None,
        )
        logger.info("Model ready: %s (deployment: %s)", config.name, deployment_name)
        try:
            ctx.gateway_handle.add_models.remote({deployment_name: config.name}).result()
        except Exception:
            logger.exception("Failed to register %s with gateway", deployment_name)
        return "deployed", None
    except Exception:
        # Did the deployment actively report a fatal init error before dying?
        try:
            fatal_err = ray.get(ctx.coordinator.pop_fatal_error.remote(deployment_name), timeout=2.0)
        except Exception:
            fatal_err = None

        ctx.deployed_this_run.pop(deployment_name, None)
        if fatal_err is not None:
            logger.error(
                "Skipping model '%s' permanently (deployment=%s): %s",
                config.name,
                deployment_name,
                fatal_err,
            )
            try:
                serve.delete(deployment_name)
            except Exception:
                logger.exception("Failed to delete failed deployment: %s", deployment_name)
            return "fatal", str(fatal_err)
        logger.exception(
            "Deploy failed for %s (deployment=%s); will retry next pass.",
            config.name,
            deployment_name,
        )
        return "transient", None
    finally:
        # Ray may already be shut down (e.g. SIGINT cleanup ran shutdown_ray);
        # the OperatorProbe death-detection will free the lock either way once
        # the driver dies.
        if ray.is_initialized():
            try:
                ray.get(ctx.coordinator.release.remote(ctx.operator_id))
            except Exception:
                logger.exception("Failed to release coordinator lock (operator=%s)", ctx.operator_id)


def run_deploy_loop(
    models: list[ModelshipModelConfig],
    ctx: DeployContext,
) -> tuple[int, list[tuple[str, str]]]:
    """Retry-pass loop: each pass tries every not-yet-deployed model. Models
    whose resources don't currently fit (or whose reservation is rejected
    because another operator holds the lock) are skipped and retried on the
    next pass. Placeable models deploy in configured order (TP>1 first)."""
    remaining = list(models)
    fatally_failed: list[tuple[str, str]] = []
    pass_count = 0
    passes_with_no_progress = 0

    while remaining:
        pass_count += 1
        made_progress = False
        for config in list(remaining):
            status, detail = try_reserve_and_deploy(config, ctx)
            if status == "deployed":
                remaining.remove(config)
                made_progress = True
            elif status == "fatal":
                fatally_failed.append((config.name, detail or ""))
                remaining.remove(config)
                made_progress = True
            # "skipped" / "transient" -> stay in `remaining` for the next pass

        if made_progress:
            passes_with_no_progress = 0
        else:
            passes_with_no_progress += 1
            if passes_with_no_progress == 1 or passes_with_no_progress % _WAITING_LOG_EVERY_N_PASSES == 0:
                logger.info(
                    "Waiting for capacity for %d model(s): %s",
                    len(remaining),
                    [c.name for c in remaining],
                )

        if remaining:
            time.sleep(_DEPLOY_RETRY_SLEEP_S)

    return pass_count, fatally_failed
