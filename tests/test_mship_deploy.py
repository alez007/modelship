"""Tests for mship_deploy.py CLI argument parsing and helpers."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modelship.deploy.actor_options import (
    build_cache_env_vars,
    build_deployment_options,
    build_passthrough_env_vars,
    resolve_plugin_wheel,
    total_cpu_reservation,
    total_gpu_reservation,
)
from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase, VllmEngineConfig
from modelship.utils import rand_suffix
from modelship.utils.cli import apply_args_to_env, parse_args


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.config is None
        assert args.reconcile is False
        assert args.gateway_name is None
        assert args.use_existing_ray_cluster is None

    def test_reconcile_flag(self):
        args = parse_args(["--reconcile"])
        assert args.reconcile is True
        assert args.replace_strategy == "blue_green"

    def test_reconcile_with_stop_start_strategy(self):
        args = parse_args(["--reconcile", "--replace-strategy", "stop_start"])
        assert args.reconcile is True
        assert args.replace_strategy == "stop_start"

    def test_config_path(self):
        args = parse_args(["--config", "/some/path/models.yaml"])
        assert args.config == "/some/path/models.yaml"

    def test_gateway_replicas(self):
        assert parse_args(["--gateway-replicas", "3"]).gateway_replicas == 3

    def test_gateway_replicas_defaults_to_none(self):
        assert parse_args([]).gateway_replicas is None

    def test_gateway_name(self):
        args = parse_args(["--gateway-name", "my-gateway"])
        assert args.gateway_name == "my-gateway"

    def test_ray_auth(self):
        assert parse_args(["--ray-auth", "none"]).ray_auth == "none"

    def test_ray_auth_defaults_to_none(self):
        assert parse_args([]).ray_auth is None

    def test_ray_port(self):
        assert parse_args(["--ray-port", "6380"]).ray_port == 6380

    def test_ray_port_defaults_to_none(self):
        assert parse_args([]).ray_port is None

    def test_address(self):
        assert parse_args(["--address", "mship-head:6380"]).address == "mship-head:6380"

    def test_address_defaults_to_none(self):
        assert parse_args([]).address is None

    def test_token(self):
        assert parse_args(["--token", "secret"]).token == "secret"

    def test_token_defaults_to_none(self):
        assert parse_args([]).token is None

    def test_node_num_cpus(self):
        assert parse_args(["--node-num-cpus", "4"]).node_num_cpus == 4

    def test_node_num_cpus_defaults_to_none(self):
        assert parse_args([]).node_num_cpus is None

    def test_node_num_gpus(self):
        assert parse_args(["--node-num-gpus", "2"]).node_num_gpus == 2

    def test_node_num_gpus_defaults_to_none(self):
        assert parse_args([]).node_num_gpus is None

    def test_responses_ttl_s(self):
        assert parse_args(["--responses-ttl-s", "60"]).responses_ttl_s == 60.0

    def test_responses_ttl_s_defaults_to_none(self):
        assert parse_args([]).responses_ttl_s is None

    def test_state_sweep_interval_s(self):
        assert parse_args(["--state-sweep-interval-s", "30"]).state_sweep_interval_s == 30.0

    def test_state_sweep_interval_s_defaults_to_none(self):
        assert parse_args([]).state_sweep_interval_s is None

    def test_all_flags_combined(self):
        args = parse_args(
            [
                "--config",
                "llm.yaml",
                "--gateway-name",
                "llm-api",
                "--reconcile",
                "--use-existing-ray-cluster",
            ]
        )
        assert args.config == "llm.yaml"
        assert args.gateway_name == "llm-api"
        assert args.reconcile is True
        assert args.use_existing_ray_cluster is True


class TestApplyArgsToEnv:
    def test_state_store_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_STATE_STORE", raising=False)
        apply_args_to_env(parse_args(["--state-store", "redis://cache:6379/0"]))
        assert os.environ["MSHIP_STATE_STORE"] == "redis://cache:6379/0"

    def test_state_store_flag_overrides_preset_env(self, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_STORE", "redis://from-env:6379/0")
        apply_args_to_env(parse_args(["--state-store", "redis://from-flag:6379/0"]))
        assert os.environ["MSHIP_STATE_STORE"] == "redis://from-flag:6379/0"

    def test_no_state_store_leaves_env_untouched(self, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_STORE", "redis://preexisting:6379/0")
        apply_args_to_env(parse_args([]))
        assert os.environ["MSHIP_STATE_STORE"] == "redis://preexisting:6379/0"

    def test_gateway_replicas_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_GATEWAY_REPLICAS", raising=False)
        apply_args_to_env(parse_args(["--gateway-replicas", "4"]))
        assert os.environ["MSHIP_GATEWAY_REPLICAS"] == "4"

    def test_ray_auth_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RAY_AUTH", raising=False)
        apply_args_to_env(parse_args(["--ray-auth", "none"]))
        assert os.environ["MSHIP_RAY_AUTH"] == "none"

    def test_ray_auth_absent_leaves_env_untouched(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RAY_AUTH", raising=False)
        apply_args_to_env(parse_args([]))
        assert "MSHIP_RAY_AUTH" not in os.environ

    def test_ray_port_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RAY_PORT", raising=False)
        apply_args_to_env(parse_args(["--ray-port", "6380"]))
        assert os.environ["MSHIP_RAY_PORT"] == "6380"

    def test_ray_port_absent_leaves_env_untouched(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RAY_PORT", raising=False)
        apply_args_to_env(parse_args([]))
        assert "MSHIP_RAY_PORT" not in os.environ

    def test_address_sets_env(self):
        # patch.dict (not monkeypatch.delenv) so the env write is reverted on exit
        # — MSHIP_ADDRESS actively changes connect_ray's branch, so a leak here
        # would silently flip every later TestConnectRay(Join) test onto the join
        # path (same hazard test_prune_ray_sessions_*_sets_env guards against).
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_ADDRESS", None)
            apply_args_to_env(parse_args(["--address", "mship-head:6380"]))
            assert os.environ["MSHIP_ADDRESS"] == "mship-head:6380"

    def test_address_absent_leaves_env_untouched(self, monkeypatch):
        monkeypatch.delenv("MSHIP_ADDRESS", raising=False)
        apply_args_to_env(parse_args([]))
        assert "MSHIP_ADDRESS" not in os.environ

    def test_token_sets_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_RAY_AUTH_TOKEN", None)
            apply_args_to_env(parse_args(["--token", "secret"]))
            assert os.environ["MSHIP_RAY_AUTH_TOKEN"] == "secret"

    def test_token_absent_leaves_env_untouched(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RAY_AUTH_TOKEN", raising=False)
        apply_args_to_env(parse_args([]))
        assert "MSHIP_RAY_AUTH_TOKEN" not in os.environ

    def test_node_num_cpus_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_NODE_NUM_CPUS", raising=False)
        apply_args_to_env(parse_args(["--node-num-cpus", "4"]))
        assert os.environ["MSHIP_NODE_NUM_CPUS"] == "4"

    def test_node_num_cpus_absent_leaves_env_untouched(self, monkeypatch):
        monkeypatch.delenv("MSHIP_NODE_NUM_CPUS", raising=False)
        apply_args_to_env(parse_args([]))
        assert "MSHIP_NODE_NUM_CPUS" not in os.environ

    def test_node_num_gpus_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_NODE_NUM_GPUS", raising=False)
        apply_args_to_env(parse_args(["--node-num-gpus", "2"]))
        assert os.environ["MSHIP_NODE_NUM_GPUS"] == "2"

    def test_node_num_gpus_absent_leaves_env_untouched(self, monkeypatch):
        monkeypatch.delenv("MSHIP_NODE_NUM_GPUS", raising=False)
        apply_args_to_env(parse_args([]))
        assert "MSHIP_NODE_NUM_GPUS" not in os.environ

    def test_prune_ray_sessions_false_sets_env(self):
        # patch.dict (not monkeypatch.delenv) so the env write is reverted on exit
        # — otherwise MSHIP_PRUNE_RAY_SESSIONS=false leaks into the prune tests.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_PRUNE_RAY_SESSIONS", None)
            apply_args_to_env(parse_args(["--prune-ray-sessions", "false"]))
            assert os.environ["MSHIP_PRUNE_RAY_SESSIONS"] == "false"

    def test_prune_ray_sessions_true_sets_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_PRUNE_RAY_SESSIONS", None)
            apply_args_to_env(parse_args(["--prune-ray-sessions", "true"]))
            assert os.environ["MSHIP_PRUNE_RAY_SESSIONS"] == "true"

    def test_prune_ray_sessions_absent_leaves_env_untouched(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_PRUNE_RAY_SESSIONS", None)
            apply_args_to_env(parse_args([]))
            assert "MSHIP_PRUNE_RAY_SESSIONS" not in os.environ

    def test_no_preflight_sets_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_PREFLIGHT", None)
            apply_args_to_env(parse_args(["--no-preflight"]))
            assert os.environ["MSHIP_PREFLIGHT"] == "false"

    def test_no_preflight_absent_leaves_env_untouched(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MSHIP_PREFLIGHT", None)
            apply_args_to_env(parse_args([]))
            assert "MSHIP_PREFLIGHT" not in os.environ

    def test_responses_ttl_s_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_RESPONSES_TTL_S", raising=False)
        apply_args_to_env(parse_args(["--responses-ttl-s", "60"]))
        assert os.environ["MSHIP_RESPONSES_TTL_S"] == "60.0"

    def test_state_sweep_interval_s_sets_env(self, monkeypatch):
        monkeypatch.delenv("MSHIP_STATE_SWEEP_INTERVAL_S", raising=False)
        apply_args_to_env(parse_args(["--state-sweep-interval-s", "30"]))
        assert os.environ["MSHIP_STATE_SWEEP_INTERVAL_S"] == "30.0"


class TestRandSuffix:
    def test_default_length(self):
        suffix = rand_suffix()
        assert len(suffix) == 5

    def test_custom_length(self):
        suffix = rand_suffix(10)
        assert len(suffix) == 10

    def test_chars_are_alphanumeric_lowercase(self):
        for _ in range(50):
            suffix = rand_suffix()
            assert all(c.islower() or c.isdigit() for c in suffix)


class TestBuildDeploymentOptions:
    def test_basic_options(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
            num_cpus=2,
        )
        opts = build_deployment_options(config)
        actor = opts["ray_actor_options"]
        assert actor["num_gpus"] == 1
        assert actor["num_cpus"] == 2
        assert "env_vars" in actor["runtime_env"]
        assert "pip" not in actor["runtime_env"]
        assert "placement_group_bundles" not in opts

    def test_with_plugin_wheel(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.custom,
            plugin="myplugin",
        )
        wheel_path = Path("/tmp/myplugin-0.1.0-py3-none-any.whl")
        opts = build_deployment_options(config, plugin_wheel=wheel_path)
        assert opts["ray_actor_options"]["runtime_env"]["pip"] == [str(wheel_path)]

    def test_llama_server_honors_num_gpus(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
            num_gpus=2,
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 2
        assert "placement_group_bundles" not in opts

    def test_llama_server_num_gpus_zero_stays_cpu(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
            num_gpus=0,
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 0

    def test_stable_diffusion_cpp_force_cpu(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.image,
            loader=ModelLoader.stable_diffusion_cpp,
            num_gpus=1,
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 0

    def test_passthrough_env_vars_forwarded_to_replicas(self, monkeypatch):
        # --no-metrics / logging / gateway set on the driver must reach the replica
        # via runtime_env, else the replica defaults to metrics-on (inconsistent).
        monkeypatch.setenv("MSHIP_METRICS", "false")
        monkeypatch.setenv("MSHIP_GATEWAY_NAME", "edge")
        monkeypatch.setenv("MSHIP_PREFLIGHT", "false")
        monkeypatch.setenv("MSHIP_RESPONSES_TTL_S", "60")
        monkeypatch.setenv("MSHIP_STATE_SWEEP_INTERVAL_S", "30")
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
        )
        env_vars = build_deployment_options(config)["ray_actor_options"]["runtime_env"]["env_vars"]
        assert env_vars["MSHIP_METRICS"] == "false"
        assert env_vars["MSHIP_GATEWAY_NAME"] == "edge"
        assert env_vars["MSHIP_PREFLIGHT"] == "false"
        assert env_vars["MSHIP_RESPONSES_TTL_S"] == "60"
        assert env_vars["MSHIP_STATE_SWEEP_INTERVAL_S"] == "30"

    def test_unset_passthrough_env_vars_not_forwarded(self, monkeypatch):
        # Unset on the driver → not forwarded, so the replica keeps its own default.
        monkeypatch.delenv("MSHIP_METRICS", raising=False)
        monkeypatch.delenv("MSHIP_PREFLIGHT", raising=False)
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
        )
        env_vars = build_deployment_options(config)["ray_actor_options"]["runtime_env"]["env_vars"]
        assert "MSHIP_METRICS" not in env_vars
        assert "MSHIP_PREFLIGHT" not in env_vars

    def test_log_level_in_passthrough_and_deployment_env(self):
        # The gateway-replica bug: MSHIP_LOG_LEVEL must flow through the shared
        # passthrough helper and into a model deployment's runtime_env alongside
        # the cache vars (which the gateway path omits but the model path keeps).
        with patch.dict(os.environ, {"MSHIP_LOG_LEVEL": "TRACE"}, clear=True):
            assert build_passthrough_env_vars()["MSHIP_LOG_LEVEL"] == "TRACE"

            config = ModelshipModelConfig(
                name="test-model",
                model="some-model",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
                num_gpus=1,
            )
            env_vars = build_deployment_options(config)["ray_actor_options"]["runtime_env"]["env_vars"]
            assert env_vars["MSHIP_LOG_LEVEL"] == "TRACE"
            # Cache vars still present (the model path keeps them).
            for key in build_cache_env_vars():
                assert key in env_vars

    def test_pipeline_parallel_uses_placement_group(self):
        # num_gpus=2 + pp=2 satisfies the world_size==num_gpus invariant; the
        # outer actor sits in bundle 0 with no GPU and vLLM workers claim the
        # rest via the inherited placement group.
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=2,
            vllm_engine_kwargs=VllmEngineConfig(pipeline_parallel_size=2),
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 0
        assert opts["placement_group_strategy"] == "STRICT_PACK"
        bundles = opts["placement_group_bundles"]
        assert len(bundles) == 2
        assert all(b["GPU"] == 1.0 for b in bundles)

    def test_tp_times_pp_builds_pg(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=4,
            vllm_engine_kwargs=VllmEngineConfig(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
            ),
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 0
        assert len(opts["placement_group_bundles"]) == 4
        assert all(b["GPU"] == 1.0 for b in opts["placement_group_bundles"])

    def test_single_slot_skips_placement_group(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=0.3,
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 0.3
        assert "placement_group_bundles" not in opts

    def test_max_ongoing_requests_omitted_by_default(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
        )
        opts = build_deployment_options(config)
        assert "max_ongoing_requests" not in opts

    def test_max_ongoing_requests_forwarded_when_set(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
            max_ongoing_requests=256,
        )
        opts = build_deployment_options(config)
        assert opts["max_ongoing_requests"] == 256

    def test_max_ongoing_requests_forwarded_for_multi_slot(self):
        # Multi-slot (PG) deploys carry the cap alongside placement_group_bundles.
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=2,
            vllm_engine_kwargs=VllmEngineConfig(tensor_parallel_size=2),
            max_ongoing_requests=64,
        )
        opts = build_deployment_options(config)
        assert opts["max_ongoing_requests"] == 64
        assert len(opts["placement_group_bundles"]) == 2


class TestReservationTotals:
    def test_single_slot_uses_actor_options(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=0.5,
            num_cpus=2,
        )
        opts = build_deployment_options(config)
        assert total_gpu_reservation(opts) == 0.5
        assert total_cpu_reservation(opts) == 2

    def test_multi_slot_sums_pg_bundles(self):
        # 4 slots, each bundle reserves num_cpus from the cluster; the outer
        # actor's CPU sits inside bundle 0 and is not additive.
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=4,
            num_cpus=2,
        )
        opts = build_deployment_options(config)
        assert total_gpu_reservation(opts) == 4
        assert total_cpu_reservation(opts) == 8


class TestRemoveApps:
    # remove_apps is defined in serve_utils (mship_deploy imports it lazily inside
    # main() now, so it's no longer a mship_deploy module attribute).
    def test_noop_on_empty_list(self):
        from modelship.deploy import serve_utils

        replica_coordinator = MagicMock()
        with patch("modelship.deploy.serve_utils.serve.delete") as mock_delete:
            serve_utils.remove_apps([], replica_coordinator, "gw")
        replica_coordinator.unregister_deployment.remote.assert_not_called()
        mock_delete.assert_not_called()

    def test_unregisters_then_deletes(self):
        from modelship.deploy import serve_utils

        replica_coordinator = MagicMock()
        apps = ["qwen-aaaaaaaaaa", "kokoro-bbbbbbbbbb"]
        with (
            patch("modelship.deploy.serve_utils.ray.get") as mock_get,
            patch("modelship.deploy.serve_utils.serve.delete") as mock_delete,
        ):
            serve_utils.remove_apps(apps, replica_coordinator, "gw")

        # Each app is dropped from the replica coordinator's registry (which bumps
        # the gateway generation so replicas stop routing) before serve.delete tears
        # it down.
        replica_coordinator.unregister_deployment.remote.assert_any_call("gw", "qwen-aaaaaaaaaa")
        replica_coordinator.unregister_deployment.remote.assert_any_call("gw", "kokoro-bbbbbbbbbb")
        mock_get.assert_called_once()  # batched ray.get over the unregister calls
        assert mock_delete.call_args_list == [(("qwen-aaaaaaaaaa",),), (("kokoro-bbbbbbbbbb",),)]

    def test_continues_on_serve_delete_error(self):
        from modelship.deploy import serve_utils

        replica_coordinator = MagicMock()
        with (
            patch("modelship.deploy.serve_utils.ray.get"),
            patch("modelship.deploy.serve_utils.serve.delete", side_effect=[Exception("gone"), None]) as mock_delete,
        ):
            serve_utils.remove_apps(["a-1234567890", "b-1234567890"], replica_coordinator, "gw")
        # Both deletes attempted even though the first raised.
        assert mock_delete.call_count == 2


class TestStartGateway:
    def _run(self, env):
        from modelship.deploy import serve_utils

        bound = MagicMock()
        options = MagicMock()
        options.return_value.bind.return_value = bound
        logging_config = MagicMock()
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(serve_utils.ModelshipAPI, "options", options),
            patch.object(serve_utils.serve, "run") as mock_run,
        ):
            serve_utils.start_gateway("gw", logging_config)
        return options, mock_run

    def test_defaults(self):
        # Ensure no leftover env from the ambient process leaks the assertion.
        options, mock_run = self._run({"MSHIP_GATEWAY_REPLICAS": "1", "MSHIP_GATEWAY_MAX_ONGOING": "1024"})
        _, kwargs = options.call_args
        assert kwargs["num_replicas"] == 1
        assert kwargs["max_ongoing_requests"] == 1024
        mock_run.assert_called_once()

    def test_env_overrides(self):
        options, _ = self._run({"MSHIP_GATEWAY_REPLICAS": "3", "MSHIP_GATEWAY_MAX_ONGOING": "256"})
        _, kwargs = options.call_args
        assert kwargs["num_replicas"] == 3
        assert kwargs["max_ongoing_requests"] == 256

    def test_forwards_log_level_to_gateway_replica(self):
        # The gateway replica must inherit MSHIP_LOG_LEVEL (and the gateway name)
        # via runtime_env, else it can't configure logging at the driver's level.
        options, _ = self._run(
            {
                "MSHIP_GATEWAY_REPLICAS": "1",
                "MSHIP_GATEWAY_MAX_ONGOING": "1024",
                "MSHIP_LOG_LEVEL": "TRACE",
            }
        )
        _, kwargs = options.call_args
        env_vars = kwargs["ray_actor_options"]["runtime_env"]["env_vars"]
        assert env_vars["MSHIP_LOG_LEVEL"] == "TRACE"

    def test_gateway_name_pinned_from_arg(self):
        # MSHIP_GATEWAY_NAME is forwarded from the gateway_name arg even when absent
        # from os.environ, so metrics stamping stays correct on isolated environments.
        from modelship.deploy import serve_utils

        bound = MagicMock()
        options = MagicMock()
        options.return_value.bind.return_value = bound
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(serve_utils.ModelshipAPI, "options", options),
            patch.object(serve_utils.serve, "run"),
        ):
            serve_utils.start_gateway("edge", MagicMock())
        _, kwargs = options.call_args
        assert kwargs["ray_actor_options"]["runtime_env"]["env_vars"]["MSHIP_GATEWAY_NAME"] == "edge"

    @pytest.mark.parametrize(
        "name, value",
        [
            ("MSHIP_GATEWAY_REPLICAS", "0"),
            ("MSHIP_GATEWAY_REPLICAS", "-2"),
            ("MSHIP_GATEWAY_MAX_ONGOING", "0"),
            ("MSHIP_GATEWAY_MAX_ONGOING", "notanint"),
        ],
    )
    def test_rejects_invalid_env(self, name, value):
        with pytest.raises(ValueError, match=name):
            self._run({name: value})


class TestResolvePluginWheel:
    def test_resolves_latest_wheel(self, tmp_path):
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()
        (wheel_dir / "myplugin-0.1.0-py3-none-any.whl").touch()
        (wheel_dir / "myplugin-0.1.1-py3-none-any.whl").touch()

        with patch.dict(os.environ, {"MSHIP_PLUGIN_WHEEL_DIR": str(wheel_dir)}):
            wheel = resolve_plugin_wheel("myplugin")
            assert wheel.name == "myplugin-0.1.1-py3-none-any.whl"
            assert wheel.is_absolute()

    def test_raises_if_no_wheel(self, tmp_path):
        import pytest

        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()

        with (
            patch.dict(os.environ, {"MSHIP_PLUGIN_WHEEL_DIR": str(wheel_dir)}),
            pytest.raises(RuntimeError, match="No wheel found for plugin 'myplugin'"),
        ):
            resolve_plugin_wheel("myplugin")


class TestConnectRay:
    def _init_call(self, env, pop=()):
        """Returns (ray.init kwargs, RAY_AUTH_MODE seen in os.environ), captured
        before patch.dict reverts it. `pop` clears env vars before the call."""
        from modelship.deploy import serve_utils

        with patch.dict(os.environ, env, clear=False):
            for key in pop:
                os.environ.pop(key, None)
            with (
                patch.object(serve_utils.ray, "init") as mock_init,
                # Don't let the own-cluster branch sweep the real /tmp/ray during tests.
                patch.object(serve_utils, "prune_ray_sessions"),
                # _ray_auth_is_safe has its own dedicated tests; decouple these from
                # the real filesystem's /tmp/ray and ~/.ray state.
                patch.object(serve_utils, "ray_auth_is_safe", return_value=True),
            ):
                serve_utils.connect_ray(20)
                auth_mode = os.environ.get("RAY_AUTH_MODE")
        _, kwargs = mock_init.call_args
        return kwargs, auth_mode

    def test_existing_cluster_connects_via_auto(self):
        kwargs, _ = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "true"})
        assert kwargs["address"] == "auto"
        # No head is started: resource/metrics kwargs must be absent.
        assert "_metrics_export_port" not in kwargs
        assert "num_cpus" not in kwargs

    def test_own_cluster_starts_head_with_metrics_port(self):
        kwargs, _ = self._init_call(
            {
                "MSHIP_USE_EXISTING_RAY_CLUSTER": "false",
                "MSHIP_METRICS": "true",
                "RAY_METRICS_EXPORT_PORT": "8079",
                "MSHIP_NODE_NUM_CPUS": "4",
            }
        )
        assert "address" not in kwargs
        assert kwargs["num_cpus"] == 4
        # Guards the private ray.init kwarg that pins Ray's metrics agent port.
        assert kwargs["_metrics_export_port"] == 8079

    def test_own_cluster_dashboard_always_on_bound_localhost(self):
        kwargs, _ = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"}, pop=("MSHIP_RAY_DASHBOARD",))
        assert kwargs["include_dashboard"] is True
        assert kwargs["dashboard_host"] == "127.0.0.1"

    def test_own_cluster_dashboard_host_overridable(self):
        kwargs, _ = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_RAY_DASHBOARD": "0.0.0.0"})
        # Still on — MSHIP_RAY_DASHBOARD only ever changes the bind host now, never on/off.
        assert kwargs["include_dashboard"] is True
        assert kwargs["dashboard_host"] == "0.0.0.0"

    def test_existing_cluster_never_sets_dashboard_kwargs(self):
        kwargs, _ = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "true"})
        assert "include_dashboard" not in kwargs
        assert "dashboard_host" not in kwargs

    def test_own_cluster_omits_metrics_port_when_disabled(self):
        kwargs, _ = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_METRICS": "false"})
        assert "address" not in kwargs
        assert "_metrics_export_port" not in kwargs

    def test_own_cluster_auth_disabled_by_default(self):
        _, auth_mode = self._init_call(
            {"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"}, pop=("MSHIP_RAY_AUTH", "RAY_AUTH_MODE")
        )
        assert auth_mode is None

    def test_own_cluster_auth_explicit_none(self):
        _, auth_mode = self._init_call(
            {"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_RAY_AUTH": "none"}, pop=("RAY_AUTH_MODE",)
        )
        assert auth_mode is None

    def test_own_cluster_auth_opt_in(self):
        _, auth_mode = self._init_call(
            {"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_RAY_AUTH": "token"}, pop=("RAY_AUTH_MODE",)
        )
        assert auth_mode == "token"

    def test_own_cluster_auth_respects_explicit_ray_auth_mode(self):
        _, auth_mode = self._init_call(
            {"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_RAY_AUTH": "token", "RAY_AUTH_MODE": "disabled"}
        )
        # setdefault: an operator's explicit RAY_AUTH_MODE always wins, even when opting into token mode.
        assert auth_mode == "disabled"

    def test_own_cluster_auth_opt_in_bails_when_unsafe(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_RAY_AUTH": "token"}, clear=False),
            patch.object(serve_utils, "prune_ray_sessions"),
            patch.object(serve_utils.ray, "init") as mock_init,
            patch.object(serve_utils, "ray_auth_is_safe", return_value=False),
        ):
            os.environ.pop("RAY_AUTH_MODE", None)
            with pytest.raises(RuntimeError, match="MSHIP_RAY_AUTH=token"):
                serve_utils.connect_ray(20)
        mock_init.assert_not_called()

    def test_own_cluster_ray_port_sets_gcs_server_port(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_RAY_PORT": "6390"}, clear=False),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions"),
            patch.object(serve_utils, "ray_auth_is_safe", return_value=True),
        ):
            os.environ.pop("RAY_GCS_SERVER_PORT", None)
            serve_utils.connect_ray(20)
            assert os.environ.get("RAY_GCS_SERVER_PORT") == "6390"

    def test_own_cluster_ray_port_absent_defaults_gcs_server_port_to_6380(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"}, clear=False),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions"),
            patch.object(serve_utils, "ray_auth_is_safe", return_value=True),
        ):
            os.environ.pop("MSHIP_RAY_PORT", None)
            os.environ.pop("RAY_GCS_SERVER_PORT", None)
            serve_utils.connect_ray(20)
            # Not Ray's own 6379 default — that collides with the recommended
            # same-host Redis state store under --network=host.
            assert os.environ.get("RAY_GCS_SERVER_PORT") == "6380"

    def test_own_cluster_ray_port_respects_explicit_gcs_server_port(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(
                os.environ,
                {
                    "MSHIP_USE_EXISTING_RAY_CLUSTER": "false",
                    "MSHIP_RAY_PORT": "6380",
                    "RAY_GCS_SERVER_PORT": "6381",
                },
                clear=False,
            ),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions"),
            patch.object(serve_utils, "ray_auth_is_safe", return_value=True),
        ):
            serve_utils.connect_ray(20)
            # setdefault: an operator's explicit RAY_GCS_SERVER_PORT always wins.
            assert os.environ["RAY_GCS_SERVER_PORT"] == "6381"

    def test_existing_cluster_never_sets_gcs_server_port(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "true", "MSHIP_RAY_PORT": "6380"}, clear=False),
            patch.object(serve_utils.ray, "init"),
        ):
            os.environ.pop("RAY_GCS_SERVER_PORT", None)
            serve_utils.connect_ray(20)
            assert "RAY_GCS_SERVER_PORT" not in os.environ

    def test_existing_cluster_never_sets_auth_mode(self):
        _, auth_mode = self._init_call(
            {"MSHIP_USE_EXISTING_RAY_CLUSTER": "true"}, pop=("MSHIP_RAY_AUTH", "RAY_AUTH_MODE")
        )
        assert auth_mode is None

    def test_prunes_stale_sessions_on_own_cluster(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "false"}, clear=False),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions") as mock_prune,
        ):
            serve_utils.connect_ray(20)
        mock_prune.assert_called_once()

    def test_skips_prune_on_existing_cluster(self):
        from modelship.deploy import serve_utils

        # We don't own the temp root on an external cluster — never sweep it.
        with (
            patch.dict(os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "true"}, clear=False),
            patch.object(serve_utils.ray, "init"),
            patch.object(serve_utils, "prune_ray_sessions") as mock_prune,
        ):
            serve_utils.connect_ray(20)
        mock_prune.assert_not_called()


@pytest.fixture
def _reset_join_node():
    """_join_ray_cluster assigns the module-level _join_node global as soon as
    Node() succeeds, so every test that goes through it leaves that global set
    unless reset — otherwise isolation would depend on test-definition order."""
    from modelship.deploy import serve_utils

    serve_utils._join_node = None
    yield
    serve_utils._join_node = None


class TestJoinRayCluster:
    """_join_ray_cluster starts THIS container's node in-process via
    ray._private.node.Node(head=False) — the same path `ray start --address`
    takes internally — instead of shelling out. These tests mock that
    Ray-internal surface (and so double as the loud-failure guard for a Ray
    bump that moves it); TestClusterJoin exercises it for real."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_join_node):
        yield

    def _join(self, env, pop=(), bootstrap="10.0.0.1:6380"):
        from modelship.deploy import serve_utils

        mock_node = MagicMock()
        mock_node.get_temp_dir_path.return_value = "/tmp/ray"
        with patch.dict(os.environ, env, clear=False):
            for key in pop:
                os.environ.pop(key, None)
            with (
                patch("ray._private.services.canonicalize_bootstrap_address", return_value=bootstrap) as mock_canon,
                patch("ray._private.services.get_node_ip_address", return_value="10.0.0.2"),
                patch("ray._private.parameter.RayParams") as mock_params,
                patch("ray._private.node.Node", return_value=mock_node) as mock_node_cls,
                patch(
                    "ray._private.authentication.authentication_token_setup.ensure_token_if_auth_enabled"
                ) as mock_ensure,
                patch("ray._private.utils.write_ray_address") as mock_write,
            ):
                result = serve_utils._join_ray_cluster("head:6380")
        return {
            "node": mock_node,
            "node_cls": mock_node_cls,
            "params_kwargs": mock_params.call_args.kwargs,
            "canon": mock_canon,
            "ensure": mock_ensure,
            "write": mock_write,
            "result": result,
        }

    def test_builds_rayparams_with_cpus_and_gpus(self):
        kw = self._join({"MSHIP_NODE_NUM_CPUS": "4", "MSHIP_NODE_NUM_GPUS": "2"})["params_kwargs"]
        assert kw["num_cpus"] == 4
        assert kw["num_gpus"] == 2

    def test_omits_resources_when_unset(self):
        kw = self._join({}, pop=("MSHIP_NODE_NUM_CPUS", "MSHIP_NODE_NUM_GPUS"))["params_kwargs"]
        assert kw["num_cpus"] is None
        assert kw["num_gpus"] is None

    def test_explicit_zero_gpus_honored(self):
        # Thin-image case: MSHIP_NODE_NUM_GPUS=0 is a real reservation, not "unset".
        kw = self._join({"MSHIP_NODE_NUM_GPUS": "0"})["params_kwargs"]
        assert kw["num_gpus"] == 0

    def test_metrics_export_port_set(self):
        kw = self._join({"MSHIP_METRICS": "true", "RAY_METRICS_EXPORT_PORT": "9999"})["params_kwargs"]
        assert kw["metrics_export_port"] == 9999

    def test_metrics_export_port_none_when_disabled(self):
        kw = self._join({"MSHIP_METRICS": "false"})["params_kwargs"]
        assert kw["metrics_export_port"] is None

    def test_passes_bootstrap_gcs_address(self):
        kw = self._join({}, bootstrap="10.9.9.9:6380")["params_kwargs"]
        assert kw["gcs_address"] == "10.9.9.9:6380"

    def test_creates_worker_node_supervised(self):
        out = self._join({})
        _, kwargs = out["node_cls"].call_args
        assert kwargs["head"] is False
        assert kwargs["shutdown_at_exit"] is True
        assert kwargs["spawn_reaper"] is True
        out["node"].check_version_info.assert_called_once()

    def test_writes_discovery_marker_with_bootstrap_address(self):
        out = self._join({}, bootstrap="10.9.9.9:6380")
        out["write"].assert_called_once_with("10.9.9.9:6380", "/tmp/ray")

    def test_sets_module_global_and_returns_node(self):
        from modelship.deploy import serve_utils

        out = self._join({})
        assert serve_utils._join_node is out["node"]
        assert out["result"] is out["node"]

    def test_calls_ensure_token_preflight(self):
        self._join({})["ensure"].assert_called_once()

    def test_unresolvable_address_raises(self):
        from modelship.deploy import serve_utils

        with (
            patch("ray._private.services.canonicalize_bootstrap_address", return_value=None),
            pytest.raises(RuntimeError, match="Could not resolve the Ray head address"),
        ):
            serve_utils._join_ray_cluster("bogus:1")


class TestConnectRayJoinBranch:
    """connect_ray's MSHIP_ADDRESS branch: brings up the local node via
    _join_ray_cluster (mocked here — TestJoinRayCluster covers its internals),
    then attaches the driver with ray.init(address='auto')."""

    @pytest.fixture(autouse=True)
    def _reset(self, _reset_join_node):
        yield

    def test_join_branch_creates_node_then_attaches_via_auto(self):
        from modelship.deploy import serve_utils

        with patch.dict(os.environ, {"MSHIP_ADDRESS": "head:6380"}, clear=False):
            os.environ.pop("MSHIP_USE_EXISTING_RAY_CLUSTER", None)
            with (
                patch.object(serve_utils, "_join_ray_cluster") as mock_join,
                patch.object(serve_utils, "prune_ray_sessions") as mock_prune,
                patch.object(serve_utils.ray, "init") as mock_init,
            ):
                serve_utils.connect_ray(20)
            mock_join.assert_called_once_with("head:6380")
            mock_prune.assert_called_once()
            # R2: address="auto", not a bare init — a bare init would silently
            # form a split-brain cluster if local discovery somehow failed.
            assert mock_init.call_args.kwargs["address"] == "auto"

    def test_address_and_existing_cluster_mutually_exclusive_raises(self):
        from modelship.deploy import serve_utils

        with (
            patch.dict(
                os.environ, {"MSHIP_USE_EXISTING_RAY_CLUSTER": "true", "MSHIP_ADDRESS": "head:6380"}, clear=False
            ),
            pytest.raises(RuntimeError, match="mutually exclusive"),
        ):
            serve_utils.connect_ray(20)


class TestLeaveRayCluster:
    @pytest.fixture(autouse=True)
    def _reset(self, _reset_join_node):
        yield

    def test_leave_tears_down_only_the_join_node(self):
        from modelship.deploy import serve_utils

        mock_node = MagicMock()
        serve_utils._join_node = mock_node
        with patch.object(serve_utils.ray, "shutdown") as mock_shutdown:
            serve_utils.leave_ray_cluster()
        mock_shutdown.assert_called_once()
        # allow_graceful lets the raylet drain hosted actors; check_alive=False
        # because a partially-started node may not have every process up.
        mock_node.kill_all_processes.assert_called_once_with(check_alive=False, allow_graceful=True)

    def test_leave_noop_when_not_joined(self):
        from modelship.deploy import serve_utils

        assert serve_utils._join_node is None
        with patch.object(serve_utils.ray, "shutdown") as mock_shutdown:
            serve_utils.leave_ray_cluster()  # must not raise with no node to stop
        mock_shutdown.assert_called_once()


class TestSuperviseJoinNode:
    @pytest.fixture(autouse=True)
    def _reset(self, _reset_join_node):
        yield

    def test_exits_nonzero_and_kills_when_core_process_dies(self):
        from modelship.deploy import serve_utils

        node = MagicMock()
        dead = MagicMock()
        dead.returncode = 1  # not a graceful SIGTERM/0 exit
        node.dead_processes.return_value = [("raylet", dead)]
        serve_utils._join_node = node
        with (
            patch.object(serve_utils.time, "sleep"),
            pytest.raises(SystemExit) as exc,
        ):
            serve_utils.supervise_join_node()
        assert exc.value.code == 1
        node.kill_all_processes.assert_called_once_with(check_alive=False, allow_graceful=False)

    def test_ignores_graceful_exits_and_keeps_supervising(self):
        from modelship.deploy import serve_utils

        node = MagicMock()
        graceful = MagicMock()
        graceful.returncode = 0  # in _GRACEFUL_EXIT_CODES — expected, not a failure
        node.dead_processes.return_value = [("agent", graceful)]
        serve_utils._join_node = node
        # First sleep returns, second breaks the otherwise-infinite loop so the
        # test can assert the graceful exit was NOT treated as a failure.
        with (
            patch.object(serve_utils.time, "sleep", side_effect=[None, RuntimeError("stop")]),
            pytest.raises(RuntimeError, match="stop"),
        ):
            serve_utils.supervise_join_node()
        node.kill_all_processes.assert_not_called()


class TestRayAuthIsSafe:
    """ray_auth_is_safe lives in the ray-free modelship.utils.ray_auth leaf
    module (so it can run before `import ray`); connect_ray imports it. The
    marker path mirrors Ray's get_ray_temp_dir() == <RAY_TMPDIR>/ray."""

    def test_true_when_token_already_exists(self, tmp_path):
        from modelship.utils import ray_auth

        home = tmp_path / "home"
        (home / ".ray").mkdir(parents=True)
        (home / ".ray" / "auth_token").write_text("abc")
        with (
            patch.dict(os.environ, {"RAY_TMPDIR": str(tmp_path)}, clear=False),
            patch.object(ray_auth.Path, "home", return_value=home),
        ):
            assert ray_auth.ray_auth_is_safe() is True

    def test_true_when_no_cluster_running(self, tmp_path):
        from modelship.utils import ray_auth

        home = tmp_path / "home"
        home.mkdir()
        with (
            patch.dict(os.environ, {"RAY_TMPDIR": str(tmp_path)}, clear=False),
            patch.object(ray_auth.Path, "home", return_value=home),
        ):
            assert ray_auth.ray_auth_is_safe() is True

    def test_false_when_attaching_to_cluster_with_no_token(self, tmp_path):
        from modelship.utils import ray_auth

        home = tmp_path / "home"
        home.mkdir()
        ray_root = tmp_path / "ray"
        ray_root.mkdir()
        (ray_root / "ray_current_cluster").write_text("127.0.0.1:6379")
        with (
            patch.dict(os.environ, {"RAY_TMPDIR": str(tmp_path)}, clear=False),
            patch.object(ray_auth.Path, "home", return_value=home),
        ):
            assert ray_auth.ray_auth_is_safe() is False

    def test_false_on_no_passwd_entry(self, tmp_path):
        from modelship.utils import ray_auth

        with (
            patch.dict(os.environ, {"RAY_TMPDIR": str(tmp_path)}, clear=False),
            patch.object(ray_auth.Path, "home", side_effect=RuntimeError),
        ):
            assert ray_auth.ray_auth_is_safe() is False


class TestResolveRayAuthEnv:
    """resolve_ray_auth_env front-runs Ray's import-time RAY_AUTH_MODE latch:
    it translates the MSHIP_* auth/join vars into RAY_AUTH_MODE/RAY_AUTH_TOKEN
    before mship_deploy imports ray. Runs with a clean auth env each time."""

    def _resolve(self, env, safe=True):
        from modelship.utils import ray_auth

        base = dict.fromkeys(
            ["MSHIP_ADDRESS", "MSHIP_USE_EXISTING_RAY_CLUSTER", "MSHIP_RAY_AUTH", "MSHIP_RAY_AUTH_TOKEN"], ""
        )
        with patch.dict(os.environ, {**base, **env}, clear=False):
            for key in ["MSHIP_ADDRESS", "MSHIP_USE_EXISTING_RAY_CLUSTER", "MSHIP_RAY_AUTH", "MSHIP_RAY_AUTH_TOKEN"]:
                if not os.environ.get(key):
                    os.environ.pop(key, None)
            os.environ.pop("RAY_AUTH_MODE", None)
            os.environ.pop("RAY_AUTH_TOKEN", None)
            with patch.object(ray_auth, "ray_auth_is_safe", return_value=safe):
                ray_auth.resolve_ray_auth_env()
            return os.environ.get("RAY_AUTH_MODE"), os.environ.get("RAY_AUTH_TOKEN")

    def test_own_head_token_safe_sets_mode(self):
        mode, _ = self._resolve({"MSHIP_RAY_AUTH": "token"}, safe=True)
        assert mode == "token"

    def test_own_head_token_unsafe_leaves_mode_unset(self):
        # Deferred to connect_ray's own re-check, which raises the clear error.
        mode, _ = self._resolve({"MSHIP_RAY_AUTH": "token"}, safe=False)
        assert mode is None

    def test_join_with_token_sets_mode_and_token(self):
        mode, token = self._resolve({"MSHIP_ADDRESS": "head:6380", "MSHIP_RAY_AUTH_TOKEN": "secret"})
        assert mode == "token"
        assert token == "secret"

    def test_join_without_token_leaves_auth_unset(self):
        mode, token = self._resolve({"MSHIP_ADDRESS": "head:6380"})
        assert mode is None
        assert token is None

    def test_existing_cluster_never_sets_mode(self):
        mode, _ = self._resolve({"MSHIP_USE_EXISTING_RAY_CLUSTER": "true", "MSHIP_RAY_AUTH": "token"})
        assert mode is None

    def test_explicit_ray_auth_mode_wins(self):
        from modelship.utils import ray_auth

        with patch.dict(os.environ, {"MSHIP_RAY_AUTH": "token", "RAY_AUTH_MODE": "disabled"}, clear=False):
            os.environ.pop("MSHIP_ADDRESS", None)
            os.environ.pop("MSHIP_USE_EXISTING_RAY_CLUSTER", None)
            with patch.object(ray_auth, "ray_auth_is_safe", return_value=True):
                ray_auth.resolve_ray_auth_env()
            # setdefault: an operator's explicit RAY_AUTH_MODE always wins.
            assert os.environ["RAY_AUTH_MODE"] == "disabled"


class TestPruneRaySessions:
    """`prune_ray_sessions` resolves the temp root via Ray's own
    `get_ray_temp_dir()`, which returns `<RAY_TMPDIR>/ray` — so pointing
    RAY_TMPDIR at a tmp dir fully isolates these tests from the real /tmp/ray."""

    def _temp_root(self, tmp_path):
        root = tmp_path / "ray"
        root.mkdir()
        return root

    def _make_session(self, root, pid, name=None):
        session = root / (name or f"session_2026-06-19_10-00-00_000000_{pid}")
        (session / "logs").mkdir(parents=True)
        (session / "logs" / "raylet.out").write_text("log")
        return session

    def test_removes_dead_pid_session(self, tmp_path):
        from modelship.deploy import serve_utils

        root = self._temp_root(tmp_path)
        dead = self._make_session(root, 111)
        with (
            patch.dict(
                os.environ,
                {"RAY_TMPDIR": str(tmp_path), "MSHIP_PRUNE_RAY_SESSIONS": "true"},
                clear=False,
            ),
            patch.object(serve_utils, "_pid_alive", return_value=False),
        ):
            serve_utils.prune_ray_sessions()
        assert not dead.exists()

    def test_keeps_live_pid_session(self, tmp_path):
        from modelship.deploy import serve_utils

        root = self._temp_root(tmp_path)
        live = self._make_session(root, 222)
        with (
            patch.dict(
                os.environ,
                {"RAY_TMPDIR": str(tmp_path), "MSHIP_PRUNE_RAY_SESSIONS": "true"},
                clear=False,
            ),
            patch.object(serve_utils, "_pid_alive", return_value=True),
        ):
            serve_utils.prune_ray_sessions()
        assert live.exists()

    def test_skips_symlink_and_non_session_entries(self, tmp_path):
        from modelship.deploy import serve_utils

        root = self._temp_root(tmp_path)
        dead = self._make_session(root, 333)
        latest = root / "session_latest"
        latest.symlink_to(dead)
        marker = root / "ray_current_cluster"
        marker.write_text("127.0.0.1:6379")
        unrelated = root / "not_a_session"
        unrelated.mkdir()
        with (
            patch.dict(
                os.environ,
                {"RAY_TMPDIR": str(tmp_path), "MSHIP_PRUNE_RAY_SESSIONS": "true"},
                clear=False,
            ),
            patch.object(serve_utils, "_pid_alive", return_value=False),
        ):
            serve_utils.prune_ray_sessions()
        assert not dead.exists()  # the real session dir is removed
        assert latest.is_symlink()  # the symlink itself survives (now dangling)
        assert marker.exists()  # non-session files untouched
        assert unrelated.exists()  # non-matching dirs untouched

    def test_disabled_via_env_keeps_everything(self, tmp_path):
        from modelship.deploy import serve_utils

        root = self._temp_root(tmp_path)
        dead = self._make_session(root, 444)
        with (
            patch.dict(
                os.environ,
                {"RAY_TMPDIR": str(tmp_path), "MSHIP_PRUNE_RAY_SESSIONS": "false"},
                clear=False,
            ),
            patch.object(serve_utils, "_pid_alive", return_value=False),
        ):
            serve_utils.prune_ray_sessions()
        assert dead.exists()

    def test_missing_temp_root_is_noop(self, tmp_path):
        from modelship.deploy import serve_utils

        # No <tmp>/ray dir exists — must not raise.
        with patch.dict(
            os.environ,
            {"RAY_TMPDIR": str(tmp_path), "MSHIP_PRUNE_RAY_SESSIONS": "true"},
            clear=False,
        ):
            serve_utils.prune_ray_sessions()

    def test_pid_alive_true_for_current_process(self):
        from modelship.deploy import serve_utils

        assert serve_utils._pid_alive(os.getpid()) is True

    def test_pid_alive_false_for_reaped_pid(self):
        import subprocess

        from modelship.deploy import serve_utils

        proc = subprocess.Popen(["true"])
        proc.wait()
        assert serve_utils._pid_alive(proc.pid) is False
