"""Tests for mship_deploy.py CLI argument parsing and helpers."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import mship_deploy
import pytest

from modelship.deploy.actor_options import (
    build_deployment_options,
    resolve_plugin_wheel,
    total_cpu_reservation,
    total_gpu_reservation,
)
from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase, VllmEngineConfig
from modelship.utils import rand_suffix
from modelship.utils.cli import parse_args


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.config is None
        assert args.redeploy is False
        assert args.gateway_name is None
        assert args.use_existing_ray_cluster is None

    def test_redeploy_flag(self):
        args = parse_args(["--redeploy"])
        assert args.redeploy is True

    def test_reconcile_flag(self):
        args = parse_args(["--reconcile"])
        assert args.reconcile is True
        assert args.replace_strategy == "blue_green"

    def test_reconcile_with_stop_start_strategy(self):
        args = parse_args(["--reconcile", "--replace-strategy", "stop_start"])
        assert args.reconcile is True
        assert args.replace_strategy == "stop_start"

    def test_redeploy_and_reconcile_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            parse_args(["--redeploy", "--reconcile"])

    def test_config_path(self):
        args = parse_args(["--config", "/some/path/models.yaml"])
        assert args.config == "/some/path/models.yaml"

    def test_gateway_name(self):
        args = parse_args(["--gateway-name", "my-gateway"])
        assert args.gateway_name == "my-gateway"

    def test_all_flags_combined(self):
        args = parse_args(
            [
                "--config",
                "llm.yaml",
                "--gateway-name",
                "llm-api",
                "--redeploy",
                "--use-existing-ray-cluster",
            ]
        )
        assert args.config == "llm.yaml"
        assert args.gateway_name == "llm-api"
        assert args.redeploy is True
        assert args.use_existing_ray_cluster is True


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

    def test_llama_cpp_force_cpu(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_cpp,
            num_gpus=1,
        )
        opts = build_deployment_options(config)
        assert opts["ray_actor_options"]["num_gpus"] == 0

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
    def test_noop_on_empty_list(self):
        gateway = MagicMock()
        with patch("mship_deploy.serve.delete") as mock_delete:
            mship_deploy.remove_apps(gateway, [])
        gateway.remove_deployments.remote.assert_not_called()
        mock_delete.assert_not_called()

    def test_unregisters_then_deletes(self):
        gateway = MagicMock()
        gateway.remove_deployments.remote.return_value.result.return_value = ["qwen"]
        apps = ["qwen-aaaaaaaaaa", "kokoro-bbbbbbbbbb"]
        with patch("mship_deploy.serve.delete") as mock_delete:
            mship_deploy.remove_apps(gateway, apps)

        # Unregister from gateway happens before serve.delete so new requests
        # stop routing before the deployment is torn down.
        gateway.remove_deployments.remote.assert_called_once_with(apps)
        assert mock_delete.call_args_list == [(("qwen-aaaaaaaaaa",),), (("kokoro-bbbbbbbbbb",),)]

    def test_continues_on_serve_delete_error(self):
        gateway = MagicMock()
        gateway.remove_deployments.remote.return_value.result.return_value = []
        with patch("mship_deploy.serve.delete", side_effect=[Exception("gone"), None]) as mock_delete:
            mship_deploy.remove_apps(gateway, ["a-1234567890", "b-1234567890"])
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
    def _init_call(self, env):
        from modelship.deploy import serve_utils

        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(serve_utils.ray, "init") as mock_init,
        ):
            serve_utils.connect_ray(20)
        _, kwargs = mock_init.call_args
        return kwargs

    def test_existing_cluster_connects_via_auto(self):
        kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "true"})
        assert kwargs["address"] == "auto"
        # No head is started: resource/metrics kwargs must be absent.
        assert "_metrics_export_port" not in kwargs
        assert "num_cpus" not in kwargs

    def test_own_cluster_starts_head_with_metrics_port(self):
        kwargs = self._init_call(
            {
                "MSHIP_USE_EXISTING_RAY_CLUSTER": "false",
                "MSHIP_METRICS": "true",
                "RAY_METRICS_EXPORT_PORT": "8079",
                "RAY_HEAD_CPU_NUM": "4",
                "MSHIP_RAY_DASHBOARD": "false",
            }
        )
        assert "address" not in kwargs
        # Dashboard off by default to save RAM; metrics still exported.
        assert kwargs["include_dashboard"] is False
        assert "dashboard_host" not in kwargs
        assert kwargs["num_cpus"] == 4
        # Guards the private ray.init kwarg that pins Ray's metrics agent port.
        assert kwargs["_metrics_export_port"] == 8079

    def test_own_cluster_enables_dashboard_when_opted_in(self):
        kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_RAY_DASHBOARD": "TRUE"})
        # Opt-in is case-insensitive and binds the dashboard on all interfaces.
        assert kwargs["include_dashboard"] is True
        assert kwargs["dashboard_host"] == "0.0.0.0"

    def test_own_cluster_omits_metrics_port_when_disabled(self):
        kwargs = self._init_call({"MSHIP_USE_EXISTING_RAY_CLUSTER": "false", "MSHIP_METRICS": "false"})
        assert "address" not in kwargs
        assert "_metrics_export_port" not in kwargs
