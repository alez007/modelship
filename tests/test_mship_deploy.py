"""Tests for mship_deploy.py CLI argument parsing and helpers."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import mship_deploy
import pytest

from modelship.deploy.actor_options import build_actor_options, resolve_plugin_wheel
from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, ModelUsecase, VllmEngineConfig
from modelship.utils import rand_suffix
from modelship.utils.cli import parse_args


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.config is None
        assert args.redeploy is False
        assert args.gateway_name is None
        assert args.ray_cluster_address is None
        assert args.ray_redis_port is None
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
                "--ray-cluster-address",
                "10.0.0.1",
                "--ray-redis-port",
                "6379",
                "--use-existing-ray-cluster",
            ]
        )
        assert args.config == "llm.yaml"
        assert args.gateway_name == "llm-api"
        assert args.redeploy is True
        assert args.ray_cluster_address == "10.0.0.1"
        assert args.ray_redis_port == "6379"
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


class TestBuildActorOptions:
    def test_basic_options(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
            num_cpus=2,
        )
        opts = build_actor_options(config)
        assert opts["num_gpus"] == 1
        assert opts["num_cpus"] == 2
        assert "env_vars" in opts["runtime_env"]
        assert "pip" not in opts["runtime_env"]

    def test_with_plugin_wheel(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.custom,
            plugin="myplugin",
        )
        wheel_path = Path("/tmp/myplugin-0.1.0-py3-none-any.whl")
        opts = build_actor_options(config, plugin_wheel=wheel_path)
        assert opts["runtime_env"]["pip"] == [str(wheel_path)]

    def test_llama_cpp_force_cpu(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_cpp,
            num_gpus=1,
        )
        opts = build_actor_options(config)
        assert opts["num_gpus"] == 0

    def test_pipeline_parallel_defaults_to_ray_backend(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
            vllm_engine_kwargs=VllmEngineConfig(pipeline_parallel_size=2),
        )
        opts = build_actor_options(config)
        # Outer actor is a coordinator; worker actors claim the GPUs via
        # VLLM_RAY_PER_WORKER_GPUS, same path as tp>1 takes.
        assert opts["num_gpus"] == 0
        assert opts["runtime_env"]["env_vars"]["VLLM_RAY_PER_WORKER_GPUS"] == "1.0"
        assert config.vllm_engine_kwargs.distributed_executor_backend == "ray"

    def test_tp_times_pp_with_mp_backend(self):
        config = ModelshipModelConfig(
            name="test-model",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
            vllm_engine_kwargs=VllmEngineConfig(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
                distributed_executor_backend="mp",
            ),
        )
        opts = build_actor_options(config)
        # mp backend: outer actor owns world_size = tp * pp GPUs directly.
        assert opts["num_gpus"] == 4


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
