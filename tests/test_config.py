"""Tests for Modelship model configuration parsing and validation."""

import pytest
from pydantic import ValidationError

from modelship.infer.infer_config import (
    AutoscalingConfig,
    LlamaServerConfig,
    ModelLoader,
    ModelshipConfig,
    ModelshipModelConfig,
    ModelUsecase,
    TransformersConfig,
    VllmEngineConfig,
)


class TestLlamaServerConfig:
    def test_defaults(self):
        config = LlamaServerConfig()
        assert config.n_ctx == 2048
        assert config.n_batch == 512
        assert config.n_gpu_layers == -1
        assert config.parallel == 1
        assert config.chat_template is None
        assert config.extra_args == []

    def test_custom_values(self):
        config = LlamaServerConfig(
            n_ctx=4096,
            n_batch=1024,
            n_gpu_layers=33,
            parallel=4,
            chat_template="chatml",
            extra_args=["--flash-attn"],
        )
        assert config.n_ctx == 4096
        assert config.n_batch == 1024
        assert config.n_gpu_layers == 33
        assert config.parallel == 4
        assert config.chat_template == "chatml"
        assert config.extra_args == ["--flash-attn"]

    def _num_gpus_model(self, num_gpus: float) -> ModelshipModelConfig:
        return ModelshipModelConfig(
            name="test-model",
            model="repo/Qwen-GGUF:*Q4_K_M.gguf",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
            num_gpus=num_gpus,
        )

    def test_num_gpus_integer_allowed(self):
        config = self._num_gpus_model(1)
        assert config.num_gpus == 1

    def test_num_gpus_zero_allowed(self):
        config = self._num_gpus_model(0)
        assert config.num_gpus == 0

    def test_num_gpus_fractional_rejected(self):
        with pytest.raises(ValidationError, match="not allowed for the llama_server loader"):
            self._num_gpus_model(0.5)

    def test_llama_server_model_config(self):
        config = ModelshipModelConfig(
            name="llama-3",
            model="meta-llama/Llama-3-8B-Instruct-GGUF:*Q4_K_M.gguf",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.llama_server,
            llama_server_config=LlamaServerConfig(parallel=4),
        )
        assert config.loader == ModelLoader.llama_server
        assert config.llama_server_config is not None
        assert config.llama_server_config.parallel == 4


class TestModelshipModelConfig:
    def test_minimal_vllm_model(self):
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-org/some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
        )
        assert config.name == "test-llm"
        assert config.loader == ModelLoader.vllm
        assert config.num_gpus == 0
        assert config.num_cpus == 0.1
        assert config.chat_template_kwargs == {}

    def test_chat_template_kwargs_round_trips(self):
        config = ModelshipModelConfig.model_validate(
            {
                "name": "qwen3",
                "model": "some-org/qwen3",
                "usecase": ModelUsecase.generate,
                "loader": ModelLoader.llama_server,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        assert config.chat_template_kwargs == {"enable_thinking": False}

    def test_custom_loader_requires_plugin(self):
        with pytest.raises(ValidationError, match="loader='custom' requires plugin"):
            ModelshipModelConfig(
                name="test-tts",
                model="some-model",
                usecase=ModelUsecase.tts,
                loader=ModelLoader.custom,
            )

    def test_custom_loader_with_plugin(self):
        config = ModelshipModelConfig(
            name="test-tts",
            model="some-model",
            usecase=ModelUsecase.tts,
            loader=ModelLoader.custom,
            plugin="kokoroonnx",
        )
        assert config.plugin == "kokoroonnx"

    def test_custom_loader_plugin_only(self):
        config = ModelshipModelConfig(
            name="test-tts",
            model="some-model",
            usecase=ModelUsecase.tts,
            loader=ModelLoader.custom,
            plugin="kokoroonnx",
        )
        assert config.plugin == "kokoroonnx"

    def test_model_required(self):
        with pytest.raises(ValidationError, match="`model:` is required for loader"):
            ModelshipModelConfig(
                name="test-llm",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
            )

    def test_loader_required(self):
        with pytest.raises(ValidationError, match="Field required"):
            ModelshipModelConfig(
                name="test-llm",
                model="some-model",
                usecase=ModelUsecase.generate,
            )

    def test_diffusers_usecase_defaults_to_image(self):
        config = ModelshipModelConfig(
            name="test-image",
            model="stabilityai/sdxl-turbo",
            loader=ModelLoader.diffusers,
        )
        assert config.usecase is ModelUsecase.image

    def test_diffusers_explicit_image_usecase_ok(self):
        config = ModelshipModelConfig(
            name="test-image",
            model="stabilityai/sdxl-turbo",
            usecase=ModelUsecase.image,
            loader=ModelLoader.diffusers,
        )
        assert config.usecase is ModelUsecase.image

    def test_diffusers_rejects_non_image_usecase(self):
        with pytest.raises(ValidationError, match="loader='diffusers' only supports usecase='image'"):
            ModelshipModelConfig(
                name="test-image",
                model="stabilityai/sdxl-turbo",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.diffusers,
            )

    def test_stable_diffusion_cpp_usecase_defaults_to_image(self):
        config = ModelshipModelConfig(
            name="test-image",
            model="org/sd-gguf:*.gguf",
            loader=ModelLoader.stable_diffusion_cpp,
        )
        assert config.usecase is ModelUsecase.image

    def test_stable_diffusion_cpp_rejects_non_image_usecase(self):
        with pytest.raises(ValidationError, match="loader='stable_diffusion_cpp' only supports usecase='image'"):
            ModelshipModelConfig(
                name="test-image",
                model="org/sd-gguf:*.gguf",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.stable_diffusion_cpp,
            )

    def test_stable_diffusion_cpp_requires_model(self):
        with pytest.raises(ValidationError, match="`model:` is required"):
            ModelshipModelConfig(
                name="test-image",
                usecase=ModelUsecase.image,
                loader=ModelLoader.stable_diffusion_cpp,
            )

    def test_gpu_allocation_fraction(self):
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=0.70,
        )
        assert config.num_gpus == 0.70

    def test_fractional_num_gpus_sets_gpu_memory_utilization(self):
        # A fractional num_gpus is the single source of truth for the VRAM share:
        # it must land on gpu_memory_utilization so the preflight + engine agree.
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=0.5,
        )
        assert config.vllm_engine_kwargs.gpu_memory_utilization == 0.5
        # and it's marked set, so it survives model_dump(exclude_unset=True)
        assert "gpu_memory_utilization" in config.vllm_engine_kwargs.model_fields_set

    def test_explicit_gpu_memory_utilization_wins_over_num_gpus(self):
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=0.5,
            vllm_engine_kwargs={"gpu_memory_utilization": 0.6},
        )
        assert config.vllm_engine_kwargs.gpu_memory_utilization == 0.6

    def test_whole_gpu_leaves_gpu_memory_utilization_default(self):
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=1,
        )
        assert config.vllm_engine_kwargs.gpu_memory_utilization == 0.9

    def test_cpu_num_gpus_lowers_gpu_memory_utilization_default(self):
        # On vLLM's CPU backend, gpu_memory_utilization means "fraction of host
        # RAM to reserve," not VRAM — the GPU-oriented 0.9 default reserves 90%
        # of node RAM and reliably raises at worker init on a real machine.
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=0,
        )
        assert config.vllm_engine_kwargs.gpu_memory_utilization == 0.4

    def test_explicit_gpu_memory_utilization_wins_over_cpu_default(self):
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=0,
            vllm_engine_kwargs={"gpu_memory_utilization": 0.6},
        )
        assert config.vllm_engine_kwargs.gpu_memory_utilization == 0.6

    def test_num_gpus_integer_required_above_one(self):
        with pytest.raises(ValidationError, match="must be integers"):
            ModelshipModelConfig(
                name="test-llm",
                model="some-model",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
                num_gpus=1.5,
            )

    def test_num_gpus_auto_derives_tp(self):
        # num_gpus=3 with default tp/pp -> tp becomes 3, num_gpus normalizes to per-slot share.
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=3,
        )
        assert config.vllm_engine_kwargs.tensor_parallel_size == 3
        assert config.num_gpus == 1.0

    def test_explicit_tp_matching_num_gpus_accepted(self):
        config = ModelshipModelConfig(
            name="test-llm",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=4,
            vllm_engine_kwargs=VllmEngineConfig(tensor_parallel_size=2, pipeline_parallel_size=2),
        )
        assert config.vllm_engine_kwargs.tensor_parallel_size == 2
        assert config.vllm_engine_kwargs.pipeline_parallel_size == 2
        assert config.num_gpus == 1.0

    def test_explicit_tp_inconsistent_with_num_gpus_rejected(self):
        with pytest.raises(ValidationError, match="does not match tensor_parallel_size"):
            ModelshipModelConfig(
                name="test-llm",
                model="some-model",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
                num_gpus=2,
                vllm_engine_kwargs=VllmEngineConfig(tensor_parallel_size=3),
            )

    def test_fractional_num_gpus_with_tp_rejected(self):
        with pytest.raises(ValidationError, match=r"fractional.*not compatible.*tensor_parallel"):
            ModelshipModelConfig(
                name="test-llm",
                model="some-model",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
                num_gpus=0.3,
                vllm_engine_kwargs=VllmEngineConfig(tensor_parallel_size=2),
            )

    def test_fractional_num_gpus_with_pp_rejected(self):
        with pytest.raises(ValidationError, match=r"fractional.*not compatible.*tensor_parallel"):
            ModelshipModelConfig(
                name="test-llm",
                model="some-model",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
                num_gpus=0.5,
                vllm_engine_kwargs=VllmEngineConfig(pipeline_parallel_size=2),
            )

    def test_num_gpus_redundant_with_tp_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="config"):
            ModelshipModelConfig(
                name="test-llm",
                model="some-model",
                usecase=ModelUsecase.generate,
                loader=ModelLoader.vllm,
                num_gpus=2,
                vllm_engine_kwargs=VllmEngineConfig(tensor_parallel_size=2),
            )
        assert any("redundant" in rec.message for rec in caplog.records)

    def test_non_vllm_loader_skips_tp_derivation(self):
        # transformers has no parallelism config; num_gpus stays as-is for the
        # loader to interpret directly (whole GPUs are fine for that path).
        config = ModelshipModelConfig(
            name="test-tts",
            model="some-model",
            usecase=ModelUsecase.tts,
            loader=ModelLoader.custom,
            plugin="myplugin",
            num_gpus=2,
        )
        assert config.num_gpus == 2

    def test_all_usecases_valid(self):
        for usecase in ModelUsecase:
            config = ModelshipModelConfig(
                name=f"test-{usecase.value}",
                model="some-model",
                usecase=usecase,
                loader=ModelLoader.vllm,
            )
            assert config.usecase == usecase

    def test_all_loaders_valid(self):
        image_only = (ModelLoader.diffusers, ModelLoader.stable_diffusion_cpp)
        for loader in ModelLoader:
            # diffusers / stable_diffusion_cpp are image-only; the rest support generate.
            usecase = ModelUsecase.image if loader in image_only else ModelUsecase.generate
            kwargs = {"name": "test", "model": "some-model", "usecase": usecase}
            if loader == ModelLoader.custom:
                kwargs["plugin"] = "test-plugin"
            config = ModelshipModelConfig(loader=loader, **kwargs)
            assert config.loader == loader


class TestVllmEngineConfig:
    def test_defaults(self):
        config = VllmEngineConfig()
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1
        assert config.dtype == "auto"
        assert config.gpu_memory_utilization == 0.9
        assert config.trust_remote_code is False

    def test_custom_values(self):
        config = VllmEngineConfig(
            tensor_parallel_size=2,
            max_model_len=12288,
            enable_auto_tool_choice=True,
            tool_call_parser="llama3_json",
        )
        assert config.tensor_parallel_size == 2
        assert config.max_model_len == 12288
        assert config.enable_auto_tool_choice is True
        assert config.tool_call_parser == "llama3_json"


class TestModelshipConfig:
    def test_multi_model_config(self):
        config = ModelshipConfig(
            models=[
                ModelshipModelConfig(
                    name="llm",
                    model="some-org/some-llm",
                    usecase=ModelUsecase.generate,
                    loader=ModelLoader.vllm,
                    num_gpus=0.70,
                ),
                ModelshipModelConfig(
                    name="tts",
                    model="some-model",
                    usecase=ModelUsecase.tts,
                    loader=ModelLoader.custom,
                    plugin="kokoroonnx",
                    num_gpus=0.05,
                ),
            ]
        )
        assert len(config.models) == 2
        assert config.models[0].name == "llm"
        assert config.models[1].name == "tts"

    def test_empty_models_list(self):
        config = ModelshipConfig(models=[])
        assert len(config.models) == 0

    def test_duplicate_names_allowed(self):
        config = ModelshipConfig(
            models=[
                ModelshipModelConfig(
                    name="kokoro",
                    model="hexgrad/Kokoro-82M",
                    usecase=ModelUsecase.tts,
                    loader=ModelLoader.custom,
                    plugin="kokoroonnx",
                    num_gpus=0.07,
                ),
                ModelshipModelConfig(
                    name="kokoro",
                    model="hexgrad/Kokoro-82M",
                    usecase=ModelUsecase.tts,
                    loader=ModelLoader.custom,
                    plugin="kokoroonnx",
                    num_gpus=0,
                ),
            ]
        )
        assert len(config.models) == 2
        assert config.models[0].name == config.models[1].name == "kokoro"

    def test_duplicate_name_and_fingerprint_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate model entries"):
            ModelshipConfig(
                models=[
                    ModelshipModelConfig(
                        name="qwen",
                        model="Qwen/Qwen-7B",
                        usecase=ModelUsecase.generate,
                        loader=ModelLoader.vllm,
                        num_gpus=0.5,
                    ),
                    ModelshipModelConfig(
                        name="qwen",
                        model="Qwen/Qwen-7B",
                        usecase=ModelUsecase.generate,
                        loader=ModelLoader.vllm,
                        num_gpus=0.5,
                    ),
                ]
            )


class TestFingerprint:
    def _cfg(self, **overrides):
        base = dict(
            name="qwen",
            model="Qwen/Qwen-7B",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_gpus=0.5,
        )
        base.update(overrides)
        return ModelshipModelConfig(**base)

    def test_stable_across_instances(self):
        assert self._cfg().fingerprint() == self._cfg().fingerprint()

    def test_changes_when_num_gpus_differs(self):
        assert self._cfg(num_gpus=0.7).fingerprint() != self._cfg(num_gpus=0.8).fingerprint()

    def test_unaffected_by_name(self):
        # Same config under a different name should fingerprint identically;
        # the name is the deployment-name prefix, not part of the hash.
        assert self._cfg(name="a").fingerprint() == self._cfg(name="b").fingerprint()

    def test_unaffected_by_num_replicas(self):
        # Replica count is a Ray Serve in-place rebind, not a config drift.
        assert self._cfg(num_replicas=1).fingerprint() == self._cfg(num_replicas=4).fingerprint()

    def test_changes_when_loader_differs(self):
        assert (
            self._cfg(loader=ModelLoader.vllm).fingerprint() != self._cfg(loader=ModelLoader.transformers).fingerprint()
        )

    def test_deployment_name_combines_name_and_fingerprint(self):
        cfg = self._cfg()
        assert cfg.deployment_name("gw") == f"{cfg.name}-{cfg.fingerprint('gw')}"
        assert len(cfg.fingerprint()) == 10

    def test_fingerprint_distinct_per_gateway(self):
        # Same config under different gateways must yield different app names so
        # they don't collide in Serve's flat global namespace.
        cfg = self._cfg()
        assert cfg.fingerprint("gw-a") != cfg.fingerprint("gw-b")
        assert cfg.deployment_name("gw-a") != cfg.deployment_name("gw-b")
        # No gateway == the gateway-independent config hash.
        assert cfg.fingerprint() == cfg.fingerprint("")


class TestTransformersConfig:
    def test_defaults(self):
        config = TransformersConfig()
        assert config.device == "cpu"
        assert config.torch_dtype == "auto"
        assert config.trust_remote_code is False
        assert config.model_kwargs == {}
        assert config.pipeline_kwargs == {}

    def test_custom_values(self):
        config = TransformersConfig(
            device="cuda:0",
            torch_dtype="float16",
            trust_remote_code=True,
            model_kwargs={"attn_implementation": "flash_attention_2"},
        )
        assert config.device == "cuda:0"
        assert config.torch_dtype == "float16"
        assert config.trust_remote_code is True
        assert config.model_kwargs == {"attn_implementation": "flash_attention_2"}

    def test_transformers_generate_model(self):
        config = ModelshipModelConfig(
            name="llm-cpu",
            model="meta-llama/Llama-3.2-1B-Instruct",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.transformers,
            num_cpus=4,
            transformers_config=TransformersConfig(torch_dtype="float32"),
        )
        assert config.loader == ModelLoader.transformers
        assert config.usecase == ModelUsecase.generate
        assert config.transformers_config.torch_dtype == "float32"

    def test_transformers_embed_model(self):
        config = ModelshipModelConfig(
            name="embed",
            model="nomic-ai/nomic-embed-text-v1.5",
            usecase=ModelUsecase.embed,
            loader=ModelLoader.transformers,
            num_cpus=2,
            transformers_config=TransformersConfig(trust_remote_code=True),
        )
        assert config.usecase == ModelUsecase.embed
        assert config.transformers_config.trust_remote_code is True

    def test_transformers_transcription_model(self):
        config = ModelshipModelConfig(
            name="whisper-cpu",
            model="openai/whisper-base",
            usecase=ModelUsecase.transcription,
            loader=ModelLoader.transformers,
            num_cpus=2,
        )
        assert config.usecase == ModelUsecase.transcription
        assert config.transformers_config is None

    def test_transformers_tts_model(self):
        config = ModelshipModelConfig(
            name="tts",
            model="microsoft/speecht5_tts",
            usecase=ModelUsecase.tts,
            loader=ModelLoader.transformers,
            num_cpus=1,
        )
        assert config.usecase == ModelUsecase.tts

    def test_transformers_config_not_required(self):
        config = ModelshipModelConfig(
            name="test",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.transformers,
        )
        assert config.transformers_config is None


class TestNumReplicas:
    def test_default_num_replicas(self):
        config = ModelshipModelConfig(
            name="test",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
        )
        assert config.num_replicas == 1

    def test_custom_num_replicas(self):
        config = ModelshipModelConfig(
            name="test",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
            num_replicas=3,
        )
        assert config.num_replicas == 3


class TestAutoscalingConfig:
    def _model(self, **overrides):
        base = dict(
            name="test",
            model="some-model",
            usecase=ModelUsecase.generate,
            loader=ModelLoader.vllm,
        )
        base.update(overrides)
        return ModelshipModelConfig(**base)

    def test_default_is_none(self):
        assert self._model().autoscaling_config is None

    def test_to_serve_dict_omits_unset_tunables(self):
        cfg = AutoscalingConfig(min_replicas=1, max_replicas=4)
        assert cfg.to_serve_dict() == {"min_replicas": 1, "max_replicas": 4}

    def test_to_serve_dict_includes_set_tunables(self):
        cfg = AutoscalingConfig(
            min_replicas=0,
            max_replicas=8,
            initial_replicas=2,
            target_ongoing_requests=5,
            upscale_delay_s=10,
            downscale_delay_s=600,
        )
        assert cfg.to_serve_dict() == {
            "min_replicas": 0,
            "max_replicas": 8,
            "initial_replicas": 2,
            "target_ongoing_requests": 5,
            "upscale_delay_s": 10,
            "downscale_delay_s": 600,
        }

    def test_scale_to_zero_allowed(self):
        cfg = AutoscalingConfig(min_replicas=0, max_replicas=3)
        assert cfg.min_replicas == 0

    def test_max_below_min_rejected(self):
        with pytest.raises(ValidationError, match=r"max_replicas .* must be >= "):
            AutoscalingConfig(min_replicas=4, max_replicas=2)

    def test_initial_outside_bounds_rejected(self):
        with pytest.raises(ValidationError, match=r"initial_replicas .* must be within"):
            AutoscalingConfig(min_replicas=1, max_replicas=4, initial_replicas=9)

    def test_negative_min_rejected(self):
        with pytest.raises(ValidationError):
            AutoscalingConfig(min_replicas=-1, max_replicas=4)

    def test_accepted_on_model(self):
        config = self._model(autoscaling_config={"min_replicas": 1, "max_replicas": 5})
        assert config.autoscaling_config is not None
        assert config.autoscaling_config.max_replicas == 5

    def test_explicit_num_replicas_with_autoscaling_rejected(self):
        with pytest.raises(ValidationError, match="either num_replicas or autoscaling_config"):
            self._model(num_replicas=2, autoscaling_config={"min_replicas": 1, "max_replicas": 4})

    def test_default_num_replicas_with_autoscaling_allowed(self):
        # An untouched num_replicas default must not trip the mutual-exclusivity check.
        config = self._model(autoscaling_config={"min_replicas": 1, "max_replicas": 4})
        assert config.autoscaling_config is not None

    def test_excluded_from_fingerprint(self):
        # Changing scaling bounds is an in-place Serve rebind, not config drift.
        a = self._model(autoscaling_config={"min_replicas": 1, "max_replicas": 2})
        b = self._model(autoscaling_config={"min_replicas": 3, "max_replicas": 9})
        plain = self._model()
        assert a.fingerprint() == b.fingerprint() == plain.fingerprint()
