"""Tests for llama_cpp GPU offload gating in LlamaCppInfer.__init__.

The guard runs before any `Llama(...)` construction in `start()`, so these
tests build the config only and never touch a real GGUF.
"""

from __future__ import annotations

from unittest.mock import patch

from modelship.infer.infer_config import LlamaCppConfig, ModelLoader, ModelshipModelConfig, ModelUsecase
from modelship.infer.llama_cpp.llama_cpp_infer import LlamaCppInfer


def _make_config(*, num_gpus: float, n_gpu_layers: int) -> ModelshipModelConfig:
    return ModelshipModelConfig(
        name="test-model",
        model="org/test-model",
        usecase=ModelUsecase.generate,
        loader=ModelLoader.llama_cpp,
        num_gpus=num_gpus,
        llama_cpp_config=LlamaCppConfig(n_gpu_layers=n_gpu_layers),
    )


class TestLlamaCppGpuOffloadGating:
    def test_honors_n_gpu_layers_when_gpu_capable_and_assigned(self):
        with patch("modelship.infer.llama_cpp.llama_cpp_infer.llama_supports_gpu_offload", return_value=True):
            infer = LlamaCppInfer(_make_config(num_gpus=1, n_gpu_layers=-1))
        assert infer._n_gpu_layers == -1

    def test_forces_cpu_when_wheel_lacks_gpu_support(self):
        with patch("modelship.infer.llama_cpp.llama_cpp_infer.llama_supports_gpu_offload", return_value=False):
            infer = LlamaCppInfer(_make_config(num_gpus=1, n_gpu_layers=-1))
        assert infer._n_gpu_layers == 0

    def test_forces_cpu_when_no_gpu_assigned_even_if_wheel_is_capable(self):
        with patch("modelship.infer.llama_cpp.llama_cpp_infer.llama_supports_gpu_offload", return_value=True):
            infer = LlamaCppInfer(_make_config(num_gpus=0, n_gpu_layers=-1))
        assert infer._n_gpu_layers == 0
