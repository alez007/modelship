"""The vllm loader rejects GGUF models at driver preflight (0.24 dropped in-tree GGUF)."""

from unittest.mock import patch

import pytest

from modelship.deploy.config import resolve_all_model_sources
from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipConfig,
    ModelshipModelConfig,
    ModelUsecase,
)


def _make_cfg(**overrides) -> ModelshipModelConfig:
    base = {
        "name": "m",
        "model": "some/repo-GGUF:*Q4_K_M.gguf",
        "usecase": ModelUsecase.generate,
        "loader": ModelLoader.vllm,
    }
    base.update(overrides)
    return ModelshipModelConfig(**base)


class TestVllmGgufGuard:
    def test_vllm_gguf_rejected(self):
        cfg = _make_cfg(loader=ModelLoader.vllm)
        with (
            patch(
                "modelship.deploy.config.resolve_model_source",
                return_value="/cache/model-Q4_K_M.gguf",
            ),
            pytest.raises(ValueError, match="GGUF"),
        ):
            resolve_all_model_sources(ModelshipConfig(models=[cfg]))

    def test_llama_cpp_gguf_allowed(self):
        cfg = _make_cfg(loader=ModelLoader.llama_cpp, num_gpus=0)
        with patch(
            "modelship.deploy.config.resolve_model_source",
            return_value="/cache/model-Q4_K_M.gguf",
        ):
            resolve_all_model_sources(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_path == "/cache/model-Q4_K_M.gguf"

    def test_vllm_non_gguf_allowed(self):
        cfg = _make_cfg(loader=ModelLoader.vllm, model="some/fp8-repo")
        with patch(
            "modelship.deploy.config.resolve_model_source",
            return_value="/cache/models--some--fp8-repo/snapshot",
        ):
            resolve_all_model_sources(ModelshipConfig(models=[cfg]))
        assert cfg._resolved_path.endswith("/snapshot")
