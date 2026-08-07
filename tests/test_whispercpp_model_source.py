"""Driver preflight must route bare built-in model names past source resolution
without importing pywhispercpp (a thin coordinator has no pywhispercpp); the name
itself is validated in the actor, against pywhispercpp's own AVAILABLE_MODELS."""

import sys
from unittest.mock import patch

import pytest

from modelship.deploy.config import resolve_all_model_sources
from modelship.infer.infer_config import (
    ModelLoader,
    ModelshipConfig,
    ModelshipModelConfig,
    ModelUsecase,
)
from modelship.infer.model_resolver import PinnedSource
from modelship.infer.whispercpp.whispercpp_infer import WhispercppInfer

_GGML_PIN = PinnedSource(
    resolved_path=None,
    repo="ggerganov/whisper.cpp",
    revision="deadbeef",
    download_filename="ggml-base.en.bin",
    download_patterns=None,
    first_shard=None,
)


def _cfg(model: str) -> ModelshipModelConfig:
    return ModelshipModelConfig(
        name="stt",
        model=model,
        usecase=ModelUsecase.transcription,
        loader=ModelLoader.whispercpp,
        num_gpus=0,
    )


@pytest.fixture
def no_pywhispercpp(monkeypatch):
    # None in sys.modules makes `import pywhispercpp...` raise, as on the thin image.
    monkeypatch.setitem(sys.modules, "pywhispercpp", None)
    monkeypatch.setitem(sys.modules, "pywhispercpp.constants", None)


@pytest.mark.parametrize("model", ["base.en", "large-v3-turbo-q5_0", "bse.en"])
def test_bare_name_skips_source_check_without_pywhispercpp(no_pywhispercpp, model):
    # A typo'd name is skipped too — the actor owns that rejection.
    cfg = _cfg(model)
    resolve_all_model_sources(ModelshipConfig(models=[cfg]))
    assert cfg._pinned_source is None


def test_repo_ref_still_resolves_as_a_source():
    cfg = _cfg("ggerganov/whisper.cpp:ggml-base.en.bin")
    with patch("modelship.deploy.config.check_model_source", return_value=_GGML_PIN) as check:
        resolve_all_model_sources(ModelshipConfig(models=[cfg]))
    check.assert_called_once()
    assert cfg._pinned_source == _GGML_PIN


def test_local_path_still_resolves_as_a_source(tmp_path):
    path = tmp_path / "ggml-base.en.bin"
    path.write_bytes(b"\x00" * 4)
    cfg = _cfg(str(path))
    with patch("modelship.deploy.config.check_model_source", return_value=_GGML_PIN) as check:
        resolve_all_model_sources(ModelshipConfig(models=[cfg]))
    check.assert_called_once()


def test_actor_rejects_unknown_builtin_name():
    pytest.importorskip("pywhispercpp.constants")
    infer = WhispercppInfer(_cfg("bse.en"))
    with pytest.raises(ValueError, match="built-in model names"):
        infer._load()
