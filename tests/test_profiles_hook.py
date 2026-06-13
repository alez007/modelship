"""Tests for the MSHIP_MODEL_STACK / --model-stack hook in resolve_config_path.

Precedence: explicit --config > profile generation > default config/models.yaml.
The profile always regenerates its own ``models_stack_<profile>.yaml`` from scratch,
so switching profiles never requires deleting an old file by hand."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from modelship.deploy.config import resolve_config_path
from modelship.deploy.profiles.budget import DeployBudget

_GIB = 1024**3


def _budget(*, cores: float, ram_gib: float) -> DeployBudget:
    return DeployBudget(cpu_units=cores, gpu_count=0, ram_bytes=int(ram_gib * _GIB), vram_bytes_per_gpu=0)


def _with_budget(budget: DeployBudget):
    return (
        patch("ray.cluster_resources", return_value={"CPU": budget.cpu_units}),
        patch("modelship.deploy.profiles.budget.detect_ram_bytes", return_value=budget.ram_bytes),
        patch("modelship.deploy.profiles.budget.detect_gpus", return_value=[]),
    )


def test_explicit_config_wins_over_env(tmp_path, monkeypatch):
    path = tmp_path / "mine.yaml"
    path.write_text("models: []\n")
    monkeypatch.setenv("MSHIP_MODEL_STACK", "everything")
    # No budget mocks needed — generation must not be attempted when --config is given.
    assert resolve_config_path(str(path), config_dir=tmp_path) == str(path)
    assert path.read_text() == "models: []\n"  # untouched


def test_explicit_config_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_config_path(str(tmp_path / "nope.yaml"), config_dir=tmp_path)


def test_profile_generates_per_profile_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MSHIP_MODEL_STACK", "chat")
    a, b, c = _with_budget(_budget(cores=8, ram_gib=16))
    with a, b, c:
        out = resolve_config_path(None, config_dir=tmp_path)
    expected = tmp_path / "models_stack_chat.yaml"
    assert out == str(expected)
    assert expected.exists()
    assert "MSHIP_MODEL_STACK=chat" in expected.read_text()


def test_profile_regenerates_fresh_each_time(tmp_path, monkeypatch):
    # A stale file from a prior run must be replaced, not preserved.
    monkeypatch.setenv("MSHIP_MODEL_STACK", "chat")
    stale = tmp_path / "models_stack_chat.yaml"
    stale.write_text("models: [stale junk]\n")
    a, b, c = _with_budget(_budget(cores=8, ram_gib=16))
    with a, b, c:
        out = resolve_config_path(None, config_dir=tmp_path)
    text = (tmp_path / "models_stack_chat.yaml").read_text()
    assert out == str(stale)
    assert "stale junk" not in text
    assert "MSHIP_MODEL_STACK=chat" in text


def test_switching_profiles_uses_distinct_files(tmp_path, monkeypatch):
    a, b, c = _with_budget(_budget(cores=8, ram_gib=16))
    with a, b, c:
        monkeypatch.setenv("MSHIP_MODEL_STACK", "chat")
        chat = resolve_config_path(None, config_dir=tmp_path)
        monkeypatch.setenv("MSHIP_MODEL_STACK", "assistant")
        assistant = resolve_config_path(None, config_dir=tmp_path)
    assert chat.endswith("models_stack_chat.yaml")
    assert assistant.endswith("models_stack_assistant.yaml")
    assert (tmp_path / "models_stack_chat.yaml").exists()
    assert (tmp_path / "models_stack_assistant.yaml").exists()


def test_unfittable_profile_exits_clean_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MSHIP_MODEL_STACK", "chat")
    a, b, c = _with_budget(_budget(cores=1, ram_gib=8))
    with a, b, c, pytest.raises(SystemExit) as exc:
        resolve_config_path(None, config_dir=tmp_path)
    assert "chat" in str(exc.value)
    assert not (tmp_path / "models_stack_chat.yaml").exists()  # no partial file


def test_unfittable_profile_removes_stale_prior_file(tmp_path, monkeypatch):
    # On a refusal, a stale file from a prior fitting run must be gone — never deployed.
    monkeypatch.setenv("MSHIP_MODEL_STACK", "chat")
    stale = tmp_path / "models_stack_chat.yaml"
    stale.write_text("models: [old]\n")
    a, b, c = _with_budget(_budget(cores=1, ram_gib=8))
    with a, b, c, pytest.raises(SystemExit):
        resolve_config_path(None, config_dir=tmp_path)
    assert not stale.exists()


def test_no_env_no_default_raises_filenotfound(tmp_path, monkeypatch):
    monkeypatch.delenv("MSHIP_MODEL_STACK", raising=False)
    with pytest.raises(FileNotFoundError):
        resolve_config_path(None, config_dir=tmp_path)


def test_no_env_uses_existing_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MSHIP_MODEL_STACK", raising=False)
    default = tmp_path / "models.yaml"
    default.write_text("models: []\n")
    assert resolve_config_path(None, config_dir=tmp_path) == str(default)


def test_unknown_profile_exits_clean(tmp_path, monkeypatch):
    monkeypatch.setenv("MSHIP_MODEL_STACK", "bogus")
    a, b, c = _with_budget(_budget(cores=8, ram_gib=16))
    with a, b, c, pytest.raises(SystemExit) as exc:
        resolve_config_path(None, config_dir=tmp_path)
    assert "bogus" in str(exc.value)


def test_path_traversal_profile_name_rejected_before_any_fs_op(tmp_path, monkeypatch):
    # `stack` is operator-supplied and goes into a filename + unlink, so a crafted
    # value must be rejected up front — no path built, no file touched.
    monkeypatch.setenv("MSHIP_MODEL_STACK", "../../evil")
    with pytest.raises(SystemExit) as exc:
        resolve_config_path(None, config_dir=tmp_path)
    assert "unknown profile" in str(exc.value)
    assert list(tmp_path.iterdir()) == []  # nothing created or deleted
