"""Tests for the generic state store and the deploy effective-config layer."""

import json

import pytest

from modelship.deploy.config import load_raw_models
from modelship.deploy.effective_config import (
    evict_failed,
    merge,
    read_effective,
    resolve_mode,
    to_config,
    write_effective,
)
from modelship.infer.infer_config import ModelshipModelConfig
from modelship.state.file import FileStateStore


def _model(name: str, **overrides) -> dict:
    """A minimal raw model dict (the form the store holds)."""
    base = {"name": name, "model": f"org/{name}", "usecase": "generate", "loader": "llama_cpp"}
    base.update(overrides)
    return base


class TestFileStateStore:
    def test_set_get_roundtrip(self, tmp_path):
        store = FileStateStore(tmp_path)
        store.set("k", {"a": 1, "b": [1, 2, 3]})
        assert store.get("k") == {"a": 1, "b": [1, 2, 3]}

    def test_missing_key_returns_none(self, tmp_path):
        assert FileStateStore(tmp_path).get("nope") is None

    def test_delete(self, tmp_path):
        store = FileStateStore(tmp_path)
        store.set("k", {"x": 1})
        store.delete("k")
        assert store.get("k") is None
        # idempotent — no error when already absent
        store.delete("k")

    def test_namespaced_key_maps_to_nested_path(self, tmp_path):
        store = FileStateStore(tmp_path)
        store.set("effective/modelship api", {"models": []})
        # spaces slugified, namespace becomes a subdirectory
        assert (tmp_path / "effective" / "modelship-api.json").exists()
        assert store.get("effective/modelship api") == {"models": []}

    def test_corrupt_file_treated_as_missing(self, tmp_path):
        store = FileStateStore(tmp_path)
        (tmp_path / "k.json").write_text("{not valid json")
        assert store.get("k") is None

    def test_write_is_atomic_no_tmp_left_behind(self, tmp_path):
        store = FileStateStore(tmp_path)
        store.set("k", {"x": 1})
        assert not list(tmp_path.glob("*.tmp"))


class TestResolveMode:
    def test_default_is_additive(self):
        assert resolve_mode(reconcile=False, redeploy=False) == "additive"

    def test_reconcile(self):
        assert resolve_mode(reconcile=True, redeploy=False) == "reconcile"

    def test_redeploy_wins(self):
        assert resolve_mode(reconcile=False, redeploy=True) == "redeploy"


class TestMerge:
    def test_additive_union(self):
        merged = merge([_model("a")], [_model("b")], "g", "additive")
        assert [m["name"] for m in merged] == ["a", "b"]

    def test_additive_dedups_identical_config(self):
        # same name + identical config = same fingerprint = idempotent skip
        merged = merge([_model("a")], [_model("a")], "g", "additive")
        assert [m["name"] for m in merged] == ["a"]

    def test_additive_keeps_same_name_different_config(self):
        # same name, different config = distinct deployment (round-robin), kept
        a1 = _model("a", num_cpus=1)
        a2 = _model("a", num_cpus=2)
        merged = merge([a1], [a2], "g", "additive")
        assert len(merged) == 2

    def test_reconcile_replaces(self):
        merged = merge([_model("a"), _model("b")], [_model("c")], "g", "reconcile")
        assert [m["name"] for m in merged] == ["c"]

    def test_redeploy_replaces(self):
        merged = merge([_model("a")], [_model("c")], "g", "redeploy")
        assert [m["name"] for m in merged] == ["c"]


class TestEvictFailed:
    def test_evicts_named_deployment(self):
        a, b = _model("a"), _model("b")
        failed = {ModelshipModelConfig.model_validate(b).deployment_name("g")}
        kept = evict_failed([a, b], "g", failed)
        assert [m["name"] for m in kept] == ["a"]

    def test_no_failures_keeps_all(self):
        models = [_model("a"), _model("b")]
        assert evict_failed(models, "g", set()) == models


class TestReadWriteEffective:
    def test_write_then_read(self, tmp_path):
        store = FileStateStore(tmp_path)
        models = [_model("a"), _model("b")]
        write_effective(store, "modelship api", models)
        assert read_effective(store, "modelship api") == models

    def test_read_absent_gateway_is_empty(self, tmp_path):
        assert read_effective(FileStateStore(tmp_path), "never-deployed") == []


class TestRawRoundTrip:
    """The reason the store holds raw dicts: a normalized vLLM config does NOT
    round-trip (num_gpus=2 -> num_gpus=1.0/tp=2, which fails re-validation). Raw
    dicts reload identically."""

    def test_multi_gpu_vllm_survives_store_roundtrip(self, tmp_path):
        raw = {"name": "x", "model": "org/x", "usecase": "generate", "loader": "vllm", "num_gpus": 2}
        store = FileStateStore(tmp_path)
        write_effective(store, "g", [raw])

        back = read_effective(store, "g")
        cfg = to_config(back)  # must not raise on the normalized-but-reloaded config
        m = cfg.models[0]
        assert m.num_gpus == 1.0
        assert m.vllm_engine_kwargs.tensor_parallel_size == 2
        # identity preserved: same fingerprint as a fresh validate of the original
        assert m.fingerprint("g") == ModelshipModelConfig.model_validate(raw).fingerprint("g")

    def test_stored_file_is_raw_not_normalized(self, tmp_path):
        # The persisted JSON keeps the user's num_gpus=2, not the normalized 1.0.
        raw = {"name": "x", "model": "org/x", "usecase": "generate", "loader": "vllm", "num_gpus": 2}
        store = FileStateStore(tmp_path)
        write_effective(store, "g", [raw])
        on_disk = json.loads((tmp_path / "effective" / "g.json").read_text())
        assert on_disk["models"][0]["num_gpus"] == 2


def _dep(name: str, gw: str = "g", **overrides) -> str:
    return ModelshipModelConfig.model_validate(_model(name, **overrides)).deployment_name(gw)


class TestComputeDeployPlan:
    """Removal must be scoped to the previous effective set, never to everything
    live — otherwise migration over pre-existing models deletes them."""

    def test_migration_keeps_legacy_live_models(self):
        from modelship.deploy.strategy import compute_deploy_plan

        # effective empty (migration); A,B,C live + the gateway app; additive adds D
        desired = to_config([_model("d")])
        existing = {_dep("a"), _dep("b"), _dep("c"), "g"}
        plan = compute_deploy_plan(desired, existing, set(), "g")
        assert plan.apps_to_remove == []  # legacy models untouched
        assert [c.name for c in plan.models_to_add] == ["d"]

    def test_reconcile_removes_dropped_effective_model(self):
        from modelship.deploy.strategy import compute_deploy_plan

        # prev effective managed a,b; new desired (reconcile) keeps only a
        desired = to_config([_model("a")])
        existing = {_dep("a"), _dep("b"), "g"}
        prev = {_dep("a"), _dep("b")}
        plan = compute_deploy_plan(desired, existing, prev, "g")
        assert plan.apps_to_remove == [_dep("b")]
        assert plan.models_to_add == []  # a already live -> skipped

    def test_additive_never_removes(self):
        from modelship.deploy.strategy import compute_deploy_plan

        # effective grew to a,b; a already live, b to add; nothing removed
        desired = to_config([_model("a"), _model("b")])
        existing = {_dep("a"), "g"}
        plan = compute_deploy_plan(desired, existing, {_dep("a")}, "g")
        assert plan.apps_to_remove == []
        assert [c.name for c in plan.models_to_add] == ["b"]

    def test_idempotent_when_all_live(self):
        from modelship.deploy.strategy import compute_deploy_plan

        desired = to_config([_model("a")])
        existing = {_dep("a"), "g"}
        plan = compute_deploy_plan(desired, existing, {_dep("a")}, "g")
        assert plan.models_to_add == []
        assert plan.apps_to_remove == []


class TestCase2AdditiveAccumulation:
    """The bug this design fixes: additive deploys accumulate beyond the last
    input, and that accumulation must survive in the effective config."""

    def test_additive_then_reconcile(self):
        a, b, c, d = _model("a"), _model("b"), _model("c"), _model("d")
        # deploy A,B,C additively
        eff = merge([], [a, b, c], "g", "additive")
        # later additive upgrade declaring only D -> effective keeps all four
        eff = merge(eff, [d], "g", "additive")
        assert sorted(m["name"] for m in eff) == ["a", "b", "c", "d"]
        # a reconcile declaring only D collapses the effective set to D
        eff = merge(eff, [d], "g", "reconcile")
        assert [m["name"] for m in eff] == ["d"]


class TestLoadRawModels:
    def _write(self, tmp_path, text: str) -> str:
        p = tmp_path / "models.yaml"
        p.write_text(text)
        return str(p)

    def test_reads_models_list(self, tmp_path):
        path = self._write(tmp_path, "models:\n  - name: a\n  - name: b\n")
        assert load_raw_models(path) == [{"name": "a"}, {"name": "b"}]

    def test_empty_file_is_empty_list(self, tmp_path):
        # yaml.safe_load("") -> None; `or {}` then `.get` yields no models.
        assert load_raw_models(self._write(tmp_path, "")) == []

    def test_missing_models_key_is_empty_list(self, tmp_path):
        assert load_raw_models(self._write(tmp_path, "other: 1\n")) == []

    def test_top_level_list_rejected(self, tmp_path):
        # A bare list at the top level has no .get(); must raise cleanly, not AttributeError.
        path = self._write(tmp_path, "- name: a\n- name: b\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_raw_models(path)

    def test_top_level_scalar_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="must be a mapping"):
            load_raw_models(self._write(tmp_path, "just a string\n"))

    def test_models_not_a_list_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="'models' must be a list"):
            load_raw_models(self._write(tmp_path, "models:\n  a: 1\n"))
