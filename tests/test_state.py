"""Tests for the state-store URI factory and the memory/redis backends.

(FileStateStore round-trip lives in test_effective_config.py.)
"""

import pytest

from modelship.state import (
    FileStateStore,
    MemoryStateStore,
    get_state_store,
    state_store_from_uri,
)
from modelship.state.redis import RedisStateStore


class TestMemoryStateStore:
    def test_set_get_roundtrip(self):
        store = MemoryStateStore()
        store.set("a/b", {"x": [1, 2]})
        assert store.get("a/b") == {"x": [1, 2]}

    def test_missing_returns_none(self):
        assert MemoryStateStore().get("nope") is None

    def test_delete(self):
        store = MemoryStateStore()
        store.set("k", {"x": 1})
        store.delete("k")
        assert store.get("k") is None
        store.delete("k")  # idempotent

    def test_isolates_stored_value_from_caller_mutation(self):
        store = MemoryStateStore()
        payload = {"x": 1}
        store.set("k", payload)
        payload["x"] = 999  # mutate the original after storing
        assert store.get("k") == {"x": 1}
        got = store.get("k")
        assert isinstance(got, dict)
        got["x"] = 7  # mutate a returned copy
        assert store.get("k") == {"x": 1}


class TestStateStoreFromUri:
    def test_memory_scheme(self):
        assert isinstance(state_store_from_uri("memory://"), MemoryStateStore)

    def test_bare_value_treated_as_scheme(self):
        assert isinstance(state_store_from_uri("memory"), MemoryStateStore)

    def test_file_scheme_uses_uri_path(self):
        store = state_store_from_uri("file:///tmp/mship-state-test")
        assert isinstance(store, FileStateStore)
        assert str(store.base_dir) == "/tmp/mship-state-test"

    def test_file_scheme_empty_path_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_DIR", "/var/lib/mship/state")
        store = state_store_from_uri("file://")
        assert isinstance(store, FileStateStore)
        assert str(store.base_dir) == "/var/lib/mship/state"

    def test_redis_scheme_builds_redis_store(self, monkeypatch):
        # Don't hit a real server: stub redis.from_url.
        import redis

        monkeypatch.setattr(redis, "from_url", lambda *a, **k: object())
        assert isinstance(state_store_from_uri("redis://cache:6379/0"), RedisStateStore)

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="unknown state-store scheme"):
            state_store_from_uri("postgres://host/db")

    def test_get_state_store_defaults_to_memory(self, monkeypatch):
        monkeypatch.delenv("MSHIP_STATE_STORE", raising=False)
        assert isinstance(get_state_store(), MemoryStateStore)

    def test_get_state_store_reads_env(self, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_STORE", "file:///tmp/mship-env-state")
        store = get_state_store()
        assert isinstance(store, FileStateStore)
        assert str(store.base_dir) == "/tmp/mship-env-state"


class TestRedisStateStore:
    @pytest.fixture
    def store(self):
        fakeredis = pytest.importorskip("fakeredis")

        s = RedisStateStore.__new__(RedisStateStore)
        s._client = fakeredis.FakeRedis(decode_responses=True)
        return s

    def test_set_get_roundtrip(self, store):
        store.set("gw/models", {"app-1": "qwen", "list": [1, 2]})
        assert store.get("gw/models") == {"app-1": "qwen", "list": [1, 2]}

    def test_missing_returns_none(self, store):
        assert store.get("absent") is None

    def test_delete(self, store):
        store.set("k", {"x": 1})
        store.delete("k")
        assert store.get("k") is None

    def test_corrupt_value_treated_as_missing(self, store):
        store._client.set("modelship/state/k", "not json{")
        assert store.get("k") is None
