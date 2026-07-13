"""Tests for the state-store URI factory and the memory/file/redis backends —
sync and async paths, TTL, prefix listing, and availability-vs-absent semantics.
"""

import json

import pytest

from modelship.state import (
    FileStateStore,
    MemoryStateStore,
    StateStoreUnavailableError,
    get_state_store,
    state_store_from_uri,
)
from modelship.state.redis import RedisStateStore


def _fake_redis_store():
    """A RedisStateStore whose sync + async clients share one fakeredis server, so
    a value written via one path is visible from the other."""
    fakeredis = pytest.importorskip("fakeredis")
    server = fakeredis.FakeServer()
    s = RedisStateStore.__new__(RedisStateStore)
    s._url = "redis://fake"
    s._sync_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    s._async_client = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
    return s


@pytest.fixture(params=["memory", "file", "redis"])
def store(request, tmp_path):
    if request.param == "memory":
        return MemoryStateStore()
    if request.param == "file":
        return FileStateStore(tmp_path)
    return _fake_redis_store()


class TestBackends:
    """Behaviour shared by every backend (parametrized via the `store` fixture)."""

    def test_set_get_roundtrip(self, store):
        store.set("a/b", {"x": [1, 2]})
        assert store.get("a/b") == {"x": [1, 2]}

    def test_missing_returns_none(self, store):
        assert store.get("nope") is None

    def test_delete(self, store):
        store.set("k", {"x": 1})
        store.delete("k")
        assert store.get("k") is None
        store.delete("k")  # idempotent

    def test_list_prefix(self, store):
        store.set("responses/u1/a", {"n": 1})
        store.set("responses/u1/b", {"n": 2})
        store.set("responses/u2/c", {"n": 3})
        store.set("other/x", {"n": 4})
        assert sorted(store.list("responses/u1")) == ["responses/u1/a", "responses/u1/b"]
        assert sorted(store.list("responses")) == [
            "responses/u1/a",
            "responses/u1/b",
            "responses/u2/c",
        ]

    @pytest.mark.asyncio
    async def test_async_mirrors_sync(self, store):
        await store.set_async("k", {"x": 1})
        assert await store.get_async("k") == {"x": 1}
        assert await store.list_async("k") == ["k"]
        # cross-path visibility: a sync write is seen by an async read
        store.set("k2", {"y": 2})
        assert await store.get_async("k2") == {"y": 2}
        await store.delete_async("k")
        assert await store.get_async("k") is None


class TestMemoryStateStore:
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

    def test_ttl_expires(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr("time.time", lambda: clock["t"])
        store = MemoryStateStore()
        store.set("k", {"x": 1}, ttl_seconds=10)
        assert store.get("k") == {"x": 1}
        clock["t"] = 1011.0
        assert store.get("k") is None
        assert store.list("k") == []


class TestFileStateStore:
    def test_ttl_expires(self, tmp_path, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr("time.time", lambda: clock["t"])
        store = FileStateStore(tmp_path)
        store.set("k", {"x": 1}, ttl_seconds=10)
        assert store.get("k") == {"x": 1}
        clock["t"] = 1011.0
        assert store.get("k") is None

    def test_reads_legacy_raw_value(self, tmp_path):
        # A pre-envelope file (raw value, no marker) still reads back as the value.
        store = FileStateStore(tmp_path)
        (tmp_path / "k.json").write_text(json.dumps({"models": ["a"]}))
        assert store.get("k") == {"models": ["a"]}

    def test_unreadable_existing_file_raises_unavailable(self, tmp_path):
        # A directory where the value file should be makes read_text raise OSError
        # (not FileNotFoundError) — an availability failure, not a missing key.
        store = FileStateStore(tmp_path)
        (tmp_path / "k.json").mkdir()
        with pytest.raises(StateStoreUnavailableError):
            store.get("k")


class TestRedisStateStore:
    def test_ttl_sets_native_expiry(self):
        store = _fake_redis_store()
        store.set("k", {"x": 1}, ttl_seconds=100)
        assert store._sync_client.pttl("modelship/state/k") > 0
        store.set("k2", {"x": 1})  # no ttl
        assert store._sync_client.pttl("modelship/state/k2") == -1  # no expiry

    def test_corrupt_value_treated_as_missing(self):
        store = _fake_redis_store()
        store._sync_client.set("modelship/state/k", "not json{")
        assert store.get("k") is None

    def test_connection_error_raises_unavailable(self):
        from redis.exceptions import ConnectionError as RedisConnectionError

        class Boom:
            def get(self, *a, **k):
                raise RedisConnectionError("down")

        store = RedisStateStore.__new__(RedisStateStore)
        store._url = "redis://fake"
        store._sync_client = Boom()
        store._async_client = None
        with pytest.raises(StateStoreUnavailableError):
            store.get("k")


class TestStateStoreFromUri:
    # state_store_from_uri wraps the chosen backend in an instrumented proxy; the
    # concrete store the URI selected is exposed as `.inner`.
    def test_memory_scheme(self):
        assert isinstance(state_store_from_uri("memory://").inner, MemoryStateStore)

    def test_bare_value_treated_as_scheme(self):
        assert isinstance(state_store_from_uri("memory").inner, MemoryStateStore)

    def test_file_scheme_uses_uri_path(self):
        store = state_store_from_uri("file:///tmp/mship-state-test").inner
        assert isinstance(store, FileStateStore)
        assert str(store.base_dir) == "/tmp/mship-state-test"

    def test_file_scheme_empty_path_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_DIR", "/var/lib/mship/state")
        store = state_store_from_uri("file://").inner
        assert isinstance(store, FileStateStore)
        assert str(store.base_dir) == "/var/lib/mship/state"

    def test_file_scheme_two_slash_path_rejected(self):
        # file://some/dir is malformed: urlparse reads "some" as the host and
        # would silently drop it. Must raise, not fall back to the default dir.
        with pytest.raises(ValueError, match="must have an empty host"):
            state_store_from_uri("file://some/dir")

    def test_redis_scheme_builds_redis_store(self):
        # Construction is inert (clients are lazy) — no server contacted.
        assert isinstance(state_store_from_uri("redis://cache:6379/0").inner, RedisStateStore)

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="unknown state-store scheme"):
            state_store_from_uri("postgres://host/db")

    def test_get_state_store_defaults_to_memory(self, monkeypatch):
        monkeypatch.delenv("MSHIP_STATE_STORE", raising=False)
        assert isinstance(get_state_store().inner, MemoryStateStore)

    def test_get_state_store_reads_env(self, monkeypatch):
        monkeypatch.setenv("MSHIP_STATE_STORE", "file:///tmp/mship-env-state")
        store = get_state_store().inner
        assert isinstance(store, FileStateStore)
        assert str(store.base_dir) == "/tmp/mship-env-state"
