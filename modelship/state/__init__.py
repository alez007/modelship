"""Generic, pluggable state stores. See base.StateStore.

A store is selected by a connection URI whose **scheme** picks the backend and
whose body carries that backend's connection:

    memory://                     dict shared cluster-wide via a Ray actor (default)
    file:///.cache/state          one JSON file per key under a directory
    redis://[:pw@]host:6379/0      one JSON value per key in Redis (rediss:// = TLS)

One arg covers a full Redis connection, and a new backend is one entry in
``_BUILDERS`` with zero new flags. ``get_state_store()`` reads the configured URI
from ``MSHIP_STATE_STORE``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import ParseResult, urlparse

from modelship.metrics import STATE_STORE_OPERATION_DURATION_SECONDS, STATE_STORE_OPERATIONS_TOTAL
from modelship.state.base import JsonValue, StateStore, StateStoreUnavailableError
from modelship.state.file import FileStateStore
from modelship.state.memory import MemoryStateStore, MemoryStoreActor

__all__ = [
    "FileStateStore",
    "JsonValue",
    "MemoryStateStore",
    "MemoryStoreActor",
    "StateStore",
    "StateStoreUnavailableError",
    "get_state_store",
    "state_store_from_uri",
]

# Env carrying the store URI. Default is in-memory: durable backends (file/redis)
# are opted into explicitly (the chart sets one for k8s).
_STATE_STORE_ENV = "MSHIP_STATE_STORE"
_DEFAULT_URI = "memory://"


class _InstrumentedStateStore(StateStore):
    """Wraps any backend to record per-op latency + ok/error counts by backend, so
    a slow/failing durable store (which silently breaks self-heal) is visible."""

    def __init__(self, inner: StateStore, backend: str) -> None:
        self._inner = inner
        self._backend = backend

    @property
    def inner(self) -> StateStore:
        """The wrapped backend (the concrete store the URI selected)."""
        return self._inner

    def __getattr__(self, name: str):
        # Fires only on a miss (get/set/delete/inner are defined). Delegate any
        # backend-specific attr to the wrapped store; guard _inner so a lookup
        # before __init__ (e.g. unpickling) raises instead of recursing.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def _record(self, op: str, start: float, result: str) -> None:
        # Best-effort: a metrics-agent hiccup must never mask the real op error.
        try:
            STATE_STORE_OPERATION_DURATION_SECONDS.observe(
                time.perf_counter() - start, tags={"backend": self._backend, "op": op}
            )
            STATE_STORE_OPERATIONS_TOTAL.inc(tags={"backend": self._backend, "op": op, "result": result})
        except Exception:
            pass

    def _run(self, op: str, fn):
        start = time.perf_counter()
        result = "ok"
        try:
            return fn()
        except Exception:
            result = "error"
            raise
        finally:
            self._record(op, start, result)

    async def _arun(self, op: str, coro):
        start = time.perf_counter()
        result = "ok"
        try:
            return await coro
        except Exception:
            result = "error"
            raise
        finally:
            self._record(op, start, result)

    def get(self, key: str) -> JsonValue | None:
        return self._run("get", lambda: self._inner.get(key))

    def set(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        self._run("set", lambda: self._inner.set(key, value, ttl_seconds=ttl_seconds))

    def delete(self, key: str) -> None:
        self._run("delete", lambda: self._inner.delete(key))

    def list(self, prefix: str) -> list[str]:
        return self._run("list", lambda: self._inner.list(prefix))

    async def get_async(self, key: str) -> JsonValue | None:
        return await self._arun("get", self._inner.get_async(key))

    async def set_async(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        await self._arun("set", self._inner.set_async(key, value, ttl_seconds=ttl_seconds))

    async def delete_async(self, key: str) -> None:
        await self._arun("delete", self._inner.delete_async(key))

    async def list_async(self, prefix: str) -> list[str]:
        return await self._arun("list", self._inner.list_async(prefix))


def _default_file_dir() -> Path:
    """The historical file-store location: ``$MSHIP_STATE_DIR`` else
    ``$MSHIP_CACHE_DIR/state`` (default ``/.cache/state``). Used when a ``file://``
    URI omits a path."""
    base = os.environ.get("MSHIP_STATE_DIR")
    if not base:
        cache = os.environ.get("MSHIP_CACHE_DIR", "/.cache")
        base = os.path.join(cache, "state")
    return Path(base)


def _build_memory(parsed: ParseResult) -> StateStore:
    # No connection body: memory:// always names the one cluster-wide actor. Any
    # netloc (e.g. memory://foo) is rejected rather than silently ignored.
    if parsed.netloc:
        raise ValueError(f"memory:// URI takes no host/path: {parsed.geturl()!r} parses host {parsed.netloc!r}.")
    return MemoryStateStore()


def _build_file(parsed: ParseResult) -> StateStore:
    # The folder is the URI path; an empty path (``file://``) falls back to the
    # default location, so the historical behaviour is just the no-path form.
    # A non-empty netloc means a two-slash path like ``file://some/dir``, whose
    # first segment urlparse reads as a host — reject it loudly so the directory
    # isn't silently dropped. Absolute paths need three slashes (``file:///dir``).
    if parsed.netloc:
        raise ValueError(
            f"file:// URI must have an empty host: {parsed.geturl()!r} parses host {parsed.netloc!r}. "
            f"Use file:///path/to/state (three slashes) for an absolute path."
        )
    base = Path(parsed.path) if parsed.path else _default_file_dir()
    return FileStateStore(base)


def _build_redis(parsed: ParseResult) -> StateStore:
    # Hand the whole URL back to redis-py (it parses host/port/db/user/password/TLS).
    from modelship.state.redis import RedisStateStore

    return RedisStateStore(parsed.geturl())


# scheme -> builder. Add a backend here (lazy-importing its client) — no CLI change.
_BUILDERS = {
    "memory": _build_memory,
    "file": _build_file,
    "redis": _build_redis,
    "rediss": _build_redis,  # TLS
}


def state_store_from_uri(uri: str) -> StateStore:
    """Construct the StateStore named by *uri* (``scheme://…``)."""
    parsed = urlparse(uri)
    # A bare value like "memory" or "file" (no "://") parses with an empty scheme
    # and the word in .path — treat it as the scheme.
    scheme = parsed.scheme or parsed.path
    builder = _BUILDERS.get(scheme)
    if builder is None:
        raise ValueError(f"unknown state-store scheme {scheme!r}; known: {sorted(_BUILDERS)}")
    return _InstrumentedStateStore(builder(parsed), backend=scheme)


def get_state_store() -> StateStore:
    """The configured default StateStore, from ``MSHIP_STATE_STORE`` (default
    ``memory://``)."""
    return state_store_from_uri(os.environ.get(_STATE_STORE_ENV) or _DEFAULT_URI)
