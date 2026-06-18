"""Redis-backed StateStore — one JSON value per key.

Durable across cluster / head death (the value lives in Redis, not the actor), so
it's the backend for the coordinator registry and effective config in k8s, where
the same Redis also backs Ray's GCS fault tolerance. Selected by the ``redis://``
(or ``rediss://`` for TLS) URI scheme; the URL carries host/port/db/user/password,
parsed natively by ``redis.from_url`` — so a password may be inlined
(``redis://:pw@host``) or injected by the deployment (e.g. a k8s Secret expanded
into the URL).
"""

import json

from modelship.logging import get_logger
from modelship.state.base import JsonValue, StateStore

logger = get_logger("startup")


def _slug(key: str) -> str:
    # Mirror the file backend's key shape: collapse the namespace path into one
    # Redis key under a shared prefix, so backends stay swappable by URI alone.
    return "modelship/state/" + "/".join(p for p in key.split("/") if p)


class RedisStateStore(StateStore):
    def __init__(self, url: str) -> None:
        # Lazy import: redis is only needed when this scheme is actually selected.
        import redis

        # decode_responses so get() returns str, not bytes, for json.loads.
        self._client = redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> JsonValue | None:
        try:
            raw = self._client.get(_slug(key))
        except Exception:
            logger.exception("Redis unreadable for key %s; treating as missing.", key)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.exception("Corrupt JSON at redis key %s; treating as missing.", key)
            return None

    def set(self, key: str, value: JsonValue) -> None:
        self._client.set(_slug(key), json.dumps(value))

    def delete(self, key: str) -> None:
        self._client.delete(_slug(key))
