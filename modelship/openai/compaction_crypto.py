"""Symmetric encryption for ``/v1/responses/compact``'s ``encrypted_content`` blobs.

Real Fernet encryption keeps the blob opaque to holders and detects tampering;
its format isn't otherwise spec'd (see the compaction plan).

The key is resolved once by the deploy driver (``ensure_key_seeded``) and
stored in the shared ``StateStore``, so every gateway/model-actor process reads
the same key regardless of node. ``_resolve_key`` never generates one on the
fly — callers outside the deploy flow (tests, scripts) must call
``ensure_key_seeded`` first.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from cryptography.fernet import Fernet, InvalidToken

from modelship.logging import get_logger

if TYPE_CHECKING:
    from modelship.state import StateStore

logger = get_logger("compaction_crypto")

_KEY_ENV = "MSHIP_COMPACTION_KEY"
_STATE_KEY = "compaction/key"

_cached_key: bytes | None = None

__all__ = ["InvalidToken", "decrypt_items", "encrypt_items", "ensure_key_seeded"]


def _validate(raw: str) -> bytes:
    key = raw.encode("ascii")
    try:
        Fernet(key)  # validate eagerly so a bad key fails fast, not mid-request
    except ValueError as e:
        raise ValueError(f"{_KEY_ENV} is not a valid Fernet key: {e}") from e
    return key


def ensure_key_seeded(store: StateStore) -> None:
    """Resolve ``MSHIP_COMPACTION_KEY`` (or generate one) and persist it to *store*,
    so every process reads the same key. No-op if the store already has one.
    """
    if store.get(_STATE_KEY) is not None:
        return
    configured = os.environ.get(_KEY_ENV)
    key = _validate(configured) if configured else Fernet.generate_key()
    store.set(_STATE_KEY, {"key": key.decode("ascii")})


def _resolve_key() -> bytes:
    global _cached_key
    if _cached_key is not None:
        return _cached_key

    from modelship.state import get_state_store

    stored = get_state_store().get(_STATE_KEY)
    raw = stored.get("key") if isinstance(stored, dict) else None
    if not isinstance(raw, str):
        # Caller outside the deploy flow (test, script) must call ensure_key_seeded() first.
        raise RuntimeError("no compaction key found in the state store")
    key = raw.encode("ascii")
    _cached_key = key
    return key


def encrypt_items(items: list[Any]) -> str:
    """Encrypt *items* (a list of Responses input items) into an opaque string."""
    fernet = Fernet(_resolve_key())
    plaintext = json.dumps(items).encode("utf-8")
    return fernet.encrypt(plaintext).decode("ascii")


def decrypt_items(blob: str) -> list[Any]:
    """Inverse of :func:`encrypt_items`.

    Raises :class:`InvalidToken` for a tampered blob, a blob encrypted under a
    different key, non-ASCII content, or malformed JSON — callers must map all of
    these to the same clean 400 without revealing which it was.
    """
    fernet = Fernet(_resolve_key())
    try:
        raw = blob.encode("ascii")
    except UnicodeEncodeError:
        raise InvalidToken from None
    plaintext = fernet.decrypt(raw)
    try:
        items = json.loads(plaintext)
    except json.JSONDecodeError:
        raise InvalidToken from None
    if not isinstance(items, list):
        raise InvalidToken
    return items
