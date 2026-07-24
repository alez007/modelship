"""Symmetric encryption for ``/v1/responses/compact``'s ``encrypted_content`` blobs.

The compact endpoint returns an opaque, server-produced string that the client
later replays verbatim as input — nothing about its format is spec'd or tested
(see the compaction plan). This is one legitimate implementation: real Fernet
encryption, so the blob is genuinely opaque to whoever holds it and tampering
is detected rather than silently decoded.

The key itself is resolved once by the deploy driver (``ensure_key_seeded``,
called from ``mship_deploy.py`` before any actor starts) and stored in the
cluster-wide ``StateStore`` — the same store the effective config and
``/v1/responses`` conversation state already use. Every reader (the gateway's
``encrypt_items`` call, a model actor's ``decrypt_items`` call) is a different
Ray process, so a per-process-random key would never round-trip: encrypting
in the gateway and decrypting in a model actor are, by construction, always
two different processes, not a "multi-replica" edge case. Reading from the
shared store instead of ``os.environ`` on every call means every process
converges on the same key regardless of which node it landed on; ``redis://``
survives a lost cluster too, exactly like ``MSHIP_RESPONSES_TTL_S``'s conversation
state.

``_resolve_key`` only ever reads the store — it does not generate a key on the
fly. If nothing is seeded yet it fails hard with a clear error rather than
silently minting a per-process key, which is exactly the failure mode this
module exists to eliminate. Callers outside the normal deploy flow (tests,
ad-hoc scripts) must call ``ensure_key_seeded`` themselves first.
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
    """Resolve ``MSHIP_COMPACTION_KEY`` (or generate one) and persist it to *store*.

    Called once by the deploy driver before any actor starts, so every gateway
    and model-actor process — regardless of which node it lands on — reads the
    same already-resolved key instead of each independently checking the env.
    A no-op if the store already has a key (redeploys, joins, self-heal).
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
        # The deploy driver seeds this before any actor starts; a caller outside
        # that flow (a test, a script) must call ensure_key_seeded() itself first.
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
