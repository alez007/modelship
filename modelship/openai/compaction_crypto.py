"""Symmetric encryption for ``/v1/responses/compact``'s ``encrypted_content`` blobs.

The compact endpoint returns an opaque, server-produced string that the client
later replays verbatim as input — nothing about its format is spec'd or tested
(see the compaction plan). This is one legitimate implementation: real Fernet
encryption, so the blob is genuinely opaque to whoever holds it and tampering
is detected rather than silently decoded.

Key resolution is lazy (only at encrypt/decrypt time) so importing the gateway
without ``MSHIP_COMPACTION_KEY`` set never fails on its own — only a missing key
at first use does, via a loud warning and an ephemeral per-process fallback.
That fallback only works single-replica / within one process lifetime: a
multi-gateway deployment MUST set the same key on every replica, or a blob
minted on one replica won't decode on another (see ``MSHIP_RESPONSES_TTL_S``
for the analogous conversation-state sharp edge).
"""

from __future__ import annotations

import json
import os
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from modelship.logging import get_logger

logger = get_logger("compaction_crypto")

_KEY_ENV = "MSHIP_COMPACTION_KEY"

_ephemeral_key: bytes | None = None

__all__ = ["InvalidToken", "decrypt_items", "encrypt_items"]


def _resolve_key() -> bytes:
    configured = os.environ.get(_KEY_ENV)
    if configured:
        return configured.encode("ascii")

    global _ephemeral_key
    if _ephemeral_key is None:
        _ephemeral_key = Fernet.generate_key()
        logger.warning(
            "%s is not set; using an ephemeral per-process compaction key. Compaction "
            "blobs will not decode after a gateway restart or on a different replica. "
            "Set %s (the same value on every replica) for production use.",
            _KEY_ENV,
            _KEY_ENV,
        )
    return _ephemeral_key


def encrypt_items(items: list[Any]) -> str:
    """Encrypt *items* (a list of Responses input items) into an opaque string."""
    fernet = Fernet(_resolve_key())
    plaintext = json.dumps(items).encode("utf-8")
    return fernet.encrypt(plaintext).decode("ascii")


def decrypt_items(blob: str) -> list[Any]:
    """Inverse of :func:`encrypt_items`.

    Raises :class:`InvalidToken` for a tampered blob, a blob encrypted under a
    different key, or malformed JSON — callers must map all three to the same
    clean 400 without revealing which it was.
    """
    fernet = Fernet(_resolve_key())
    plaintext = fernet.decrypt(blob.encode("ascii"))
    try:
        items = json.loads(plaintext)
    except json.JSONDecodeError:
        raise InvalidToken from None
    if not isinstance(items, list):
        raise InvalidToken
    return items
