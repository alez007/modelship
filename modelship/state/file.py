"""File-backed StateStore — one JSON file per key under a base directory.

The default backend. Durable in every environment without extra infra: the base
dir defaults into the model cache, which is already a mounted volume (Docker), a
PVC (k8s), or a local dir — so it survives the Ray cluster dying. Plain atomic
JSON writes (vs. an embedded DB) stay reliable on NFS-style RWX volumes and match
the ``JsonValue`` contract exactly.

Each file holds an envelope ``{_MARKER: {"exp": <epoch|null>}, "value": <value>}``
so a TTL can travel with the value; a file without the marker is read as a legacy
raw value (back-compat with pre-envelope effective-config state).
"""

import contextlib
import json
import re
import time
import uuid
from pathlib import Path

from modelship.logging import get_logger
from modelship.state.base import JsonValue, StateStore, StateStoreUnavailableError

logger = get_logger("startup")

# Key segments may contain arbitrary text (e.g. a gateway name with spaces);
# slugify each before using it as a path component.
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Envelope marker. Deliberately unusual so a real stored value never collides.
_MARKER = "__mship_state_v1__"


def _slug(segment: str) -> str:
    return _SLUG_RE.sub("-", segment).strip("-") or "_"


def _unwrap(doc: JsonValue) -> tuple[JsonValue | None, float | None]:
    """Return (value, expires_at) from a stored doc; a doc without the marker is a
    legacy raw value with no expiry."""
    if isinstance(doc, dict) and _MARKER in doc:
        meta = doc.get(_MARKER) or {}
        return doc.get("value"), meta.get("exp") if isinstance(meta, dict) else None
    return doc, None


class FileStateStore(StateStore):
    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)

    def _path(self, key: str) -> Path:
        parts = [_slug(p) for p in key.split("/") if p]
        if not parts:
            raise ValueError(f"empty state key: {key!r}")
        *dirs, leaf = parts
        return self.base_dir.joinpath(*dirs, f"{leaf}.json")

    def get(self, key: str) -> JsonValue | None:
        path = self._path(key)
        try:
            raw = path.read_text()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StateStoreUnavailableError(f"reading state at {path}") from exc
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError:
            logger.exception("Corrupt state at %s; treating as missing.", path)
            return None
        value, expires_at = _unwrap(doc)
        if expires_at is not None and time.time() >= expires_at:
            with contextlib.suppress(OSError):
                path.unlink()  # opportunistic cleanup; harmless if it fails
            return None
        return value

    def set(self, key: str, value: JsonValue, *, ttl_seconds: float | None = None) -> None:
        path = self._path(key)
        expires_at = time.time() + ttl_seconds if ttl_seconds is not None else None
        doc = {_MARKER: {"exp": expires_at}, "value": value}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replace: a crash mid-write never leaves a torn file the next
            # read would choke on. Unique suffix avoids concurrent writers to the
            # same key colliding on one tmp file.
            tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
            tmp.write_text(json.dumps(doc, indent=2))
            tmp.replace(path)
        except OSError as exc:
            raise StateStoreUnavailableError(f"writing state at {path}") from exc

    def delete(self, key: str) -> None:
        try:
            self._path(key).unlink(missing_ok=True)
        except OSError as exc:
            raise StateStoreUnavailableError(f"deleting state {key!r}") from exc

    def list(self, prefix: str) -> list[str]:
        if not self.base_dir.exists():
            return []
        try:
            files = list(self.base_dir.rglob("*.json"))
        except OSError as exc:
            raise StateStoreUnavailableError(f"listing state under {self.base_dir}") from exc
        keys = []
        for path in files:
            if path.name.endswith(".tmp"):
                continue
            # Reconstruct the (slugged) key from the path; lossy for keys with
            # unsafe chars, so list() round-trips only already-safe segments.
            key = "/".join(path.relative_to(self.base_dir).with_suffix("").parts)
            if not prefix or key == prefix or key.startswith(f"{prefix}/"):
                keys.append(key)
        return keys
