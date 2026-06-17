"""File-backed StateStore — one JSON file per key under a base directory.

The default backend. Durable in every environment without extra infra: the base
dir defaults into the model cache, which is already a mounted volume (Docker), a
PVC (k8s), or a local dir — so it survives the Ray cluster dying. Plain atomic
JSON writes (vs. an embedded DB) stay reliable on NFS-style RWX volumes and match
the ``JsonValue`` contract exactly.
"""

import json
import re
from pathlib import Path

from modelship.logging import get_logger
from modelship.state.base import JsonValue, StateStore

logger = get_logger("startup")

# Key segments may contain arbitrary text (e.g. a gateway name with spaces);
# slugify each before using it as a path component.
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(segment: str) -> str:
    return _SLUG_RE.sub("-", segment).strip("-") or "_"


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
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.exception("Corrupt/unreadable state at %s; treating as missing.", path)
            return None

    def set(self, key: str, value: JsonValue) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace: a crash mid-write never leaves a torn file the next read
        # would choke on.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(value, indent=2))
        tmp.replace(path)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
