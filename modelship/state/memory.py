"""In-memory StateStore — a process-local dict.

The default backend. Holds nothing across process death, so it suits self-hosted
/ Docker deployments where a restart replaces the config anyway, and any actor
that wants the StateStore interface without durable infra. Selected by the
``memory://`` URI scheme.
"""

import copy

from modelship.state.base import JsonValue, StateStore


class MemoryStateStore(StateStore):
    def __init__(self) -> None:
        self._data: dict[str, JsonValue] = {}

    def get(self, key: str) -> JsonValue | None:
        value = self._data.get(key)
        # Deep-copy on the way out so callers can't mutate stored state in place.
        return copy.deepcopy(value) if value is not None else None

    def set(self, key: str, value: JsonValue) -> None:
        self._data[key] = copy.deepcopy(value)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)
