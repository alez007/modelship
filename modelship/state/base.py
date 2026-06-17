"""Generic durable state store.

A pluggable key→value store shared across the codebase: the deploy driver uses it
for the per-gateway *effective config* (durable desired state for self-heal), and
actors can use it for their own state (e.g. ``/v1/responses``). Keys are
``/``-separated namespace paths; values are JSON/YAML-serializable (``dict`` or
``list``).

Backends differ in durability, so each caller picks the one its use needs: the
default file backend survives cluster death (required for the effective config),
while an in-memory / Ray-actor backend would suit ephemeral actor state.
"""

from abc import ABC, abstractmethod

# JSON/YAML-serializable value. Kept deliberately narrow so every backend (file,
# Ray-actor, ConfigMap, Redis) can store it without custom encoders.
JsonValue = dict | list


class StateStore(ABC):
    """Key→value store. Keys are ``/``-separated namespace paths."""

    @abstractmethod
    def get(self, key: str) -> JsonValue | None:
        """Return the value for *key*, or ``None`` if absent/unreadable."""

    @abstractmethod
    def set(self, key: str, value: JsonValue) -> None:
        """Persist *value* under *key*, replacing any existing value."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove *key* if present (no error if absent)."""
