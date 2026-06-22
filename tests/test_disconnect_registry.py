"""Disconnect propagation through the shared DisconnectRegistry actor.

Ray Serve is mocked out; the registry actor is stood in for by a fake whose
``.remote()`` mimics Ray actor-method dispatch (fire-and-forget for set/clear,
awaitable for is_set) so the keying contract can be exercised without a cluster.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modelship.infer.infer_config import RawRequestProxy, RequestWatcher


class _FakeRegistry:
    """Stand-in for the DisconnectRegistry actor handle."""

    def __init__(self):
        self.disconnected: set[str] = set()
        self.set = self._Method(self._set)
        self.is_set = self._Method(self._is_set)
        self.clear = self._Method(self._clear)

    async def _set(self, rid):
        self.disconnected.add(rid)

    async def _is_set(self, rid):
        return rid in self.disconnected

    async def _clear(self, rid):
        self.disconnected.discard(rid)

    class _Method:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args):
            return asyncio.ensure_future(self._fn(*args))


@pytest.mark.asyncio
async def test_proxies_keyed_independently_on_one_registry():
    reg = _FakeRegistry()
    p1 = RawRequestProxy(reg, {}, "req-1")
    p2 = RawRequestProxy(reg, {}, "req-2")

    await reg.set.remote("req-1")

    assert await p1.is_disconnected() is True
    assert await p2.is_disconnected() is False

    await reg.clear.remote("req-1")
    assert await p1.is_disconnected() is False


@pytest.mark.asyncio
async def test_watcher_sets_on_disconnect_and_clears_on_stop():
    reg = _FakeRegistry()
    raw_request = MagicMock()
    raw_request.is_disconnected = AsyncMock(return_value=True)

    with patch("modelship.infer.infer_config.get_disconnect_registry", return_value=reg):
        watcher = RequestWatcher(raw_request, "req-9", model="m", endpoint="e")
        await watcher._task  # watch loop records the disconnect, then breaks

    assert "req-9" in reg.disconnected

    watcher.stop()
    await asyncio.sleep(0)  # let the fire-and-forget clear run
    assert "req-9" not in reg.disconnected
