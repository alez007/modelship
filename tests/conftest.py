"""Shared test fixtures."""

from unittest.mock import MagicMock, patch

import pytest

import modelship.infer.infer_config as infer_config


@pytest.fixture(autouse=True)
def neutralize_request_watcher():
    """Stub the disconnect registry and watch loop so route-handler tests don't spin
    up a real Ray cluster. A test module that exercises the real watcher/registry can
    override this by defining a same-named autouse fixture of its own."""

    async def _noop_watch(self):
        return

    with (
        patch.object(infer_config, "get_disconnect_registry", return_value=MagicMock()),
        patch.object(infer_config.RequestWatcher, "_watch", _noop_watch),
    ):
        yield
