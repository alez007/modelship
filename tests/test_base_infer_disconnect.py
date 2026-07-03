"""Unit tests for BaseInfer.run_cancellable, the generic non-stream disconnect guard.

Non-streaming Ray Serve calls don't get a socket to watch (unlike streaming,
where Starlette's own StreamingResponse races disconnect against the body
iterator and cancellation propagates all the way down automatically).
run_cancellable polls RawRequestProxy.is_disconnected() alongside an arbitrary
coroutine and cancels whichever loses, calling the on_generation_aborted()
hook so a loader can free engine-side resources. No GPU/Ray needed — both
race participants are plain asyncio coroutines against a minimal BaseInfer
subclass.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from modelship.infer.base_infer import _DISCONNECT_POLL_INTERVAL_S, BaseInfer, ClientDisconnectedError


class _FakeRawRequest:
    def __init__(self, *, disconnect_after: float | None):
        self._disconnect_after = disconnect_after
        self._start = asyncio.get_event_loop().time()

    async def is_disconnected(self) -> bool:
        if self._disconnect_after is None:
            return False
        return asyncio.get_event_loop().time() - self._start >= self._disconnect_after


class _Infer(BaseInfer):
    """Minimal concrete BaseInfer — only run_cancellable/on_generation_aborted are exercised."""

    def __init__(self):
        super().__init__(MagicMock())
        self.aborted = False

    def shutdown(self) -> None:
        pass

    async def start(self) -> None:
        pass

    async def warmup(self) -> None:
        pass

    async def on_generation_aborted(self) -> None:
        self.aborted = True


@pytest.mark.asyncio
async def test_work_finishing_first_returns_result_without_aborting():
    async def fast_work():
        await asyncio.sleep(0.01)
        return "done"

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=None)  # client never disconnects

    result = await infer.run_cancellable(fast_work(), raw_request)

    assert result == "done"
    assert infer.aborted is False


@pytest.mark.asyncio
async def test_disconnect_cancels_work_and_calls_abort_hook():
    cancelled = asyncio.Event()

    async def slow_work():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=0.0)  # disconnected immediately

    with pytest.raises(ClientDisconnectedError):
        await infer.run_cancellable(slow_work(), raw_request)

    assert cancelled.is_set()
    assert infer.aborted is True


@pytest.mark.asyncio
async def test_work_exception_propagates_without_aborting():
    async def failing_work():
        raise ValueError("boom")

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=None)

    with pytest.raises(ValueError, match="boom"):
        await infer.run_cancellable(failing_work(), raw_request)

    assert infer.aborted is False


@pytest.mark.asyncio
async def test_disconnect_poll_interval_does_not_starve_fast_work():
    """A slow poll interval must not delay returning once work finishes —
    run_cancellable awaits asyncio.wait with FIRST_COMPLETED, not the poll loop."""

    async def fast_work():
        return 42

    infer = _Infer()
    raw_request = _FakeRawRequest(disconnect_after=None)

    start = asyncio.get_event_loop().time()
    result = await infer.run_cancellable(fast_work(), raw_request)
    elapsed = asyncio.get_event_loop().time() - start

    assert result == 42
    assert elapsed < _DISCONNECT_POLL_INTERVAL_S
