"""Streaming-aware latency instrumentation in ModelDeployment.generate()."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from modelship.infer.model_deployment import ModelDeployment

# Bypass the @serve.deployment wrapper.
_ModelDeployment = ModelDeployment.func_or_class


def _patch_gen_metric(mock):
    """Patch GENERATION_DURATION_SECONDS on generate()'s own globals.

    `@serve.deployment` cloudpickles the class, so the unwrapped method carries a
    reconstructed globals dict — patching the module attribute wouldn't reach it."""
    return patch.dict(_ModelDeployment.generate.__globals__, {"GENERATION_DURATION_SECONDS": mock})


def _make_deployment(infer):
    """Build a ModelDeployment without running its async __init__."""
    inst = _ModelDeployment.__new__(_ModelDeployment)
    inst.config = MagicMock()
    inst.config.name = "test-model"
    inst.infer = infer
    return inst


@pytest.mark.asyncio
async def test_streaming_generation_observed_after_drain():
    """For streaming, GENERATION_DURATION_SECONDS must be observed after the
    async generator is exhausted, capturing the full decode time."""
    delay = 0.02

    async def fake_stream():
        for i in range(3):
            await asyncio.sleep(delay)
            yield f"chunk{i}"

    infer = MagicMock()
    infer.create_chat_completion = MagicMock(return_value=_awaitable(fake_stream()))

    dep = _make_deployment(infer)
    gen = MagicMock()
    with _patch_gen_metric(gen):
        chunks = [c async for c in dep.generate(MagicMock(), {}, MagicMock(), "rid")]

    assert chunks == ["chunk0", "chunk1", "chunk2"]
    gen.observe.assert_called_once()
    observed = gen.observe.call_args.args[0]
    assert observed >= delay * 3


@pytest.mark.asyncio
async def test_non_streaming_generation_observed_once():
    """Non-streaming results are observed exactly once and still yielded."""
    infer = MagicMock()
    infer.create_chat_completion = MagicMock(return_value=_awaitable("full-response"))

    dep = _make_deployment(infer)
    gen = MagicMock()
    with _patch_gen_metric(gen):
        chunks = [c async for c in dep.generate(MagicMock(), {}, MagicMock(), "rid")]

    assert chunks == ["full-response"]
    gen.observe.assert_called_once()


def _awaitable(value):
    async def _coro():
        return value

    return _coro()
