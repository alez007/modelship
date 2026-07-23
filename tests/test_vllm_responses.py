"""Unit tests for VllmInfer.create_response's own wiring: gating, request
translation error mapping, and pre-generation-vs-mid-stream error handling.

Full DTO-shaping correctness (render_and_params -> build_choices/stream_chat_completion
against a real tokenizer) is covered by test_vllm_engine_ops.py's GPU-free
real-pipeline tests and by a manual end-to-end run against a real engine (the
parser-migration roadmap's standing convention for each phase) — this file
mocks `engine_ops` out entirely to isolate VllmInfer.create_response's own logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vllm.exceptions import VLLMValidationError as VllmValidationError

from modelship.infer.vllm.capabilities import VllmCapabilities
from modelship.infer.vllm.vllm_infer import VllmInfer
from modelship.openai.protocol import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    ErrorResponse,
    ResponsesRequest,
)


class _FakeRawRequest:
    """Never disconnects — is_watchable=False keeps it out of BaseInfer's
    per-replica disconnect pump entirely (same as RawRequestProxy(registry=None)),
    which also sidesteps needing a real DisconnectRegistry actor in these tests."""

    request_id = "req-1"
    is_watchable = False

    async def is_disconnected(self) -> bool:
        return False


def _make_infer() -> VllmInfer:
    """Build a VllmInfer with __init__/start bypassed; only the fields
    create_response reads are populated (mirrors test_vllm_vision.py)."""
    infer = VllmInfer.__new__(VllmInfer)
    infer._caps = VllmCapabilities(supports_image=False, supports_audio=False)
    infer.model_config = MagicMock()
    infer.model_config.name = "m"
    infer.model_config.chat_template_kwargs = {}
    infer.openai_serving_render = MagicMock()
    infer._tokenizer = MagicMock()
    infer.engine = MagicMock()
    infer._enable_auto_tools = True
    return infer


@pytest.mark.asyncio
async def test_no_render_pipeline_falls_back_to_not_supported():
    infer = VllmInfer.__new__(VllmInfer)  # no openai_serving_render attribute at all
    request = ResponsesRequest(model="m", input="hi")

    result = await infer.create_response(request, raw_request=_FakeRawRequest())

    assert isinstance(result, ErrorResponse)
    assert result._http_status == 404


@pytest.mark.asyncio
async def test_background_rejected_before_engine_ops_touched():
    infer = _make_infer()
    request = ResponsesRequest(model="m", input="hi", background=True)

    with patch("modelship.infer.vllm.vllm_infer.engine_ops") as mock_ops:
        result = await infer.create_response(request, raw_request=_FakeRawRequest())

    assert isinstance(result, ErrorResponse)
    assert result._http_status == 400
    mock_ops.render_and_params.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_reasoning_effort_returns_400():
    infer = _make_infer()
    request = ResponsesRequest(model="m", input="hi", reasoning={"effort": "turbo"})

    result = await infer.create_response(request, raw_request=_FakeRawRequest())

    assert isinstance(result, ErrorResponse)
    assert result._http_status == 400


@pytest.mark.asyncio
async def test_stream_prevalidation_error_returns_plain_error_not_generator():
    """A VllmValidationError from render_and_params must short-circuit to a
    plain ErrorResponse before any generator is created — the client gets a
    400 body, not a broken/empty event stream."""
    infer = _make_infer()
    request = ResponsesRequest(model="m", input="hi", stream=True)

    with patch("modelship.infer.vllm.vllm_infer.engine_ops") as mock_ops:
        mock_ops.build_vllm_request.return_value = MagicMock()
        mock_ops.render_and_params = AsyncMock(side_effect=VllmValidationError("too long", parameter="messages"))
        result = await infer.create_response(request, raw_request=_FakeRawRequest())

    assert isinstance(result, ErrorResponse)
    assert result._http_status == 400


@pytest.mark.asyncio
async def test_stream_success_produces_native_responses_events():
    infer = _make_infer()
    request = ResponsesRequest(model="m", input="hi", stream=True)

    async def fake_stream(*_args, **_kwargs):
        yield ChatCompletionStreamResponse(
            model="m", choices=[ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(content="hi!"))]
        )
        yield ChatCompletionStreamResponse(
            model="m",
            choices=[ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(), finish_reason="stop")],
        )

    with patch("modelship.infer.vllm.vllm_infer.engine_ops") as mock_ops:
        mock_ops.build_vllm_request.return_value = MagicMock()
        mock_ops.render_and_params = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_ops.stream_chat_completion = MagicMock(side_effect=fake_stream)
        result = await infer.create_response(request, raw_request=_FakeRawRequest())
        body = "".join([chunk async for chunk in result])

    assert "event: response.created" in body
    assert "event: response.output_text.delta" in body
    assert "event: response.completed" in body
    assert body.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_stream_mid_stream_exception_emits_failed_event():
    infer = _make_infer()
    request = ResponsesRequest(model="m", input="hi", stream=True)

    async def fake_stream(*_args, **_kwargs):
        yield ChatCompletionStreamResponse(
            model="m", choices=[ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(content="partial"))]
        )
        raise RuntimeError("engine blew up")

    with patch("modelship.infer.vllm.vllm_infer.engine_ops") as mock_ops:
        mock_ops.build_vllm_request.return_value = MagicMock()
        mock_ops.render_and_params = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_ops.stream_chat_completion = MagicMock(side_effect=fake_stream)
        result = await infer.create_response(request, raw_request=_FakeRawRequest())
        body = "".join([chunk async for chunk in result])

    assert "event: response.failed" in body
    assert "event: response.completed" not in body
    # A failed stream still terminates the SSE connection on its own; no [DONE]
    # sentinel follows (that's only emitted on the clean-finish path).
    assert "[DONE]" not in body
