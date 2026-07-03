"""Tests for vLLM chat-completion handling of vision (image_url) message parts.

The boundary we can verify without a real GPU/model is the protocol hop:
modelship ``ChatCompletionRequest`` -> ``model_dump()`` -> ``VllmChatCompletionRequest``
(vLLM's own pydantic model). If image parts survive that, vLLM's downstream
multimodal preprocessing runs against the same shape it does in upstream
deployments. Wrapper-level concerns (normalize_chat_messages gating) are
tested via :class:`VllmInfer.create_chat_completion` with serving_chat stubbed
out — the goal there is the 400-rejection path, not the inference call.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest as VllmChatCompletionRequest

from modelship.infer.vllm.capabilities import VllmCapabilities
from modelship.infer.vllm.vllm_infer import VllmInfer
from modelship.openai.protocol import ChatCompletionRequest, ErrorResponse

# ---------------------------------------------------------------------------
# Protocol hop — exercise the real vLLM ChatCompletionRequest validator
# ---------------------------------------------------------------------------


def test_image_url_part_survives_protocol_hop_into_vllm_request():
    """An OpenAI-style image_url part on our request must round-trip into
    ``VllmChatCompletionRequest`` unchanged — that's the shape vLLM's
    multimodal preprocessing reads."""
    request = ChatCompletionRequest(
        model="qwen-vl",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                ],
            }
        ],
    )

    vllm_request = VllmChatCompletionRequest(**request.model_dump())

    parts = vllm_request.messages[0]["content"]
    assert isinstance(parts, list)
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"] == "https://example.com/cat.png"
    text_parts = [p for p in parts if p.get("type") == "text"]
    assert len(text_parts) == 1
    assert text_parts[0]["text"] == "describe"


def test_data_uri_image_survives_protocol_hop():
    """Base64 data URIs are the common alternative to remote URLs; vLLM must
    accept them on the same code path."""
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    request = ChatCompletionRequest(
        model="qwen-vl",
        messages=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": data_uri}}],
            }
        ],
    )

    vllm_request = VllmChatCompletionRequest(**request.model_dump())

    parts = vllm_request.messages[0]["content"]
    assert parts[0]["image_url"]["url"] == data_uri


# ---------------------------------------------------------------------------
# Wrapper gating — exercise VllmInfer.create_chat_completion's 400 path
# ---------------------------------------------------------------------------


def _make_infer(*, supports_image: bool) -> VllmInfer:
    """Build a VllmInfer with __init__/start bypassed; only the fields the
    chat-completion path reads are populated."""
    infer = VllmInfer.__new__(VllmInfer)
    infer._caps = VllmCapabilities(supports_image=supports_image)  # type: ignore[attr-defined]
    infer.model_config = MagicMock(chat_template_kwargs={})
    infer.serving_chat = MagicMock()
    infer.serving_chat.create_chat_completion = AsyncMock(return_value=MagicMock())
    return infer


@pytest.mark.asyncio
async def test_image_part_rejected_on_text_only_model_with_400():
    """A text-only model receiving image_url returns a 400 BadRequest with
    our error envelope — same shape transformers and llama.cpp produce."""
    infer = _make_infer(supports_image=False)
    request = ChatCompletionRequest(
        model="llm",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                ],
            }
        ],
    )

    result = await infer.create_chat_completion(request, raw_request=MagicMock())

    assert isinstance(result, ErrorResponse)
    assert result._http_status == 400
    assert "image" in result.error.message.lower()
    # The reject must happen before we hand off to vLLM.
    infer.serving_chat.create_chat_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_only_request_reaches_serving_chat_on_vlm():
    """Sanity check: a plain text request on a VLM is not blocked by the
    gating layer. Streaming, since non-stream no longer calls serving_chat
    (see test_vllm_engine_ops.py / test_integration.py::TestChatCapable for
    the engine_ops-based non-stream path)."""
    infer = _make_infer(supports_image=True)
    request = ChatCompletionRequest(
        model="qwen-vl",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

    await infer.create_chat_completion(request, raw_request=MagicMock())

    infer.serving_chat.create_chat_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_model_chat_template_kwargs_merged_into_vllm_request():
    """The model's chat_template_kwargs default reaches the vLLM request that
    serving_chat renders from. Streaming — see test above for why."""
    infer = _make_infer(supports_image=False)
    infer.model_config.chat_template_kwargs = {"enable_thinking": False}
    request = ChatCompletionRequest(model="llm", messages=[{"role": "user", "content": "hi"}], stream=True)

    await infer.create_chat_completion(request, raw_request=MagicMock())

    vllm_request = infer.serving_chat.create_chat_completion.await_args.args[0]
    assert vllm_request.chat_template_kwargs == {"enable_thinking": False}
