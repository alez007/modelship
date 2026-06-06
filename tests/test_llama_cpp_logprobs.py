"""The llama.cpp loader cannot yet thread token logprobs into the OpenAI
response, so it must reject requests that ask for them rather than silently
dropping the fields (which left clients waiting for logprobs that never came).
"""

import pytest

from modelship.infer.infer_config import RawRequestProxy
from modelship.infer.llama_cpp.openai.serving_chat import OpenAIServingChat
from modelship.openai.protocol import ChatCompletionRequest, ErrorResponse


def _serving() -> OpenAIServingChat:
    # The logprobs guard runs at the very top of create_chat_completion, before
    # anything reads _caps / renderer / llama, so a bare instance is enough.
    return OpenAIServingChat.__new__(OpenAIServingChat)


def _request(**overrides) -> ChatCompletionRequest:
    payload = {"model": "x", "messages": [{"role": "user", "content": "hi"}], **overrides}
    return ChatCompletionRequest(**payload)


def _raw_request() -> RawRequestProxy:
    return RawRequestProxy(None, {})


class TestLogprobsRejected:
    @pytest.mark.asyncio
    async def test_logprobs_true_returns_400(self):
        chat = _serving()
        result = await chat.create_chat_completion(_request(logprobs=True), _raw_request())
        assert isinstance(result, ErrorResponse)
        assert result._http_status == 400
        assert "logprobs" in result.error.message

    @pytest.mark.asyncio
    async def test_top_logprobs_alone_returns_400(self):
        # A client may set top_logprobs without logprobs=True; still unsupported.
        chat = _serving()
        result = await chat.create_chat_completion(_request(top_logprobs=5), _raw_request())
        assert isinstance(result, ErrorResponse)
        assert result._http_status == 400

    @pytest.mark.asyncio
    async def test_logprobs_false_is_not_rejected_here(self):
        # The default (logprobs=False, top_logprobs=0) must fall through the guard
        # so normal requests still work; it then proceeds past this point and
        # touches _caps, which the bare instance lacks → AttributeError, not an
        # ErrorResponse. That confirms the guard let it through.
        chat = _serving()
        with pytest.raises(AttributeError):
            await chat.create_chat_completion(_request(), _raw_request())
