"""Route-level tests for /v1/responses (Phase A, non-streaming)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modelship.openai.api import ModelshipAPI
from modelship.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ResponsesRequest,
    UsageInfo,
)

_ModelshipAPI = ModelshipAPI.func_or_class


@pytest.fixture
def api():
    with patch("modelship.openai.api.serve.get_replica_context") as mock_ctx:
        mock_ctx.return_value.app_name = "test-gateway"
        return _ModelshipAPI("test-gateway")


def _raw_request():
    raw = MagicMock()
    raw.headers = {}
    return raw


class TestResponsesRoute:
    @pytest.mark.asyncio
    async def test_adapts_request_and_reuses_handle_response(self, api):
        handle = MagicMock()
        remote = handle.generate.options.return_value.remote
        api.models = {"m": {"m-a1b2c": handle}}
        api._round_robin = {"m": 0}

        request = ResponsesRequest(model="m", input="hi", instructions="be terse")

        with (
            patch("modelship.openai.api.RequestWatcher"),
            patch.object(api, "_handle_response", new=AsyncMock(return_value="OK")) as hr,
        ):
            result = await api.create_response(request, _raw_request())

        assert result == "OK"
        # The chat request handed to the actor must be the translated shape.
        chat_request = remote.call_args.args[0]
        assert isinstance(chat_request, ChatCompletionRequest)
        assert chat_request.messages == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
        assert chat_request.stream is False
        hr.assert_awaited_once()
        # endpoint label flows through to _handle_response for metrics.
        assert hr.call_args.args[3] == "create_response"

    @pytest.mark.asyncio
    async def test_stream_true_drives_streaming_translation(self, api):
        # stream=True sets stream + include_usage on the chat request and routes
        # the chat SSE chunks through the Responses event translator, yielding a
        # text/event-stream of Responses events.
        handle = MagicMock()

        async def gen():
            chunk = ChatCompletionStreamResponse(
                model="m",
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0, delta=DeltaMessage(content="hello!"), finish_reason="stop"
                    )
                ],
            )
            yield f"data: {json.dumps(chunk.model_dump(mode='json'))}\n\n"
            yield "data: [DONE]\n\n"

        handle.generate.options.return_value.remote.return_value = gen()
        api.models = {"m": {"m-a1b2c": handle}}
        api._round_robin = {"m": 0}

        request = ResponsesRequest(model="m", input="hi", stream=True)
        with patch("modelship.openai.api.RequestWatcher"):
            result = await api.create_response(request, _raw_request())

        assert result.media_type == "text/event-stream"
        # The chat request handed to the loader must be streaming with usage on.
        chat_request = handle.generate.options.return_value.remote.call_args.args[0]
        assert chat_request.stream is True
        assert chat_request.stream_options is not None and chat_request.stream_options.include_usage is True

        body = "".join([chunk async for chunk in result.body_iterator])
        assert "event: response.created" in body
        assert "event: response.output_text.delta" in body
        assert "event: response.completed" in body

    @pytest.mark.asyncio
    async def test_previous_response_id_rejected_400(self, api):
        request = ResponsesRequest(model="m", input="hi", previous_response_id="resp_1")
        result = await api.create_response(request, _raw_request())
        assert result.status_code == 400
        body = json.loads(bytes(result.body))
        assert "previous_response_id" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_param_returns_400_not_500(self, api):
        # An invalid reasoning.effort fails when constructing the
        # ChatCompletionRequest (pydantic ValidationError, not a ValueError);
        # the route must convert it to a 400, not let it 500.
        request = ResponsesRequest(model="m", input="hi", reasoning={"effort": "turbo"})
        result = await api.create_response(request, _raw_request())
        assert result.status_code == 400
        body = json.loads(bytes(result.body))
        assert body["error"]["type"] == "invalid_request_error"

    @pytest.mark.asyncio
    async def test_end_to_end_adaptation_through_handle_response(self, api):
        # Drive the real _handle_response with a mock generator yielding a chat
        # response, and assert the body comes back in Responses shape.
        handle = MagicMock()

        async def gen():
            yield ChatCompletionResponse(
                model="m",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content="hello!"),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            )

        handle.generate.options.return_value.remote.return_value = gen()
        api.models = {"m": {"m-a1b2c": handle}}
        api._round_robin = {"m": 0}

        request = ResponsesRequest(model="m", input="hi")
        with patch("modelship.openai.api.RequestWatcher"):
            result = await api.create_response(request, _raw_request())

        body = json.loads(bytes(result.body))
        assert body["object"] == "response"
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"][0]["text"] == "hello!"
        assert body["usage"]["input_tokens"] == 1
