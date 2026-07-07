"""Route-level tests for /v1/responses.

The route does no chat<->Responses translation and no pre-validation of its
own: it hands the original `ResponsesRequest` straight to `handle.respond`
and threads the result through the shared `_handle_response`, exactly like
`create_chat_completion`'s route. Feature-support validation (e.g. rejecting
`previous_response_id`) happens inside the deployment's `create_response` —
a rejection surfaces here as an ordinary leading `ErrorResponse` from the
handle, the same path any other loader-side error takes.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modelship.openai.api import ModelshipAPI
from modelship.openai.protocol import ResponsesRequest, create_error_response
from modelship.openai.protocol.responses import ResponseObject, ResponseOutputMessage, ResponseOutputText, ResponseUsage

_ModelshipAPI = ModelshipAPI.func_or_class


@pytest.fixture
def api():
    with (
        patch("modelship.openai.api.serve.get_replica_context") as mock_ctx,
        # See test_api.py's api fixture: stub configure_logging (via the cloudpickled
        # class's globals) so gateway instantiation doesn't leak global logging state.
        patch.dict(_ModelshipAPI._handle_response.__globals__, {"configure_logging": lambda: None}),
    ):
        mock_ctx.return_value.app_name = "test-gateway"
        inst = _ModelshipAPI("test-gateway")
        # Tests set api.models directly; mark the watch loop started so the routing
        # accessors (_get_handle) don't try to reconcile from a coordinator.
        inst._watch_task = MagicMock()
        return inst


def _raw_request():
    raw = MagicMock()
    raw.headers = {}
    return raw


class TestResponsesRoute:
    @pytest.mark.asyncio
    async def test_dispatches_original_request_to_respond(self, api):
        handle = MagicMock()
        remote = handle.respond.options.return_value.remote
        api.models = {"m": {"m-a1b2c": handle}}
        api._round_robin = {"m": 0}

        request = ResponsesRequest(model="m", input="hi", instructions="be terse")

        with patch.object(api, "_handle_response", new=AsyncMock(return_value="OK")) as hr:
            result = await api.create_response(request, _raw_request())

        assert result == "OK"
        # No gateway-side translation: the original ResponsesRequest is handed
        # straight to the deployment, which owns chat<->Responses shaping now.
        assert remote.call_args.args[0] is request
        hr.assert_awaited_once()
        # endpoint label flows through to _handle_response for metrics.
        assert hr.call_args.args[3] == "create_response"

    @pytest.mark.asyncio
    async def test_stream_true_dispatches_and_returns_event_stream(self, api):
        # Streaming has no gateway-side translation anymore: the deployment
        # (BaseInfer.create_response) is responsible for producing Responses SSE
        # events directly. The route only needs to thread them through.
        handle = MagicMock()

        async def gen():
            yield 'event: response.created\ndata: {"type": "response.created"}\n\n'
            yield 'event: response.completed\ndata: {"type": "response.completed"}\n\n'

        handle.respond.options.return_value.remote.return_value = gen()
        api.models = {"m": {"m-a1b2c": handle}}
        api._round_robin = {"m": 0}

        request = ResponsesRequest(model="m", input="hi", stream=True)
        result = await api.create_response(request, _raw_request())

        assert result.media_type == "text/event-stream"
        assert handle.respond.options.return_value.remote.call_args.args[0] is request

        body = "".join([chunk async for chunk in result.body_iterator])
        assert "event: response.created" in body
        assert "event: response.completed" in body

    @pytest.mark.asyncio
    async def test_previous_response_id_rejected_400(self, api):
        # The deployment rejects unsupported features (its create_response calls
        # responses_request_to_chat internally); the route just relays whatever
        # ErrorResponse comes back as the first item from the handle, same as any
        # other loader-side rejection.
        handle = MagicMock()

        async def gen():
            yield create_error_response("previous_response_id is not supported")

        handle.respond.options.return_value.remote.return_value = gen()
        api.models = {"m": {"m-a1b2c": handle}}
        api._round_robin = {"m": 0}

        request = ResponsesRequest(model="m", input="hi", previous_response_id="resp_1")
        result = await api.create_response(request, _raw_request())
        assert result.status_code == 400
        body = json.loads(bytes(result.body))
        assert "previous_response_id" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_param_returns_400_not_500(self, api):
        # An invalid reasoning.effort fails when the deployment constructs the
        # ChatCompletionRequest (pydantic ValidationError, not a ValueError);
        # that must surface as a 400 ErrorResponse, not a 500.
        handle = MagicMock()

        async def gen():
            yield create_error_response("bad reasoning.effort value", err_type="invalid_request_error")

        handle.respond.options.return_value.remote.return_value = gen()
        api.models = {"m": {"m-a1b2c": handle}}
        api._round_robin = {"m": 0}

        request = ResponsesRequest(model="m", input="hi", reasoning={"effort": "turbo"})
        result = await api.create_response(request, _raw_request())
        assert result.status_code == 400
        body = json.loads(bytes(result.body))
        assert body["error"]["type"] == "invalid_request_error"

    @pytest.mark.asyncio
    async def test_end_to_end_through_handle_response(self, api):
        # Drive the real _handle_response with a mock generator yielding an
        # already-built ResponseObject (what the deployment now returns
        # directly, with no gateway-side chat->Responses translation left).
        handle = MagicMock()

        async def gen():
            yield ResponseObject(
                model="m",
                output=[ResponseOutputMessage(content=[ResponseOutputText(text="hello!")])],
                usage=ResponseUsage(input_tokens=1, output_tokens=2, total_tokens=3),
            )

        handle.respond.options.return_value.remote.return_value = gen()
        api.models = {"m": {"m-a1b2c": handle}}
        api._round_robin = {"m": 0}

        request = ResponsesRequest(model="m", input="hi")
        result = await api.create_response(request, _raw_request())

        body = json.loads(bytes(result.body))
        assert body["object"] == "response"
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"][0]["text"] == "hello!"
        assert body["usage"]["input_tokens"] == 1
