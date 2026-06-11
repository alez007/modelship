"""Tests for the streaming Responses <-> chat-completions translator (Phase A2).

The translator consumes the chat SSE chunk stream every loader emits and emits
the Responses event protocol. These tests drive it directly with synthesized
chat chunks (no Ray, no loader) and assert the event sequence/shape.
"""

import json

import pytest

from modelship.openai.protocol import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ResponsesRequest,
    UsageInfo,
)
from modelship.openai.protocol.responses.streaming import (
    ResponsesStreamTranslator,
    _parse_chat_sse,
    responses_stream_from_chat,
)


def _req(**overrides) -> ResponsesRequest:
    payload = {"model": "m", "input": "hi"}
    payload.update(overrides)
    return ResponsesRequest(**payload)


def _chunk(delta: DeltaMessage | None = None, *, finish_reason=None, usage=None) -> ChatCompletionStreamResponse:
    choices = []
    if delta is not None or finish_reason is not None:
        choices = [
            ChatCompletionResponseStreamChoice(index=0, delta=delta or DeltaMessage(), finish_reason=finish_reason)
        ]
    return ChatCompletionStreamResponse(model="m", choices=choices, usage=usage)


def _events(translator: ResponsesStreamTranslator, chunks) -> list[dict]:
    """Run start -> process(chunks) -> finish and return parsed event payloads."""
    raw: list[str] = []
    raw.extend(translator.start())
    for c in chunks:
        raw.extend(translator.process(c))
    raw.extend(translator.finish())
    out = []
    for line in raw:
        # each is "event: <type>\ndata: {json}\n\n"
        data_line = next(ln for ln in line.splitlines() if ln.startswith("data:"))
        out.append(json.loads(data_line[len("data:") :].strip()))
    return out


def _types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


class TestSseParsing:
    def test_parses_data_chunk(self):
        chunk = _chunk(DeltaMessage(content="hi"))
        raw = f"data: {json.dumps(chunk.model_dump(mode='json'))}\n\n"
        parsed = _parse_chat_sse(raw)
        assert parsed is not None
        assert parsed.choices[0].delta.content == "hi"

    def test_done_sentinel_is_none(self):
        assert _parse_chat_sse("data: [DONE]\n\n") is None

    def test_non_data_line_is_none(self):
        assert _parse_chat_sse(": keep-alive\n\n") is None


class TestTextStream:
    def test_envelope_and_text_event_sequence(self):
        translator = ResponsesStreamTranslator(_req())
        chunks = [
            _chunk(DeltaMessage(role="assistant")),
            _chunk(DeltaMessage(content="Hel")),
            _chunk(DeltaMessage(content="lo")),
            _chunk(finish_reason="stop", usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3)),
        ]
        events = _events(translator, chunks)
        assert _types(events) == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]

    def test_text_deltas_carry_pieces_and_done_carries_full(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(
            translator,
            [_chunk(DeltaMessage(content="Hel")), _chunk(DeltaMessage(content="lo"), finish_reason="stop")],
        )
        deltas = [e["delta"] for e in events if e["type"] == "response.output_text.delta"]
        assert deltas == ["Hel", "lo"]
        done = next(e for e in events if e["type"] == "response.output_text.done")
        assert done["text"] == "Hello"

    def test_sequence_numbers_are_monotonic(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(translator, [_chunk(DeltaMessage(content="x"), finish_reason="stop")])
        seqs = [e["sequence_number"] for e in events]
        assert seqs == list(range(len(events)))

    def test_completed_response_carries_full_text_and_usage(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(
            translator,
            [
                _chunk(DeltaMessage(content="hello world")),
                _chunk(finish_reason="stop", usage=UsageInfo(prompt_tokens=4, completion_tokens=6, total_tokens=10)),
            ],
        )
        completed = events[-1]
        assert completed["type"] == "response.completed"
        resp = completed["response"]
        assert resp["status"] == "completed"
        assert resp["output"][0]["type"] == "message"
        assert resp["output"][0]["content"][0]["text"] == "hello world"
        assert resp["usage"]["input_tokens"] == 4
        assert resp["usage"]["output_tokens"] == 6

    def test_item_id_is_stable_across_events(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(translator, [_chunk(DeltaMessage(content="x"), finish_reason="stop")])
        msg_events = [e for e in events if e.get("item_id", "").startswith("msg_")]
        ids = {e["item_id"] for e in msg_events}
        assert len(ids) == 1
        # and the same id appears on the final output item
        item = next(e for e in events if e["type"] == "response.output_item.done")["item"]
        assert item["id"] == next(iter(ids))


class TestReasoningStream:
    def test_reasoning_emits_summary_events_before_message(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(
            translator,
            [
                _chunk(DeltaMessage(reasoning="be")),
                _chunk(DeltaMessage(reasoning="cause")),
                _chunk(DeltaMessage(content="answer"), finish_reason="stop"),
            ],
        )
        types = _types(events)
        assert "response.reasoning_summary_text.delta" in types
        # reasoning item opens before the message item
        assert types.index("response.reasoning_summary_part.added") < types.index("response.content_part.added")
        done = next(e for e in events if e["type"] == "response.reasoning_summary_text.done")
        assert done["text"] == "because"

    def test_completed_has_reasoning_then_message(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(
            translator,
            [_chunk(DeltaMessage(reasoning="think")), _chunk(DeltaMessage(content="ans"), finish_reason="stop")],
        )
        out = events[-1]["response"]["output"]
        assert out[0]["type"] == "reasoning"
        assert out[0]["summary"][0]["text"] == "think"
        assert out[1]["type"] == "message"


class TestToolCallStream:
    def test_tool_call_args_stream_and_finalize(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(
            translator,
            [
                _chunk(
                    DeltaMessage(
                        tool_calls=[DeltaToolCall(index=0, id="call_1", function=DeltaFunctionCall(name="get_weather"))]
                    )
                ),
                _chunk(
                    DeltaMessage(tool_calls=[DeltaToolCall(index=0, function=DeltaFunctionCall(arguments='{"city":'))])
                ),
                _chunk(
                    DeltaMessage(tool_calls=[DeltaToolCall(index=0, function=DeltaFunctionCall(arguments='"NYC"}'))])
                ),
                _chunk(finish_reason="tool_calls"),
            ],
        )
        types = _types(events)
        assert "response.function_call_arguments.delta" in types
        arg_deltas = [e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta"]
        assert "".join(arg_deltas) == '{"city":"NYC"}'
        done = next(e for e in events if e["type"] == "response.function_call_arguments.done")
        assert done["arguments"] == '{"city":"NYC"}'
        # completed output carries the function_call item with name + call_id
        fc = next(o for o in events[-1]["response"]["output"] if o["type"] == "function_call")
        assert fc["name"] == "get_weather"
        assert fc["call_id"] == "call_1"
        assert fc["arguments"] == '{"city":"NYC"}'
        # tool_calls finish reason is still a completed response
        assert events[-1]["type"] == "response.completed"

    def test_interleaved_content_after_tool_call_is_merged_into_one_message(self):
        # answer -> tool_call -> more answer: the trailing text appends to the
        # already-open message item (not dropped, not a second message).
        translator = ResponsesStreamTranslator(_req())
        events = _events(
            translator,
            [
                _chunk(DeltaMessage(content="before ")),
                _chunk(
                    DeltaMessage(tool_calls=[DeltaToolCall(index=0, id="call_1", function=DeltaFunctionCall(name="f"))])
                ),
                _chunk(DeltaMessage(content="after"), finish_reason="tool_calls"),
            ],
        )
        out = events[-1]["response"]["output"]
        messages = [o for o in out if o["type"] == "message"]
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "before after"


class TestTerminalStatus:
    def test_length_finish_reason_is_incomplete(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(translator, [_chunk(DeltaMessage(content="x"), finish_reason="length")])
        assert events[-1]["type"] == "response.incomplete"
        assert events[-1]["response"]["status"] == "incomplete"
        assert events[-1]["response"]["incomplete_details"] == {"reason": "max_output_tokens"}

    def test_content_filter_finish_reason_is_incomplete(self):
        translator = ResponsesStreamTranslator(_req())
        events = _events(translator, [_chunk(DeltaMessage(content="x"), finish_reason="content_filter")])
        assert events[-1]["type"] == "response.incomplete"
        assert events[-1]["response"]["incomplete_details"] == {"reason": "content_filter"}


class TestStreamWrapper:
    @pytest.mark.asyncio
    async def test_wraps_chat_sse_strings_into_events(self):
        async def gen():
            yield f"data: {json.dumps(_chunk(DeltaMessage(content='hi')).model_dump(mode='json'))}\n\n"
            yield f"data: {json.dumps(_chunk(finish_reason='stop').model_dump(mode='json'))}\n\n"
            yield "data: [DONE]\n\n"

        out = [e async for e in responses_stream_from_chat(gen(), _req())]
        assert all(isinstance(e, str) for e in out)
        assert out[0].startswith("event: response.created")
        assert "event: response.completed" in out[-1]

    @pytest.mark.asyncio
    async def test_pre_stream_error_object_passes_through(self):
        from modelship.openai.protocol import create_error_response

        err = create_error_response("nope")

        async def gen():
            yield err

        out = [e async for e in responses_stream_from_chat(gen(), _req())]
        # Passed straight through (not turned into a 200 event stream) so
        # _handle_response can render the proper HTTP error.
        assert out == [err]
