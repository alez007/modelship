"""Responses API (``/v1/responses``) schemas and the request-side chat adapter.

Each loader implements its own ``create_response`` natively (shaping straight
from its parsed chat output — see ``utils.responses.build_responses_items_from_parsed``),
non-streaming and streaming alike; there is no generic response-side fallback.
The one thing every loader shares is the request-side translation: an incoming
``ResponsesRequest`` is turned into a ``ChatCompletionRequest`` via
``responses_request_to_chat`` before the loader re-derives its own chat request
internally.

Thin re-exporter over the leaf submodules (mirrors the parent ``protocol``
package): :mod:`.schemas` holds the pydantic models, :mod:`.adapter` the
request-side translation plus the shared response-envelope helpers, :mod:`.streaming`
the streaming translator. The submodules import cleanly — ``adapter`` pulls in
``schemas``, ``streaming`` pulls in ``adapter`` + ``schemas``, and none reach back
into the top-level ``protocol`` package — so no import cycle exists here.
"""

from modelship.openai.protocol.responses.adapter import (
    UnsupportedResponsesFeatureError,
    responses_request_to_chat,
)
from modelship.openai.protocol.responses.schemas import (
    CompactionItem,
    CompactRequest,
    CompactResource,
    ResponseFunctionToolCall,
    ResponseInputItem,
    ResponseInputTokensDetails,
    ResponseObject,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseOutputTokensDetails,
    ResponseReasoningItem,
    ResponseReasoningSummary,
    ResponseReasoningText,
    ResponsesRequest,
    ResponseUsage,
)
from modelship.openai.protocol.responses.streaming import (
    TERMINAL_EVENT_TYPES,
    ResponsesStreamTranslator,
    error_ws_frame,
    frame_sse,
    store_failure_event,
)

__all__ = [
    "TERMINAL_EVENT_TYPES",
    "CompactRequest",
    "CompactResource",
    "CompactionItem",
    "ResponseFunctionToolCall",
    "ResponseInputItem",
    "ResponseInputTokensDetails",
    "ResponseObject",
    "ResponseOutputItem",
    "ResponseOutputMessage",
    "ResponseOutputText",
    "ResponseOutputTokensDetails",
    "ResponseReasoningItem",
    "ResponseReasoningSummary",
    "ResponseReasoningText",
    "ResponseUsage",
    "ResponsesRequest",
    "ResponsesStreamTranslator",
    "UnsupportedResponsesFeatureError",
    "error_ws_frame",
    "frame_sse",
    "responses_request_to_chat",
    "store_failure_event",
]
