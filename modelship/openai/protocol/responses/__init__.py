"""Responses API (``/v1/responses``) schemas and the stateless chat adapter.

Phase A implements ``/v1/responses`` as a stateless adapter over the existing
chat-completion pipeline rather than a new inference path: the request is
translated to a :class:`ChatCompletionRequest`, run through the unchanged
``handle.generate`` deployment method, and the chat result is translated back
into a Responses :class:`ResponseObject`.

Thin re-exporter over the leaf submodules (mirrors the parent ``protocol``
package): :mod:`.schemas` holds the pydantic models, :mod:`.adapter` the
translation logic. The two submodules import cleanly in either order —
``adapter`` pulls in ``schemas`` itself, and neither reaches back into the
top-level ``protocol`` package — so no import cycle exists here.
"""

from modelship.openai.protocol.responses.adapter import (
    UnsupportedResponsesFeatureError,
    chat_response_to_responses,
    responses_request_to_chat,
)
from modelship.openai.protocol.responses.schemas import (
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

__all__ = [
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
    "UnsupportedResponsesFeatureError",
    "chat_response_to_responses",
    "responses_request_to_chat",
]
