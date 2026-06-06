"""OpenAI-compatible error models and the error-response factory."""

from http import HTTPStatus

from pydantic import PrivateAttr

from modelship.openai.protocol.base import OpenAIBaseModel


class ErrorInfo(OpenAIBaseModel):
    message: str
    type: str
    param: str | None = None
    # OpenAI error code identifier (e.g. "context_length_exceeded"). None when we
    # haven't mapped the underlying failure to a specific OpenAI-known code.
    code: str | None = None


class ErrorResponse(OpenAIBaseModel):
    error: ErrorInfo

    # HTTP status code carried for the gateway to set on the JSONResponse. Lives
    # outside `error` because OpenAI's spec has no HTTP status field on the
    # error object — the spec exposes status purely via the HTTP layer.
    _http_status: int = PrivateAttr(default=500)


def create_error_response(
    message: str | Exception,
    err_type: str = "invalid_request_error",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    param: str | None = None,
    code: str | None = None,
) -> ErrorResponse:
    if isinstance(message, Exception):
        exc = message
        if isinstance(exc, ValueError | TypeError | OverflowError):
            err_type = "invalid_request_error"
            status_code = HTTPStatus.BAD_REQUEST
            param = None
        elif isinstance(exc, NotImplementedError):
            err_type = "api_error"
            status_code = HTTPStatus.NOT_IMPLEMENTED
            param = None
        else:
            err_type = "api_error"
            status_code = HTTPStatus.INTERNAL_SERVER_ERROR
            param = None
        message = str(exc)

    resp = ErrorResponse(
        error=ErrorInfo(
            message=message,
            type=err_type,
            code=code,
            param=param,
        )
    )
    resp._http_status = status_code.value
    return resp


__all__ = [
    "ErrorInfo",
    "ErrorResponse",
    "create_error_response",
]
