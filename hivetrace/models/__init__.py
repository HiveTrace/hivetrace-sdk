"""
Package of Pydantic models for HiveTrace SDK.
"""

# Request models
from .requests import (
    BaseRequest,
    FunctionCallRequest,
    InputRequest,
    MessageRequest,
    OutputRequest,
)

# Response models
from .responses import (
    BaseResponse,
    ConnectionErrorResponse,
    ErrorResponse,
    HivetraceResponse,
    HTTPErrorResponse,
    JSONDecodeErrorResponse,
    ProcessResponse,
    RequestErrorResponse,
    SuccessResponse,
    TimeoutErrorResponse,
    UnexpectedErrorResponse,
    ValidationErrorResponse,
)

__all__ = [
    # Request models
    "BaseRequest",
    "MessageRequest",
    "InputRequest",
    "OutputRequest",
    "FunctionCallRequest",
    # Response models
    "BaseResponse",
    "HivetraceResponse",
    "SuccessResponse",
    "ProcessResponse",
    "ErrorResponse",
    "ConnectionErrorResponse",
    "TimeoutErrorResponse",
    "RequestErrorResponse",
    "HTTPErrorResponse",
    "JSONDecodeErrorResponse",
    "ValidationErrorResponse",
    "UnexpectedErrorResponse",
]
