from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class BaseResponse(BaseModel):
    """Base response model."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)


class SuccessResponse(BaseResponse):
    """Success response from HiveTrace API."""

    success: bool = Field(default=True, description="Success flag")
    timestamp: Optional[str] = Field(default=None, description="Timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")


class ProcessResponse(BaseResponse):
    """
    Response model for `/process_request/` and `/process_response/`.
    Matches the API payload shape (no SDK-added fields).
    """

    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")
    schema_version: Optional[str] = Field(default=None, description="Schema version")
    status: Optional[str] = Field(default=None, description="Status string")
    errors: List[Any] = Field(default_factory=list, description="Errors list")
    tokens: Optional[Dict[str, Any]] = Field(default=None, description="Token usage info")
    guardrails: Optional[Dict[str, Any]] = Field(default=None, description="Guardrails result")
    custom_policy: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom policy result"
    )
    dataclean: Optional[Dict[str, Any]] = Field(default=None, description="Data cleaning result")


class ErrorResponse(BaseResponse):
    """Base response model with error."""

    success: bool = Field(default=False, description="Success flag")
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    details: str = Field(..., description="Error details")
    status_code: Optional[int] = Field(default=None, description="HTTP status code")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")


class ConnectionErrorResponse(ErrorResponse):
    """Response on connection error."""

    error_type: Literal["connection_error"] = "connection_error"
    error: Literal["Connection failed"] = "Connection failed"


class TimeoutErrorResponse(ErrorResponse):
    """Response on timeout error."""

    error_type: Literal["timeout_error"] = "timeout_error"
    error: Literal["Request timeout"] = "Request timeout"


class RequestErrorResponse(ErrorResponse):
    """Response on request error."""

    error_type: Literal["request_error"] = "request_error"
    error: Literal["Request error"] = "Request error"


class HTTPErrorResponse(ErrorResponse):
    """Response on HTTP error."""

    error_type: Literal["http_error"] = "http_error"
    status_code: int = Field(..., description="HTTP status code")


class JSONDecodeErrorResponse(ErrorResponse):
    """Response on JSON decoding error."""

    error_type: Literal["json_decode_error"] = "json_decode_error"
    error: Literal["Invalid JSON response"] = "Invalid JSON response"


class ValidationErrorResponse(ErrorResponse):
    """Response on validation error."""

    error_type: Literal["validation_error"] = "validation_error"
    field_errors: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Field validation errors"
    )


class UnexpectedErrorResponse(ErrorResponse):
    """Response on unexpected error."""

    error_type: str = Field(..., description="Type of unexpected error")
    error: Literal["Unexpected error"] = "Unexpected error"


HivetraceResponse = Union[
    ProcessResponse,
    SuccessResponse,
    ConnectionErrorResponse,
    TimeoutErrorResponse,
    RequestErrorResponse,
    HTTPErrorResponse,
    JSONDecodeErrorResponse,
    ValidationErrorResponse,
    UnexpectedErrorResponse,
    Dict[str, Any],
]
