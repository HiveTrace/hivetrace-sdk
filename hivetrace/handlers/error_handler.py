from typing import Optional

import httpx

from ..models.responses import (
    ConnectionErrorResponse,
    HivetraceResponse,
    HTTPErrorResponse,
    JSONDecodeErrorResponse,
    RequestErrorResponse,
    TimeoutErrorResponse,
    UnexpectedErrorResponse,
    ValidationErrorResponse,
)


class ErrorHandler:
    """
    Error handler for converting exceptions to structured responses.
    """

    @staticmethod
    def handle_http_error(
        error: httpx.HTTPStatusError, request_id: Optional[str] = None
    ) -> HTTPErrorResponse:
        """Handles HTTP errors."""
        return HTTPErrorResponse(
            error=f"HTTP error {error.response.status_code}",
            details=error.response.text,
            status_code=error.response.status_code,
            request_id=request_id,
        )

    @staticmethod
    def handle_connection_error(
        error: httpx.ConnectError, request_id: Optional[str] = None
    ) -> ConnectionErrorResponse:
        """Handles connection errors."""
        return ConnectionErrorResponse(
            details=str(error),
            request_id=request_id,
        )

    @staticmethod
    def handle_timeout_error(
        error: httpx.TimeoutException, request_id: Optional[str] = None
    ) -> TimeoutErrorResponse:
        """Handles timeout errors."""
        return TimeoutErrorResponse(
            details=str(error),
            request_id=request_id,
        )

    @staticmethod
    def handle_request_error(
        error: httpx.RequestError, request_id: Optional[str] = None
    ) -> RequestErrorResponse:
        """Handles request errors."""
        return RequestErrorResponse(
            details=str(error),
            request_id=request_id,
        )

    @staticmethod
    def handle_json_decode_error(
        error: ValueError, request_id: Optional[str] = None
    ) -> JSONDecodeErrorResponse:
        """Handles JSON decoding errors."""
        return JSONDecodeErrorResponse(
            details=str(error),
            request_id=request_id,
        )

    @staticmethod
    def handle_validation_error(
        error: Exception,
        field_errors: Optional[list] = None,
        request_id: Optional[str] = None,
    ) -> ValidationErrorResponse:
        """Handles validation errors."""
        return ValidationErrorResponse(
            error="Validation failed",
            details=str(error),
            field_errors=field_errors,
            request_id=request_id,
        )

    @staticmethod
    def handle_unexpected_error(
        error: Exception, request_id: Optional[str] = None
    ) -> UnexpectedErrorResponse:
        """Handles unexpected errors."""
        return UnexpectedErrorResponse(
            error_type=type(error).__name__,
            details=str(error),
            request_id=request_id,
        )

    @classmethod
    def handle_error(
        cls, error: Exception, request_id: Optional[str] = None
    ) -> HivetraceResponse:
        """
        Universal error handler.

        Automatically determines the error type and calls the appropriate handler.
        """
        if isinstance(error, httpx.HTTPStatusError):
            return cls.handle_http_error(error, request_id)
        elif isinstance(error, httpx.ConnectError):
            return cls.handle_connection_error(error, request_id)
        elif isinstance(error, httpx.TimeoutException):
            return cls.handle_timeout_error(error, request_id)
        elif isinstance(error, httpx.RequestError):
            return cls.handle_request_error(error, request_id)
        elif isinstance(error, ValueError) and "JSON" in str(error):
            return cls.handle_json_decode_error(error, request_id)
        else:
            return cls.handle_unexpected_error(error, request_id)
