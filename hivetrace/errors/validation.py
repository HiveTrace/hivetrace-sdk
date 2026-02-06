"""
Validation exceptions for HiveTrace SDK.
"""

from .base import HiveTraceError


class ValidationError(HiveTraceError):
    """Base exception for validation errors."""

    pass


class InvalidParameterError(ValidationError):
    """Exception for invalid parameter."""

    def __init__(self, parameter: str, message: str | None = None):
        message = message or f"Invalid parameter: {parameter}"
        super().__init__(message)
        self.parameter = parameter


class MissingParameterError(ValidationError):
    """Exception for missing required parameter."""

    def __init__(self, parameter: str):
        super().__init__(f"Missing required parameter: {parameter}")
        self.parameter = parameter


class InvalidFormatError(ValidationError):
    """Exception for invalid data format."""

    def __init__(self, field: str, expected_format: str):
        super().__init__(f"Invalid format for {field}. Expected: {expected_format}")
        self.field = field
        self.expected_format = expected_format
