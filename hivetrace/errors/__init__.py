"""
Package of exceptions for HiveTrace SDK.
"""

from .api import (
    APIError,
    HTTPError,
    JSONDecodeError,
    RateLimitError,
)
from .base import (
    ConfigurationError,
    HiveTraceError,
    MissingConfigError,
    UnauthorizedError,
)

# Network exceptions
from .network import (
    ConnectionError,
    NetworkError,
    RequestError,
    TimeoutError,
)

# Validation exceptions
from .validation import (
    InvalidFormatError,
    InvalidParameterError,
    MissingParameterError,
    ValidationError,
)

__all__ = [
    # Base exceptions
    "HiveTraceError",
    "ConfigurationError",
    "MissingConfigError",
    "UnauthorizedError",
    # Validation exceptions
    "ValidationError",
    "InvalidParameterError",
    "MissingParameterError",
    "InvalidFormatError",
    # Network exceptions
    "NetworkError",
    "ConnectionError",
    "TimeoutError",
    "RequestError",
    # API exceptions
    "APIError",
    "HTTPError",
    "JSONDecodeError",
    "RateLimitError",
]
