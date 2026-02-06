"""
Network exceptions for HiveTrace SDK.
"""

from .base import HiveTraceError


class NetworkError(HiveTraceError):
    """Base exception for network errors."""

    pass


class ConnectionError(NetworkError):
    """Exception for connection errors."""

    def __init__(self, message: str = "Failed to connect to server"):
        super().__init__(message)


class TimeoutError(NetworkError):
    """Exception for timeout errors."""

    def __init__(self, message: str = "Request timeout"):
        super().__init__(message)


class RequestError(NetworkError):
    """Exception for request errors."""

    def __init__(self, message: str = "Request failed"):
        super().__init__(message)
