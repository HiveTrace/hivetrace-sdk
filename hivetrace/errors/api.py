"""
API исключения для HiveTrace SDK.
"""

from .base import HiveTraceError


class APIError(HiveTraceError):
    """Базовое исключение для API ошибок."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class HTTPError(APIError):
    """Исключение для HTTP ошибок."""

    def __init__(self, status_code: int, message: str | None = None):
        message = message or f"HTTP error {status_code}"
        super().__init__(message, status_code)


class JSONDecodeError(APIError):
    """Исключение при ошибке декодирования JSON."""

    def __init__(self, message: str = "Invalid JSON response"):
        super().__init__(message)


class RateLimitError(APIError):
    """Исключение при превышении лимита запросов."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, 429)
