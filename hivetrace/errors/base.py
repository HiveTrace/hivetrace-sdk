"""
Base exceptions for HiveTrace SDK.
"""


class HiveTraceError(Exception):
    """Base class for all HiveTrace SDK exceptions."""

    pass


class ConfigurationError(HiveTraceError):
    """Base exception for configuration errors."""

    pass


class MissingConfigError(ConfigurationError):
    """Exception for missing configuration parameter."""

    def __init__(self, param: str):
        super().__init__(f"Config parameter '{param}' is missing")
        self.parameter = param


class UnauthorizedError(HiveTraceError):
    """Exception for authorization errors."""

    def __init__(self, message: str = "Invalid or expired access token"):
        super().__init__(message)
