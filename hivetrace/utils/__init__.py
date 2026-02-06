"""
Utility package initialization.
"""

from hivetrace.utils.uuid_generator import generate_uuid

# Error helpers
from .error_helpers import (
    get_error_details,
    get_error_type,
    get_status_code,
    is_connection_error,
    is_error_response,
    is_http_error,
    is_json_decode_error,
    is_request_error,
    is_success_response,
    is_timeout_error,
    is_validation_error,
)

__all__ = [
    # UUID utilities
    "generate_uuid",
    # Error helpers
    "is_error_response",
    "is_success_response",
    "get_error_type",
    "get_error_details",
    "get_status_code",
    "is_connection_error",
    "is_timeout_error",
    "is_http_error",
    "is_json_decode_error",
    "is_request_error",
    "is_validation_error",
]
