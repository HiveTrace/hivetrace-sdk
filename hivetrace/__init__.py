"""
HiveTrace SDK - Python client for monitoring LLM applications.
"""

# Main clients
from .client import AsyncHivetraceSDK, BaseHivetraceSDK, SyncHivetraceSDK

# Exceptions (only exception classes)
from .errors import (
    APIError,
    ConfigurationError,
    ConnectionError,
    HiveTraceError,
    HTTPError,
    InvalidFormatError,
    InvalidParameterError,
    JSONDecodeError,
    MissingConfigError,
    MissingParameterError,
    NetworkError,
    RateLimitError,
    RequestError,
    TimeoutError,
    UnauthorizedError,
    ValidationError,
)

# Handlers
from .handlers import ErrorHandler, ResponseBuilder

# Data models (Pydantic)
from .models import (
    FunctionCallRequest,
    HivetraceResponse,
    InputRequest,
    OutputRequest,
    ProcessResponse,
    SuccessResponse,
)

# Utils
from .utils import (
    generate_uuid,
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
    # Main classes
    "AsyncHivetraceSDK",
    "SyncHivetraceSDK",
    "BaseHivetraceSDK",
    # Exceptions
    "HiveTraceError",
    "ConfigurationError",
    "MissingConfigError",
    "UnauthorizedError",
    "ValidationError",
    "InvalidParameterError",
    "MissingParameterError",
    "InvalidFormatError",
    "NetworkError",
    "ConnectionError",
    "TimeoutError",
    "RequestError",
    "APIError",
    "HTTPError",
    "JSONDecodeError",
    "RateLimitError",
    # Models
    "HivetraceResponse",
    "SuccessResponse",
    "ProcessResponse",
    "InputRequest",
    "OutputRequest",
    "FunctionCallRequest",
    # Handlers
    "ErrorHandler",
    "ResponseBuilder",
    # Utils
    "generate_uuid",
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

# Optional adapters
try:
    from hivetrace.adapters.crewai import CrewAIAdapter as _CrewAIAdapter
    from hivetrace.adapters.crewai import trace as _crewai_trace

    CrewAIAdapter = _CrewAIAdapter
    crewai_trace = _crewai_trace
    trace = _crewai_trace

    __all__.extend(["CrewAIAdapter", "crewai_trace", "trace"])
except ImportError:
    pass

try:
    from hivetrace.adapters.langchain import (
        LangChainAdapter as _LangChainAdapter,
    )
    from hivetrace.adapters.langchain import (
        run_with_tracing as _run_with_tracing,
    )
    from hivetrace.adapters.langchain import (
        run_with_tracing_async as _run_with_tracing_async,
    )
    from hivetrace.adapters.langchain import (
        trace as _langchain_trace,
    )

    LangChainAdapter = _LangChainAdapter
    langchain_trace = _langchain_trace
    run_with_tracing = _run_with_tracing
    run_with_tracing_async = _run_with_tracing_async

    __all__.extend(
        [
            "LangChainAdapter",
            "langchain_trace",
            "run_with_tracing",
            "run_with_tracing_async",
        ]
    )
except ImportError:
    pass
