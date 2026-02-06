from typing import Any, Dict, Optional

from ..models.responses import HivetraceResponse, ProcessResponse


class ResponseBuilder:
    """
    Response builder for creating structured responses.
    """

    @staticmethod
    def build_process_response(
        data: Optional[Dict[str, Any]] = None,
    ) -> ProcessResponse:
        """Builds a process response directly from API payload."""
        return ProcessResponse(**(data or {}))

    @staticmethod
    def build_response_from_api(
        api_response: Dict[str, Any],
        request_id: Optional[str] = None,
        *,
        endpoint: Optional[str] = None,
    ) -> HivetraceResponse:
        """
        Builds a typed response from API data when appropriate.

        We keep the SDK "transparent" by only parsing into `ProcessResponse` for
        `/process_*` endpoints. For other endpoints we return the raw dict, since
        their payload shapes can vary.
        """

        if endpoint and endpoint.startswith("/process_"):
            return ResponseBuilder.build_process_response(api_response)

        return api_response
