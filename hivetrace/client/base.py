import os
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Dict, List, Optional, Tuple, Union

import httpx
from pydantic import ValidationError

from ..errors import InvalidParameterError, MissingConfigError
from ..models import (
    FunctionCallRequest,
    HivetraceResponse,
    InputRequest,
)


class BaseHivetraceSDK(ABC):
    """
    Base class for HiveTrace SDK with common business logic.

    Contains validation, payload building and other common functionality,
    but delegates HTTP operations to specific implementations.
    """

    _DEFAULT_TIMEOUT = httpx.Timeout(connect=120.0, read=120.0, write=120.0, pool=120.0)

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or self._load_config_from_env()
        self.hivetrace_url = self._get_required_config("HIVETRACE_URL")
        self.hivetrace_access_token = self._get_required_config(
            "HIVETRACE_ACCESS_TOKEN"
        )
        self._closed = False

    def _load_config_from_env(self) -> Dict[str, Any]:
        return {
            "HIVETRACE_URL": os.getenv("HIVETRACE_URL", "").strip(),
            "HIVETRACE_ACCESS_TOKEN": os.getenv("HIVETRACE_ACCESS_TOKEN", "").strip(),
        }

    def _get_required_config(self, key: str) -> str:
        value = self.config.get(key, "").strip()
        if not value:
            raise MissingConfigError(key)

        if key == "HIVETRACE_URL":
            if not (value.startswith("http://") or value.startswith("https://")):
                raise InvalidParameterError(
                    parameter="HIVETRACE_URL",
                    message=f"Invalid URL format for {key}. Must start with http:// or https://",
                )
            return value.rstrip("/")

        return value

    def _build_request_args(
        self, endpoint: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        url = f"{self.hivetrace_url}/{endpoint.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self.hivetrace_access_token}"}
        return {
            "url": url,
            "json": payload,
            "headers": headers,
            "timeout": self._DEFAULT_TIMEOUT,
        }

    def _build_files_request_args(
        self,
        endpoint: str,
        files: List[Tuple[str, bytes, str]],
        files_field_name: str = "attached_files",
    ) -> Dict[str, Any]:
        """Builds request args for multipart file upload."""
        url = f"{self.hivetrace_url}/{endpoint.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self.hivetrace_access_token}"}
        return {
            "url": url,
            "files": self._prepare_files_param(files, files_field_name),
            "headers": headers,
            "timeout": self._DEFAULT_TIMEOUT,
        }

    @staticmethod
    def _prepare_files_param(
        files: List[Tuple[str, bytes, str]],
        files_field_name: str = "attached_files",
    ) -> List[Tuple[str, Tuple[str, bytes, str]]]:
        files_param: List[Tuple[str, Tuple[str, bytes, str]]] = []
        for file_tuple in files:
            files_param.append((files_field_name, file_tuple))
        return files_param

    @staticmethod
    def _extract_analysis_id(response: Any) -> Optional[str]:
        """Extracts analysis id from API response if present."""
        try:
            if isinstance(response, dict):
                request_id = response.get("request_id")
                if request_id:
                    return str(request_id)
            elif hasattr(response, "request_id"):
                request_id = getattr(response, "request_id", None)
                if request_id:
                    return str(request_id)
        except Exception:
            return None
        return None

    @staticmethod
    def _is_failure_response(response: Any) -> bool:
        try:
            if isinstance(response, dict):
                if response.get("success") is False:
                    return True
                status = response.get("status")
                if isinstance(status, str) and "success" not in status.lower():
                    return True
                if response.get("error") or response.get("error_type"):
                    return True
                return False

            if hasattr(response, "success") and getattr(response, "success") is False:
                return True
            status = getattr(response, "status", None)
            if isinstance(status, str) and "success" not in status.lower():
                return True
            if getattr(response, "error", None) or getattr(response, "error_type", None):
                return True
        except Exception:
            return False

        return False

    def _build_message_payload(
        self,
        application_id: str,
        message: str,
        additional_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Builds payload for message with validation through Pydantic."""
        try:
            request_model = InputRequest(
                application_id=application_id,
                message=message,
                additional_parameters=additional_parameters,
            )
            return request_model.model_dump()
        except ValidationError as e:
            raise InvalidParameterError(
                parameter="request_data", message=f"Validation failed: {e}"
            ) from e

    def _build_function_call_payload(
        self,
        application_id: str,
        tool_call_id: str,
        func_name: str,
        func_args: str,
        func_result: Optional[Union[Dict, str]] = None,
        additional_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Builds payload for function call with validation through Pydantic."""
        try:
            request_model = FunctionCallRequest(
                application_id=application_id,
                tool_call_id=tool_call_id,
                func_name=func_name,
                func_args=func_args,
                func_result=func_result,
                additional_parameters=additional_parameters,
            )
            return request_model.model_dump()
        except ValidationError as e:
            raise InvalidParameterError(
                parameter="request_data", message=f"Validation failed: {e}"
            ) from e

    @abstractmethod
    def input(
        self,
        application_id: str,
        message: str,
        additional_parameters: Optional[Dict[str, Any]] = None,
        files: Optional[List[Tuple[str, bytes, str]]] = None,
    ) -> Union[HivetraceResponse, Awaitable[HivetraceResponse]]:
        """Sends user request to HiveTrace."""
        pass

    @abstractmethod
    def output(
        self,
        application_id: str,
        message: str,
        additional_parameters: Optional[Dict[str, Any]] = None,
        files: Optional[List[Tuple[str, bytes, str]]] = None,
    ) -> Union[HivetraceResponse, Awaitable[HivetraceResponse]]:
        """Sends LLM response to HiveTrace."""
        pass

    @abstractmethod
    def function_call(
        self,
        application_id: str,
        tool_call_id: str,
        func_name: str,
        func_args: str,
        func_result: Optional[Union[Dict, str]] = None,
        additional_parameters: Optional[Dict[str, Any]] = None,
    ) -> Union[HivetraceResponse, Awaitable[HivetraceResponse]]:
        """Sends function call to HiveTrace."""
        pass

    @abstractmethod
    def close(self) -> Union[None, Awaitable[None]]:
        """Closes HTTP session and frees resources."""
        pass
