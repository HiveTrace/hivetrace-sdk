from typing import Any, Dict, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseRequest(BaseModel):
    """Base request model."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    @field_validator("application_id", check_fields=False)
    @classmethod
    def validate_application_id(cls, v: str) -> str:
        """Validation of UUID."""
        try:
            UUID(v)
            return v
        except ValueError:
            raise ValueError("application_id must be a valid UUID")


class MessageRequest(BaseRequest):
    """Base model for requests with message and additional parameters."""

    application_id: str = Field(..., description="Application ID (UUID)")
    message: str = Field(..., min_length=1, description="Message text")
    additional_parameters: Optional[Dict[str, Any]] = Field(
        None, description="Additional parameters"
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validation of message."""
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class InputRequest(MessageRequest):
    """Model for user input request."""

    pass


class OutputRequest(MessageRequest):
    """Model for LLM response request."""

    pass


class FunctionCallRequest(BaseRequest):
    """Model for function call request."""

    application_id: str = Field(..., description="Application ID (UUID)")
    tool_call_id: str = Field(..., description="Tool call ID")
    func_name: str = Field(..., min_length=1, description="Function name")
    func_args: str = Field(..., description="Function arguments in JSON")
    func_result: Optional[Union[Dict[str, Any], str]] = Field(
        None, description="Function execution result"
    )
    additional_parameters: Optional[Dict[str, Any]] = Field(
        None, description="Additional parameters"
    )

    @field_validator("func_name")
    @classmethod
    def validate_func_name(cls, v: str) -> str:
        """Validation of function name."""
        if not v.strip():
            raise ValueError("Function name cannot be empty")
        return v.strip()

    @field_validator("tool_call_id")
    @classmethod
    def validate_tool_call_id(cls, v: str) -> str:
        """Validation of tool call ID."""
        if not v.strip():
            raise ValueError("Tool call ID cannot be empty")
        return v.strip()
