"""
Base adapter class for HiveTrace integrations.
"""

from typing import Any, Coroutine, Dict, Optional, cast


class BaseAdapter:
    """
    Base class for all integration adapters for Hivetrace.

    This class defines the common interface and utilities used by all adapters.
    Specific adapters should inherit from this class and implement the required methods.
    """

    def __init__(
        self,
        hivetrace,
        application_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the base adapter.

        Parameters:
        - hivetrace: The hivetrace instance for logging
        - application_id: ID of the application in Hivetrace
        - user_id: ID of the user in the conversation
        - session_id: ID of the session in the conversation
        """
        self.trace = hivetrace
        self.application_id = application_id
        self.user_id = user_id
        self.session_id = session_id
        self.async_mode = self.trace.async_mode

    def _prepare_and_log(
        self,
        log_method_name_stem: str,
        is_async: bool,
        message_content: Optional[str] = None,
        tool_call_details: Optional[Dict[str, Any]] = None,
        additional_params_from_caller: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Helper method to prepare parameters and log to Hivetrace.

        Parameters:
        - log_method_name_stem: Base name of the logging method ('input', 'output', 'function_call')
        - is_async: Whether to use async logging methods
        - message_content: Content of the message to log (for input/output)
        - tool_call_details: Details of tool/function call (for function_call)
        - additional_params_from_caller: Additional parameters to include in the log
        """
        final_additional_params = additional_params_from_caller or {}
        if hasattr(self, "user_id") and self.user_id is not None and self.user_id != "":
            final_additional_params.setdefault("user_id", self.user_id)
        if (
            hasattr(self, "session_id")
            and self.session_id is not None
            and self.session_id != ""
        ):
            final_additional_params.setdefault("session_id", self.session_id)

        log_kwargs = {
            "application_id": self.application_id,
            "additional_parameters": final_additional_params,
        }

        if log_method_name_stem in ["input", "output"]:
            if message_content is None:
                print(f"Warning: message_content is None for {log_method_name_stem}")
                return
            log_kwargs["message"] = message_content
        elif log_method_name_stem == "function_call":
            if tool_call_details is None:
                print("Warning: tool_call_details is None for function_call")
                return
            merged_tool_details = dict(tool_call_details)
            ap = dict(merged_tool_details.get("additional_parameters", {}) or {})
            if (
                hasattr(self, "user_id")
                and self.user_id is not None
                and self.user_id != ""
            ):
                ap.setdefault("user_id", self.user_id)
            if (
                hasattr(self, "session_id")
                and self.session_id is not None
                and self.session_id != ""
            ):
                ap.setdefault("session_id", self.session_id)
            if ap:
                merged_tool_details["additional_parameters"] = ap
            log_kwargs.update(merged_tool_details)
        else:
            print(f"Error: Unsupported log_method_name_stem: {log_method_name_stem}")
            return

        # Both SyncHivetraceSDK and AsyncHivetraceSDK expose the same method names
        # (input/output/function_call). In async mode they are coroutines.
        method_to_call_name = log_method_name_stem

        try:
            actual_log_method = getattr(self.trace, method_to_call_name)
            if is_async:
                import asyncio
                import inspect

                try:
                    maybe_coro = actual_log_method(**log_kwargs)
                except TypeError:
                    # Fallback: call without kwargs if signature mismatch (defensive)
                    maybe_coro = actual_log_method()

                if inspect.isawaitable(maybe_coro):
                    if inspect.iscoroutine(maybe_coro):
                        asyncio.create_task(cast(Coroutine[Any, Any, Any], maybe_coro))
                    else:
                        async def _await_and_drop() -> None:
                            await maybe_coro

                        asyncio.create_task(_await_and_drop())
                else:
                    # If the method is unexpectedly sync in async mode, call directly
                    # to avoid dropping the log.
                    pass
            else:
                actual_log_method(**log_kwargs)
        except AttributeError:
            print(f"Error: Hivetrace object does not have method {method_to_call_name}")
        except Exception as e:
            print(f"Error logging {log_method_name_stem} to Hivetrace: {e}")

    async def input_async(
        self, message: str, additional_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Asynchronously logs user input to Hivetrace.
        """
        if not self.async_mode:
            raise RuntimeError("Cannot use async methods when SDK is in sync mode")

        self._prepare_and_log(
            "input",
            True,
            message_content=message,
            additional_params_from_caller=additional_params,
        )

    def input(
        self, message: str, additional_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Synchronously logs user input to Hivetrace.
        """
        if self.async_mode:
            raise RuntimeError("Cannot use sync methods when SDK is in async mode")

        self._prepare_and_log(
            "input",
            False,
            message_content=message,
            additional_params_from_caller=additional_params,
        )

    async def output_async(
        self,
        message: str,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Asynchronously logs agent output to Hivetrace.
        """
        if not self.async_mode:
            raise RuntimeError("Cannot use async methods when SDK is in sync mode")

        self._prepare_and_log(
            "output",
            True,
            message_content=message,
            additional_params_from_caller=additional_params,
        )

    def output(
        self,
        message: str,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Synchronously logs agent output to Hivetrace.
        """
        if self.async_mode:
            raise RuntimeError("Cannot use sync methods when SDK is in async mode")

        self._prepare_and_log(
            "output",
            False,
            message_content=message,
            additional_params_from_caller=additional_params,
        )
