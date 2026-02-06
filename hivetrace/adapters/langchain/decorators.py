from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional, TypeVar, Union

from hivetrace.adapters.langchain.adapter import LangChainAdapter

F = TypeVar("F", bound=Callable[..., Any])


def _create_run_wrapper(
    orchestrator: Any,
    adapter: LangChainAdapter,
) -> Callable[[str], Any]:
    original_run = orchestrator.run

    @functools.wraps(original_run)
    def _wrapped_run(*args: Any, **kwargs: Any):
        result = original_run(*args, **kwargs)

        callback = getattr(orchestrator, "logging_callback", None)
        if callback is not None:
            try:
                adapter.send_log_data(callback)
            except Exception as exc:
                print(f"[LangChainDecorator] Error sending logs: {exc}")

        return result

    return _wrapped_run


def trace(
    hivetrace,
    application_id: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Callable[[F], F]:
    if callable(hivetrace) and not hasattr(hivetrace, "input"):
        raise ValueError(
            "trace requires at least the 'hivetrace' parameter. Use @trace(hivetrace=...)."
        )

    def _decorator(obj: Union[F, Callable[..., Any]]):
        is_method_like = (
            "." in getattr(obj, "__qualname__", "")
            and (inspect.signature(obj).parameters.keys())
            and next(iter(inspect.signature(obj).parameters.keys())) == "self"
        )

        if callable(obj) and not is_method_like:

            @functools.wraps(obj)
            def _wrapper(*args: Any, **kwargs: Any):
                orchestrator_instance = obj(*args, **kwargs)

                adapter = LangChainAdapter(
                    hivetrace=hivetrace,
                    application_id=application_id,
                    user_id=user_id,
                    session_id=session_id,
                )

                orchestrator_instance.run = _create_run_wrapper(
                    orchestrator_instance, adapter
                )

                return orchestrator_instance

            return _wrapper

        @functools.wraps(obj)
        def _method_wrapper(self, *args: Any, **kwargs: Any):
            if not hasattr(self, "_hivetrace_langchain_adapter"):
                setattr(
                    self,
                    "_hivetrace_langchain_adapter",
                    LangChainAdapter(
                        hivetrace=hivetrace,
                        application_id=application_id,
                        user_id=user_id,
                        session_id=session_id,
                    ),
                )

            adapter_instance: LangChainAdapter = getattr(
                self, "_hivetrace_langchain_adapter"
            )

            result = obj(self, *args, **kwargs)

            callback = getattr(self, "logging_callback", None)
            if callback is not None:
                try:
                    adapter_instance.send_log_data(callback)
                except Exception as exc:
                    print(f"[LangChainDecorator] Error sending logs: {exc}")

            return result

        return _method_wrapper

    return _decorator
