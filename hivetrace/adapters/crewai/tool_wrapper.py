"""Tool wrapping utilities for CrewAI adapter."""

import functools
from typing import Any, Callable, cast

from hivetrace.utils.uuid_generator import generate_uuid


def wrap_tool_function(
    func: Callable,
    func_name: str,
    agent_role: str,
    adapter_instance,
) -> Callable:
    """Wraps a tool function to monitor its calls, attributing to the specified agent_role."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_str = [str(arg) for arg in args]
        kwargs_str = [f"{k}={v}" for k, v in kwargs.items()]
        all_args = ", ".join(args_str + kwargs_str)

        result = func(*args, **kwargs)

        agent_mapping = adapter_instance._get_agent_mapping(agent_role)
        tool_call_id = generate_uuid()

        tool_call_details = {
            "tool_call_id": tool_call_id,
            "func_name": func_name,
            "func_args": all_args,
            "func_result": str(result),
        }

        agent_params = {
            "agents": {
                agent_mapping["id"]: {
                    "name": agent_role,
                    "description": agent_mapping["description"],
                }
            },
        }

        adapter_instance._prepare_and_log(
            "function_call",
            adapter_instance.async_mode,
            tool_call_details=tool_call_details,
            additional_params_from_caller=agent_params,
            force_log=False,
        )
        return result

    return wrapper


def wrap_tool(tool: Any, agent_role: str, adapter_instance) -> Any:
    """Wraps a tool's _run method to monitor its calls."""
    if not (hasattr(tool, "_run") and callable(tool._run)):
        return tool

    run_callable = cast(Any, tool._run)
    if getattr(run_callable, "_is_hivetrace_wrapped", False):
        return tool

    tool_name = getattr(tool, "name", "unknown_tool")
    wrapped = wrap_tool_function(tool._run, tool_name, agent_role, adapter_instance)
    setattr(cast(Any, wrapped), "_is_hivetrace_wrapped", True)
    tool._run = wrapped

    return tool
