from typing import Any, Dict, Optional, TypeVar

from hivetrace.adapters.base_adapter import BaseAdapter
from hivetrace.adapters.langchain.callback import AgentLoggingCallback

T = TypeVar("T")


class LangChainAdapter(BaseAdapter):
    def __init__(
        self,
        hivetrace,
        application_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        super().__init__(hivetrace, application_id, user_id, session_id)
        self._forced_agent_conversation_id: Optional[str] = None

    def output(
        self, message: str, additional_params: Optional[Dict[str, Any]] = None
    ) -> None:
        self._prepare_and_log(
            "output",
            self.async_mode,
            message_content=message,
            additional_params_from_caller=additional_params,
        )

    def function_call_callback(self, tool_call_details: Dict[str, Any]) -> None:
        self._prepare_and_log(
            "function_call",
            self.async_mode,
            tool_call_details=tool_call_details,
            additional_params_from_caller=tool_call_details,
        )

    def _get_conversation_id(self, callback: Any) -> Optional[str]:
        if self._forced_agent_conversation_id:
            return self._forced_agent_conversation_id

        root_name: Optional[str] = getattr(callback, "default_root_name", None)
        if not root_name or not hasattr(callback, "agents_log"):
            return None

        agents_log: Dict[str, Any] = getattr(callback, "agents_log", {})
        if root_name not in agents_log:
            return None

        return agents_log[root_name].get("agent_info", {}).get("id")

    def _get_message_for_output(self, agent_info: Dict[str, Any]) -> str:
        return str(
            agent_info.get("agent_response")
            or agent_info.get("agent_answer")
            or agent_info.get("tool_response")
            or agent_info.get("message")
            or ""
        )

    def send_log_data(self, callback: Any) -> None:
        if not hasattr(callback, "agents_log"):
            print("[LangChainAdapter] У переданного callback нет поля 'agents_log'.")
            return

        agents_log: Dict[str, Any] = getattr(callback, "agents_log") or {}
        agent_conversation_id = self._get_conversation_id(callback)

        for agent_name, agent_entry in agents_log.items():
            agent_info: Dict[str, Any] = agent_entry.get("agent_info", {})
            if not agent_info:
                continue

            message_for_output = self._get_message_for_output(agent_info)
            additional_params = self._build_additional_params(
                agent_name,
                agent_info,
                agent_conversation_id=agent_conversation_id,
            )

            self._prepare_and_log(
                "output",
                self.async_mode,
                message_content=message_for_output,
                additional_params_from_caller=additional_params,
            )

            self._process_tool_calls(
                agent_entry, agent_name, agent_info, agent_conversation_id
            )

    async def send_log_data_async(self, callback: Any) -> None:
        """Async variant: awaits logging to ensure delivery with AsyncHivetraceSDK."""
        if not hasattr(callback, "agents_log"):
            print("[LangChainAdapter] У переданного callback нет поля 'agents_log'.")
            return

        agents_log: Dict[str, Any] = getattr(callback, "agents_log") or {}
        agent_conversation_id = self._get_conversation_id(callback)

        for agent_name, agent_entry in agents_log.items():
            agent_info: Dict[str, Any] = agent_entry.get("agent_info", {})
            if not agent_info:
                continue

            message_for_output = self._get_message_for_output(agent_info)
            additional_params = self._build_additional_params(
                agent_name,
                agent_info,
                agent_conversation_id=agent_conversation_id,
            )

            try:
                # Expect AsyncHivetraceSDK with async .output
                await self.trace.output(
                    self.application_id, message_for_output, additional_params
                )
            except Exception as e:  # pragma: no cover
                print(f"[LangChainAdapter] Async output log error: {e}")

            # Process tool calls
            for tool_call in agent_entry.get("tool_call_info", []):
                tool_call_details = {
                    "tool_call_id": tool_call.get("id"),
                    "func_name": tool_call.get("tool"),
                    "func_args": tool_call.get("tool_input"),
                    "func_result": tool_call.get("tool_response")
                    or tool_call.get("tool_answer"),
                }

                func_call_additional = {
                    "agent_parent_id": tool_call.get("agent_parent_id"),
                    "agents": {
                        tool_call.get("agent_id", ""): {
                            "name": tool_call.get("agent"),
                            "description": "",
                            "agent_parent_id": tool_call.get("parent"),
                        }
                    },
                }

                if agent_conversation_id:
                    func_call_additional["agent_conversation_id"] = (
                        agent_conversation_id
                    )

                try:
                    await self.trace.function_call(
                        self.application_id,
                        tool_call_details["tool_call_id"],
                        tool_call_details["func_name"],
                        tool_call_details["func_args"],
                        tool_call_details["func_result"],
                        func_call_additional,
                    )
                except Exception as e:  # pragma: no cover
                    print(f"[LangChainAdapter] Async function_call log error: {e}")

    def _process_tool_calls(
        self,
        agent_entry: Dict[str, Any],
        agent_name: str,
        agent_info: Dict[str, Any],
        agent_conversation_id: Optional[str],
    ) -> None:
        for tool_call in agent_entry.get("tool_call_info", []):
            tool_call_details = {
                "tool_call_id": tool_call.get("id"),
                "func_name": tool_call.get("tool"),
                "func_args": tool_call.get("tool_input"),
                "func_result": tool_call.get("tool_response")
                or tool_call.get("tool_answer"),
            }

            func_call_additional = {
                "agent_parent_id": tool_call.get("agent_parent_id"),
                "agents": {
                    tool_call.get("agent_id", ""): {
                        "name": tool_call.get("agent"),
                        "description": "",
                        "agent_parent_id": tool_call.get("parent"),
                    }
                },
            }

            if agent_conversation_id:
                func_call_additional["agent_conversation_id"] = agent_conversation_id

            self._prepare_and_log(
                "function_call",
                self.async_mode,
                tool_call_details=tool_call_details,
                additional_params_from_caller=func_call_additional,
            )

    class _HiveTraceLoggingCallback(AgentLoggingCallback):
        def __init__(self, adapter: "LangChainAdapter", *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._adapter = adapter

        def _get_conversation_id(self) -> Optional[str]:
            if not (hasattr(self, "default_root_name") and hasattr(self, "agents_log")):
                return None

            if self.default_root_name not in self.agents_log:
                return None

            return (
                self.agents_log[self.default_root_name].get("agent_info", {}).get("id")
            )

        def _flush_agent_output(self, agent_name: str) -> None:
            agent_entry = self.agents_log.get(agent_name)
            if not agent_entry:
                return

            agent_info = agent_entry.get("agent_info", {})
            message_for_output = self._adapter._get_message_for_output(agent_info)
            if not message_for_output:
                return

            conversation_id = self._get_conversation_id()
            additional_params = self._adapter._build_additional_params(
                agent_name,
                agent_info,
                agent_conversation_id=conversation_id,
            )

            self._adapter._prepare_and_log(
                "output",
                self._adapter.async_mode,
                message_content=message_for_output,
                additional_params_from_caller=additional_params,
            )

        def _flush_tool_call(self, agent_name: str) -> None:
            agent_entry = self.agents_log.get(agent_name)
            if not agent_entry:
                return

            for tool_call in agent_entry.get("tool_call_info", []):
                if tool_call.get("status") != "completed" or tool_call.get(
                    "_sent_to_hivetrace"
                ):
                    continue

                tool_call_details = {
                    "tool_call_id": tool_call.get("id"),
                    "func_name": tool_call.get("tool"),
                    "func_args": tool_call.get("tool_input"),
                    "func_result": tool_call.get("tool_response")
                    or tool_call.get("tool_answer"),
                }

                agent_info = agent_entry.get("agent_info", {})
                conversation_id = self._get_conversation_id()

                additional_params = self._adapter._build_additional_params(
                    agent_name,
                    agent_info,
                    agent_conversation_id=conversation_id,
                )

                if tool_call.get("agent_parent_id") is not None:
                    additional_params["agent_parent_id"] = tool_call.get(
                        "agent_parent_id"
                    )

                self._adapter._prepare_and_log(
                    "function_call",
                    self._adapter.async_mode,
                    tool_call_details=tool_call_details,
                    additional_params_from_caller=additional_params,
                )

                tool_call["_sent_to_hivetrace"] = True

        def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
            super().on_tool_end(*args, **kwargs)
            run_id = kwargs.get("run_id")
            agent_name = self.current_agent_run_ids.get(str(run_id))
            if agent_name:
                self._flush_tool_call(agent_name)

        def on_agent_finish(self, finish: Any, *args: Any, **kwargs: Any) -> None:
            run_id = kwargs.get("run_id")
            super().on_agent_finish(finish, *args, **kwargs)

            agent_name = self.current_agent_run_ids.get(str(run_id))
            if agent_name:
                self._flush_agent_output(agent_name)
                self._flush_tool_call(agent_name)

    def create_logging_callback(
        self, **callback_kwargs: Any
    ) -> "_HiveTraceLoggingCallback":
        return LangChainAdapter._HiveTraceLoggingCallback(self, **callback_kwargs)

    def _build_additional_params(
        self,
        agent_name: str,
        agent_info: Dict[str, Any],
        agent_conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        agent_id = agent_info.get("id")
        if agent_id is None:
            return {"agent_info": agent_info}

        agent_entry: Dict[str, Any] = {
            "name": agent_name,
            "description": agent_info.get("description", ""),
        }

        parent_id_val = agent_info.get("agent_parent_id")
        if parent_id_val is not None:
            agent_entry["agent_parent_id"] = parent_id_val

        additional_params: Dict[str, Any] = {
            "agents": {agent_id: agent_entry},
        }

        effective_conversation_id = (
            agent_conversation_id or self._forced_agent_conversation_id
        )
        if effective_conversation_id:
            additional_params["agent_conversation_id"] = effective_conversation_id

        if "is_final_answer" in agent_info:
            additional_params["is_final_answer"] = agent_info["is_final_answer"]

        if self.user_id is not None:
            additional_params["user_id"] = self.user_id
        if self.session_id is not None:
            additional_params["session_id"] = self.session_id

        return additional_params
