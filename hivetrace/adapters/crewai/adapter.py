"""
The main implementation of the CrewAI adapter.
"""

import asyncio
from typing import Any, Dict, Optional

from crewai import Agent, Crew, Task

from hivetrace.adapters.base_adapter import BaseAdapter
from hivetrace.adapters.crewai.monitored_agent import MonitoredAgent
from hivetrace.adapters.crewai.monitored_crew import MonitoredCrew
from hivetrace.adapters.crewai.tool_wrapper import wrap_tool
from hivetrace.adapters.utils.logging import process_agent_params
from hivetrace.utils.uuid_generator import generate_uuid


class CrewAIAdapter(BaseAdapter):
    """Adapter for monitoring CrewAI agents with Hivetrace."""

    def __init__(
        self,
        hivetrace,
        application_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id_mapping: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        super().__init__(hivetrace, application_id, user_id, session_id)
        self.agent_id_mapping = agent_id_mapping or {}
        self.agents_info: Dict[str, Dict[str, Any]] = {}
        self._runtime_user_id: Optional[str] = None
        self._runtime_session_id: Optional[str] = None
        self._runtime_agents_conversation_id: Optional[str] = None
        self._current_parent_agent_id: Optional[str] = None
        self._last_active_agent_id: Optional[str] = None
        self._conversation_started = False
        self._first_agent_logged = False
        self._recent_messages: list[int] = []
        self._max_recent_messages = 5

    def _reset_conversation_state(self):
        """Reset conversation state for new command execution."""
        self._conversation_started = False
        self._first_agent_logged = False
        self._current_parent_agent_id = None
        self._last_active_agent_id = None

    def _set_current_parent(self, agent_id: str):
        """Set current agent as parent for subsequent operations."""
        self._current_parent_agent_id = agent_id
        self._last_active_agent_id = agent_id

    def _clear_current_parent(self):
        """Clear current parent when agent finishes work."""
        self._current_parent_agent_id = None

    def _get_current_parent_id(self) -> Optional[str]:
        """Get ID of current parent agent."""
        return self._current_parent_agent_id

    def _get_last_active_agent_id(self) -> Optional[str]:
        """Get ID of last active agent."""
        return self._last_active_agent_id

    def _set_runtime_context(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_conversation_id: Optional[str] = None,
    ):
        """Set execution context for runtime parameters."""
        self._runtime_user_id = user_id
        self._runtime_session_id = session_id
        self._runtime_agents_conversation_id = agent_conversation_id

    def _get_effective_user_id(self) -> Optional[str]:
        return self._runtime_user_id or self.user_id

    def _get_effective_session_id(self) -> Optional[str]:
        return self._runtime_session_id or self.session_id

    def _get_effective_agents_conversation_id(self) -> Optional[str]:
        return self._runtime_agents_conversation_id

    def _should_skip_deduplication(
        self, message_content: Optional[str], force_log: bool
    ) -> bool:
        """Determine if deduplication should be skipped."""
        if force_log or not message_content:
            return True

        skip_patterns = ["[", "Thought", "working on"]
        return any(pattern in message_content for pattern in skip_patterns)

    def _handle_deduplication(self, message_content: str) -> bool:
        """Handle message deduplication. Returns True if message should be skipped."""
        message_hash = hash(message_content)
        if message_hash in self._recent_messages:
            return True

        self._recent_messages.append(message_hash)
        if len(self._recent_messages) > self._max_recent_messages:
            self._recent_messages.pop(0)
        return False

    def _prepare_effective_params(
        self, additional_params_from_caller: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare parameters with runtime values."""
        params = (
            additional_params_from_caller.copy()
            if additional_params_from_caller
            else {}
        )

        for key, getter in [
            ("user_id", self._get_effective_user_id),
            ("session_id", self._get_effective_session_id),
            ("agent_conversation_id", self._get_effective_agents_conversation_id),
        ]:
            value = getter()
            if value:
                params.setdefault(key, value)

        params.setdefault("is_final_answer", False)
        return params

    def _handle_agent_parent_id(self, params: Dict[str, Any]) -> None:
        """Add parent_id to agents if needed."""
        agents = params.get("agents")
        if not isinstance(agents, dict):
            return

        if not self._first_agent_logged and not self._conversation_started:
            self._first_agent_logged = True
            self._conversation_started = True
        elif self._current_parent_agent_id:
            for agent_info in agents.values():
                if isinstance(agent_info, dict):
                    agent_info["agent_parent_id"] = self._current_parent_agent_id

    def _build_log_kwargs(
        self,
        log_type: str,
        message_content: Optional[str],
        tool_call_details: Optional[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Build arguments for logging method."""
        base_kwargs = {
            "application_id": self.application_id,
            "additional_parameters": params,
        }

        if log_type in ["input", "output"]:
            if message_content is None:
                return None
            base_kwargs["message"] = message_content
        elif log_type == "function_call":
            if tool_call_details is None:
                return None
            base_kwargs.update(tool_call_details)
        else:
            return None

        return base_kwargs

    def _execute_log(
        self, log_type: str, is_async: bool, kwargs: Dict[str, Any]
    ) -> None:
        """Execute logging with error handling."""
        method_name = f"{log_type}{'_async' if is_async else ''}"

        try:
            method = getattr(self.trace, method_name)
            if is_async:
                asyncio.create_task(method(**kwargs))
            else:
                method(**kwargs)
        except AttributeError:
            print(f"Error: Hivetrace object does not have method {method_name}")
        except Exception as e:
            print(f"Error logging {log_type} to Hivetrace: {e}")

    def _prepare_and_log(
        self,
        log_type: str,
        is_async: bool,
        message_content: Optional[str] = None,
        tool_call_details: Optional[Dict[str, Any]] = None,
        additional_params_from_caller: Optional[Dict[str, Any]] = None,
        force_log: bool = False,
    ) -> None:
        """Central logging method with deduplication and parameter handling."""
        if (
            not self._should_skip_deduplication(message_content, force_log)
            and message_content
        ):
            if self._handle_deduplication(message_content):
                return

        params = self._prepare_effective_params(additional_params_from_caller)
        self._handle_agent_parent_id(params)

        kwargs = self._build_log_kwargs(
            log_type, message_content, tool_call_details, params
        )
        if kwargs:
            self._execute_log(log_type, is_async, kwargs)

    def _get_agent_mapping(self, role: str) -> Dict[str, str]:
        """Get agent ID and description from mapping."""
        mapping = self.agent_id_mapping.get(role, {})

        if isinstance(mapping, dict):
            return {
                "id": mapping.get("id", generate_uuid()),
                "description": mapping.get("description", ""),
            }
        elif isinstance(mapping, str):
            return {"id": mapping, "description": ""}

        return {"id": generate_uuid(), "description": ""}

    def _log_output(
        self, message: str, additional_params: Optional[Dict[str, Any]], is_async: bool
    ):
        """Common output logging logic."""
        processed_params = process_agent_params(additional_params)
        self._prepare_and_log(
            "output",
            is_async,
            message_content=message,
            additional_params_from_caller=processed_params,
        )

    async def output_async(
        self, message: str, additional_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Asynchronous logging of agent output."""
        if not self.async_mode:
            raise RuntimeError("Cannot use async methods when SDK is in sync mode")
        self._log_output(message, additional_params, True)

    def output(
        self, message: str, additional_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Synchronous logging of agent output."""
        if self.async_mode:
            raise RuntimeError("Cannot use sync methods when SDK is in async mode")
        self._log_output(message, additional_params, False)

    def agent_callback(self, message: Any) -> None:
        """Callback for agent actions."""
        if isinstance(message, dict) and message.get("type") == "agent_thought":
            self._handle_agent_thought_message(message)
        else:
            self._handle_generic_agent_message(message)

    def _handle_agent_thought_message(self, message: Dict[str, Any]) -> None:
        """Handle agent thought type messages."""
        role = message.get("role", "")
        agent_mapping = self._get_agent_mapping(role)
        final_agent_id = message.get("agent_id") or agent_mapping["id"]

        agent_info = {
            final_agent_id: {
                "name": message.get("agent_name", role),
                "description": agent_mapping.get("description")
                or message.get("agent_description", "Agent thought"),
            }
        }

        message_text = f"Thought from agent {role}: {message['thought']}"
        self._prepare_and_log(
            "input",
            self.async_mode,
            message_content=message_text,
            additional_params_from_caller={"agents": agent_info},
            force_log=True,
        )
        self._set_current_parent(final_agent_id)

    def _handle_generic_agent_message(self, message: Any) -> None:
        """Handle generic agent messages."""
        message_text = str(message)
        self._prepare_and_log(
            "input",
            self.async_mode,
            message_content=message_text,
            additional_params_from_caller={"agents": self.agents_info},
            force_log=True,
        )

    def task_callback(self, message: Any) -> None:
        """Handler for task messages."""
        if not hasattr(message, "__dict__"):
            self._handle_simple_task_message(message)
            return

        agent_info, current_agent_id = self._extract_agent_info_from_task(message)
        message_content = self._extract_message_content_from_task(message)

        if message_content:
            self._prepare_and_log(
                "output",
                self.async_mode,
                message_content=message_content,
                additional_params_from_caller={"agents": agent_info},
                force_log=True,
            )

        if current_agent_id:
            self._set_current_parent(current_agent_id)

    def _extract_agent_info_from_task(
        self, message: Any
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """Extract agent information from task message."""
        agent_info = {}
        current_agent_id = None
        current_agent_role = ""

        if hasattr(message, "agent"):
            agent_value = message.agent
            if isinstance(agent_value, str):
                current_agent_role = agent_value
            elif hasattr(agent_value, "role"):
                current_agent_role = agent_value.role

            if current_agent_role:
                agent_mapping = self._get_agent_mapping(current_agent_role)
                current_agent_id = agent_mapping["id"]

                description = agent_mapping["description"] or (
                    getattr(agent_value, "goal", "")
                    if hasattr(agent_value, "goal")
                    else "Task agent"
                )

                agent_info = {
                    current_agent_id: {
                        "name": current_agent_role,
                        "description": description,
                    }
                }

        return agent_info, current_agent_id

    def _extract_message_content_from_task(self, message: Any) -> str:
        """Extract message content from task message."""
        if hasattr(message, "raw") and message.raw:
            return str(message.raw)

        message_parts = []
        for field in ["status", "step", "action", "observation", "thought"]:
            if hasattr(message, field):
                value = getattr(message, field)
                if value:
                    message_parts.append(f"{field}: {str(value)}")

        return "; ".join(message_parts) if message_parts else str(message)

    def _handle_simple_task_message(self, message: Any) -> None:
        """Handle simple task messages without attributes."""
        message_text = f"[Task] {str(message)}"
        self._prepare_and_log(
            "output",
            self.async_mode,
            message_content=message_text,
            additional_params_from_caller={"agents": {}},
            force_log=True,
        )

    def _wrap_agent(self, agent: Agent) -> Agent:
        """Wrap agent for monitoring."""
        agent_mapping = self._get_agent_mapping(agent.role)
        agent_id = agent_mapping["id"]

        agent_props = agent.__dict__.copy()
        original_tools = getattr(agent, "tools", [])
        agent_props["tools"] = [
            wrap_tool(tool, agent.role, self) for tool in original_tools
        ]

        for key in ["id", "agent_executor", "agent_ops_agent_id"]:
            agent_props.pop(key, None)

        return MonitoredAgent(
            adapter_instance=self,
            callback_func=self.agent_callback,
            agent_id=agent_id,
            **agent_props,
        )

    def _wrap_task(self, task: Task) -> Task:
        """Wrap task for monitoring."""
        original_callback = task.callback

        def combined_callback(message):
            self.task_callback(message)
            if original_callback:
                original_callback(message)

        task.callback = combined_callback
        return task

    def wrap_crew(self, crew: Crew) -> Crew:
        """Add monitoring to CrewAI crew."""
        self._reset_conversation_state()

        agents_info = {}
        for agent in crew.agents:
            if hasattr(agent, "role"):
                agent_mapping = self._get_agent_mapping(agent.role)
                agent_id = agent_mapping["id"]
                description = agent_mapping["description"] or getattr(agent, "goal", "")
                agents_info[agent_id] = {
                    "name": agent.role,
                    "description": description,
                }

        self.agents_info = agents_info

        wrapped_agents = [self._wrap_agent(agent) for agent in crew.agents]
        wrapped_tasks = [self._wrap_task(task) for task in crew.tasks]

        return MonitoredCrew(
            original_crew_agents=wrapped_agents,
            original_crew_tasks=wrapped_tasks,
            original_crew_verbose=crew.verbose,
            manager_llm=getattr(crew, "manager_llm", None),
            memory=getattr(crew, "memory", None),
            process=getattr(crew, "process", None),
            config=getattr(crew, "config", None),
            adapter=self,
        )
