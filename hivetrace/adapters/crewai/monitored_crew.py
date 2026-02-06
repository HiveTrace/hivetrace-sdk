"""
Monitored Crew implementation for CrewAI.
"""

from typing import Any, Dict, Optional

from crewai import Crew


class MonitoredCrew(Crew):
    """A monitored version of CrewAI's Crew class that logs all actions to HiveTrace."""

    model_config = {"extra": "allow"}

    def __init__(
        self,
        adapter,
        original_crew_agents,
        original_crew_tasks,
        original_crew_verbose,
        **kwargs,
    ):
        super().__init__(
            agents=original_crew_agents,
            tasks=original_crew_tasks,
            verbose=original_crew_verbose,
            **kwargs,
        )
        self._adapter = adapter

    def _prepare_runtime_context(
        self,
        user_id: Optional[str],
        session_id: Optional[str],
        agent_conversation_id: Optional[str],
    ) -> None:
        """Setup runtime context and reset conversation state."""
        self._adapter._reset_conversation_state()

        if user_id or session_id or agent_conversation_id:
            self._adapter._set_runtime_context(
                user_id=user_id,
                session_id=session_id,
                agent_conversation_id=agent_conversation_id,
            )

    def _build_log_params(
        self,
        user_id: Optional[str],
        session_id: Optional[str],
        agent_conversation_id: Optional[str],
    ) -> Dict[str, Any]:
        """Build parameters for logging."""
        agent_info = {}

        # Получаем ID последнего активного агента
        last_active_agent_id = None
        if hasattr(self._adapter, "_get_last_active_agent_id"):
            last_active_agent_id = self._adapter._get_last_active_agent_id()

        # Логируем только последнего активного агента
        if last_active_agent_id:
            for agent in self.agents:
                if (
                    hasattr(agent, "agent_id")
                    and agent.agent_id == last_active_agent_id
                ):
                    if hasattr(agent, "role"):
                        agent_info[agent.agent_id] = {
                            "name": agent.role,
                            "description": getattr(agent, "goal", ""),
                        }
                    break

        params = {
            "agents": agent_info,
            "is_final_answer": True,
        }

        # Add runtime parameters if provided
        if user_id:
            params["user_id"] = user_id
        if session_id:
            params["session_id"] = session_id
        if agent_conversation_id:
            params["agent_conversation_id"] = agent_conversation_id

        return params

    def _log_kickoff_result(
        self,
        result: Any,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_conversation_id: Optional[str] = None,
    ) -> None:
        """Log the final result of the crew execution."""
        # Log even if result is empty - this is part of the monitoring logic
        final_message = f"[Final Result] {str(result)}"
        additional_params = self._build_log_params(
            user_id, session_id, agent_conversation_id
        )

        self._adapter._prepare_and_log(
            "output",
            self._adapter.async_mode,
            message_content=final_message,
            additional_params_from_caller=additional_params,
            force_log=True,
        )

    def kickoff(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_conversation_id: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Start the crew's work and log the result."""
        self._prepare_runtime_context(user_id, session_id, agent_conversation_id)

        if inputs is not None:
            result = super().kickoff(inputs=inputs, *args, **kwargs)
        else:
            result = super().kickoff(*args, **kwargs)

        self._log_kickoff_result(result, user_id, session_id, agent_conversation_id)
        return result

    async def kickoff_async(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_conversation_id: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Start the crew's work asynchronously and log the result."""
        if not hasattr(super(), "kickoff_async"):
            raise NotImplementedError(
                "Async kickoff is not supported by the underlying crew's superclass"
            )

        self._prepare_runtime_context(user_id, session_id, agent_conversation_id)

        if inputs is not None:
            result = await super().kickoff_async(inputs=inputs, *args, **kwargs)
        else:
            result = await super().kickoff_async(*args, **kwargs)

        self._log_kickoff_result(result, user_id, session_id, agent_conversation_id)
        return result
