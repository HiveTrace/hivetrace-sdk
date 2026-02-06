"""
Monitored Agent implementation for CrewAI.
"""

from typing import Any, Callable, Optional

from crewai import Agent, Task


class MonitoredAgent(Agent):
    """
    A monitored version of CrewAI's Agent class that logs all actions to HiveTrace.
    """

    model_config = {"extra": "allow"}

    def __init__(
        self,
        adapter_instance: Any,
        callback_func: Callable[[Any], None],
        agent_id: str,
        **kwargs,
    ):
        """
        Initialize the monitored agent.

        Parameters:
        - adapter_instance: The CrewAI adapter instance
        - callback_func: Function to call for agent callbacks
        - agent_id: ID of the agent
        - **kwargs: Additional parameters for the Agent class
        """
        if "id" in kwargs:
            del kwargs["id"]

        super().__init__(**kwargs)
        self._adapter_instance = adapter_instance
        self.callback_func = callback_func
        self.agent_id = agent_id
        self._last_thought: Optional[str] = None

    def execute_task(self, task: Task) -> str:
        """
        Override execute_task to capture thoughts and manage agent parent tracking.

        Parameters:
        - task: The task to execute

        Returns:
        - Result of the task execution
        """
        agent_role = self.role if hasattr(self, "role") else "UnknownRole"

        if hasattr(self._adapter_instance, "_set_current_parent"):
            self._adapter_instance._set_current_parent(self.agent_id)

        try:
            result = super().execute_task(task)

            if hasattr(self, "_last_thought") and self._last_thought:
                agent_goal = self.goal if hasattr(self, "goal") else "Agent thought"

                self.callback_func(
                    {
                        "type": "agent_thought",
                        "agent_id": self.agent_id,
                        "role": agent_role,
                        "thought": self._last_thought,
                        "agent_name": agent_role,
                        "agent_description": agent_goal,
                    }
                )
                self._last_thought = None

            return result

        finally:
            if hasattr(self._adapter_instance, "_clear_current_parent"):
                self._adapter_instance._clear_current_parent(agent_role)

    def _think(self, thought: str) -> None:
        """
        Override _think to capture thoughts.

        Parameters:
        - thought: The thought to capture
        """
        agent_role = self.role if hasattr(self, "role") else "UnknownRole"

        self._last_thought = thought

        if hasattr(self, "callback_func"):
            agent_goal = self.goal if hasattr(self, "goal") else "Agent thought"

            self.callback_func(
                {
                    "type": "agent_thought",
                    "agent_id": self.agent_id,
                    "role": agent_role,
                    "thought": thought,
                    "agent_name": agent_role,
                    "agent_description": agent_goal,
                }
            )
        super()._think(thought)
