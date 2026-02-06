"""
Decorator implementation for CrewAI tracking.
"""

import functools
from typing import Callable, Dict, Optional

from hivetrace.adapters.crewai.adapter import CrewAIAdapter


def trace(
    hivetrace,
    application_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    agent_id_mapping: Optional[Dict[str, Dict[str, str]]] = None,
):
    """
    Decorator for tracking CrewAI crews.

    Args:
        hivetrace: Hivetrace instance for logging
        application_id: Application ID in Hivetrace
        user_id: User ID in Hivetrace
        session_id: Session ID in Hivetrace
        agent_id_mapping: Maps agent role names to metadata dict with 'id' and 'description'
                         Example: {"Content Planner": {"id": "planner-123", "description": "Creates content plans"}}

    Returns:
        Decorator function that wraps crew setup functions
    """
    if callable(hivetrace):
        raise ValueError(
            "trace() missing required argument 'hivetrace'. "
            "Use @trace(hivetrace=your_instance) instead of @trace"
        )

    if application_id is None:
        raise ValueError("application_id is required")

    adapter = CrewAIAdapter(
        hivetrace=hivetrace,
        application_id=application_id,
        user_id=user_id,
        session_id=session_id,
        agent_id_mapping=agent_id_mapping or {},
    )

    def decorator(crew_setup_func: Callable):
        """Wraps crew setup function to return monitored crew."""

        @functools.wraps(crew_setup_func)
        def wrapper(*args, **kwargs):
            crew = crew_setup_func(*args, **kwargs)
            return adapter.wrap_crew(crew)

        return wrapper

    return decorator
