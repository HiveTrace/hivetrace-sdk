"""
CrewAI adapter package.
"""

from hivetrace.adapters.crewai.adapter import CrewAIAdapter
from hivetrace.adapters.crewai.decorators import trace

__all__ = ["CrewAIAdapter", "trace"]

crewai_trace = trace
__all__.append("crewai_trace")
