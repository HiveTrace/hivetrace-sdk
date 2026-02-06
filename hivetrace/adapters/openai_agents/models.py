"""
Models for OpenAI Agents hivetrace adapter.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from hivetrace.utils.uuid_generator import generate_agent_uuid


@dataclass
class Call:
    """
    Base class for all calls.
    """

    span_parent_id: Optional[str] = None
    type: Literal["agent", "tool", "handoff"] = "agent"
    name: str = ""
    input: Optional[str] = None
    output: Optional[str] = None
    instructions: Optional[str] = None
    from_agent: Optional[str] = None
    to_agent: Optional[str] = None

    def to_dict(self):
        return {
            "span_parent_id": self.span_parent_id,
            "type": self.type,
            "name": self.name,
            "input": self.input,
            "output": self.output,
            "instructions": self.instructions,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
        }


@dataclass
class AgentCall(Call):
    """
    Call for an agent.
    """

    type: Literal["agent"] = "agent"

    @property
    def agent_uuid(self) -> str:
        return generate_agent_uuid(self.name)


@dataclass
class ToolCall(Call):
    """
    Call for a tool.
    """

    type: Literal["tool"] = "tool"


@dataclass
class HandoffCall(Call):
    """
    Call for a handoff.
    """

    type: Literal["handoff"] = "handoff"
    name: str = "handoff_call"
