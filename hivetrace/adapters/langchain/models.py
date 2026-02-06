from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class AgentStats:
    """Agent stats"""

    calls_made: int = 0
    calls_received: int = 0
    unique_children: Set[str] = field(default_factory=set)
    unique_parents: Set[str] = field(default_factory=set)
    depth_level: int = 0
    call_index: int = 0
    is_leaf: bool = True


@dataclass
class AgentInfo:
    """Agent info"""

    id: str
    role: str
    parent: Optional[str]
    agent_parent_id: Optional[str]
    description: str
    message: str
    level: int
    timestamp: str
    children: List[str] = field(default_factory=list)
    agent_response: Optional[str] = None
    agent_answer: Optional[str] = None
    tool_response: Optional[str] = None
    next_call: Optional[str] = None
    is_final_answer: bool = False
    behavior_stats: Optional[Dict[str, Any]] = None


@dataclass
class ToolCallInfo:
    """Tool call info"""

    id: str
    agent: str
    agent_id: str
    parent: Optional[str]
    agent_parent_id: Optional[str]
    tool: str
    tool_input: str
    run_id: str
    timestamp: str
    parent_run_id: Optional[str] = None
    tool_response: Optional[str] = None
    tool_answer: Optional[str] = None
    flow_direction: Optional[str] = None
    result_data: Optional[str] = None
    status: str = "started"
    end_timestamp: Optional[str] = None
    result_sent_to: Optional[str] = None
