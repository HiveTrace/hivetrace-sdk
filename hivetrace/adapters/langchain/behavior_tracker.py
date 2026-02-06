from typing import Dict, List

from .models import AgentStats


class AgentBehaviorTracker:
    """Class for tracking agent behavior"""

    def __init__(self):
        self.agent_behavior: Dict[str, AgentStats] = {}
        self.execution_order: List[str] = []
        self.agent_stack: List[str] = []

    def update_stats(self, agent_name: str, action: str) -> None:
        """Updates the agent's statistics"""
        if agent_name not in self.agent_behavior:
            self.agent_behavior[agent_name] = AgentStats(
                calls_made=0,
                calls_received=0,
                unique_children=set(),
                unique_parents=set(),
                depth_level=0,
                call_index=len(self.execution_order),
                is_leaf=True,
            )
            self.execution_order.append(agent_name)

        stats = self.agent_behavior[agent_name]

        if action == "start":
            stats.calls_made += 1
            if self.agent_stack:
                parent = self.agent_stack[-1]
                stats.unique_parents.add(parent)
                if parent in self.agent_behavior:
                    self.agent_behavior[parent].unique_children.add(agent_name)
                    self.agent_behavior[parent].is_leaf = False
        elif action == "end":
            stats.calls_received += 1
            stats.depth_level = len(self.agent_stack)

    def determine_agent_role(self, agent_name: str) -> str:
        """Determines the role of an agent based on its behavior"""
        if agent_name not in self.agent_behavior:
            return "processing_node"

        stats = self.agent_behavior[agent_name]

        if not stats.unique_parents and not stats.unique_children:
            return "root_node"
        if not stats.unique_children:
            return "leaf_node"
        if len(stats.unique_children) > 2 and len(stats.unique_parents) > 1:
            return "hub_node"
        if len(stats.unique_parents) > 1 and len(stats.unique_children) == 1:
            return "bridge_node"
        if len(stats.unique_children) > 1 and len(stats.unique_parents) == 1:
            return "splitter_node"
        if len(stats.unique_parents) > 1 and len(stats.unique_children) > 1:
            return "collector_node"
        return "processing_node"
