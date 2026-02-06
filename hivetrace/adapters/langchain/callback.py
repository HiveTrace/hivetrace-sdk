import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler

from .behavior_tracker import AgentBehaviorTracker
from .models import AgentInfo, ToolCallInfo


class AgentLoggingCallback(BaseCallbackHandler):
    """Callback for logging agent actions"""

    def __init__(
        self,
        log_filename: str = "agent_calls_log.json",
        agent_descriptions: Optional[Dict[str, str]] = None,
        default_root_name: str = "RootNode",
        predefined_agent_ids: Optional[Dict[str, str]] = None,
    ):
        self.behavior_tracker = AgentBehaviorTracker()
        self._agents_log: Dict[str, Dict[str, Any]] = {}
        self.current_agent_run_ids: Dict[str, str] = {}
        self.call_id_mapping: Dict[str, str] = {}
        self.log_filename = log_filename
        self.default_root_name = default_root_name
        self.agent_descriptions = agent_descriptions or {}
        self.predefined_agent_ids = predefined_agent_ids or {}

        self.tool_names = {
            "calculator",
            "statistics_calculator",
            "text_analyzer",
            "text_formatter",
        }

    @property
    def agents_log(self) -> Dict[str, Dict[str, Any]]:
        """Returns the agent log for the current call"""
        return self._agents_log

    def _get_conversation_id(self) -> Optional[str]:
        """Returns the conversation ID for the current call"""
        if not (hasattr(self, "default_root_name") and hasattr(self, "_agents_log")):
            return None

        if self.default_root_name not in self._agents_log:
            return None

        return self._agents_log[self.default_root_name].get("agent_info", {}).get("id")

    def _save_to_file_immediately(self) -> None:
        """Saves the current state of the logs to a file"""
        try:
            with open(self.log_filename, "w", encoding="utf-8") as f:
                json.dump(self.agents_log, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def reset(self) -> None:
        """Resets the callback"""
        self._agents_log.clear()
        self.current_agent_run_ids.clear()
        self.call_id_mapping.clear()
        self.behavior_tracker = AgentBehaviorTracker()

    def _get_universal_description(self, agent_name: str) -> str:
        """Returns the universal agent description"""
        if agent_name in self.agent_descriptions:
            return self.agent_descriptions[agent_name]

        agent_role = self.behavior_tracker.determine_agent_role(agent_name)
        stats = self.behavior_tracker.agent_behavior.get(agent_name)

        role_descriptions = {
            "root_node": f"Root node of the system {agent_name} (entry point)",
            "leaf_node": f"Leaf node {agent_name} (final executor)",
            "hub_node": f"Hub node {agent_name} (coordinates {len(stats.unique_children) if stats else 0} agents)",
            "bridge_node": f"Bridge node {agent_name} (transfers data between levels)",
            "splitter_node": f"Splitter node {agent_name} (splits tasks)",
            "collector_node": f"Collector node {agent_name} (merges results)",
            "processing_node": f"Processing node {agent_name} (executes logic)",
        }

        return role_descriptions.get(agent_role, f"Node {agent_name}")

    def _get_parent_id(self, parent_name: str) -> Optional[str]:
        """Returns the ID of the parent agent"""
        if parent_name in self.predefined_agent_ids:
            return self.predefined_agent_ids[parent_name]
        if parent_name in self.agents_log:
            return self.agents_log[parent_name]["agent_info"]["id"]
        return None

    def _create_agent_entry(
        self,
        agent_name: str,
        agent_id: str,
        parent_agent: str,
        input_str: str,
        level: int,
    ) -> Dict[str, Any]:
        """Creates an entry for an agent"""
        return {
            "agent_info": AgentInfo(
                id=agent_id,
                role=self.behavior_tracker.determine_agent_role(agent_name),
                parent=parent_agent,
                agent_parent_id=self._get_parent_id(parent_agent),
                description=self._get_universal_description(agent_name),
                message=input_str,
                level=level,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            ).__dict__,
            "tool_call_info": [],
        }

    def _create_tool_call_info(
        self,
        call_id: str,
        agent_name: str,
        agent_id: str,
        parent_agent: str,
        input_str: str,
        run_id: str,
        parent_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates information about a tool call"""
        return ToolCallInfo(
            id=call_id,
            agent=agent_name,
            agent_id=agent_id,
            parent=parent_agent,
            agent_parent_id=self._get_parent_id(parent_agent),
            tool=agent_name,
            tool_input=input_str,
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        ).__dict__

    def _create_return_entry(
        self,
        original_call_id: str,
        agent_name: str,
        parent_agent: str,
        agent_id: str,
        output: str,
        run_id: str,
        parent_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates an entry for the return result"""
        return ToolCallInfo(
            id=f"return_{original_call_id}",
            agent=agent_name,
            agent_id=agent_id,
            parent=parent_agent,
            agent_parent_id=self._get_parent_id(parent_agent),
            tool="return_result",
            tool_input=f"result from {agent_name}",
            tool_answer=f"sent to {parent_agent}",
            flow_direction="return",
            result_data=output,
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            status="completed",
        ).__dict__

    def _process_agent_start(
        self,
        agent_name: str,
        input_str: str,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
    ) -> None:
        """Processes the start of an agent"""
        if agent_name in self.tool_names:
            return

        parent_agent = (
            self.behavior_tracker.agent_stack[-1]
            if self.behavior_tracker.agent_stack
            else self.default_root_name
        )
        agent_id = self.predefined_agent_ids.get(agent_name, str(uuid.uuid4()))
        level = len(self.behavior_tracker.agent_stack)

        self.behavior_tracker.update_stats(agent_name, "start")
        self.behavior_tracker.agent_stack.append(agent_name)
        self.current_agent_run_ids[str(run_id)] = agent_name

        if agent_name not in self._agents_log:
            self._agents_log[agent_name] = self._create_agent_entry(
                agent_name, agent_id, parent_agent, input_str, level
            )

        if parent_agent != self.default_root_name and parent_agent in self._agents_log:
            children = self._agents_log[parent_agent]["agent_info"]["children"]
            if agent_name not in children:
                children.append(agent_name)

        call_id = str(uuid.uuid4())[:8]
        self.call_id_mapping[str(run_id)] = call_id

        tool_call_info = self._create_tool_call_info(
            call_id,
            agent_name,
            agent_id,
            parent_agent,
            input_str,
            run_id,
            parent_run_id,
        )

        self._agents_log[agent_name]["tool_call_info"].append(tool_call_info)

    def _process_agent_end(
        self,
        agent_name: str,
        output: str,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
    ) -> None:
        """Processes the end of an agent"""
        if agent_name in self.tool_names:
            return

        self.behavior_tracker.update_stats(agent_name, "end")

        if agent_name in self._agents_log:
            tool_calls = self._agents_log[agent_name]["tool_call_info"]
            original_call_id = None

            for tool_call in reversed(tool_calls):
                if tool_call["status"] == "started" and tool_call["run_id"] == str(
                    run_id
                ):
                    tool_call.update(
                        {
                            "tool_response": output,
                            "tool_answer": output,
                            "status": "completed",
                            "end_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                    original_call_id = tool_call["id"]
                    break

            agent_info = self._agents_log[agent_name]["agent_info"]
            agent_info.update(
                {
                    "tool_response": output,
                    "agent_answer": output,
                    "agent_response": output,
                }
            )

            parent_agent = (
                self.behavior_tracker.agent_stack[-2]
                if len(self.behavior_tracker.agent_stack) >= 2
                else self.default_root_name
            )

            if original_call_id and parent_agent != self.default_root_name:
                agent_id_current = agent_info.get("id")
                return_entry = self._create_return_entry(
                    original_call_id,
                    agent_name,
                    parent_agent,
                    agent_id_current,
                    output,
                    run_id,
                    None,
                )

                self._agents_log[agent_name]["tool_call_info"].append(return_entry)

                for tool_call in self._agents_log[agent_name]["tool_call_info"]:
                    if tool_call["id"] == original_call_id:
                        tool_call["result_sent_to"] = parent_agent
                        break

            agent_info["next_call"] = (
                self.behavior_tracker.agent_stack[-1]
                if self.behavior_tracker.agent_stack
                else None
            )

    def _finalize_agent_roles(self) -> None:
        """Finalizes the roles of agents"""
        for agent_name in self._agents_log:
            final_role = self.behavior_tracker.determine_agent_role(agent_name)
            agent_info = self._agents_log[agent_name]["agent_info"]

            agent_info["role"] = final_role
            agent_info["description"] = self._get_universal_description(agent_name)

            if agent_name in self.behavior_tracker.agent_behavior:
                stats = self.behavior_tracker.agent_behavior[agent_name]
                agent_info["behavior_stats"] = {
                    "calls_made": stats.calls_made,
                    "calls_received": stats.calls_received,
                    "children_count": len(stats.unique_children),
                    "parents_count": len(stats.unique_parents),
                    "depth_level": stats.depth_level,
                    "execution_order": stats.call_index,
                    "is_leaf": stats.is_leaf,
                }

            if agent_name != self.default_root_name:
                agent_info["is_final_answer"] = False

    def _create_final_root_entry(self, final_output: str) -> None:
        """Creates the final entry for the root agent"""
        main_agents = [
            name
            for name, agent_data in self._agents_log.items()
            if name != self.default_root_name
            and agent_data.get("agent_info", {}).get("parent") == self.default_root_name
        ]

        if self.default_root_name not in self._agents_log:
            root_id = self.predefined_agent_ids.get(
                self.default_root_name,
                str(uuid.uuid4())[:8],
            )
            self.predefined_agent_ids[self.default_root_name] = root_id

            self._agents_log[self.default_root_name] = {
                "agent_info": AgentInfo(
                    id=root_id,
                    role="processing_node",
                    parent=None,
                    agent_parent_id=None,
                    description=self._get_universal_description(self.default_root_name),
                    children=main_agents,
                    message="coordination of system nodes",
                    agent_response=final_output,
                    agent_answer=final_output,
                    level=-1,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    is_final_answer=True,
                ).__dict__,
                "tool_call_info": [],
            }
        else:
            root_info = self._agents_log[self.default_root_name]["agent_info"]
            root_info.update(
                {
                    "children": main_agents,
                    "agent_response": final_output,
                    "agent_answer": final_output,
                    "is_final_answer": True,
                }
            )

        final_entry = ToolCallInfo(
            id="final_result",
            agent=self.default_root_name,
            agent_id=self._agents_log[self.default_root_name]["agent_info"]["id"],
            parent=None,
            agent_parent_id=None,
            tool="final_answer",
            tool_input="merging results of nodes",
            tool_answer=final_output,
            flow_direction="final",
            run_id="root_final",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            status="completed",
        ).__dict__

        self._agents_log[self.default_root_name]["tool_call_info"].append(final_entry)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handler for the start of tool work"""
        agent_name = serialized.get("name", serialized.get("id", "unknown"))
        self._process_agent_start(agent_name, input_str, run_id, parent_run_id)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Handler for the end of tool work"""
        if not self.behavior_tracker.agent_stack:
            return

        agent_name = self.behavior_tracker.agent_stack.pop()
        self._process_agent_end(agent_name, output, run_id, parent_run_id)

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Handler for tool error"""
        if not self.behavior_tracker.agent_stack:
            return

        agent_name = self.behavior_tracker.agent_stack.pop()

        if agent_name in self.agents_log:
            tool_calls = self.agents_log[agent_name]["tool_call_info"]
            for tool_call in reversed(tool_calls):
                if tool_call["status"] == "started" and tool_call["run_id"] == str(
                    run_id
                ):
                    tool_call.update(
                        {
                            "tool_answer": f"ERROR: {str(error)}",
                            "status": "error",
                            "end_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                    break

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Handler for agent action"""
        agent_name = action.tool
        input_str = (
            str(action.tool_input.get("query", action.tool_input))
            if isinstance(action.tool_input, dict)
            else str(action.tool_input)
        )
        self._process_agent_start(agent_name, input_str, run_id, parent_run_id)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Handler for the end of agent work"""
        if not self.behavior_tracker.agent_stack:
            return

        agent_name = self.behavior_tracker.agent_stack.pop()
        self.behavior_tracker.update_stats(agent_name, "end")

        output = (
            str(finish.return_values.get("output", finish.return_values))
            if isinstance(finish.return_values, dict)
            else str(finish.return_values)
        )

        if agent_name in self.agents_log:
            tool_calls = self.agents_log[agent_name]["tool_call_info"]
            original_call_id = None

            for tool_call in reversed(tool_calls):
                if tool_call["status"] == "started":
                    tool_call.update(
                        {
                            "tool_answer": output,
                            "status": "completed",
                            "end_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                    original_call_id = tool_call["id"]
                    break

            agent_info = self.agents_log[agent_name]["agent_info"]
            agent_info.update(
                {
                    "agent_response": output,
                    "agent_answer": output,
                }
            )

            parent_agent = (
                self.behavior_tracker.agent_stack[-1]
                if self.behavior_tracker.agent_stack
                else self.default_root_name
            )

            if original_call_id and parent_agent != self.default_root_name:
                agent_id_current = agent_info.get("id")
                return_entry = self._create_return_entry(
                    original_call_id,
                    agent_name,
                    parent_agent,
                    agent_id_current,
                    output,
                    run_id,
                    parent_run_id,
                )

                self.agents_log[agent_name]["tool_call_info"].append(return_entry)

                for tool_call in self.agents_log[agent_name]["tool_call_info"]:
                    if tool_call["id"] == original_call_id:
                        tool_call["result_sent_to"] = parent_agent
                        break

            agent_parent = agent_info.get("parent")

            if (
                agent_parent == self.default_root_name
                and self.default_root_name in self.agents_log
            ):
                children = self.agents_log[self.default_root_name]["agent_info"][
                    "children"
                ]
                if agent_name not in children:
                    children.append(agent_name)

            agent_info["next_call"] = (
                self.behavior_tracker.agent_stack[-1]
                if self.behavior_tracker.agent_stack
                else None
            )

            if not self.behavior_tracker.agent_stack:
                agent_info["is_final_answer"] = True
                self._create_final_root_entry(output)
                self._finalize_agent_roles()
