import uuid

from hivetrace.adapters.base_adapter import BaseAdapter
from hivetrace.adapters.openai_agents.models import AgentCall, Call


class OpenaiAgentsAdapter(BaseAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_traces(self, trace_calls: dict[str, Call | None], conversation_uuid: str):
        _trace_calls = self._join_handoff_spans(trace_calls)
        _trace_calls = self._join_agent_calling_tool_spans(_trace_calls)
        source_agent: AgentCall | None = None
        for call in _trace_calls.values():
            if isinstance(call, AgentCall):
                source_agent = call
                break
        if source_agent is None:
            return

        self._log_start_message(source_agent, conversation_uuid)

        for trace_call in _trace_calls.values():
            if trace_call is None or trace_call.span_parent_id is None:
                continue

            parent = _trace_calls.get(trace_call.span_parent_id)
            if not isinstance(parent, AgentCall):
                continue

            if isinstance(trace_call, AgentCall):
                if trace_call.output is None:
                    continue
                additional_params = {
                    "agent_conversation_id": conversation_uuid,
                    "is_final_answer": False,
                    "agents": {
                        trace_call.agent_uuid: {
                            "agent_parent_id": parent.agent_uuid,
                            "name": trace_call.name,
                            "description": trace_call.instructions,
                        },
                    },
                }
                self.output(
                    message=trace_call.output,
                    additional_params=additional_params,
                )

            elif trace_call.type == "tool":
                if trace_call.output is None:
                    continue
                self._prepare_and_log(
                    log_method_name_stem="function_call",
                    is_async=False,
                    tool_call_details={
                        "application_id": self.application_id,
                        "tool_call_id": str(uuid.uuid4()),
                        "func_name": trace_call.name,
                        "func_args": f"{trace_call.input}",
                        "func_result": f"{trace_call.output}",
                        "additional_parameters": {
                            "agent_conversation_id": conversation_uuid,
                            "agents": {
                                parent.agent_uuid: {
                                    "name": parent.name,
                                    "description": parent.instructions,
                                },
                            },
                        },
                    },
                )
        self._log_final_message(source_agent, conversation_uuid)

    def _join_agent_calling_tool_spans(
        self, trace_calls: dict[str, Call | None]
    ) -> dict[str, Call | None]:
        for span_id, span in trace_calls.items():
            if span is None:
                continue
            if span.type == "agent" and span.span_parent_id is not None:
                parent = trace_calls[span.span_parent_id]
                if parent is None:
                    continue
                if parent.type == "tool":
                    trace_calls[span.span_parent_id] = None
                    span.span_parent_id = parent.span_parent_id
                    span.input = (
                        parent.input if span.input is None else span.input
                    )
                    span.output = (
                        parent.output if span.output is None else span.output
                    )
        return trace_calls

    def _join_handoff_spans(
        self, trace_calls: dict[str, Call | None]
    ) -> dict[str, Call | None]:
        for span in reversed(trace_calls.values()):
            if span is None:
                continue
            if span.type == "handoff" and span.span_parent_id is not None:
                parent = trace_calls[span.span_parent_id]
                child = next(
                    (
                        call
                        for call in trace_calls.values()
                        if call is not None and call.name == span.to_agent
                    ),
                    None,
                )
                if parent is None:
                    continue
                if child is None:
                    continue
                child.span_parent_id = span.span_parent_id
                if parent.output is None:
                    parent.output = child.output
                if parent.input is None:
                    parent.input = child.input
        return trace_calls

    def _log_start_message(self, trace_call: AgentCall, conversation_uuid: str):
        if trace_call.input is None:
            return
        self.input(
            message=trace_call.input,
            additional_params={
                "agent_conversation_id": conversation_uuid,
                "agents": {
                    trace_call.agent_uuid: {
                        "name": trace_call.name,
                        "description": trace_call.instructions,
                    },
                },
            },
        )

    def _log_final_message(self, trace_call: AgentCall, conversation_uuid: str):
        if trace_call.output is None:
            return
        self.output(
            message=trace_call.output,
            additional_params={
                "agent_conversation_id": conversation_uuid,
                "is_final_answer": True,
                "agents": {
                    trace_call.agent_uuid: {
                        "name": trace_call.name,
                        "description": trace_call.instructions,
                    },
                },
            },
        )
