"""
Tracing processor for OpenAI Agents.
"""

import os

from agents.tracing import TracingProcessor
from agents.tracing.span_data import (
    AgentSpanData,
    FunctionSpanData,
    HandoffSpanData,
    ResponseSpanData,
)
from agents.tracing.spans import Span
from agents.tracing.traces import Trace
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage

from hivetrace import SyncHivetraceSDK
from hivetrace.adapters.openai_agents.adapter import OpenaiAgentsAdapter
from hivetrace.adapters.openai_agents.models import (
    AgentCall,
    Call,
    HandoffCall,
    ToolCall,
)
from hivetrace.utils.uuid_generator import generate_uuid


class HivetraceOpenAIAgentProcessor(TracingProcessor):
    """
    Tracing processor for OpenAI Agents.

    This class is responsible for tracing the execution of OpenAI Agents.
    It is used to log the traces of the agents and tools.

    Attributes:
        conversation_uuid: str
            The UUID of the conversation.

        adapter: OpenaiAgentsAdapter
            The adapter for the OpenAI Agents.

        _trace_calls: dict[str, Call | None]
            The trace calls.
    """

    conversation_uuid: str = ""

    def __init__(
        self,
        hivetrace_instance: SyncHivetraceSDK | None = None,
        application_id: str | None = os.getenv("HIVETRACE_APPLICATION_ID"),
    ):
        if not application_id:
            raise ValueError("HIVETRACE_APPLICATION_ID is not set")

        if not hivetrace_instance:
            hivetrace_instance = SyncHivetraceSDK(
                config={
                    "HIVETRACE_URL": os.getenv("HIVETRACE_URL"),
                    "HIVETRACE_ACCESS_TOKEN": os.getenv("HIVETRACE_ACCESS_TOKEN"),
                }
            )
        self.adapter = OpenaiAgentsAdapter(
            hivetrace=hivetrace_instance,
            application_id=application_id,
        )
        self._trace_calls: dict[str, Call | None] = {}

    def on_trace_start(self, trace: Trace):
        metadata = trace.metadata or {}
        user_id = metadata.get("user_id") or os.getenv("USER_ID")
        session_id = metadata.get("session_id") or os.getenv("SESSION_ID")
        self.adapter.user_id = str(user_id) if user_id else None
        self.adapter.session_id = str(session_id) if session_id else None
        self.conversation_uuid = generate_uuid()

    def on_trace_end(self, _: Trace):
        self.adapter.log_traces(self._trace_calls, self.conversation_uuid)

    def on_span_start(self, span: Span):
        if span.span_id not in self._trace_calls:
            # Save start of agent call
            if (
                isinstance(span.span_data, AgentSpanData)
                and span.span_data.type == "agent"
            ):
                self._trace_calls[span.span_id] = AgentCall(
                    span_parent_id=span.parent_id,
                    name=getattr(span.span_data, "name", "Unknown"),
                )

            # Save start of tool call
            elif (
                isinstance(span.span_data, FunctionSpanData)
                and span.span_data.type == "function"
            ):
                self._trace_calls[span.span_id] = ToolCall(
                    span_parent_id=span.parent_id,
                    name=getattr(span.span_data, "name", "Unknown"),
                )

    def on_span_end(self, span: Span):
        # Save end of handoff call
        if (
            isinstance(span.span_data, HandoffSpanData)
            and span.span_data.type == "handoff"
        ):
            if (
                span.span_data.from_agent is not None
                and span.span_data.to_agent is not None
            ):
                self._trace_calls[span.span_id] = HandoffCall(
                    span_parent_id=span.parent_id,
                    from_agent=span.span_data.from_agent,
                    to_agent=span.span_data.to_agent,
                )

        # Save input and output for tool
        if (
            isinstance(span.span_data, FunctionSpanData)
            and span.span_data.type == "function"
        ):
            call = self._trace_calls.get(span.span_id)
            if call is not None:
                call.input = span.span_data.input
                call.output = span.span_data.output

        # Save input and output for agent
        elif (
            isinstance(span.span_data, ResponseSpanData)
            and span.span_data.type == "response"
        ):
            response = span.span_data.response
            if not response or not response.output:
                return
            if isinstance(response.output[0], ResponseOutputMessage):
                parent_call = self._trace_calls.get(span.parent_id)
                if parent_call is None:
                    return
                parent_call.input = span.span_data.input[0]["content"]
                parent_call.output = response.output[0].content[0].text
                parent_call.instructions = response.instructions
            elif isinstance(response.output[0], ResponseFunctionToolCall):
                parent_call = self._trace_calls.get(span.parent_id)
                if parent_call is not None:
                    parent_call.instructions = response.instructions

    def shutdown(self):
        self._trace_calls = {}

    def force_flush(self):
        self._trace_calls = {}
