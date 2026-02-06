from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

from .adapter import LangChainAdapter


@contextmanager
def _ensure_hivetrace_client(hivetrace_instance: Any):
    """Yield provided hivetrace client or create a temporary SyncHivetraceSDK from ENV."""
    if hivetrace_instance is not None:
        yield hivetrace_instance
        return

    try:
        from hivetrace import SyncHivetraceSDK
    except Exception as exc:
        raise RuntimeError(
            "SyncHivetraceSDK is not available. Install hivetrace or provide 'hivetrace' instance."
        ) from exc

    client = SyncHivetraceSDK()
    try:
        yield client
    finally:
        if hasattr(client, "close"):
            client.close()


def run_with_tracing(
    orchestrator: Any,
    query: str,
    *,
    hivetrace: Any | None = None,
    application_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    agents_mapping: Optional[Dict[str, Dict[str, str]]] = None,
    conversation_id: Optional[str] = None,
    input_additional_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Execute LangChain orchestrator with HiveTrace monitoring in one call.

    Steps:
    1) Log user input to HiveTrace (with agents mapping and metadata)
    2) Run orchestrator.run(query)
    3) Send LangChain logs from orchestrator.logging_callback

    Returns orchestrator result.
    """

    if not hasattr(orchestrator, "run"):
        raise ValueError("orchestrator must have a .run(str) method")

    callback = getattr(orchestrator, "logging_callback", None)
    if callback is None:
        raise ValueError(
            "orchestrator must expose 'logging_callback' for sending LangChain logs"
        )

    conv_id = conversation_id or str(uuid.uuid4())

    additional: Dict[str, Any] = {
        "agent_conversation_id": conv_id,
        "user_id": user_id,
        "session_id": session_id,
    }

    if agents_mapping is None:
        try:
            predefined = getattr(callback, "predefined_agent_ids", None) or {}
            if isinstance(predefined, dict) and predefined:
                additional["agents"] = {
                    agent_uuid: {"name": agent_name, "description": ""}
                    for agent_name, agent_uuid in predefined.items()
                }
        except Exception:
            pass
    else:
        additional["agents"] = agents_mapping

    if input_additional_params:
        additional.update(input_additional_params)

    app_id = application_id or os.getenv("HIVETRACE_APP_ID")
    if not app_id:
        raise ValueError(
            "application_id is required (pass parameter or set HIVETRACE_APP_ID in environment)"
        )

    with _ensure_hivetrace_client(hivetrace) as ht_client:
        adapter = LangChainAdapter(
            hivetrace=ht_client,
            application_id=app_id,
            user_id=user_id,
            session_id=session_id,
        )

        adapter._forced_agent_conversation_id = conv_id

        adapter.input(query, additional)
        result = orchestrator.run(query)

        try:
            root_name = getattr(callback, "default_root_name", None)
            agents_log = getattr(callback, "agents_log", {}) or {}
            if root_name and root_name not in agents_log:
                if hasattr(callback, "_create_final_root_entry"):
                    callback._create_final_root_entry(str(result))
                if hasattr(callback, "_finalize_agent_roles"):
                    callback._finalize_agent_roles()
        except Exception:
            pass

        try:
            root_name = getattr(callback, "default_root_name", None)
            agents_log = getattr(callback, "agents_log", {}) or {}
            root_info = (
                agents_log.get(root_name, {}).get("agent_info", {}) if root_name else {}
            )
            root_id = root_info.get("id")
            if root_id:
                final_additional = {
                    "agents": {
                        root_id: {"name": root_name or "Root", "description": ""}
                    },
                    "agent_conversation_id": conv_id,
                    "is_final_answer": True,
                }
                adapter.output(str(result), final_additional)
        except Exception:
            pass

        adapter.send_log_data(callback)

        return result


async def run_with_tracing_async(
    orchestrator: Any,
    query: str,
    *,
    hivetrace: Any | None = None,
    application_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    agents_mapping: Optional[Dict[str, Dict[str, str]]] = None,
    conversation_id: Optional[str] = None,
    input_additional_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Async variant: uses AsyncHivetraceSDK semantics and orchestrator.arun.
    """

    if not hasattr(orchestrator, "arun"):
        raise ValueError(
            "orchestrator must have an .arun(str) coroutine for async integration"
        )

    callback = getattr(orchestrator, "logging_callback", None)
    if callback is None:
        raise ValueError(
            "orchestrator must expose 'logging_callback' for sending LangChain logs"
        )

    conv_id = conversation_id or str(uuid.uuid4())

    additional: Dict[str, Any] = {
        "agent_conversation_id": conv_id,
        "user_id": user_id,
        "session_id": session_id,
    }

    if agents_mapping is None:
        try:
            predefined = getattr(callback, "predefined_agent_ids", None) or {}
            if isinstance(predefined, dict) and predefined:
                additional["agents"] = {
                    agent_uuid: {"name": agent_name, "description": ""}
                    for agent_name, agent_uuid in predefined.items()
                }
        except Exception:
            pass
    else:
        additional["agents"] = agents_mapping

    if input_additional_params:
        additional.update(input_additional_params)

    app_id = application_id or os.getenv("HIVETRACE_APP_ID")
    if not app_id:
        raise ValueError(
            "application_id is required (pass parameter or set HIVETRACE_APP_ID in environment)"
        )

    if hivetrace is None:
        from hivetrace import AsyncHivetraceSDK

        async with AsyncHivetraceSDK() as ht_client:
            adapter = LangChainAdapter(
                hivetrace=ht_client,
                application_id=app_id,
                user_id=user_id,
                session_id=session_id,
            )
            adapter._forced_agent_conversation_id = conv_id
            await ht_client.input(app_id, query, additional)
            result = await orchestrator.arun(query)
            try:
                root_name = getattr(callback, "default_root_name", None)
                agents_log = getattr(callback, "agents_log", {}) or {}
                root_info = (
                    agents_log.get(root_name, {}).get("agent_info", {})
                    if root_name
                    else {}
                )
                root_id = root_info.get("id")
                if root_id:
                    final_additional = {
                        "agents": {
                            root_id: {"name": root_name or "Root", "description": ""}
                        },
                        "agent_conversation_id": conv_id,
                        "is_final_answer": True,
                    }
                    await ht_client.output(app_id, str(result), final_additional)
            except Exception:
                pass
            await adapter.send_log_data_async(callback)
            return result
    else:
        adapter = LangChainAdapter(
            hivetrace=hivetrace,
            application_id=app_id,
            user_id=user_id,
            session_id=session_id,
        )
        adapter._forced_agent_conversation_id = conv_id
        await hivetrace.input(app_id, query, additional)
        result = await orchestrator.arun(query)
        try:
            root_name = getattr(callback, "default_root_name", None)
            agents_log = getattr(callback, "agents_log", {}) or {}
            root_info = (
                agents_log.get(root_name, {}).get("agent_info", {}) if root_name else {}
            )
            root_id = root_info.get("id")
            if root_id:
                final_additional = {
                    "agents": {
                        root_id: {"name": root_name or "Root", "description": ""}
                    },
                    "agent_conversation_id": conv_id,
                    "is_final_answer": True,
                }
                await hivetrace.output(app_id, str(result), final_additional)
        except Exception:
            pass
        await adapter.send_log_data_async(callback)
        return result
