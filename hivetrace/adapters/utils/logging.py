"""
Module for common logging utilities used across adapters.
"""

from typing import Any, Dict, Optional


def process_agent_params(
    additional_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process and format agent parameters for consistent logging.

    Ensures agent information is properly formatted for HiveTrace.

    Parameters:
    - additional_params: The parameters to process, potentially containing agent information

    Returns:
    - Processed parameters with properly formatted agent information
    """
    processed_params = additional_params or {}

    if "agents" in processed_params and processed_params["agents"]:
        agent_uuid = next(iter(processed_params["agents"]))
        agent_info_val = processed_params["agents"][agent_uuid]
        if isinstance(agent_info_val, dict) and "name" in agent_info_val:
            processed_params["agents"] = {
                agent_uuid: {
                    "name": agent_info_val["name"],
                    "description": agent_info_val.get("description", ""),
                }
            }

    return processed_params
