"""
UUID generation and management utilities.
"""

import uuid


def generate_uuid() -> str:
    """
    Generate a new UUID string.
    """
    return str(uuid.uuid4())


def generate_agent_uuid(agent_name: str) -> str:
    """
    Generate a UUID for an agent from its name.
    """
    return str(uuid.uuid3(uuid.NAMESPACE_DNS, agent_name))
