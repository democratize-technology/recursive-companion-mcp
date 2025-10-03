"""
Custom decorators for Recursive Companion MCP tools
"""

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


def inject_client_context(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to inject client_id context for multi-client scenarios.

    In a production environment with authentication, this would extract
    the authenticated user ID. For now, it provides a default client_id.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # If client_id not provided, use default
        if "client_id" not in kwargs:
            kwargs["client_id"] = "default"

        return await func(*args, **kwargs)  # type: ignore[misc, return-value]

    return wrapper  # type: ignore[misc, return-value]
