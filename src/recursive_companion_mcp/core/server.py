"""
MCP Server setup and core decorators for Recursive Companion
"""

import logging
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from mcp.server.fastmcp import FastMCP

# Configure logging to stderr only - NEVER stdout in MCP servers
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Type variable for decorators
T = TypeVar("T")

# Create FastMCP instance
mcp = FastMCP("recursive-companion")


def handle_tool_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to handle standard error patterns for MCP tools.

    Provides consistent error handling and logging across all tool functions.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_name = func.__name__
        try:
            return await func(*args, **kwargs)  # type: ignore[misc, return-value]
        except ValueError as e:
            logger.warning(f"Validation error in {tool_name}: {e}")
            return f"❌ **Invalid input**: {str(e)}"
        except KeyError as e:
            logger.error(f"Configuration error in {tool_name}: Missing key {e}")
            return f"❌ **Configuration error**: Missing required field {e}. Check your environment variables."
        except ImportError as e:
            logger.error(f"Import error in {tool_name}: {e}")
            error_msg = f"❌ **Missing dependency**: {str(e)}. "
            if "boto3" in str(e):
                error_msg += "Install with: pip install boto3"
            return error_msg
        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}")
            return f"❌ **Unexpected error in {tool_name}**: {type(e).__name__}: {str(e)}"

    return wrapper  # type: ignore[misc, return-value]


def format_output(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to format tool output for optimal LLM consumption.

    Converts dict results to formatted strings for better LLM parsing.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = await func(*args, **kwargs)  # type: ignore[misc, return-value]

        # If already a string (formatted or error), return as-is
        if isinstance(result, str):
            return result

        # If dict result from legacy functions, convert to string
        if isinstance(result, dict):
            if result.get("error"):
                return f"❌ **Error**: {result['error']}"

            # Format success response
            if result.get("success"):
                session_id = result.get("session_id", "")
                status = result.get("status", "")

                output = []
                output.append("✅ **Success**")

                if session_id:
                    output.append(f"\n**Session ID:** `{session_id}`")
                if status:
                    output.append(f"**Status:** {status}")

                # Add any message
                if result.get("message"):
                    output.append(f"\n{result['message']}")

                # Add session ID footer
                if session_id:
                    output.append(f"\n---\n*Session ID: {session_id}*")

                return "\n".join(output)

            # Fallback to JSON representation
            import json

            return json.dumps(result, indent=2)

        # Return other types as-is
        return result

    return wrapper  # type: ignore[misc, return-value]


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
