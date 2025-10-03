"""
Session management tools: list and get current session
"""

import logging
import sys

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from ..core import format_output, handle_tool_errors, mcp
from ..decorators import inject_client_context
from ..formatting import format_current_session, format_session_list
from .refinement import get_incremental_engine


@mcp.tool(description="List all active refinement sessions.")
@format_output
@handle_tool_errors
@inject_client_context
async def list_refinement_sessions(client_id: str = "default") -> str:
    """
    List all active refinement sessions.

    Args:
        client_id: Client identifier for session isolation (auto-injected)

    Returns:
        Formatted list of active sessions
    """
    engine, _ = get_incremental_engine()

    sessions = engine.session_manager.list_active_sessions()
    return format_session_list(sessions, len(sessions))


@mcp.tool(description="Get the current refinement session status without needing the ID")
@format_output
@handle_tool_errors
@inject_client_context
async def current_session(client_id: str = "default") -> str:
    """
    Get current session status.

    Args:
        client_id: Client identifier for session isolation (auto-injected)

    Returns:
        Formatted current session status
    """
    engine, tracker = get_incremental_engine()

    current_session_id = tracker.get_current_session()

    if not current_session_id:
        # Try to find the most recent session
        sessions = engine.session_manager.list_active_sessions()
        if sessions:
            recent = sessions[0]
            return format_current_session(
                {
                    "success": True,
                    "message": "No current session set, showing most recent",
                    "session": recent,
                }
            )

        return format_current_session(
            {"success": False, "message": "No active sessions. Start one with start_refinement."}
        )

    # Get status of current session
    result = await engine.get_status(current_session_id)
    return format_current_session(result)
