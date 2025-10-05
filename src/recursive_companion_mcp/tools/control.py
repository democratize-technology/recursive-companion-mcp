"""
Control tools: abort refinement
"""

import logging
import sys

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from ..core import format_output, handle_tool_errors
from ..core.server import mcp  # Import mcp directly from server to avoid circular import
from ..decorators import inject_client_context
from ..formatting import format_abort_result
from .refinement import get_incremental_engine


@mcp.tool(description="Stop refinement and get the best result so far")
@format_output
@handle_tool_errors
@inject_client_context
async def abort_refinement(
    session_id: str = "",
    client_id: str = "default",
) -> str:
    """
    Abort refinement and get best result.

    Args:
        session_id: Optional session ID (uses current if not provided)
        client_id: Client identifier for session isolation (auto-injected)

    Returns:
        Formatted best result so far
    """
    engine, tracker = get_incremental_engine()

    # Use current session if not specified
    if not session_id:
        session_id = tracker.get_current_session()

    if not session_id:
        return "❌ **No session specified and no current session active**"

    result = await engine.abort_refinement(session_id)
    return format_abort_result(result)
