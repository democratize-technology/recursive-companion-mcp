"""
Results tool: get final refined answer
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
from ..formatting import format_final_result
from .refinement import get_incremental_engine


@mcp.tool(description="Get the final refined answer once convergence is achieved.")
@format_output
@handle_tool_errors
@inject_client_context
async def get_final_result(
    session_id: str,
    client_id: str = "default",
) -> str:
    """
    Get the final refined answer.

    Args:
        session_id: The refinement session ID
        client_id: Client identifier for session isolation (auto-injected)

    Returns:
        Formatted string with final refined answer
    """
    engine, _ = get_incremental_engine()

    result = await engine.get_final_result(session_id)
    return format_final_result(result)
