"""
Convenience tools: quick_refine for one-shot refinement
"""

import asyncio
import logging
import sys
import time

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from ..core import format_output, handle_tool_errors, mcp
from ..decorators import inject_client_context
from ..formatting import format_quick_refine
from .refinement import get_incremental_engine


@mcp.tool(
    description=(
        "Start and auto-continue a refinement until complete. "
        "Best for simple refinements that don't need step-by-step control."
    )
)
@format_output
@handle_tool_errors
@inject_client_context
async def quick_refine(
    prompt: str,
    max_wait: float = 30,
    client_id: str = "default",
) -> str:
    """
    Quick refinement with auto-continue.

    Args:
        prompt: The question to refine
        max_wait: Max seconds to wait (default: 30)
        client_id: Client identifier for session isolation (auto-injected)

    Returns:
        Formatted final refined answer or best result if timeout
    """
    engine, tracker = get_incremental_engine()

    # Start refinement
    start_result = await engine.start_refinement(prompt)
    if not start_result.get("success"):
        return format_quick_refine(start_result)

    session_id = start_result["session_id"]
    tracker.set_current_session(session_id, prompt)

    # Auto-continue until done or timeout
    start_time = time.time()
    iterations = 0
    last_preview = ""

    while (time.time() - start_time) < max_wait:
        continue_result = await engine.continue_refinement(session_id)
        iterations += 1

        # Track the latest draft preview
        if continue_result.get("draft_preview"):
            last_preview = continue_result["draft_preview"]
            logger.debug(f"Quick refine iteration {iterations}: preview length {len(last_preview)}")

        if continue_result.get("status") in ["completed", "converged"]:
            return format_quick_refine(
                {
                    "success": True,
                    "final_answer": continue_result.get("final_answer", ""),
                    "iterations": iterations,
                    "time_taken": round(time.time() - start_time, 1),
                    "convergence_score": continue_result.get("convergence_score", 0),
                }
            )

        await asyncio.sleep(0.1)  # Small delay between steps

    # Timeout - abort and return best so far
    abort_result = await engine.abort_refinement(session_id)
    return format_quick_refine(
        {
            "success": True,
            "status": "timeout",
            "message": f"Stopped after {max_wait}s",
            "final_answer": abort_result.get("final_answer", last_preview),
            "iterations": iterations,
            "time_taken": round(time.time() - start_time, 1),
            "convergence_score": 0,
        }
    )
