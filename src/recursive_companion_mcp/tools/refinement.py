"""
Refinement tools: start, continue, and get status
"""

import logging
import sys

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Import modules from package structure
from ..clients.bedrock import BedrockClient
from ..core import format_output, handle_tool_errors, mcp
from ..core.domains import DomainDetector
from ..core.session_manager import SessionTracker
from ..core.validation import SecurityValidator
from ..decorators import inject_client_context
from ..engines.incremental import IncrementalRefineEngine
from ..formatting import (
    format_refinement_continue,
    format_refinement_start,
    format_refinement_status,
)

# Initialize global engine (will be created on first use)
_incremental_engine = None
_session_tracker = None


def get_incremental_engine():
    """Lazy initialization of incremental engine"""
    global _incremental_engine, _session_tracker
    if _incremental_engine is None:
        bedrock_client = BedrockClient()
        domain_detector = DomainDetector()
        security_validator = SecurityValidator()

        _incremental_engine = IncrementalRefineEngine(
            bedrock_client, domain_detector, security_validator
        )
        _session_tracker = SessionTracker()

    return _incremental_engine, _session_tracker


@mcp.tool(
    description=(
        "Start a new incremental refinement session. "
        "Returns immediately with a session ID. "
        "Use continue_refinement to proceed step by step."
    )
)
@format_output
@handle_tool_errors
@inject_client_context
async def start_refinement(
    prompt: str,
    domain: str = "auto",
    client_id: str = "default",
) -> str:
    """
    Start a new refinement session.

    Args:
        prompt: The question or task to refine
        domain: Domain for specialized prompts (auto|technical|marketing|strategy|legal|financial|general)
        client_id: Client identifier for session isolation (auto-injected)

    Returns:
        Formatted string with session ID and next action instructions
    """
    engine, tracker = get_incremental_engine()

    result = await engine.start_refinement(prompt, domain)

    # Track current session for better UX
    if result.get("success"):
        tracker.set_current_session(result["session_id"], prompt)

    return format_refinement_start(result)


@mcp.tool(
    description=(
        "Continue an active refinement session by one step. "
        "Each call performs one action: draft, critique, or revise. "
        "If no session_id provided, continues the current session."
    )
)
@format_output
@handle_tool_errors
@inject_client_context
async def continue_refinement(
    session_id: str = "",
    client_id: str = "default",
) -> str:
    """
    Continue a refinement session by one step.

    Args:
        session_id: The refinement session ID (optional, uses current if not provided)
        client_id: Client identifier for session isolation (auto-injected)

    Returns:
        Formatted string with step results and convergence status
    """
    engine, tracker = get_incremental_engine()

    # Use current session if not specified
    if not session_id:
        session_id = tracker.get_current_session()

    if not session_id:
        active_sessions = engine.session_manager.list_active_sessions() if engine else []
        error_msg = "No session_id provided and no current session"

        return f"""❌ **{error_msg}**

**Current Session ID:** None
**Active Sessions:** {len(active_sessions)}

**Suggestion:** Use `start_refinement` to create a new session.

**Tip:** After `start_refinement`, `continue_refinement` will auto-track the session."""

    result = await engine.continue_refinement(session_id)
    return format_refinement_continue(result)


@mcp.tool(description="Get the current status of a refinement session.")
@format_output
@handle_tool_errors
@inject_client_context
async def get_refinement_status(
    session_id: str,
    client_id: str = "default",
) -> str:
    """
    Get refinement session status.

    Args:
        session_id: The refinement session ID
        client_id: Client identifier for session isolation (auto-injected)

    Returns:
        Formatted string with session status and progress
    """
    engine, _ = get_incremental_engine()

    result = await engine.get_status(session_id)
    return format_refinement_status(result)
