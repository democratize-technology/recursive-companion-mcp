"""
Response formatting utilities for LLM-optimized output
"""

from typing import Any


def format_refinement_start(result: dict[str, Any]) -> str:
    """Format start_refinement response"""
    if not result.get("success"):
        return f"❌ **Error**: {result.get('error', 'Unknown error')}"

    session_id = result.get("session_id", "N/A")
    domain = result.get("domain", "auto")
    status = result.get("status", "started")

    return f"""✅ **Refinement Session Started**

**Session ID:** `{session_id}`
**Status:** {status}
**Domain:** {domain}

**Next Action:** Use `continue_refinement` to proceed with refinement cycles.

*Session ID: {session_id}*"""


def format_refinement_continue(result: dict[str, Any]) -> str:
    """Format continue_refinement response"""
    if not result.get("success"):
        return f"❌ **Error**: {result.get('error', 'Unknown error')}"

    session_id = result.get("session_id", "N/A")
    status = result.get("status", "in_progress")
    current_step = result.get("current_step", "")
    iteration = result.get("iteration", 0)
    convergence_score = result.get("convergence_score", 0.0)

    output = f"""✅ **Refinement Step Completed**

**Session ID:** `{session_id}`
**Status:** {status}
**Current Step:** {current_step}
**Iteration:** {iteration}
**Convergence Score:** {convergence_score:.2%}
"""

    # Add draft preview if available
    if result.get("draft_preview"):
        preview = result["draft_preview"][:200]
        output += f"\n**Draft Preview:**\n{preview}...\n"

    # Add completion notice if converged
    if status in ["completed", "converged"]:
        output += "\n✨ **Refinement has converged!** Use `get_final_result` to retrieve the refined answer.\n"
    else:
        output += "\n**Next:** Use `continue_refinement` again to continue refining.\n"

    output += f"\n---\n*Session ID: {session_id}*"
    return output


def format_refinement_status(result: dict[str, Any]) -> str:
    """Format get_refinement_status response"""
    if not result.get("success"):
        return f"❌ **Error**: {result.get('error', 'Unknown error')}"

    session_id = result.get("session_id", "N/A")
    status = result.get("status", "unknown")
    iteration = result.get("iteration", 0)
    convergence_score = result.get("convergence_score", 0.0)

    return f"""📊 **Refinement Status**

**Session ID:** `{session_id}`
**Status:** {status}
**Iteration:** {iteration}
**Convergence Score:** {convergence_score:.2%}

*Session ID: {session_id}*"""


def format_final_result(result: dict[str, Any]) -> str:
    """Format get_final_result response"""
    if not result.get("success"):
        return f"❌ **Error**: {result.get('error', 'Unknown error')}"

    session_id = result.get("session_id", "N/A")
    final_answer = result.get("final_answer", "")
    iterations = result.get("iterations", 0)
    convergence_score = result.get("convergence_score", 0.0)

    return f"""✨ **Final Refined Answer**

**Session ID:** `{session_id}`
**Iterations:** {iterations}
**Convergence Score:** {convergence_score:.2%}

---

{final_answer}

---
*Session ID: {session_id}*"""


def format_session_list(sessions: list[dict[str, Any]], count: int) -> str:
    """Format list_refinement_sessions response"""
    if count == 0:
        return "📋 **No Active Sessions**\n\nNo refinement sessions are currently active. Use `start_refinement` to create one."

    output = [f"📋 **Active Refinement Sessions** ({count})"]
    output.append("")

    for i, session in enumerate(sessions[:10], 1):  # Limit to 10
        session_id = session.get("session_id", "N/A")
        status = session.get("status", "unknown")
        created = session.get("created_at", "")

        output.append(f"{i}. **{session_id[:8]}...** - Status: {status} (Created: {created})")

    if count > 10:
        output.append(f"\n*... and {count - 10} more sessions*")

    return "\n".join(output)


def format_current_session(result: dict[str, Any]) -> str:
    """Format current_session response"""
    if result.get("success") is False:
        return result.get("message", "❌ No current session")

    # If showing most recent
    if result.get("message") and "most recent" in result["message"]:
        session = result.get("session", {})
        session_id = session.get("session_id", "N/A")
        status = session.get("status", "unknown")

        return f"""📌 **Most Recent Session** (no current session set)

**Session ID:** `{session_id}`
**Status:** {status}

*This is the most recent session, but not explicitly set as current.*"""

    # Current session status
    return format_refinement_status(result)


def format_abort_result(result: dict[str, Any]) -> str:
    """Format abort_refinement response"""
    if not result.get("success"):
        return f"❌ **Error**: {result.get('error', 'Unknown error')}"

    session_id = result.get("session_id", "N/A")
    final_answer = result.get("final_answer", "")
    iterations = result.get("iterations", 0)

    return f"""⏹️ **Refinement Aborted**

**Session ID:** `{session_id}`
**Iterations Completed:** {iterations}

**Best Result So Far:**

{final_answer}

*Session ID: {session_id}*"""


def format_quick_refine(result: dict[str, Any]) -> str:
    """Format quick_refine response"""
    if not result.get("success"):
        return f"❌ **Error**: {result.get('error', 'Unknown error')}"

    final_answer = result.get("final_answer", "")
    iterations = result.get("iterations", 0)
    time_taken = result.get("time_taken", 0)
    convergence_score = result.get("convergence_score", 0.0)

    status = result.get("status", "completed")
    if status == "timeout":
        header = "⏱️ **Quick Refinement** (Timeout)"
        note = result.get("message", "Stopped after time limit")
    else:
        header = "✨ **Quick Refinement Complete**"
        note = None

    output = f"""{header}

**Iterations:** {iterations}
**Time Taken:** {time_taken}s
**Convergence Score:** {convergence_score:.2%}
"""

    if note:
        output += f"\n{note}\n"

    output += f"""
---

{final_answer}
"""

    return output
