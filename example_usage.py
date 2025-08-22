"""
Example usage of Recursive Companion MCP
This shows how the tool processes refinement requests
"""

# Example: Starting a refinement session
start_request = {
    "tool": "start_refinement",
    "arguments": {
        "prompt": "Write a technical specification for a user authentication system",
        "domain": "technical",  # optional, defaults to "auto"
    },
}

start_response = {
    "success": True,
    "session_id": "abc-123-def-456",
    "status": "started",
    "domain": "technical",
    "message": "Refinement session started. Use continue_refinement to proceed.",
    "next_action": "continue_refinement",
}

# Example: Continuing refinement (no session_id needed after recent improvements!)
continue_request = {
    "tool": "continue_refinement",
    "arguments": {},  # Uses current session automatically
}

continue_response = {
    "success": True,
    "status": "draft_complete",
    "iteration": 1,
    "progress": {
        "step": "2/11",
        "percent": 18,
        "current_action": "Analyzing for improvements",
        "iteration": "1/5",
        "convergence": "0.0%",
        "status_emoji": "🔍",
    },
    "message": "✍️ Initial draft generated. Ready for critiques.",
    "draft_preview": "# User Authentication System Specification...",
    "next_action": "continue_refinement",
    "continue_needed": True,
}

# Example: Quick refinement for simple questions
quick_request = {
    "tool": "quick_refine",
    "arguments": {"prompt": "What is OAuth 2.0?", "max_wait": 30},  # seconds
}

# The refinement process:
# 1. Generate initial draft (✍️)
# 2. Create parallel critiques (🔍)
# 3. Synthesize into revision (✨)
# 4. Check convergence score
# 5. Repeat until threshold met (✅)

print("See the README for complete tool documentation")
