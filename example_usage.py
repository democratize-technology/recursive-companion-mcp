"""
Example usage of Recursive Companion MCP in Python code
This shows how the tool processes refinement requests
"""

# Example request that would be sent to the MCP server:
example_request = {
    "tool": "refine_answer",
    "arguments": {
        "prompt": "Write a technical specification for a user authentication system",
        "domain": "technical",
        "convergence_threshold": 0.95,
        "max_iterations": 5,
        "urgency": "normal"
    }
}

# Example response structure:
example_response = {
    "success": True,
    "refined_answer": "# User Authentication System Specification\n\n## Overview\n[refined content here]...",
    "metadata": {
        "domain": "technical",
        "iterations": 3,
        "convergence_achieved": True,
        "final_similarity": 0.962,
        "convergence_threshold": 0.95,
        "elapsed_time_seconds": 12.5,
        "urgency": "normal",
        "model": "anthropic.claude-3-sonnet-20240229-v1:0"
    },
    "thinking_history": {
        "initial_draft": "[initial draft content]",
        "critiques": [
            "Critique 1: The security section needs more detail...",
            "Critique 2: Consider adding rate limiting...",
            "Critique 3: The API design should follow REST principles..."
        ],
        "revisions": [
            "[revision 1 content]",
            "[revision 2 content]",
            "[revision 3 content]"
        ],
        "similarity_scores": [0.832, 0.921, 0.962]
    }
}

# The tool works by:
# 1. Generating an initial draft based on the domain
# 2. Creating multiple critiques in parallel
# 3. Synthesizing critiques into a revision
# 4. Measuring similarity between versions
# 5. Repeating until convergence threshold is met

print("See this file for example request/response structures")