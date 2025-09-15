"""Progress tracking and status management for refinement sessions.

This module provides utilities for tracking refinement progress,
formatting status information, and managing session state displays.
"""

from typing import Any

from refinement_types import RefinementSession, RefinementStatus


class ProgressTracker:
    """Manages progress tracking and status formatting for refinement sessions."""

    @staticmethod
    def format_progress(session: "RefinementSession") -> dict[str, Any]:
        """Create detailed progress information for a refinement session.

        Args:
            session: The refinement session to track

        Returns:
            Dictionary containing progress metrics and status information
        """
        # Estimate steps: draft(1) + (critique(1) + revise(1)) * iterations
        estimated_total_steps = 1 + (2 * session.max_iterations)
        current_step = 1 + (2 * session.current_iteration)

        if session.status == RefinementStatus.DRAFTING:
            current_step = 1
        elif session.status == RefinementStatus.CRITIQUING:
            current_step = 2 + (2 * (session.current_iteration - 1))
        elif session.status == RefinementStatus.REVISING:
            current_step = 3 + (2 * (session.current_iteration - 1))

        return {
            "step": f"{current_step}/{estimated_total_steps}",
            "percent": round((current_step / estimated_total_steps) * 100),
            "current_action": ProgressTracker.get_action_description(session.status),
            "iteration": f"{session.current_iteration}/{session.max_iterations}",
            "convergence": f"{session.convergence_score:.1%}",
            "status_emoji": ProgressTracker.get_status_emoji(session.status),
        }

    @staticmethod
    def get_action_description(status: "RefinementStatus") -> str:
        """Get human-friendly action descriptions for refinement status.

        Args:
            status: The current refinement status

        Returns:
            Human-readable description of the current action
        """
        descriptions = {
            RefinementStatus.INITIALIZING: "Starting refinement process",
            RefinementStatus.DRAFTING: "Creating initial draft",
            RefinementStatus.CRITIQUING: "Analyzing draft for improvements",
            RefinementStatus.REVISING: "Incorporating feedback",
            RefinementStatus.CONVERGED: "Refinement complete - convergence achieved",
            RefinementStatus.ABORTED: "Refinement aborted by user",
            RefinementStatus.TIMEOUT: "Maximum iterations reached",
            RefinementStatus.ERROR: "Error occurred during refinement",
        }
        return descriptions.get(status, "Processing")

    @staticmethod
    def get_status_emoji(status: "RefinementStatus") -> str:
        """Get fun status indicator emojis for refinement status.

        Args:
            status: The current refinement status

        Returns:
            Emoji representing the current status
        """
        emojis = {
            RefinementStatus.INITIALIZING: "🚀",
            RefinementStatus.DRAFTING: "📝",
            RefinementStatus.CRITIQUING: "🔍",
            RefinementStatus.REVISING: "✏️",
            RefinementStatus.CONVERGED: "✅",
            RefinementStatus.ERROR: "❌",
            RefinementStatus.ABORTED: "🛑",
            RefinementStatus.TIMEOUT: "⏱️",
        }
        return emojis.get(status, "⏳")
