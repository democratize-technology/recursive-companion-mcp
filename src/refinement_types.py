"""Shared type definitions for the refinement engine."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RefinementStatus(Enum):
    """Status of a refinement session"""

    INITIALIZING = "initializing"
    DRAFTING = "drafting"
    CRITIQUING = "critiquing"
    REVISING = "revising"
    CONVERGED = "converged"
    ERROR = "error"
    ABORTED = "aborted"
    TIMEOUT = "timeout"


@dataclass
class RefinementSession:
    """Represents an active refinement session"""

    session_id: str
    prompt: str
    domain: str
    status: RefinementStatus
    current_iteration: int
    max_iterations: int
    convergence_threshold: float
    current_draft: str = ""
    previous_draft: str = ""
    critiques: list[str] = field(default_factory=list)
    convergence_score: float = 0.0
    iterations_history: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary for JSON serialization"""
        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "domain": self.domain,
            "status": self.status.value,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "convergence_score": round(self.convergence_score, 4),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error_message": self.error_message,
            "metadata": self.metadata,
            "draft_preview": (
                self.current_draft[:200] + "..."
                if len(self.current_draft) > 200
                else self.current_draft
            ),
            "iterations_completed": len(self.iterations_history),
        }
