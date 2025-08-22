#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2025 Recursive Companion Contributors
# Based on work by Hank Besser (https://github.com/hankbesser/recursive-companion)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Session management for refinement operations.
Tracks current and recent sessions for better UX.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config import config

logger = logging.getLogger(__name__)


@dataclass
class RefinementIteration:
    """Represents a single iteration in the refinement process."""

    iteration_number: int
    draft: str
    critiques: list[str]
    revision: str
    convergence_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RefinementResult:
    """Complete result of the refinement process."""

    final_answer: str
    domain: str
    iterations: list[RefinementIteration]
    total_iterations: int
    convergence_achieved: bool
    execution_time: float
    metadata: dict[str, Any]


class SessionTracker:
    """Tracks active and recent refinement sessions."""

    def __init__(self):
        self.current_session_id: str | None = None
        self.session_history: list[dict[str, Any]] = []
        self.max_history = 5

    def set_current_session(self, session_id: str, prompt: str) -> None:
        """
        Set the current active session.

        Args:
            session_id: The session ID
            prompt: The prompt for this session
        """
        self.current_session_id = session_id

        # Add to history
        preview = (
            prompt[: config.prompt_preview_length] + "..."
            if len(prompt) > config.prompt_preview_length
            else prompt
        )
        self.session_history.insert(
            0,
            {
                "session_id": session_id,
                "prompt_preview": preview,
                "started_at": datetime.utcnow().isoformat(),
            },
        )

        # Trim history
        if len(self.session_history) > self.max_history:
            self.session_history = self.session_history[: self.max_history]

    def get_current_session(self) -> str | None:
        """Get the current session ID."""
        return self.current_session_id

    def get_session_history(self) -> list[dict[str, Any]]:
        """Get the session history."""
        return self.session_history

    def clear_current_session(self) -> None:
        """Clear the current session."""
        self.current_session_id = None
