"""
Input validation and security utilities.
Handles prompt validation and security checks.
"""

import re
from typing import Tuple

from config import config


class SecurityValidator:
    """Handles input validation and security checks."""

    @staticmethod
    def validate_prompt(prompt: str) -> Tuple[bool, str]:
        """
        Validate prompt for security and constraints.

        Args:
            prompt: The prompt to validate

        Returns:
            Tuple of (is_valid, validation_message)
        """
        if not prompt or len(prompt.strip()) < config.min_prompt_length:
            return False, f"Prompt too short (minimum {config.min_prompt_length} characters)"

        if len(prompt) > config.max_prompt_length:
            return False, f"Prompt too long (maximum {config.max_prompt_length} characters)"

        # Check for potential injection patterns
        dangerous_patterns = [
            r"ignore\s+previous\s+instructions",
            r"system\s+prompt",
            r"<\s*script",
            r"javascript:",
            r"eval\s*\(",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return False, "Potentially dangerous content detected"

        return True, "Valid"
