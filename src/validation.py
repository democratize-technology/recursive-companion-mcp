#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2025 Jeremy
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
Input validation and security utilities.
Handles prompt validation and security checks.
"""

import re

from config import config


class SecurityValidator:
    """Handles input validation and security checks."""

    @staticmethod
    def validate_prompt(prompt: str) -> tuple[bool, str]:
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
