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
Error handling utilities with AI-assistant-friendly responses.
Provides structured error responses that help AI assistants understand and recover from errors.
"""

import logging
from typing import Any

from config import config

logger = logging.getLogger(__name__)


def create_ai_error_response(error: Exception, context: str) -> dict[str, Any]:
    """
    Create error response with AI-actionable hints.

    Args:
        error: The exception that occurred
        context: Context about where the error occurred

    Returns:
        Dict with error details and AI-friendly recovery hints
    """
    error_type = type(error).__name__
    error_msg = str(error)

    # Common error patterns with AI-helpful diagnostics
    response = {"success": False, "error": error_msg, "error_type": error_type, "context": context}

    # AWS credential errors
    if "credentials" in error_msg.lower() or "aws" in error_msg.lower():
        response.update(
            {
                "_ai_diagnosis": "AWS credentials issue detected",
                "_ai_actions": [
                    "Check if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set",
                    f"Verify AWS_REGION is correct (current: {config.aws_region})",
                    "Test with: aws sts get-caller-identity",
                ],
                "_human_action": "Run 'aws configure' to set up AWS credentials",
            }
        )

    # Bedrock model errors
    elif "ResourceNotFoundException" in error_msg or "model" in error_msg.lower():
        response.update(
            {
                "_ai_diagnosis": "AWS Bedrock model not available",
                "_ai_context": {
                    "current_region": config.aws_region,
                    "requested_model": config.bedrock_model_id,
                    "critique_model": config.critique_model_id,
                },
                "_ai_suggestion": "Try us-east-1 or us-west-2 regions",
                "_human_action": "Change AWS_REGION in .env or enable model in AWS console",
            }
        )

    # Timeout errors
    elif error_type == "TimeoutError":
        response.update(
            {
                "_ai_diagnosis": "Operation exceeded timeout",
                "_ai_suggestion": "For long refinements, use quick_refine with higher max_wait",
                "_ai_alternative": "Or use start_refinement + continue_refinement for control",
                "_human_action": "Try quick_refine with max_wait=60",
            }
        )

    # Session errors
    elif "session" in error_msg.lower() or error_type == "KeyError":
        response.update(
            {
                "_ai_diagnosis": "Session not found or invalid",
                "_ai_suggestion": "Check active sessions with list_refinement_sessions",
                "_ai_recovery": "Start fresh with start_refinement",
                "_human_action": "Verify session ID or start a new session",
            }
        )

    else:
        # Generic helpful hints
        response.update(
            {
                "_ai_diagnosis": f"Unexpected error in {context}",
                "_ai_suggestion": "Check server logs for details",
                "_ai_context": {"error_type": error_type},
            }
        )

    return response
