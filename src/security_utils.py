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
Security utilities for sanitizing sensitive information from logs and error messages.
"""

import re
from typing import Any


class CredentialSanitizer:
    """Comprehensive sanitizer for AWS credentials and sensitive data."""

    # Patterns for various AWS credential formats
    PATTERNS = {
        "aws_access_key": re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{7,}", re.IGNORECASE),
        "aws_secret_key": re.compile(
            r"(?:aws_secret_access_key|secret_key|SecretAccessKey)[\s=:]+[\"\']?([A-Za-z0-9+/]{40})[\"\']?",
            re.IGNORECASE,
        ),
        "aws_session_token": re.compile(
            r"(?:aws_session_token|session_token|SessionToken)[\s=:]+[\"\']?([A-Za-z0-9+/=]{100,})[\"\']?",
            re.IGNORECASE,
        ),
        "arn": re.compile(r"arn:aws:iam::\d{12}:(?:user|role)/[^\s]+", re.IGNORECASE),
        "authorization_header": re.compile(
            r"(?:Authorization|X-Amz-Security-Token)[\s:]+[\"\']?([^\s\"\']+)[\"\']?",
            re.IGNORECASE,
        ),
        "base64_credentials": re.compile(
            r"(?:Basic|Bearer)\s+([A-Za-z0-9+/]{20,}={0,2})", re.IGNORECASE
        ),
        "generic_api_key": re.compile(
            r"(?:api[_-]?key|apikey|api_secret)[\s=:]+[\"\']?([A-Za-z0-9_\-]+)[\"\']?",
            re.IGNORECASE,
        ),
    }

    # Sensitive field names to redact in structured data
    SENSITIVE_FIELDS = {
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "access_key",
        "secret_key",
        "private_key",
        "client_secret",
        "authorization",
        "auth",
        "credentials",
        "certificate",
        "x-amz-security-token",
        "x-api-key",
    }

    @classmethod
    def sanitize_string(cls, text: str, replacement: str = "[REDACTED]") -> str:
        """
        Sanitize sensitive information from a string.

        Args:
            text: String to sanitize
            replacement: Replacement text for sensitive data

        Returns:
            Sanitized string
        """
        if not text:
            return text

        sanitized = text

        # Apply all regex patterns
        for pattern_name, pattern in cls.PATTERNS.items():
            if pattern_name in ["aws_access_key", "arn"]:
                # Direct replacement for patterns without capture groups
                sanitized = pattern.sub(f"[REDACTED_{pattern_name.upper()}]", sanitized)
            else:
                # Replace captured groups
                sanitized = pattern.sub(
                    lambda m, name=pattern_name: m.group(0).replace(
                        m.group(1) if m.lastindex else m.group(0),
                        f"[REDACTED_{name.upper()}]",
                    ),
                    sanitized,
                )

        # Additional safety: redact any remaining long base64-like strings
        # that might be credentials
        sanitized = re.sub(
            r"(?<![A-Za-z0-9+/])([A-Za-z0-9+/]{35,}={0,2})(?![A-Za-z0-9+/])",
            "[REDACTED_POSSIBLE_CREDENTIAL]",
            sanitized,
        )

        return sanitized

    @classmethod
    def sanitize_dict(cls, data: dict[str, Any], max_depth: int = 10) -> dict[str, Any]:
        """
        Recursively sanitize sensitive fields in a dictionary.

        Args:
            data: Dictionary to sanitize
            max_depth: Maximum recursion depth

        Returns:
            Sanitized dictionary
        """
        if max_depth <= 0:
            return {"error": "Max recursion depth reached"}

        sanitized = {}
        for key, value in data.items():
            # Check if field name indicates sensitive data
            if any(sensitive in key.lower() for sensitive in cls.SENSITIVE_FIELDS):
                # If the value is a dict or list, still recursively sanitize it
                # but also check individual field names
                if isinstance(value, dict):
                    sanitized[key] = cls.sanitize_dict(value, max_depth - 1)
                elif isinstance(value, list):
                    sanitized[key] = cls.sanitize_list(value, max_depth - 1)
                else:
                    sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_dict(value, max_depth - 1)
            elif isinstance(value, list):
                sanitized[key] = cls.sanitize_list(value, max_depth - 1)
            elif isinstance(value, str):
                sanitized[key] = cls.sanitize_string(value)
            else:
                sanitized[key] = value

        return sanitized

    @classmethod
    def sanitize_list(cls, data: list[Any], max_depth: int = 10) -> list[Any]:
        """
        Recursively sanitize sensitive data in a list.

        Args:
            data: List to sanitize
            max_depth: Maximum recursion depth

        Returns:
            Sanitized list
        """
        if max_depth <= 0:
            return ["[Max recursion depth reached]"]

        sanitized = []
        for item in data:
            if isinstance(item, dict):
                sanitized.append(cls.sanitize_dict(item, max_depth - 1))
            elif isinstance(item, list):
                sanitized.append(cls.sanitize_list(item, max_depth - 1))
            elif isinstance(item, str):
                sanitized.append(cls.sanitize_string(item))
            else:
                sanitized.append(item)

        return sanitized

    @classmethod
    def sanitize_error(cls, error: Exception) -> str:
        """
        Sanitize an exception message and its string representation.

        Args:
            error: Exception to sanitize

        Returns:
            Sanitized error message
        """
        # Get the error message
        error_msg = str(error)

        # Sanitize the message
        sanitized_msg = cls.sanitize_string(error_msg)

        # Also check for sensitive data in exception attributes
        if hasattr(error, "__dict__"):
            for _attr, value in error.__dict__.items():
                if isinstance(value, str):
                    value_sanitized = cls.sanitize_string(value)
                    if value != value_sanitized:
                        sanitized_msg = sanitized_msg.replace(value, value_sanitized)

        return sanitized_msg

    @classmethod
    def sanitize_boto3_error(cls, error: Exception) -> dict[str, Any]:
        """
        Sanitize boto3 client errors which often contain credentials.

        Args:
            error: Boto3 ClientError exception

        Returns:
            Sanitized error information
        """
        result = {
            "error_type": error.__class__.__name__,
            "error_message": cls.sanitize_string(str(error)),
        }

        # Handle boto3 ClientError specifically
        if hasattr(error, "response"):
            response = error.response
            if isinstance(response, dict):
                # Sanitize the response metadata
                sanitized_response = {}

                # Extract safe fields
                if "Error" in response:
                    sanitized_response["Error"] = {
                        "Code": response["Error"].get("Code", "Unknown"),
                        "Message": cls.sanitize_string(response["Error"].get("Message", "")),
                    }

                if "ResponseMetadata" in response:
                    metadata = response["ResponseMetadata"]
                    sanitized_response["ResponseMetadata"] = {
                        "HTTPStatusCode": metadata.get("HTTPStatusCode"),
                        "RequestId": metadata.get("RequestId", "[REDACTED]"),
                    }

                result["response"] = sanitized_response

        return result
