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
Configuration module for Recursive Companion MCP Server
Centralizes all configuration values and environment variables
"""

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServerConfig:
    """Configuration for the MCP server"""

    # AWS Configuration
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))

    # Model Configuration
    bedrock_model_id: str = field(
        default_factory=lambda: os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
        )
    )
    critique_model_id: str = field(
        default_factory=lambda: os.getenv(
            "CRITIQUE_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
        )
    )
    embedding_model_id: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1")
    )

    # Refinement Configuration
    max_iterations: int = field(default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "10")))
    convergence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONVERGENCE_THRESHOLD", "0.98"))
    )
    parallel_critiques: int = field(
        default_factory=lambda: int(os.getenv("PARALLEL_CRITIQUES", "3"))
    )
    quick_refine_timeout: int = field(
        default_factory=lambda: int(os.getenv("QUICK_REFINE_TIMEOUT", "30"))
    )

    # Cache Configuration
    embedding_cache_size: int = 1000
    embedding_cache_trim_to: int = 900
    cache_key_length: int = 16

    # Display Configuration
    prompt_preview_length: int = 50
    draft_preview_length: int = 200
    max_active_sessions_display: int = 3

    # Thread Pool Configuration
    executor_max_workers: int = 4

    # API Limits
    max_tokens: int = 4096
    embedding_max_tokens: int = 8192

    # Temperature Settings
    default_temperature: float = 0.7
    critique_temperature: float = 0.3

    # Prompt Validation
    max_prompt_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_PROMPT_LENGTH", "10000"))
    )
    min_prompt_length: int = field(
        default_factory=lambda: int(os.getenv("MIN_PROMPT_LENGTH", "10"))
    )

    # Request Timeout
    request_timeout: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "300")))

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "aws_region": self.aws_region,
            "bedrock_model_id": self.bedrock_model_id,
            "critique_model_id": self.critique_model_id,
            "embedding_model_id": self.embedding_model_id,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "parallel_critiques": self.parallel_critiques,
            "quick_refine_timeout": self.quick_refine_timeout,
            "embedding_cache_size": self.embedding_cache_size,
            "max_tokens": self.max_tokens,
            "default_temperature": self.default_temperature,
        }


# Global configuration instance
config = ServerConfig()
