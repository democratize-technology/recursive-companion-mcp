"""Configuration management for the refinement engine.

This module provides centralized configuration management for model names,
domain-specific system prompts, and other refinement engine settings.
"""

import os

from domains import get_domain_system_prompt


class ConfigurationManager:
    """Manages configuration settings for the refinement engine."""

    @staticmethod
    def get_model_name() -> str:
        """Get the current model name for performance hints.

        Returns:
            The configured Bedrock model ID
        """
        return os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

    @staticmethod
    def get_domain_system_prompt(domain: str) -> str:
        """Get domain-specific system prompt.

        Args:
            domain: The domain type (technical, marketing, legal, etc.)

        Returns:
            Domain-specific system prompt
        """
        return get_domain_system_prompt(domain)

    @staticmethod
    def get_default_max_iterations() -> int:
        """Get default maximum iterations for refinement.

        Returns:
            Default maximum number of refinement iterations
        """
        return int(os.getenv("MAX_ITERATIONS", "10"))

    @staticmethod
    def get_default_convergence_threshold() -> float:
        """Get default convergence threshold.

        Returns:
            Default convergence threshold for refinement completion
        """
        return float(os.getenv("CONVERGENCE_THRESHOLD", "0.98"))

    @staticmethod
    def get_parallel_critiques_count() -> int:
        """Get number of parallel critiques to generate.

        Returns:
            Number of parallel critique perspectives
        """
        return int(os.getenv("PARALLEL_CRITIQUES", "3"))

    @staticmethod
    def is_cot_enhancement_enabled() -> bool:
        """Check if Chain of Thought enhancement is enabled.

        Returns:
            True if CoT enhancement should be used
        """
        return os.getenv("COT_ENHANCEMENT", "true").lower() == "true"
