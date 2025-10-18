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
Base cognitive enhancement class for thinking tools.
Provides common functionality like convergence detection and iteration tracking.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .convergence import create_detector_for_tool

logger = logging.getLogger(__name__)


@dataclass
class CognitiveConfig:
    """Configuration for cognitive enhancement"""

    tool_name: str
    max_iterations: int = 10
    convergence_threshold: float | None = None
    enable_convergence: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"


@dataclass
class IterationResult:
    """Result of a single iteration"""

    iteration: int
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    convergence_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CognitiveEnhancer:
    """
    Base class for adding cognitive enhancements to thinking tools.

    Provides:
    - Convergence detection
    - Iteration tracking
    - Performance logging
    - Common patterns for iterative thinking
    """

    def __init__(self, config: CognitiveConfig):
        self.config = config
        self.convergence_detector = None
        self.iteration_history: list[IterationResult] = []
        self.start_time = datetime.utcnow()
        self._setup_logging()
        self._setup_convergence()

    def _setup_logging(self):
        """Setup logging for this cognitive enhancer"""
        if self.config.enable_logging:
            log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
            logger.setLevel(log_level)
            logger.info(f"Cognitive enhancer initialized for {self.config.tool_name}")

    def _setup_convergence(self):
        """Setup convergence detection"""
        if self.config.enable_convergence:
            self.convergence_detector = create_detector_for_tool(self.config.tool_name)
            logger.info(
                f"Convergence detection enabled with threshold {self.convergence_detector.config.threshold}"
            )

    async def check_convergence(self, current: str, previous: str) -> tuple[bool, float]:
        """
        Check if current iteration has converged with previous

        Returns:
            Tuple of (converged, similarity_score)
        """
        if not self.config.enable_convergence or not self.convergence_detector:
            return False, 0.0

        return await self.convergence_detector.is_converged(current, previous)

    def add_iteration(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> IterationResult:
        """Add an iteration to the history"""
        iteration = IterationResult(
            iteration=len(self.iteration_history) + 1,
            content=content,
            metadata=metadata or {},
        )
        self.iteration_history.append(iteration)

        logger.debug(f"Added iteration {iteration.iteration} with {len(content)} characters")
        return iteration

    async def process_with_convergence(
        self,
        processor_func: Callable[[str, int], str],
        initial_input: str,
        max_iterations: int | None = None,
    ) -> dict[str, Any]:
        """
        Process with automatic convergence detection

        Args:
            processor_func: Function that takes (input, iteration) and returns output
            initial_input: Starting input
            max_iterations: Override default max iterations

        Returns:
            Dict with final result and metadata
        """
        max_iter = max_iterations or self.config.max_iterations
        current_input = initial_input

        for iteration in range(1, max_iter + 1):
            logger.info(f"Starting iteration {iteration}/{max_iter}")

            # Process current iteration
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, processor_func, current_input, iteration
                )
            except Exception as e:
                logger.error(f"Processing failed at iteration {iteration}: {e}")
                break

            # Add to history
            iteration_result = self.add_iteration(result, {"iteration": iteration})

            # Check convergence if we have a previous iteration
            if len(self.iteration_history) >= 2:
                previous_content = self.iteration_history[-2].content
                converged, score = await self.check_convergence(result, previous_content)

                # Update convergence score in current iteration
                iteration_result.convergence_score = score

                if converged:
                    logger.info(f"Converged at iteration {iteration} with score {score:.4f}")
                    break
                logger.info(
                    f"Iteration {iteration} similarity: {score:.4f} (threshold: {self.convergence_detector.config.threshold:.4f})"
                )

            current_input = result

        return self._create_final_result()

    def _create_final_result(self) -> dict[str, Any]:
        if not self.iteration_history:
            return {
                "success": False,
                "error": "No iterations completed",
                "total_time": (datetime.utcnow() - self.start_time).total_seconds(),
            }

        final_iteration = self.iteration_history[-1]
        convergence_scores = [
            r.convergence_score for r in self.iteration_history if r.convergence_score is not None
        ]

        return {
            "success": True,
            "final_result": final_iteration.content,
            "total_iterations": len(self.iteration_history),
            "convergence_achieved": (
                final_iteration.convergence_score >= self.convergence_detector.config.threshold
                if self.convergence_detector
                else False
            ),
            "final_convergence_score": final_iteration.convergence_score,
            "convergence_history": convergence_scores,
            "total_time": (datetime.utcnow() - self.start_time).total_seconds(),
            "tool_name": self.config.tool_name,
            "metadata": {
                "max_iterations": self.config.max_iterations,
                "convergence_enabled": self.config.enable_convergence,
                "threshold": (
                    self.convergence_detector.config.threshold
                    if self.convergence_detector
                    else None
                ),
            },
        }

    def get_iteration_history(self) -> list[dict[str, Any]]:
        return [
            {
                "iteration": r.iteration,
                "content_preview": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                "content_length": len(r.content),
                "timestamp": r.timestamp.isoformat(),
                "convergence_score": r.convergence_score,
                "metadata": r.metadata,
            }
            for r in self.iteration_history
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            "tool_name": self.config.tool_name,
            "total_iterations": len(self.iteration_history),
            "total_time": (datetime.utcnow() - self.start_time).total_seconds(),
            "convergence_enabled": self.config.enable_convergence,
        }

        if self.convergence_detector:
            stats["convergence_stats"] = self.convergence_detector.get_stats()

        if self.iteration_history:
            scores = [
                r.convergence_score
                for r in self.iteration_history
                if r.convergence_score is not None
            ]
            if scores:
                stats["convergence_summary"] = {
                    "avg_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                    "min_score": min(scores),
                    "final_score": scores[-1] if scores else None,
                }

        return stats


class EnhancedThinkingTool(ABC):
    """
    Abstract base class for thinking tools with cognitive enhancement

    Inherit from this class to get automatic convergence detection
    and iteration tracking for your thinking tool.
    """

    def __init__(self, tool_name: str, config: CognitiveConfig | None = None):
        self.tool_name = tool_name
        self.config = config or CognitiveConfig(tool_name=tool_name)
        self.enhancer = CognitiveEnhancer(self.config)

    @abstractmethod
    async def process_iteration(self, input_data: str, iteration: int) -> str:
        """
        Process a single iteration - must be implemented by subclass

        Args:
            input_data: Current input to process
            iteration: Current iteration number (1-based)

        Returns:
            Processed output for this iteration
        """

    async def process(self, initial_input: str) -> dict[str, Any]:
        """
        Main processing method with automatic convergence detection

        Args:
            initial_input: Starting input

        Returns:
            Final result with metadata
        """
        logger.info(f"Starting {self.tool_name} processing")

        return await self.enhancer.process_with_convergence(self.process_iteration, initial_input)

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics"""
        return self.enhancer.get_stats()

    def get_history(self) -> list[dict[str, Any]]:
        """Get iteration history"""
        return self.enhancer.get_iteration_history()


# Decorator for adding convergence to existing functions
def with_convergence(tool_name: str, threshold: float | None = None):
    """
    Decorator to add convergence detection to any iterative function

    Usage:
        @with_convergence("my_tool", threshold=0.9)
        async def my_iterative_function(input_data):
            # Your function that might need multiple iterations
            pass
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            config = CognitiveConfig(tool_name=tool_name, convergence_threshold=threshold)
            enhancer = CognitiveEnhancer(config)

            # Simple single-shot processing with logging
            result = await func(*args, **kwargs)
            enhancer.add_iteration(str(result))

            return {"result": result, "enhanced_stats": enhancer.get_stats()}

        return wrapper

    return decorator


# Utility functions for common convergence patterns
async def iterate_until_convergence(
    processor: Callable[[str], str],
    initial_input: str,
    tool_name: str = "generic",
    max_iterations: int = 10,
    threshold: float = 0.95,
) -> dict[str, Any]:
    """
    Utility function to iterate any processor until convergence

    Args:
        processor: Function that processes one iteration
        initial_input: Starting input
        tool_name: Name for logging
        max_iterations: Maximum iterations
        threshold: Convergence threshold

    Returns:
        Final result with convergence metadata
    """
    config = CognitiveConfig(
        tool_name=tool_name,
        max_iterations=max_iterations,
        convergence_threshold=threshold,
    )
    enhancer = CognitiveEnhancer(config)

    return await enhancer.process_with_convergence(
        lambda x, i: processor(x), initial_input, max_iterations
    )
