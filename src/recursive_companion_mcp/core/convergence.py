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
Reusable convergence detection module for thinking tools.
Extracted from recursive-companion and adapted for general use.
"""

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import boto3
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceConfig:
    """Configuration for convergence detection"""

    threshold: float = 0.95
    embedding_model_id: str = "amazon.titan-embed-text-v1"
    aws_region: str = "us-east-1"
    cache_size: int = 1000
    max_text_length: int = 8000


class EmbeddingService:
    """AWS Bedrock embedding service with caching"""

    def __init__(self, config: ConvergenceConfig):
        self.config = config
        self.bedrock_runtime = None
        self._embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self):
        """Lazy initialization of Bedrock client"""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            try:
                self.bedrock_runtime = boto3.client(
                    service_name="bedrock-runtime", region_name=self.config.aws_region
                )
                self._initialized = True
                logger.info(f"Embedding service initialized in region {self.config.aws_region}")
            except Exception as e:
                logger.error(f"Failed to initialize embedding service: {e}")
                raise

    async def get_embedding(self, text: str) -> list[float]:
        # Truncate text if too long
        if len(text) > self.config.max_text_length:
            text = text[: self.config.max_text_length]

        # Create cache key (MD5 used for non-security caching purpose)
        text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()  # nosec

        if text_hash in self._embedding_cache:
            self._cache_hits += 1
            self._embedding_cache.move_to_end(text_hash)
            logger.debug(f"Embedding cache hit. Hit rate: {self.get_cache_hit_rate():.2%}")
            return self._embedding_cache[text_hash]

        self._cache_misses += 1

        # Ensure initialized
        await self._ensure_initialized()

        try:
            body = json.dumps({"inputText": text, "dimensions": 1536, "normalize": True})

            response = self.bedrock_runtime.invoke_model(
                modelId=self.config.embedding_model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )

            result = json.loads(response["body"].read())
            embedding = result.get("embedding", [])

            # Cache the result
            self._embedding_cache[text_hash] = embedding

            # Trim cache if too large
            if len(self._embedding_cache) > self.config.cache_size:
                # Remove oldest entry
                self._embedding_cache.popitem(last=False)

            return embedding

        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

    def get_cache_hit_rate(self) -> float:
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    def get_cache_stats(self) -> dict[str, Any]:
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self.get_cache_hit_rate(),
            "entries": len(self._embedding_cache),
        }


class ConvergenceDetector:
    """Drop-in convergence detection for thinking tools"""

    def __init__(self, config: ConvergenceConfig | None = None):
        self.config = config or ConvergenceConfig()
        self.embedding_service = EmbeddingService(self.config)
        self._convergence_history = []

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def is_converged(
        self, current: str, previous: str, threshold: float | None = None
    ) -> tuple[bool, float]:
        """
        Check if iterations have converged

        Args:
            current: Current iteration text
            previous: Previous iteration text
            threshold: Override default threshold

        Returns:
            Tuple of (converged, similarity_score)
        """
        if not current or not previous:
            return False, 0.0

        # Use provided threshold or default
        threshold = threshold or self.config.threshold

        try:
            # Get embeddings
            current_emb = await self.embedding_service.get_embedding(current)
            previous_emb = await self.embedding_service.get_embedding(previous)

            # Calculate similarity
            score = self.cosine_similarity(current_emb, previous_emb)

            self._convergence_history.append(
                {
                    "score": score,
                    "threshold": threshold,
                    "converged": score >= threshold,
                }
            )

            logger.info(
                f"Convergence check: score={score:.4f}, threshold={threshold:.4f}, converged={score >= threshold}"
            )

            return score >= threshold, score

        except Exception as e:
            logger.error(f"Convergence check failed: {e}")
            return False, 0.0

    def get_convergence_history(self) -> list[dict[str, Any]]:
        return self._convergence_history.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get convergence detection statistics"""
        if not self._convergence_history:
            return {"total_checks": 0}

        scores = [h["score"] for h in self._convergence_history]
        convergences = [h["converged"] for h in self._convergence_history]

        return {
            "total_checks": len(self._convergence_history),
            "convergences": sum(convergences),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "embedding_stats": self.embedding_service.get_cache_stats(),
        }


# Tool-specific convergence detectors with optimized thresholds
def create_detector_for_tool(tool_name: str) -> ConvergenceDetector:
    """Create optimized convergence detector for specific tools"""

    # Tool-specific thresholds based on implementation guide
    thresholds = {
        "devil-advocate": 0.70,  # Lower - we want diversity
        "decision-matrix": 0.90,  # Moderate
        "conversation-tree": 0.85,  # Moderate - some diversity good
        "rubber-duck": 0.95,  # High - stop loops
        "hindsight": 0.95,  # High - stable perspective
        "context-switcher": 0.85,  # Moderate - some diversity good
    }

    threshold = thresholds.get(tool_name, 0.95)  # Default to high

    config = ConvergenceConfig(threshold=threshold)
    detector = ConvergenceDetector(config)

    logger.info(f"Created convergence detector for {tool_name} with threshold {threshold}")
    return detector


# Simple convergence check for basic use cases
async def simple_convergence_check(current: str, previous: str, threshold: float = 0.95) -> bool:
    """Simple convergence check without full detector setup"""
    detector = ConvergenceDetector(ConvergenceConfig(threshold=threshold))
    converged, _ = await detector.is_converged(current, previous)
    return converged


# Fallback convergence check (no embeddings needed)
def basic_text_convergence(
    current: str, previous: str, threshold: float = 0.95
) -> tuple[bool, float]:
    """
    Basic convergence check using simple text similarity
    Fallback when embeddings are not available
    """
    if not current or not previous:
        return False, 0.0

    # Simple character-level similarity
    max_len = max(len(current), len(previous))
    if max_len == 0:
        return True, 1.0

    # Calculate Levenshtein distance approximation
    common_chars = sum(c1 == c2 for c1, c2 in zip(current, previous, strict=False))
    similarity = common_chars / max_len

    return similarity >= threshold, similarity
