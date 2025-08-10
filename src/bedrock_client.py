"""
AWS Bedrock client wrapper for model operations.
Handles text generation and embeddings with proper error handling and caching.
"""

import asyncio
import hashlib
import json
import logging
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import boto3
import numpy as np

from config import config
from security_utils import CredentialSanitizer
from circuit_breaker import CircuitBreakerConfig, CircuitBreakerOpenError, circuit_manager

logger = logging.getLogger(__name__)


class BedrockClient:
    """Wrapper for AWS Bedrock operations with async support and caching."""

    def __init__(self):
        """Initialize without blocking - credentials will be validated on first use."""
        self.bedrock_runtime = None
        # Use OrderedDict for LRU cache implementation
        self._embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._cache_memory_bytes = 0
        self._max_cache_memory_bytes = 50 * 1024 * 1024  # 50MB limit
        self._executor = ThreadPoolExecutor(max_workers=config.executor_max_workers)
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._cache_hits = 0
        self._cache_misses = 0

        # Initialize circuit breakers for different AWS operations
        breaker_config = CircuitBreakerConfig(
            failure_threshold=3,  # Open after 3 consecutive failures
            success_threshold=2,  # Close after 2 consecutive successes in half-open
            timeout=30.0,  # Try half-open after 30 seconds
            failure_rate_threshold=0.5,  # Open if 50% of calls fail
            min_calls=5,  # Need at least 5 calls before evaluating rate
            tracked_exceptions=(Exception,),  # Track all exceptions
            excluded_exceptions=(
                KeyboardInterrupt,
                SystemExit,
                asyncio.CancelledError,
                ValueError,  # Don't track validation errors
            ),
        )

        self._generation_breaker = circuit_manager.get_or_create(
            "bedrock_generation", breaker_config
        )
        self._embedding_breaker = circuit_manager.get_or_create("bedrock_embedding", breaker_config)

    async def _ensure_initialized(self):
        """Ensure client is initialized (async lazy initialization)."""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:  # Double-check after acquiring lock
                return

            try:
                # Create the Bedrock runtime client
                self.bedrock_runtime = boto3.client(
                    service_name="bedrock-runtime", region_name=config.aws_region
                )

                # Test connection asynchronously
                await self._test_connection_async()

                self._initialized = True
                logger.info(
                    f"AWS Bedrock client initialized successfully in region {config.aws_region}"
                )

            except Exception as e:
                # Use comprehensive sanitizer for all error information
                sanitized_error = CredentialSanitizer.sanitize_boto3_error(e)
                error_msg = sanitized_error.get("error_message", "Unknown error")
                logger.error(f"Failed to initialize AWS Bedrock client: {error_msg}")
                # Don't include original exception to prevent credential leakage
                raise ValueError(f"AWS Bedrock initialization failed: {error_msg}")

    async def _test_connection_async(self):
        """Test AWS Bedrock connection asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._test_connection_sync)
        except Exception as e:
            sanitized_msg = CredentialSanitizer.sanitize_error(e)
            logger.warning(f"Could not verify Bedrock access: {sanitized_msg}")

    def _test_connection_sync(self):
        """Synchronous connection test for executor."""
        bedrock = boto3.client(service_name="bedrock", region_name=config.aws_region)
        bedrock.list_foundation_models(byProvider="Anthropic", maxResults=1)

    def _invoke_model_sync(self, model: str, body: dict) -> dict:
        """Synchronous model invocation for thread pool executor."""
        response = self.bedrock_runtime.invoke_model(modelId=model, body=json.dumps(body))
        return response

    async def _invoke_model_with_circuit_breaker(self, model: str, body: dict) -> dict:
        """Invoke model with circuit breaker protection."""
        loop = asyncio.get_event_loop()

        async def invoke():
            return await loop.run_in_executor(self._executor, self._invoke_model_sync, model, body)

        # Fallback function returns None to indicate service unavailable
        async def fallback():
            logger.warning(f"Circuit breaker open for model {model}, returning fallback")
            return None

        return await self._generation_breaker.call(invoke, fallback=fallback)

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        model_override: Optional[str] = None,
    ) -> str:
        """
        Generate text using Claude via Bedrock with optimized async handling.

        Args:
            prompt: The user prompt
            system_prompt: System prompt for the model
            temperature: Generation temperature (0.0-1.0)
            model_override: Override the default model

        Returns:
            Generated text response

        Raises:
            ValueError: If response format is invalid
            Exception: For other Bedrock errors
        """
        try:
            # Ensure initialized before using
            await self._ensure_initialized()

            model = model_override or config.bedrock_model_id
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": config.max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system_prompt:
                body["system"] = system_prompt

            # Use circuit breaker for protection against AWS failures
            response = await self._invoke_model_with_circuit_breaker(model, body)

            # Check if circuit breaker returned fallback (None)
            if response is None:
                raise CircuitBreakerOpenError(
                    "AWS Bedrock service unavailable (circuit breaker open). "
                    "Please try again in a few moments."
                )

            response_body = json.loads(response["body"].read())
            return response_body["content"][0]["text"]

        except json.JSONDecodeError as e:
            sanitized_msg = CredentialSanitizer.sanitize_error(e)
            logger.error(f"Invalid JSON response from Bedrock: {sanitized_msg}")
            raise ValueError("Invalid response format from Bedrock model")

        except Exception as e:
            # Sanitize any potential credentials in error messages
            sanitized_error = CredentialSanitizer.sanitize_boto3_error(e)
            logger.error(f"Bedrock generation error: {sanitized_error}")
            # Re-raise with sanitized message to prevent credential leakage
            raise RuntimeError(
                f"Generation failed: {sanitized_error.get('error_message', 'Unknown error')}"
            )

    def _get_embedding_uncached_sync(self, text: str) -> List[float]:
        """Synchronous embedding generation for executor."""
        response = self.bedrock_runtime.invoke_model(
            modelId=config.embedding_model_id, body=json.dumps({"inputText": text})
        )
        response_body = json.loads(response["body"].read())
        return response_body["embedding"]

    async def _get_embedding_uncached(self, text: str) -> Optional[List[float]]:
        """Get text embedding using Titan with circuit breaker protection."""
        try:
            loop = asyncio.get_event_loop()

            async def get_embedding():
                return await loop.run_in_executor(
                    self._executor, self._get_embedding_uncached_sync, text
                )

            # Fallback returns None if circuit is open
            async def fallback():
                logger.warning("Embedding circuit breaker open, returning None")
                return None

            result = await self._embedding_breaker.call(get_embedding, fallback=fallback)

            if result is None:
                raise CircuitBreakerOpenError(
                    "AWS Bedrock embedding service unavailable (circuit breaker open)"
                )

            return result

        except CircuitBreakerOpenError:
            raise

        except json.JSONDecodeError as e:
            sanitized_msg = CredentialSanitizer.sanitize_error(e)
            logger.error(f"Invalid JSON response from Bedrock: {sanitized_msg}")
            raise ValueError("Invalid response format from Bedrock embedding model")

        except Exception as e:
            # Sanitize error to prevent credential leakage
            sanitized_error = CredentialSanitizer.sanitize_boto3_error(e)
            logger.error(f"Embedding generation error: {sanitized_error}")
            raise RuntimeError(
                f"Embedding failed: {sanitized_error.get('error_message', 'Unknown error')}"
            )

    async def get_embedding(self, text: str) -> List[float]:
        """
        Get text embedding with LRU caching and memory management.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Create hash for caching
        text_hash = hashlib.sha256(text.encode()).hexdigest()[: config.cache_key_length]

        # Check cache first (LRU - move to end if hit)
        if text_hash in self._embedding_cache:
            self._cache_hits += 1
            # Move to end (most recently used)
            self._embedding_cache.move_to_end(text_hash)
            logger.debug(f"Cache hit for embedding. Hit rate: {self.get_cache_hit_rate():.2%}")
            return self._embedding_cache[text_hash]

        self._cache_misses += 1

        # Ensure initialized before using
        await self._ensure_initialized()

        # Generate embedding if not cached - now uses async with circuit breaker
        embedding = await self._get_embedding_uncached(text)

        # Calculate memory size of this embedding (float = 4 bytes typically)
        embedding_size = len(embedding) * 4 + sys.getsizeof(text_hash)

        # Evict oldest entries if adding this would exceed memory limit
        while (
            self._cache_memory_bytes + embedding_size > self._max_cache_memory_bytes
            and len(self._embedding_cache) > 0
        ):
            # Remove least recently used (first item)
            oldest_key, oldest_embedding = self._embedding_cache.popitem(last=False)
            removed_size = len(oldest_embedding) * 4 + sys.getsizeof(oldest_key)
            self._cache_memory_bytes -= removed_size
            logger.debug(
                f"Evicted cache entry to stay within memory limit. Cache size: {len(self._embedding_cache)}"
            )

        # Cache the result
        self._embedding_cache[text_hash] = embedding
        self._cache_memory_bytes += embedding_size

        # Also respect the original count-based limit
        if len(self._embedding_cache) > config.embedding_cache_size:
            # Remove oldest entries
            while len(self._embedding_cache) > config.embedding_cache_trim_to:
                oldest_key, oldest_embedding = self._embedding_cache.popitem(last=False)
                removed_size = len(oldest_embedding) * 4 + sys.getsizeof(oldest_key)
                self._cache_memory_bytes -= removed_size

        return embedding

    def get_cache_hit_rate(self) -> float:
        """Get the cache hit rate for monitoring."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self.get_cache_hit_rate(),
            "entries": len(self._embedding_cache),
            "memory_bytes": self._cache_memory_bytes,
            "memory_mb": self._cache_memory_bytes / (1024 * 1024),
            "max_memory_mb": self._max_cache_memory_bytes / (1024 * 1024),
        }

    def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics for monitoring."""
        return {
            "generation": self._generation_breaker.get_stats(),
            "embedding": self._embedding_breaker.get_stats(),
        }

    def calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0.0-1.0)
        """
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def cleanup(self):
        """Explicit cleanup method for resources."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True, cancel_futures=True)
            logger.info("Thread pool executor shut down cleanly")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        self.cleanup()
        return False

    def __del__(self):
        """Fallback cleanup on deletion."""
        try:
            if hasattr(self, "_executor") and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass  # Best effort cleanup
