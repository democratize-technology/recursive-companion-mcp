"""
AWS Bedrock client wrapper for model operations.
Handles text generation and embeddings with proper error handling and caching.
"""

import asyncio
import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import boto3
import numpy as np

from config import config

logger = logging.getLogger(__name__)


class BedrockClient:
    """Wrapper for AWS Bedrock operations with async support and caching."""

    def __init__(self):
        """Initialize without blocking - credentials will be validated on first use."""
        self.bedrock_runtime = None
        self._embedding_cache: Dict[str, List[float]] = {}
        self._executor = ThreadPoolExecutor(max_workers=config.executor_max_workers)
        self._initialized = False
        self._init_lock = asyncio.Lock()

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
                # Scrub any sensitive information from error message
                error_msg = self._sanitize_error_message(str(e))
                logger.error(f"Failed to initialize AWS Bedrock client: {error_msg}")
                raise ValueError(f"AWS Bedrock initialization failed: {error_msg}")

    def _sanitize_error_message(self, error_msg: str) -> str:
        """Remove sensitive information from error messages."""
        # Remove potential access keys, secret keys, and session tokens
        # Pattern for AWS access key IDs (AKIA followed by 16 alphanumeric)
        error_msg = re.sub(r"AKIA[A-Z0-9]{16}", "[REDACTED_ACCESS_KEY]", error_msg)
        # Pattern for potential secret keys (40 character base64-like strings)
        error_msg = re.sub(r"[A-Za-z0-9+/]{40}", "[REDACTED_SECRET]", error_msg)
        # Remove any key=value pairs that might contain credentials
        error_msg = re.sub(
            r"(aws_access_key_id|aws_secret_access_key|aws_session_token)=[^\s]+",
            r"\1=[REDACTED]",
            error_msg,
            flags=re.IGNORECASE,
        )
        return error_msg

    async def _test_connection_async(self):
        """Test AWS Bedrock connection asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._test_connection_sync)
        except Exception as e:
            logger.warning(
                f"Could not verify Bedrock access: {self._sanitize_error_message(str(e))}"
            )

    def _test_connection_sync(self):
        """Synchronous connection test for executor."""
        bedrock = boto3.client(service_name="bedrock", region_name=config.aws_region)
        bedrock.list_foundation_models(byProvider="Anthropic", maxResults=1)

    def _invoke_model_sync(self, model: str, body: dict) -> dict:
        """Synchronous model invocation for thread pool executor."""
        response = self.bedrock_runtime.invoke_model(modelId=model, body=json.dumps(body))
        return response

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

            # Use the dedicated thread pool executor for better performance
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor, self._invoke_model_sync, model, body
            )

            response_body = json.loads(response["body"].read())
            return response_body["content"][0]["text"]

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Bedrock: {e}")
            raise ValueError("Invalid response format from Bedrock model")

        except Exception as e:
            logger.error(f"Bedrock generation error: {e}")
            raise

    def _get_embedding_uncached(self, text: str) -> List[float]:
        """Get text embedding using Titan without caching."""
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=config.embedding_model_id, body=json.dumps({"inputText": text})
            )
            response_body = json.loads(response["body"].read())
            return response_body["embedding"]

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Bedrock: {e}")
            raise ValueError("Invalid response format from Bedrock embedding model")

        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise

    async def get_embedding(self, text: str) -> List[float]:
        """
        Get text embedding with proper caching and optimized async handling.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Create hash for caching
        text_hash = hashlib.sha256(text.encode()).hexdigest()[: config.cache_key_length]

        # Check cache first
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]

        # Ensure initialized before using
        await self._ensure_initialized()

        # Generate embedding if not cached - use dedicated thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(self._executor, self._get_embedding_uncached, text)

        # Cache the result
        self._embedding_cache[text_hash] = embedding

        # Limit cache size to prevent memory issues
        if len(self._embedding_cache) > config.embedding_cache_size:
            # Remove oldest entries to trim down to embedding_cache_trim_to
            current_keys = list(self._embedding_cache.keys())
            # Keep only the last embedding_cache_trim_to entries
            keys_to_remove = current_keys[: -config.embedding_cache_trim_to]
            for key in keys_to_remove:
                del self._embedding_cache[key]

        return embedding

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

    def __del__(self):
        """Cleanup thread pool executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
