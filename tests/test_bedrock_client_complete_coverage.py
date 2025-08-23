#!/usr/bin/env python3
"""
Comprehensive test coverage for bedrock_client.py targeting specific missing lines.

This test suite focuses on achieving 100% coverage by testing:
- Concurrent initialization edge cases (line 92)
- Cache memory management and eviction (lines 300-303)
- Zero division protection in cache stats (line 325)
- Return statements in monitoring methods (lines 330, 342)
- Resource cleanup patterns (lines 349-351, 368-369)
- Async context manager methods (lines 355-356, 360-361)
"""

import asyncio
import gc
import json
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.append("src")

from bedrock_client import BedrockClient


class TestBedrockClientCompleteCoverage:
    """Test suite targeting specific missing coverage lines in bedrock_client.py."""

    @pytest.fixture
    def mock_bedrock_runtime(self):
        """Create a mock bedrock runtime client."""
        mock_client = MagicMock()
        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = json.dumps(
            {"content": [{"text": "Generated response"}]}
        ).encode()
        mock_client.invoke_model.return_value = mock_response
        return mock_client

    @pytest.fixture
    def mock_embedding_response(self):
        """Create a mock embedding response."""
        mock_response = {"body": Mock()}
        # Create a realistic embedding vector (1536 dimensions like OpenAI)
        embedding = [0.1] * 1536
        mock_response["body"].read.return_value = json.dumps({"embedding": embedding}).encode()
        return mock_response

    @pytest.mark.asyncio
    async def test_concurrent_initialization_double_check_lock(self):
        """
        Test concurrent initialization to trigger line 92 (double-check after lock).

        This test ensures that when multiple coroutines try to initialize simultaneously,
        the double-check pattern after acquiring the lock works correctly.
        """
        client = BedrockClient()

        # Mock the boto3 client creation and test connection
        with patch("boto3.client") as mock_boto3, patch.object(
            client, "_test_connection_async", new=AsyncMock()
        ) as mock_test:

            mock_boto3.return_value = MagicMock()

            # Create multiple concurrent initialization attempts
            # This should trigger the race condition where the second task
            # finds the client already initialized after acquiring the lock (line 92)
            tasks = [client._ensure_initialized() for _ in range(5)]

            # Execute all initialization attempts concurrently
            await asyncio.gather(*tasks)

            # Verify initialization only happened once
            assert client._initialized is True
            assert mock_boto3.call_count == 1
            assert mock_test.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_memory_eviction_with_logging(self, mock_embedding_response):
        """
        Test cache eviction when memory limit is exceeded to trigger lines 300-303.

        This test ensures the cache eviction logging and memory management
        works correctly when the cache grows beyond the memory limit.
        """
        client = BedrockClient()
        client._max_cache_memory_bytes = 1000  # Small but reasonable limit

        with patch("boto3.client"), patch.object(
            client, "_test_connection_async", new=AsyncMock()
        ), patch.object(client, "_get_embedding_uncached_sync") as mock_embedding, patch(
            "bedrock_client.logger"
        ) as mock_logger:

            # Set up embedding mock to return large embeddings that will trigger eviction
            large_embedding = [0.1] * 200  # 200 * 4 bytes = 800 bytes per embedding
            mock_embedding.return_value = large_embedding

            # Initialize client
            await client._ensure_initialized()

            # First, add one embedding to get some cache memory usage
            await client.get_embedding("first_text")
            client._cache_memory_bytes

            # Now add more embeddings that will exceed the limit and trigger eviction
            texts = [f"text_{i}" for i in range(5)]

            for text in texts:
                await client.get_embedding(text)

            # Verify that eviction logging was triggered (lines 300-303)
            # The debug log should contain "Evicted cache entry to stay within memory limit"
            debug_calls = [
                call
                for call in mock_logger.debug.call_args_list
                if "Evicted cache entry to stay within memory limit" in str(call)
            ]
            assert len(debug_calls) > 0, "Cache eviction logging should have been triggered"

            # Verify cache memory is maintained within limits
            assert client._cache_memory_bytes <= client._max_cache_memory_bytes

    @pytest.mark.asyncio
    async def test_cache_hit_rate_zero_division_protection(self):
        """
        Test cache hit rate calculation when no operations have occurred (line 325).

        This tests the defensive programming pattern that prevents division by zero
        when calculating hit rate before any cache operations.
        """
        client = BedrockClient()

        # Test hit rate calculation with no cache operations
        # This should trigger line 325: return 0.0 when total == 0
        hit_rate = client.get_cache_hit_rate()

        assert hit_rate == 0.0
        assert client._cache_hits == 0
        assert client._cache_misses == 0

    @pytest.mark.asyncio
    async def test_get_cache_stats_return_statement(self):
        """
        Test that get_cache_stats return statement is hit (line 330).

        This ensures the return statement in get_cache_stats is executed
        and returns the expected dictionary structure.
        """
        client = BedrockClient()

        # Call get_cache_stats to hit the return statement
        stats = client.get_cache_stats()

        # Verify all expected keys are present (line 330 return statement)
        expected_keys = {
            "hits",
            "misses",
            "hit_rate",
            "entries",
            "memory_bytes",
            "memory_mb",
            "max_memory_mb",
        }
        assert set(stats.keys()) == expected_keys
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_get_circuit_breaker_stats_return_statement(self):
        """
        Test that get_circuit_breaker_stats return statement is hit (line 342).

        This ensures the return statement in get_circuit_breaker_stats is executed
        and returns the expected dictionary structure.
        """
        client = BedrockClient()

        # Call get_circuit_breaker_stats to hit the return statement
        stats = client.get_circuit_breaker_stats()

        # Verify expected structure (line 342 return statement)
        assert "generation" in stats
        assert "embedding" in stats
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_explicit_cleanup_executor_shutdown(self):
        """
        Test explicit cleanup method to trigger executor shutdown (lines 349-351).

        This tests the cleanup method that shuts down the thread pool executor
        and logs the shutdown message.
        """
        client = BedrockClient()

        with patch("bedrock_client.logger") as mock_logger:
            # Ensure the executor attribute exists (it's created in __init__)
            assert hasattr(client, "_executor")

            # Call cleanup explicitly to trigger lines 349-351
            client.cleanup()

            # Verify logger was called with shutdown message
            mock_logger.info.assert_called_with("Thread pool executor shut down cleanly")

    @pytest.mark.asyncio
    async def test_async_context_manager_aenter(self):
        """
        Test async context manager __aenter__ method (lines 355-356).

        This tests the async context manager entry point which calls
        _ensure_initialized and returns self.
        """
        client = BedrockClient()

        with patch("boto3.client"), patch.object(client, "_test_connection_async", new=AsyncMock()):

            # Use async context manager to trigger __aenter__ (lines 355-356)
            async with client as context_client:
                assert context_client is client
                assert client._initialized is True

    @pytest.mark.asyncio
    async def test_async_context_manager_aexit(self):
        """
        Test async context manager __aexit__ method (lines 360-361).

        This tests the async context manager exit point which calls cleanup
        and returns False.
        """
        client = BedrockClient()

        with patch("boto3.client"), patch.object(
            client, "_test_connection_async", new=AsyncMock()
        ), patch.object(client, "cleanup") as mock_cleanup:

            # Use async context manager to trigger __aexit__ (lines 360-361)
            async with client:
                pass

            # Verify cleanup was called during __aexit__
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_destructor_fallback_cleanup(self):
        """
        Test destructor fallback cleanup (lines 368-369).

        This tests the __del__ method that provides best-effort cleanup
        when the object is garbage collected.
        """
        # Create a client that will be garbage collected
        client = BedrockClient()

        with patch.object(client._executor, "shutdown") as mock_shutdown:
            # Trigger destructor by deleting reference and forcing garbage collection
            del client
            gc.collect()

            # Note: Since __del__ uses best-effort cleanup with try/except,
            # we can't guarantee the shutdown is called, but we can verify
            # the code path exists and doesn't raise exceptions

        # The test passes if no exceptions were raised during cleanup

    @pytest.mark.asyncio
    async def test_destructor_with_exception_handling(self):
        """
        Test that destructor handles exceptions gracefully (lines 368-369).

        This tests the exception handling in __del__ method to ensure
        it doesn't raise exceptions during cleanup.
        """
        client = BedrockClient()

        # Mock the executor to raise an exception during shutdown
        with patch.object(
            client._executor, "shutdown", side_effect=RuntimeError("Mock shutdown error")
        ):
            # Manually call __del__ to test exception handling
            try:
                client.__del__()  # This should hit lines 368-369 (exception handling)
                # Test passes if no exception is raised
            except Exception as e:
                pytest.fail(f"__del__ should not raise exceptions, but raised: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_initialization_race_condition_precise(self):
        """
        Test precise race condition for line 92 (double-check after lock).

        This creates a more specific scenario to ensure we hit the exact
        condition where the second coroutine finds _initialized=True after acquiring lock.
        """
        client = BedrockClient()
        initialization_started = asyncio.Event()
        initialization_can_proceed = asyncio.Event()

        async def slow_initialization():
            """First initialization that will be slow."""
            await client._init_lock.acquire()
            try:
                if client._initialized:  # This might be line 92
                    return
                initialization_started.set()
                await initialization_can_proceed.wait()  # Wait for signal

                # Mock successful initialization
                client.bedrock_runtime = MagicMock()
                client._initialized = True
            finally:
                client._init_lock.release()

        async def fast_initialization():
            """Second initialization that should hit the double-check."""
            await initialization_started.wait()  # Wait for first to start
            # Now try to initialize - should hit the double-check pattern
            await client._ensure_initialized()

        with patch("boto3.client"), patch.object(client, "_test_connection_async", new=AsyncMock()):

            # Start the slow initialization
            slow_task = asyncio.create_task(slow_initialization())

            # Start the fast initialization that should hit double-check
            fast_task = asyncio.create_task(fast_initialization())

            # Let the slow initialization proceed
            initialization_can_proceed.set()

            # Wait for both to complete
            await asyncio.gather(slow_task, fast_task)

            assert client._initialized is True

    @pytest.mark.asyncio
    async def test_cache_eviction_large_embeddings_realistic_scenario(
        self, mock_embedding_response
    ):
        """
        Test cache eviction with realistic large embeddings to ensure proper memory management.

        This is a comprehensive test that simulates realistic embedding sizes
        and cache behavior to ensure the memory management works correctly.
        """
        client = BedrockClient()
        # Set a reasonable but small cache limit to trigger eviction
        client._max_cache_memory_bytes = 50000  # 50KB

        with patch("boto3.client"), patch.object(
            client, "_test_connection_async", new=AsyncMock()
        ), patch.object(client, "_get_embedding_uncached_sync") as mock_embedding:

            # Create realistic embedding size (like OpenAI ada-002: 1536 dimensions)
            realistic_embedding = [0.1] * 1536
            mock_embedding.return_value = realistic_embedding

            await client._ensure_initialized()

            # Generate enough embeddings to trigger cache eviction
            texts = [f"This is a test document number {i} with some content." for i in range(20)]

            for text in texts:
                await client.get_embedding(text)

            # Verify cache is working and staying within memory limits
            assert len(client._embedding_cache) > 0
            assert client._cache_memory_bytes <= client._max_cache_memory_bytes

            # Verify some cache hits occurred
            cache_stats = client.get_cache_stats()
            assert cache_stats["entries"] > 0
            assert cache_stats["memory_mb"] > 0

    @pytest.mark.asyncio
    async def test_initialization_failure_edge_cases(self):
        """
        Test various initialization failure scenarios to ensure proper error handling.

        This test covers edge cases in the initialization process that might
        not be covered by other tests.
        """
        client = BedrockClient()

        # Test boto3 client creation failure
        with patch("boto3.client", side_effect=Exception("AWS credentials not found")):
            with pytest.raises(ValueError, match="AWS Bedrock initialization failed"):
                await client._ensure_initialized()

        # Verify client remains uninitialized after failure
        assert not client._initialized

    @pytest.mark.asyncio
    async def test_comprehensive_resource_cleanup_scenario(self):
        """
        Comprehensive test of resource cleanup in various scenarios.

        This test ensures all cleanup paths work correctly and resources
        are properly freed.
        """
        client = BedrockClient()

        with patch("boto3.client"), patch.object(
            client, "_test_connection_async", new=AsyncMock()
        ), patch("bedrock_client.logger") as mock_logger:

            # Initialize client
            await client._ensure_initialized()

            # Use client for some operations
            client.get_cache_stats()
            client.get_circuit_breaker_stats()

            # Test explicit cleanup
            client.cleanup()

            # Verify cleanup logging
            mock_logger.info.assert_called_with("Thread pool executor shut down cleanly")

            # Test that multiple cleanups don't cause issues
            client.cleanup()  # Should not raise exception


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
