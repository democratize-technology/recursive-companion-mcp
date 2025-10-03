"""
Surgical tests for ConvergenceDetector to achieve 100% coverage.
Specifically targets missing lines: 74, 287
"""

from unittest.mock import patch

import pytest

from recursive_companion_mcp.legacy.convergence import EmbeddingService, basic_text_convergence

# sys.path removed - using package imports


class TestConvergenceCoverage:
    """Surgical tests targeting specific missing lines in ConvergenceDetector."""

    @pytest.fixture
    def embedding_service(self):
        """Create test embedding service"""
        from convergence import ConvergenceConfig

        return EmbeddingService(ConvergenceConfig())

    @pytest.mark.asyncio
    async def test_ensure_initialized_already_initialized(self, embedding_service):
        """Test line 74: Early return when already initialized"""
        # First call to initialize
        await embedding_service._ensure_initialized()

        # Verify it's initialized
        assert embedding_service._initialized is True

        # Mock the boto3.client to track if it's called again
        with patch("convergence.boto3.client") as mock_client:
            # Second call should hit line 74 and return early
            await embedding_service._ensure_initialized()

            # boto3.client should NOT have been called again
            mock_client.assert_not_called()

    def test_basic_text_convergence_edge_case_line_287(self):
        """Test line 287: Return True, 1.0 when max_len == 0"""
        # Line 287 is actually unreachable with normal string input because:
        # - Empty strings are falsy and caught by line 281
        # - Non-empty strings have len > 0
        # However, we can trigger it by bypassing the type hints with custom objects

        class TruthyZeroLength:
            """Object that is truthy but has length 0"""

            def __len__(self):
                return 0

            def __bool__(self):
                return True

        obj1 = TruthyZeroLength()
        obj2 = TruthyZeroLength()

        # This should pass line 281 (truthy) but trigger line 287 (max_len == 0)
        converged, similarity = basic_text_convergence(obj1, obj2, threshold=0.9)

        # This should trigger line 287: return True, 1.0
        assert converged is True
        assert similarity == 1.0

    @pytest.mark.asyncio
    async def test_double_initialization_race_condition(self, embedding_service):
        """Test initialization race condition handling"""
        import asyncio

        # Create multiple concurrent initialization calls
        tasks = [embedding_service._ensure_initialized() for _ in range(3)]

        # Execute them concurrently
        await asyncio.gather(*tasks)

        # Should still be properly initialized only once
        assert embedding_service._initialized is True
        assert embedding_service.bedrock_runtime is not None

    def test_basic_text_convergence_edge_cases(self):
        """Test edge cases in string similarity that don't hit line 287"""
        # Test cases where strings are not empty
        test_cases = [
            ("a", "a", True, 1.0),  # Identical single chars
            ("a", "b", False, 0.0),  # Different single chars
            ("hello", "hello", True, 1.0),  # Identical strings
            ("ab", "ba", False, 0.0),  # Different order
        ]

        for current, previous, expected_converged, expected_similarity in test_cases:
            converged, similarity = basic_text_convergence(current, previous, threshold=0.9)

            assert converged == expected_converged
            assert abs(similarity - expected_similarity) < 0.001  # Float comparison

    @pytest.mark.asyncio
    async def test_initialization_with_different_embedding_services(self):
        """Test that each embedding service instance initializes independently"""
        from convergence import ConvergenceConfig

        service1 = EmbeddingService(ConvergenceConfig())
        service2 = EmbeddingService(ConvergenceConfig())

        # Initialize first service
        await service1._ensure_initialized()
        assert service1._initialized is True
        assert service2._initialized is False

        # Second service should still need initialization
        await service2._ensure_initialized()
        assert service2._initialized is True

    def test_similarity_threshold_variations(self):
        """Test our custom objects with different thresholds to ensure line 287 works"""

        class TruthyZeroLength:
            def __len__(self):
                return 0

            def __bool__(self):
                return True

        obj1 = TruthyZeroLength()
        obj2 = TruthyZeroLength()

        # Test with different thresholds - should always return True, 1.0 for line 287
        for threshold in [0.1, 0.5, 0.8, 0.95, 1.0]:
            converged, similarity = basic_text_convergence(obj1, obj2, threshold=threshold)

            # Line 287 should always return True, 1.0 regardless of threshold
            assert converged is True
            assert similarity == 1.0
