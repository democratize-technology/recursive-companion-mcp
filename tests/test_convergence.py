"""
Comprehensive tests for convergence.py module to achieve 90%+ coverage.
Tests cover AWS Bedrock integration, caching, mathematical operations, and error handling.
"""

import asyncio
import hashlib
import json
from collections import OrderedDict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from recursive_companion_mcp.core.convergence import (
    ConvergenceConfig,
    ConvergenceDetector,
    EmbeddingService,
    basic_text_convergence,
    create_detector_for_tool,
    simple_convergence_check,
)

# sys.path removed - using package imports


class TestConvergenceConfig:
    """Test ConvergenceConfig dataclass"""

    def test_default_configuration(self):
        """Test default config values"""
        config = ConvergenceConfig()

        assert config.threshold == 0.95
        assert config.embedding_model_id == "amazon.titan-embed-text-v1"
        assert config.aws_region == "us-east-1"
        assert config.cache_size == 1000
        assert config.max_text_length == 8000

    def test_custom_configuration(self):
        """Test custom config values"""
        config = ConvergenceConfig(
            threshold=0.9,
            embedding_model_id="custom-model",
            aws_region="us-west-2",
            cache_size=500,
            max_text_length=4000,
        )

        assert config.threshold == 0.9
        assert config.embedding_model_id == "custom-model"
        assert config.aws_region == "us-west-2"
        assert config.cache_size == 500
        assert config.max_text_length == 4000


class TestEmbeddingService:
    """Test EmbeddingService class"""

    def test_initialization(self):
        """Test EmbeddingService initialization"""
        config = ConvergenceConfig()
        service = EmbeddingService(config)

        assert service.config == config
        assert service.bedrock_runtime is None
        assert isinstance(service._embedding_cache, OrderedDict)
        assert service._cache_hits == 0
        assert service._cache_misses == 0
        assert service._initialized is False

    @pytest.mark.asyncio
    async def test_ensure_initialized_success(self):
        """Test successful initialization"""
        config = ConvergenceConfig()
        service = EmbeddingService(config)

        with patch("boto3.client") as mock_boto:
            mock_client = Mock()
            mock_boto.return_value = mock_client

            await service._ensure_initialized()

            assert service._initialized is True
            assert service.bedrock_runtime == mock_client
            mock_boto.assert_called_once_with(
                service_name="bedrock-runtime", region_name="us-east-1"
            )

    @pytest.mark.asyncio
    async def test_ensure_initialized_failure(self):
        """Test initialization failure"""
        config = ConvergenceConfig()
        service = EmbeddingService(config)

        with patch("boto3.client") as mock_boto:
            mock_boto.side_effect = Exception("AWS error")

            with pytest.raises(Exception, match="AWS error"):
                await service._ensure_initialized()

            assert service._initialized is False

    @pytest.mark.asyncio
    async def test_ensure_initialized_double_call(self):
        """Test that double initialization doesn't re-initialize"""
        config = ConvergenceConfig()
        service = EmbeddingService(config)

        with patch("boto3.client") as mock_boto:
            mock_client = Mock()
            mock_boto.return_value = mock_client

            # First call
            await service._ensure_initialized()
            # Second call
            await service._ensure_initialized()

            # Should only call boto3.client once
            assert mock_boto.call_count == 1

    @pytest.mark.asyncio
    async def test_ensure_initialized_concurrent_calls(self):
        """Test concurrent initialization calls"""
        config = ConvergenceConfig()
        service = EmbeddingService(config)

        with patch("boto3.client") as mock_boto:
            mock_client = Mock()
            mock_boto.return_value = mock_client

            # Start multiple concurrent initialization tasks
            tasks = [service._ensure_initialized() for _ in range(5)]
            await asyncio.gather(*tasks)

            # Should only call boto3.client once due to lock
            assert mock_boto.call_count == 1
            assert service._initialized is True

    @pytest.mark.asyncio
    async def test_get_embedding_cache_hit(self):
        """Test embedding retrieval with cache hit"""
        config = ConvergenceConfig()
        service = EmbeddingService(config)

        # Pre-populate cache
        text = "test text"
        text_hash = hashlib.md5(text.encode()).hexdigest()
        expected_embedding = [0.1, 0.2, 0.3]
        service._embedding_cache[text_hash] = expected_embedding

        with patch("recursive_companion_mcp.core.convergence.logger") as mock_logger:
            embedding = await service.get_embedding(text)

            assert embedding == expected_embedding
            assert service._cache_hits == 1
            assert service._cache_misses == 0
            mock_logger.debug.assert_called_once()

            # Verify cache entry was moved to end (LRU)
            assert list(service._embedding_cache.keys())[-1] == text_hash

    @pytest.mark.asyncio
    async def test_get_embedding_cache_miss(self):
        """Test embedding retrieval with cache miss"""
        config = ConvergenceConfig()
        service = EmbeddingService(config)

        text = "test text"
        expected_embedding = [0.1, 0.2, 0.3]

        with patch.object(service, "_ensure_initialized") as mock_init:
            mock_runtime = Mock()
            mock_response = {"body": Mock()}
            mock_response["body"].read.return_value = json.dumps({"embedding": expected_embedding})
            mock_runtime.invoke_model.return_value = mock_response
            service.bedrock_runtime = mock_runtime

            embedding = await service.get_embedding(text)

            assert embedding == expected_embedding
            assert service._cache_hits == 0
            assert service._cache_misses == 1

            # Verify embedding was cached
            text_hash = hashlib.md5(text.encode()).hexdigest()
            assert service._embedding_cache[text_hash] == expected_embedding

            mock_init.assert_called_once()
            mock_runtime.invoke_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_embedding_text_truncation(self):
        """Test text truncation for oversized input"""
        config = ConvergenceConfig(max_text_length=10)
        service = EmbeddingService(config)

        long_text = "a" * 20
        expected_embedding = [0.1, 0.2, 0.3]

        with patch.object(service, "_ensure_initialized"):
            mock_runtime = Mock()
            mock_response = {"body": Mock()}
            mock_response["body"].read.return_value = json.dumps({"embedding": expected_embedding})
            mock_runtime.invoke_model.return_value = mock_response
            service.bedrock_runtime = mock_runtime

            await service.get_embedding(long_text)

            # Verify text was truncated
            call_args = mock_runtime.invoke_model.call_args[1]
            body_data = json.loads(call_args["body"])
            assert len(body_data["inputText"]) == 10

    @pytest.mark.asyncio
    async def test_get_embedding_api_failure(self):
        """Test embedding API failure"""
        config = ConvergenceConfig()
        service = EmbeddingService(config)

        with patch.object(service, "_ensure_initialized"):
            mock_runtime = Mock()
            mock_runtime.invoke_model.side_effect = Exception("API error")
            service.bedrock_runtime = mock_runtime

            with patch("recursive_companion_mcp.core.convergence.logger") as mock_logger:
                with pytest.raises(Exception, match="API error"):
                    await service.get_embedding("test text")

                mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_embedding_cache_size_management(self):
        """Test cache size management and LRU eviction"""
        config = ConvergenceConfig(cache_size=2)
        service = EmbeddingService(config)

        expected_embedding = [0.1, 0.2, 0.3]

        with patch.object(service, "_ensure_initialized"):
            mock_runtime = Mock()
            mock_response = {"body": Mock()}
            mock_response["body"].read.return_value = json.dumps({"embedding": expected_embedding})
            mock_runtime.invoke_model.return_value = mock_response
            service.bedrock_runtime = mock_runtime

            # Add items to fill cache
            await service.get_embedding("text1")
            await service.get_embedding("text2")
            assert len(service._embedding_cache) == 2

            # Add one more to trigger eviction
            await service.get_embedding("text3")
            assert len(service._embedding_cache) == 2

            # First item should be evicted
            text1_hash = hashlib.md5(b"text1").hexdigest()
            assert text1_hash not in service._embedding_cache

    def test_get_cache_hit_rate_no_requests(self):
        """Test cache hit rate with no requests"""
        config = ConvergenceConfig()
        service = EmbeddingService(config)

        assert service.get_cache_hit_rate() == 0.0

    def test_get_cache_hit_rate_with_requests(self):
        """Test cache hit rate calculation"""
        config = ConvergenceConfig()
        service = EmbeddingService(config)

        service._cache_hits = 7
        service._cache_misses = 3

        assert service.get_cache_hit_rate() == 0.7

    def test_get_cache_stats(self):
        """Test cache statistics"""
        config = ConvergenceConfig()
        service = EmbeddingService(config)

        service._cache_hits = 10
        service._cache_misses = 5
        service._embedding_cache["key1"] = [0.1, 0.2]
        service._embedding_cache["key2"] = [0.3, 0.4]

        stats = service.get_cache_stats()

        assert stats == {"hits": 10, "misses": 5, "hit_rate": 2 / 3, "entries": 2}


class TestConvergenceDetector:
    """Test ConvergenceDetector class"""

    def test_initialization_default_config(self):
        """Test initialization with default config"""
        detector = ConvergenceDetector()

        assert isinstance(detector.config, ConvergenceConfig)
        assert detector.config.threshold == 0.95
        assert isinstance(detector.embedding_service, EmbeddingService)
        assert detector._convergence_history == []

    def test_initialization_custom_config(self):
        """Test initialization with custom config"""
        config = ConvergenceConfig(threshold=0.9)
        detector = ConvergenceDetector(config)

        assert detector.config == config
        assert detector.config.threshold == 0.9

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity with identical vectors"""
        detector = ConvergenceDetector()

        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0, 3.0]

        similarity = detector.cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-10

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors"""
        detector = ConvergenceDetector()

        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]

        similarity = detector.cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-10

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity with opposite vectors"""
        detector = ConvergenceDetector()

        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]

        similarity = detector.cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-10

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector"""
        detector = ConvergenceDetector()

        vec1 = [0.0, 0.0]
        vec2 = [1.0, 2.0]

        similarity = detector.cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_cosine_similarity_both_zero_vectors(self):
        """Test cosine similarity with both zero vectors"""
        detector = ConvergenceDetector()

        vec1 = [0.0, 0.0]
        vec2 = [0.0, 0.0]

        similarity = detector.cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    @pytest.mark.asyncio
    async def test_is_converged_empty_strings(self):
        """Test convergence check with empty strings"""
        detector = ConvergenceDetector()

        # Test empty current
        converged, score = await detector.is_converged("", "previous")
        assert converged is False
        assert score == 0.0

        # Test empty previous
        converged, score = await detector.is_converged("current", "")
        assert converged is False
        assert score == 0.0

        # Test both empty
        converged, score = await detector.is_converged("", "")
        assert converged is False
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_is_converged_success(self):
        """Test successful convergence check"""
        detector = ConvergenceDetector()

        current_embedding = [0.8, 0.6]
        previous_embedding = [0.6, 0.8]

        # Mock embedding service
        mock_service = AsyncMock()
        mock_service.get_embedding.side_effect = [current_embedding, previous_embedding]
        detector.embedding_service = mock_service

        with patch("recursive_companion_mcp.core.convergence.logger") as mock_logger:
            converged, score = await detector.is_converged("current", "previous", threshold=0.5)

            assert isinstance(converged, bool)
            assert isinstance(score, float)
            assert len(detector._convergence_history) == 1

            history_entry = detector._convergence_history[0]
            assert history_entry["score"] == score
            assert history_entry["threshold"] == 0.5
            assert history_entry["converged"] == converged

            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_converged_embedding_failure(self):
        """Test convergence check with embedding failure"""
        detector = ConvergenceDetector()

        # Mock embedding service to fail
        mock_service = AsyncMock()
        mock_service.get_embedding.side_effect = Exception("Embedding failed")
        detector.embedding_service = mock_service

        with patch("recursive_companion_mcp.core.convergence.logger") as mock_logger:
            converged, score = await detector.is_converged("current", "previous")

            assert converged is False
            assert score == 0.0
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_converged_default_threshold(self):
        """Test convergence check with default threshold"""
        detector = ConvergenceDetector()

        # Mock embedding service
        mock_service = AsyncMock()
        mock_service.get_embedding.side_effect = [[1.0, 0.0], [1.0, 0.0]]
        detector.embedding_service = mock_service

        converged, score = await detector.is_converged("current", "previous")

        # Should use default threshold from config
        history_entry = detector._convergence_history[0]
        assert history_entry["threshold"] == detector.config.threshold

    def test_get_convergence_history_empty(self):
        """Test getting convergence history when empty"""
        detector = ConvergenceDetector()

        history = detector.get_convergence_history()
        assert history == []

    def test_get_convergence_history_with_data(self):
        """Test getting convergence history with data"""
        detector = ConvergenceDetector()

        # Manually add history entries
        detector._convergence_history = [
            {"score": 0.8, "threshold": 0.9, "converged": False},
            {"score": 0.95, "threshold": 0.9, "converged": True},
        ]

        history = detector.get_convergence_history()

        assert len(history) == 2
        assert history[0]["score"] == 0.8
        assert history[1]["converged"] is True

    def test_get_stats_empty_history(self):
        """Test getting stats with empty history"""
        detector = ConvergenceDetector()

        stats = detector.get_stats()
        assert stats == {"total_checks": 0}

    def test_get_stats_with_history(self):
        """Test getting stats with convergence history"""
        detector = ConvergenceDetector()

        # Add history entries
        detector._convergence_history = [
            {"score": 0.8, "threshold": 0.9, "converged": False},
            {"score": 0.95, "threshold": 0.9, "converged": True},
            {"score": 0.7, "threshold": 0.9, "converged": False},
        ]

        # Mock embedding service stats
        mock_service = Mock()
        mock_service.get_cache_stats.return_value = {"cache": "stats"}
        detector.embedding_service = mock_service

        stats = detector.get_stats()

        assert stats["total_checks"] == 3
        assert stats["convergences"] == 1
        assert stats["avg_score"] == (0.8 + 0.95 + 0.7) / 3
        assert stats["max_score"] == 0.95
        assert stats["min_score"] == 0.7
        assert stats["embedding_stats"] == {"cache": "stats"}


class TestToolSpecificDetectors:
    """Test tool-specific detector creation"""

    def test_create_detector_for_known_tools(self):
        """Test creating detectors for known tools"""
        tools_and_thresholds = {
            "devil-advocate": 0.70,
            "decision-matrix": 0.90,
            "conversation-tree": 0.85,
            "rubber-duck": 0.95,
            "hindsight": 0.95,
            "context-switcher": 0.85,
        }

        for tool_name, expected_threshold in tools_and_thresholds.items():
            with patch("recursive_companion_mcp.core.convergence.logger") as mock_logger:
                detector = create_detector_for_tool(tool_name)

                assert isinstance(detector, ConvergenceDetector)
                assert detector.config.threshold == expected_threshold
                mock_logger.info.assert_called_once()

    def test_create_detector_for_unknown_tool(self):
        """Test creating detector for unknown tool (uses default)"""
        with patch("recursive_companion_mcp.core.convergence.logger") as mock_logger:
            detector = create_detector_for_tool("unknown-tool")

            assert detector.config.threshold == 0.95  # Default threshold
            mock_logger.info.assert_called_once()


class TestSimpleConvergenceCheck:
    """Test simple_convergence_check utility function"""

    @pytest.mark.asyncio
    async def test_simple_convergence_check(self):
        """Test simple convergence check utility"""
        with patch(
            "recursive_companion_mcp.core.convergence.ConvergenceDetector"
        ) as mock_detector_class:
            mock_detector = AsyncMock()  # Use AsyncMock for async method
            mock_detector.is_converged.return_value = (True, 0.96)
            mock_detector_class.return_value = mock_detector

            result = await simple_convergence_check("current", "previous", 0.9)

            assert result is True
            mock_detector_class.assert_called_once()
            config_arg = mock_detector_class.call_args[0][0]
            assert config_arg.threshold == 0.9

    @pytest.mark.asyncio
    async def test_simple_convergence_check_default_threshold(self):
        """Test simple convergence check with default threshold"""
        with patch(
            "recursive_companion_mcp.core.convergence.ConvergenceDetector"
        ) as mock_detector_class:
            mock_detector = AsyncMock()  # Use AsyncMock for async method
            mock_detector.is_converged.return_value = (False, 0.9)
            mock_detector_class.return_value = mock_detector

            result = await simple_convergence_check("current", "previous")

            assert result is False
            config_arg = mock_detector_class.call_args[0][0]
            assert config_arg.threshold == 0.95


class TestBasicTextConvergence:
    """Test basic_text_convergence fallback function"""

    def test_basic_text_convergence_empty_strings(self):
        """Test basic text convergence with empty strings"""
        # Empty current
        converged, score = basic_text_convergence("", "previous")
        assert converged is False
        assert score == 0.0

        # Empty previous
        converged, score = basic_text_convergence("current", "")
        assert converged is False
        assert score == 0.0

        # Both empty
        converged, score = basic_text_convergence("", "")
        assert converged is False
        assert score == 0.0

    def test_basic_text_convergence_both_empty_edge_case(self):
        """Test edge case where both strings are empty (max_len = 0)"""
        converged, score = basic_text_convergence("", "", threshold=0.95)
        assert converged is False
        assert score == 0.0

    def test_basic_text_convergence_identical_strings(self):
        """Test basic text convergence with identical strings"""
        text = "identical text"
        converged, score = basic_text_convergence(text, text)

        assert converged is True
        assert score == 1.0

    def test_basic_text_convergence_similar_strings(self):
        """Test basic text convergence with similar strings"""
        current = "hello world"
        previous = "hello world!"  # One extra character

        converged, score = basic_text_convergence(current, previous, threshold=0.9)

        # Should have high similarity but not perfect
        assert 0.9 <= score < 1.0
        assert converged == (score >= 0.9)

    def test_basic_text_convergence_different_strings(self):
        """Test basic text convergence with very different strings"""
        current = "hello"
        previous = "world"

        converged, score = basic_text_convergence(current, previous, threshold=0.5)

        assert score < 0.5
        assert converged is False

    def test_basic_text_convergence_different_lengths(self):
        """Test basic text convergence with different length strings"""
        current = "short"
        previous = "much longer string"

        converged, score = basic_text_convergence(current, previous)

        # Similarity should be based on max length
        max_len = max(len(current), len(previous))
        common_chars = sum(c1 == c2 for c1, c2 in zip(current, previous, strict=False))
        expected_score = common_chars / max_len

        assert abs(score - expected_score) < 1e-10

    def test_basic_text_convergence_threshold_edge_cases(self):
        """Test basic text convergence with edge case thresholds"""
        current = "test"
        previous = "test"

        # Threshold of 1.0 should work with identical strings
        converged, score = basic_text_convergence(current, previous, threshold=1.0)
        assert converged is True
        assert score == 1.0

        # Threshold of 0.0 should always converge
        current = "different"
        previous = "strings"
        converged, score = basic_text_convergence(current, previous, threshold=0.0)
        assert converged is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.convergence"])
