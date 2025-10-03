"""
Surgical tests for RefineEngine to achieve 100% coverage.
Specifically targets missing lines: 132-133, 189-194
"""

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from recursive_companion_mcp.legacy.bedrock_client import BedrockClient
from recursive_companion_mcp.legacy.convergence import ConvergenceDetector
from recursive_companion_mcp.legacy.domains import DomainDetector
from recursive_companion_mcp.legacy.refine_engine import RefineEngine
from recursive_companion_mcp.legacy.validation import SecurityValidator

# sys.path removed - using package imports


class TestRefineEngineCoverage:
    """Surgical tests targeting specific missing lines in RefineEngine."""

    @pytest.fixture
    def mock_bedrock(self):
        """Mock Bedrock client"""
        client = Mock(spec=BedrockClient)
        client.generate_text = AsyncMock(return_value="Generated text")
        client.get_embedding = AsyncMock(return_value=[0.1] * 100)
        return client

    @pytest.fixture
    def mock_domain_detector(self):
        """Mock domain detector"""
        detector = Mock(spec=DomainDetector)
        detector.detect_domain = Mock(return_value="technical")
        return detector

    @pytest.fixture
    def mock_validator(self):
        """Mock security validator"""
        validator = Mock(spec=SecurityValidator)
        validator.validate_prompt = Mock(return_value=(True, "Valid"))
        return validator

    @pytest.fixture
    def mock_convergence(self):
        """Mock convergence detector"""
        detector = Mock(spec=ConvergenceDetector)
        detector.cosine_similarity = Mock(return_value=0.95)
        return detector

    @pytest.fixture
    def refine_engine(self, mock_bedrock, mock_domain_detector, mock_validator, mock_convergence):
        """Create RefineEngine with all mocked dependencies"""
        engine = RefineEngine(mock_bedrock)
        engine.domain_detector = mock_domain_detector
        engine.validator = mock_validator
        engine.convergence_detector = mock_convergence
        return engine

    @pytest.mark.asyncio
    async def test_domain_auto_detection_logging(self, refine_engine, caplog):
        """Test lines 132-133: Domain auto-detection with logging"""
        # Setup domain detector to return specific domain
        refine_engine.domain_detector.detect_domain.return_value = "marketing"

        # Use return_value instead of side_effect to always return the same thing
        refine_engine.bedrock.generate_text = AsyncMock(return_value="Generated content")
        refine_engine.bedrock.get_embedding = AsyncMock(return_value=[0.1] * 100)

        # Set convergence to trigger after first iteration
        refine_engine.convergence_detector.cosine_similarity.return_value = 0.99

        # Clear any previous logs
        caplog.clear()

        # Call refine with domain="auto" to trigger auto-detection
        with caplog.at_level(logging.INFO):
            result = await refine_engine.refine("Test prompt for auto-detection", domain="auto")

        # Verify domain detection was called
        refine_engine.domain_detector.detect_domain.assert_called_once_with(
            "Test prompt for auto-detection"
        )

        # Verify logging occurred (lines 132-133)
        log_messages = [record.message for record in caplog.records]
        assert any("Auto-detected domain: marketing" in msg for msg in log_messages)

        # Verify result has the detected domain
        assert result.domain == "marketing"

    @pytest.mark.asyncio
    async def test_asyncio_timeout_error_handling(self, refine_engine):
        """Test lines 189-191: asyncio.TimeoutError exception handling"""
        # Mock bedrock to raise AsyncioTimeoutError during draft generation
        refine_engine.bedrock.generate_text = AsyncMock(
            side_effect=TimeoutError("Operation timed out")
        )

        # Call refine and expect TimeoutError to be re-raised
        with pytest.raises(TimeoutError) as exc_info:
            await refine_engine.refine("Test prompt")

        # Verify the error message matches expected format
        assert "Refinement exceeded" in str(exc_info.value)
        assert "seconds" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generic_exception_handling(self, refine_engine):
        """Test lines 192-194: Generic exception handling"""
        # Mock bedrock to raise a generic exception during embedding generation
        refine_engine.bedrock.generate_text = AsyncMock(return_value="Draft text")
        refine_engine.bedrock.get_embedding = AsyncMock(
            side_effect=ValueError("Invalid embedding request")
        )

        # Call refine and expect the generic exception to be re-raised
        with pytest.raises(ValueError) as exc_info:
            await refine_engine.refine("Test prompt")

        # Verify the original exception is preserved
        assert "Invalid embedding request" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_exception_during_revision_synthesis(self, refine_engine):
        """Test generic exception during revision synthesis"""
        # Setup successful draft and critiques, but fail on revision
        refine_engine.bedrock.generate_text = AsyncMock(
            side_effect=[
                "Initial draft",
                "Critique 1",
                "Critique 2",
                "Critique 3",  # Critiques succeed
                RuntimeError("Revision synthesis failed"),  # Revision fails
            ]
        )

        with pytest.raises(RuntimeError) as exc_info:
            await refine_engine.refine("Test prompt")

        assert "Revision synthesis failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_domain_detection_with_different_domains(self, refine_engine, caplog):
        """Test domain auto-detection with various domain types for complete logging coverage"""
        domain_types = ["technical", "business", "creative", "academic"]

        for domain_type in domain_types:
            # Reset mocks
            refine_engine.domain_detector.detect_domain.return_value = domain_type
            refine_engine.bedrock.generate_text = AsyncMock(return_value="Generated content")
            refine_engine.bedrock.get_embedding = AsyncMock(return_value=[0.1] * 100)
            refine_engine.convergence_detector.cosine_similarity.return_value = 0.99

            caplog.clear()

            with caplog.at_level(logging.INFO):
                result = await refine_engine.refine(f"Test prompt for {domain_type}", domain="auto")

            # Verify logging for each domain type
            log_messages = [record.message for record in caplog.records]
            assert any(f"Auto-detected domain: {domain_type}" in msg for msg in log_messages)
            assert result.domain == domain_type

    @pytest.mark.asyncio
    async def test_timeout_error_with_config_message(self, refine_engine):
        """Test that timeout error includes config timeout value"""
        # Mock config to have a specific timeout value
        with patch("refine_engine.config") as mock_config:
            mock_config.request_timeout = 30
            mock_config.max_iterations = 5
            mock_config.convergence_threshold = 0.98
            mock_config.parallel_critiques = 3
            mock_config.bedrock_model_id = "test-model"
            mock_config.embedding_model_id = "test-embedding"
            mock_config.critique_model_id = "test-critique"

            refine_engine.bedrock.generate_text = AsyncMock(side_effect=TimeoutError())

            with pytest.raises(TimeoutError) as exc_info:
                await refine_engine.refine("Test prompt")

            # Verify timeout message includes the config value
            assert "30 seconds" in str(exc_info.value)
