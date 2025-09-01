#!/usr/bin/env python3

"""
Surgical tests for incremental_engine.py - targeting specific missing lines to achieve 100% coverage.
These tests focus on defensive programming branches and error handling edge cases.

Missing lines to cover: 46-47, 165-167, 211-213, 292, 311, 325, 339, 837
"""

import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add src to path like other tests
sys.path.insert(0, "./src")


@pytest.fixture
def mock_bedrock():
    """Mock bedrock client"""
    mock = AsyncMock()
    mock.generate_text.return_value = "Generated response"
    mock.get_embedding.return_value = [0.1, 0.2, 0.3]
    mock.bedrock_client = Mock()
    return mock


@pytest.fixture
def mock_domain_detector():
    """Mock domain detector"""
    mock = Mock()
    mock.detect_domain.return_value = "technical"
    return mock


@pytest.fixture
def mock_validator():
    """Mock validator"""
    mock = Mock()
    mock.validate_prompt.return_value = (True, "Valid")
    return mock


@pytest.mark.asyncio
class TestIncrementalEngineSurgical:
    """Surgical tests targeting specific missing lines in incremental_engine.py"""

    async def test_cot_not_available_line_292(
        self, mock_bedrock, mock_domain_detector, mock_validator
    ):
        """
        Test Chain of Thought not available scenario - targets line 292

        Line 292: return [] when COT not available in get_cot_tools()

        Note: Lines 46-47 (ImportError handler) are extremely difficult to test
        due to import-time execution complexity. This test covers the related
        functionality when COT_AVAILABLE is False.
        """
        from incremental_engine import IncrementalRefineEngine

        # Test the case where COT_AVAILABLE is False by patching it
        with patch("incremental_engine.COT_AVAILABLE", False):
            engine = IncrementalRefineEngine(mock_bedrock, mock_domain_detector, mock_validator)

            # This should hit line 292: return [] when COT not available
            tools = engine.get_cot_tools()
            assert tools == []

    async def test_session_persistence_exception_lines_165_167(
        self, mock_bedrock, mock_domain_detector, mock_validator
    ):
        """
        Test session persistence failure - targets lines 165-167

        Lines 165-167: Exception handling in _persist_session() method
        """
        from incremental_engine import SessionManager

        # Create session manager with mocked persistence that fails
        with patch("incremental_engine.persistence_manager") as mock_pm:
            # Make save_session raise an exception
            mock_pm.save_session.side_effect = ValueError("Serialization failed")

            session_manager = SessionManager()

            # Create a session (this will trigger _persist_session)
            with patch("incremental_engine.logger") as mock_logger:
                session = await session_manager.create_session("test prompt", "technical", {})

                # Verify the error was logged (lines 165-167 were executed)
                mock_logger.error.assert_called()
                error_call = mock_logger.error.call_args
                assert "Failed to persist session" in str(error_call)
                assert session.session_id in str(error_call)

    async def test_session_reconstruction_exception_lines_211_213(
        self, mock_bedrock, mock_domain_detector, mock_validator
    ):
        """
        Test session reconstruction failure - targets lines 211-213

        Lines 211-213: Exception handling in _reconstruct_session() method
        """
        from incremental_engine import SessionManager

        session_manager = SessionManager()

        # Create malformed session data that will cause reconstruction to fail
        malformed_data = {
            "session_id": "test-id",
            "prompt": "test prompt",
            "domain": "technical",
            "status": "invalid_status",  # This will cause ValueError when creating RefinementStatus
            "current_iteration": "not_a_number",  # This will cause issues
            "created_at": "invalid_date_format",  # This will cause datetime parsing to fail
        }

        with patch("incremental_engine.logger") as mock_logger:
            # This should trigger the exception handler in _reconstruct_session
            result = session_manager._reconstruct_session(malformed_data)

            # Verify it returns None and logs error (lines 211-213)
            assert result is None
            mock_logger.error.assert_called()
            error_call = mock_logger.error.call_args
            assert "Failed to reconstruct session" in str(error_call)

    async def test_cot_processing_no_messages_line_311(
        self, mock_bedrock, mock_domain_detector, mock_validator
    ):
        """
        Test CoT processing with no messages - targets line 311

        Line 311: return "No response generated" when no messages found
        """
        from incremental_engine import IncrementalRefineEngine

        engine = IncrementalRefineEngine(mock_bedrock, mock_domain_detector, mock_validator)

        # Mock COT as available but create request with empty messages
        with patch("incremental_engine.COT_AVAILABLE", True):
            request = {
                "messages": [],
                "system": [{"text": "system prompt"}],
            }  # Empty messages list

            # This should hit line 311: return "No response generated"
            result = await engine._process_with_cot(None, request)
            assert result == "No response generated"

    async def test_cot_processing_no_text_content_line_325(
        self, mock_bedrock, mock_domain_detector, mock_validator
    ):
        """
        Test CoT processing with no text content in result - targets line 325

        Line 325: return "No response generated" when no text content extracted
        """
        from incremental_engine import IncrementalRefineEngine

        engine = IncrementalRefineEngine(mock_bedrock, mock_domain_detector, mock_validator)

        # Mock CoT processor that returns result without text content
        mock_processor = AsyncMock()
        mock_processor.process_tool_loop.return_value = {
            "output": {
                "message": {
                    "content": [
                        {"type": "image", "data": "..."},  # No text content
                        {"type": "other", "value": "..."},  # Still no text
                    ]
                }
            }
        }

        with patch("incremental_engine.COT_AVAILABLE", True):
            request = {
                "messages": [{"role": "user", "content": [{"text": "test"}]}],
                "system": [{"text": "system"}],
            }

            # This should hit line 325: return "No response generated"
            result = await engine._process_with_cot(mock_processor, request)
            assert result == "No response generated"

    async def test_cot_processing_complete_failure_line_339(
        self, mock_bedrock, mock_domain_detector, mock_validator
    ):
        """
        Test CoT processing complete failure - targets line 339

        Line 339: return "Error processing request" when basic generation also fails
        """
        from incremental_engine import IncrementalRefineEngine

        engine = IncrementalRefineEngine(mock_bedrock, mock_domain_detector, mock_validator)

        # Mock CoT processor that raises exception
        mock_processor = AsyncMock()
        mock_processor.process_tool_loop.side_effect = Exception("CoT processing failed")

        # Mock bedrock generate_text to also fail (no messages case)
        mock_bedrock.generate_text.side_effect = Exception("Bedrock also failed")

        with patch("incremental_engine.COT_AVAILABLE", True):
            with patch("incremental_engine.logger"):
                request = {
                    "messages": [],  # Empty messages to trigger second failure path
                    "system": [{"text": "system"}],
                }

                # This should hit line 339: return "Error processing request"
                result = await engine._process_with_cot(mock_processor, request)
                assert result == "Error processing request"

    async def test_convergence_score_above_90_percent_line_837(
        self, mock_bedrock, mock_domain_detector, mock_validator
    ):
        """
        Test convergence score > 0.9 scenario - targets line 837

        Line 837: _ai_prediction assignment when convergence_score > 0.9
        """
        from domains import DomainDetector
        from incremental_engine import IncrementalRefineEngine, RefinementStatus
        from validation import SecurityValidator

        # Use real domain detector and validator like other tests
        real_detector = DomainDetector()
        real_validator = SecurityValidator()
        engine = IncrementalRefineEngine(mock_bedrock, real_detector, real_validator)

        # Start a refinement session
        result = await engine.start_refinement("Test prompt for convergence", "technical")
        session_id = result["session_id"]

        # Get the session and advance it to revising state with high convergence
        session = await engine.session_manager.get_session(session_id)
        session.status = RefinementStatus.REVISING
        session.current_iteration = 2
        session.current_draft = "Current draft"
        session.critiques = ["Good critique"]

        # Mock generate_text for revision step
        mock_bedrock.generate_text.return_value = "Revised draft content"

        # Mock embeddings to return high similarity (> 0.9 but < threshold)
        mock_bedrock.get_embedding.side_effect = [
            [0.9, 0.1, 0.0],  # revision embedding
            [0.95, 0.05, 0.0],  # current draft embedding
        ]

        # Mock cosine similarity to return > 0.9 but < convergence threshold
        with patch.object(
            engine.convergence_detector, "cosine_similarity", return_value=0.92
        ):  # > 0.9 but < 0.95 threshold
            # Continue refinement - this should trigger the revision step
            result = await engine.continue_refinement(session_id)

            # Verify the AI prediction was set for scores > 0.9 (line 837)
            assert result["success"] is True
            assert "_ai_prediction" in result
            assert "Likely to converge in 1-2 more iterations" in result["_ai_prediction"]
            assert "_ai_suggestion" in result
            assert "Consider abort_refinement" in result["_ai_suggestion"]
