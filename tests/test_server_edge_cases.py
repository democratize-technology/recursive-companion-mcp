#!/usr/bin/env python3
"""
Test suite specifically targeting missing server.py coverage lines.
Focuses on defensive programming patterns, error handling, and edge cases.

MISSING LINES TARGETED: 225, 272-274, 287, 302-304, 316, 331-333, 346,
366-368, 413, 428, 440, 456-458, 471, 487, 503-504, 583-594
"""
import json
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, "./src")

import server


class TestEngineInitializationFailures:
    """Test all 'if not incremental_engine' defensive checks"""

    @pytest.fixture
    def reset_engine(self):
        """Reset the global engine state for isolated testing"""
        original_engine = server.incremental_engine
        server.incremental_engine = None
        yield
        server.incremental_engine = original_engine

    @pytest.mark.asyncio
    async def test_start_refinement_no_engine(self, reset_engine):
        """Test line 225: start_refinement when incremental_engine is None"""
        result = await server.handle_call_tool("start_refinement", {"prompt": "test"})

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["error"] == "Incremental engine not initialized"
        assert response["success"] is False

    @pytest.mark.asyncio
    async def test_get_refinement_status_no_engine(self, reset_engine):
        """Test line 287: get_refinement_status when incremental_engine is None"""
        result = await server.handle_call_tool("get_refinement_status", {"session_id": "test"})

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["error"] == "Incremental engine not initialized"
        assert response["success"] is False

    @pytest.mark.asyncio
    async def test_get_final_result_no_engine(self, reset_engine):
        """Test line 316: get_final_result when incremental_engine is None"""
        result = await server.handle_call_tool("get_final_result", {"session_id": "test"})

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["error"] == "Incremental engine not initialized"
        assert response["success"] is False

    @pytest.mark.asyncio
    async def test_list_refinement_sessions_no_engine(self, reset_engine):
        """Test line 346: list_refinement_sessions when incremental_engine is None"""
        result = await server.handle_call_tool("list_refinement_sessions", {})

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["error"] == "Incremental engine not initialized"
        assert response["success"] is False

    @pytest.mark.asyncio
    async def test_abort_refinement_no_engine(self, reset_engine):
        """Test line 428: abort_refinement when incremental_engine is None"""
        result = await server.handle_call_tool("abort_refinement", {"session_id": "test"})

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["error"] == "Incremental engine not initialized"
        assert response["success"] is False

    @pytest.mark.asyncio
    async def test_quick_refine_no_engine(self, reset_engine):
        """Test line 471: quick_refine when incremental_engine is None"""
        result = await server.handle_call_tool("quick_refine", {"prompt": "test"})

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["error"] == "Incremental engine not initialized"
        assert response["success"] is False


class TestSpecificMissingLines:
    """Test the last remaining missing lines"""

    @pytest.fixture
    def reset_engine(self):
        """Reset the global engine state for isolated testing"""
        original_engine = server.incremental_engine
        server.incremental_engine = None
        yield
        server.incremental_engine = original_engine

    @pytest.mark.asyncio
    async def test_continue_refinement_no_engine_line_225(self, reset_engine):
        """Test line 225 specifically: continue_refinement when incremental_engine is None"""
        # This should hit the exact path for line 225
        result = await server.handle_call_tool("continue_refinement", {"session_id": "test"})

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["error"] == "Incremental engine not initialized"
        assert response["success"] is False

    @pytest.mark.asyncio
    async def test_current_session_successful_status_line_413(self):
        """Test line 413: current_session successful get_status call"""
        # Store original state
        original_engine = server.incremental_engine
        original_session_id = server.session_tracker.current_session_id

        # Create a mock engine with successful get_status
        mock_engine = Mock()
        mock_engine.get_status = AsyncMock(
            return_value={
                "success": True,
                "session_id": "test-session",
                "status": "active",
                "iterations": 1,
            }
        )
        server.incremental_engine = mock_engine

        # Set a current session
        server.session_tracker.current_session_id = "test-session"

        try:
            result = await server.handle_call_tool("current_session", {})

            # This should hit line 413 (successful return path)
            assert len(result) == 1
            response = json.loads(result[0].text)
            assert response["success"] is True
            assert response["session_id"] == "test-session"

            # Verify get_status was called with the current session ID
            mock_engine.get_status.assert_called_once_with("test-session")
        finally:
            # Restore original state
            server.incremental_engine = original_engine
            server.session_tracker.current_session_id = original_session_id


class TestExceptionHandling:
    """Test exception handling paths in each tool function"""

    @pytest.fixture
    def mock_failing_engine(self):
        """Create a mock engine that throws exceptions"""
        mock_engine = Mock()
        mock_engine.continue_refinement = AsyncMock(side_effect=Exception("Connection timeout"))
        mock_engine.get_status = AsyncMock(side_effect=Exception("Session not found"))
        mock_engine.get_final_result = AsyncMock(side_effect=Exception("Network error"))
        mock_engine.session_manager.list_active_sessions = Mock(
            side_effect=Exception("Database error")
        )
        mock_engine.abort_refinement = AsyncMock(side_effect=Exception("Abort failed"))

        original_engine = server.incremental_engine
        server.incremental_engine = mock_engine
        yield mock_engine
        server.incremental_engine = original_engine

    @pytest.mark.asyncio
    async def test_continue_refinement_exception_handling(self, mock_failing_engine):
        """Test lines 272-274: continue_refinement exception handler"""
        result = await server.handle_call_tool("continue_refinement", {"session_id": "test"})

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["error"] == "Failed to continue refinement: Connection timeout"
        assert response["success"] is False

    @pytest.mark.asyncio
    async def test_get_refinement_status_exception_handling(self, mock_failing_engine):
        """Test lines 302-304: get_refinement_status exception handler"""
        result = await server.handle_call_tool("get_refinement_status", {"session_id": "test"})

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["error"] == "Failed to get status: Session not found"
        assert response["success"] is False

    @pytest.mark.asyncio
    async def test_get_final_result_exception_handling(self, mock_failing_engine):
        """Test lines 331-333: get_final_result exception handler"""
        result = await server.handle_call_tool("get_final_result", {"session_id": "test"})

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["error"] == "Failed to get final result: Network error"
        assert response["success"] is False

    @pytest.mark.asyncio
    async def test_list_refinement_sessions_exception_handling(self, mock_failing_engine):
        """Test lines 366-368: list_refinement_sessions exception handler"""
        result = await server.handle_call_tool("list_refinement_sessions", {})

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["error"] == "Failed to list sessions: Database error"
        assert response["success"] is False

    @pytest.mark.asyncio
    async def test_abort_refinement_exception_handling(self, mock_failing_engine):
        """Test lines 456-458: abort_refinement exception handler"""
        result = await server.handle_call_tool("abort_refinement", {"session_id": "test"})

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["error"] == "Failed to abort refinement: Abort failed"
        assert response["success"] is False


class TestCurrentSessionEdgeCases:
    """Test current_session functionality edge cases"""

    @pytest.fixture
    def mock_engine_with_sessions(self):
        """Create a mock engine with session management"""
        mock_engine = Mock()
        mock_session_manager = Mock()
        mock_engine.session_manager = mock_session_manager

        original_engine = server.incremental_engine
        server.incremental_engine = mock_engine
        yield mock_engine, mock_session_manager
        server.incremental_engine = original_engine

    @pytest.mark.asyncio
    async def test_current_session_exception_handling(self, mock_engine_with_sessions):
        """Test line 413: current_session exception handler when getting status fails"""
        mock_engine, mock_session_manager = mock_engine_with_sessions

        # Set up session tracker to have a current session
        server.session_tracker.set_current_session("test-session", "test prompt")

        # Mock get_status to throw exception
        mock_engine.get_status = AsyncMock(side_effect=Exception("Status retrieval failed"))

        result = await server.handle_call_tool("current_session", {})

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["success"] is False
        assert response["error"] == "Failed to get current session: Status retrieval failed"

        # Clean up
        server.session_tracker.current_session_id = None

    @pytest.mark.asyncio
    async def test_abort_refinement_no_session(self):
        """Test line 440: abort_refinement when no session is provided or current"""
        # Temporarily store original engine and session state
        original_engine = server.incremental_engine
        original_session_id = server.session_tracker.current_session_id

        # Set up a mock engine (but we won't actually call its methods due to early return)
        mock_engine = Mock()
        mock_engine.abort_refinement = AsyncMock()
        server.incremental_engine = mock_engine

        # Ensure no current session
        server.session_tracker.current_session_id = None

        try:
            result = await server.handle_call_tool("abort_refinement", {})

            assert len(result) == 1
            response = json.loads(result[0].text)
            assert response["success"] is False
            assert response["error"] == "No session specified and no current session active"
        finally:
            # Restore original engine and session state
            server.incremental_engine = original_engine
            server.session_tracker.current_session_id = original_session_id


class TestQuickRefineEdgeCases:
    """Test quick_refine specific edge cases and logging paths"""

    @pytest.fixture
    def mock_quick_refine_engine(self):
        """Create engine that supports quick_refine testing"""
        mock_engine = Mock()
        mock_engine.start_refinement = AsyncMock()
        mock_engine.continue_refinement = AsyncMock()
        mock_engine.abort_refinement = AsyncMock()

        original_engine = server.incremental_engine
        server.incremental_engine = mock_engine
        yield mock_engine
        server.incremental_engine = original_engine

    @pytest.mark.asyncio
    async def test_quick_refine_start_failure(self, mock_quick_refine_engine):
        """Test line 487: quick_refine when start_refinement fails"""
        # Mock start_refinement to fail
        mock_quick_refine_engine.start_refinement.return_value = {
            "success": False,
            "error": "Validation failed",
        }

        result = await server.handle_call_tool("quick_refine", {"prompt": "test"})

        assert len(result) == 1
        response = json.loads(result[0].text)
        # Should return the failed start result directly
        assert response["success"] is False
        assert response["error"] == "Validation failed"

    @pytest.mark.asyncio
    @patch("server.time.time")
    async def test_quick_refine_with_draft_preview_logging(
        self, mock_time, mock_quick_refine_engine
    ):
        """Test lines 503-504: quick_refine draft preview logging"""
        # Mock time progression for timeout logic
        mock_time.side_effect = [0, 0.1, 0.2, 50]  # Start, iteration 1, iteration 2, timeout

        # Mock successful start
        mock_quick_refine_engine.start_refinement.return_value = {
            "success": True,
            "session_id": "test-session",
        }

        # Mock continue_refinement responses with draft_preview
        mock_quick_refine_engine.continue_refinement.side_effect = [
            {"status": "in_progress", "draft_preview": "First draft preview content"},
            {"status": "in_progress", "draft_preview": "Second draft preview content"},
            {"status": "in_progress", "draft_preview": "Third draft preview content"},
        ]

        # Mock abort for timeout
        mock_quick_refine_engine.abort_refinement.return_value = {"final_answer": "Timeout result"}

        with patch("server.logger") as mock_logger:
            result = await server.handle_call_tool(
                "quick_refine", {"prompt": "test", "max_wait": 30}
            )

            # Verify the draft preview logging occurred (lines 503-504)
            assert mock_logger.debug.call_count >= 1
            # Check that at least one debug call mentioned preview length
            debug_calls = [call.args[0] for call in mock_logger.debug.call_args_list]
            assert any("preview length" in call for call in debug_calls)

        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["success"] is True
        assert response["status"] == "timeout"


class TestMainFunctionEdgeCases:
    """Test main function initialization and logging paths"""

    @patch("server.logger")
    @patch("server.BedrockClient")
    @patch("server.RefineEngine")
    @patch("server.IncrementalRefineEngine")
    @patch("server.DomainDetector")
    @patch("server.SecurityValidator")
    @patch("server.boto3.client")
    @patch("server.stdio_server")
    async def test_main_function_successful_initialization(
        self,
        mock_stdio_server,
        mock_boto3_client,
        mock_security_validator,
        mock_domain_detector,
        mock_incremental_engine,
        mock_refine_engine,
        mock_bedrock_client,
        mock_logger,
    ):
        """Test lines 583-594: main function successful initialization with logging"""

        # Mock successful initialization
        mock_bedrock_instance = Mock()
        mock_bedrock_client.return_value = mock_bedrock_instance

        mock_refine_instance = Mock()
        mock_refine_engine.return_value = mock_refine_instance

        mock_incremental_instance = Mock()
        mock_incremental_engine.return_value = mock_incremental_instance

        mock_domain_instance = Mock()
        mock_domain_detector.return_value = mock_domain_instance

        mock_security_instance = Mock()
        mock_security_validator.return_value = mock_security_instance

        # Mock Bedrock connection test
        mock_bedrock_test = Mock()
        mock_bedrock_test.list_foundation_models.return_value = {"models": []}
        mock_boto3_client.return_value = mock_bedrock_test

        # Mock stdio_server context manager
        mock_streams = (Mock(), Mock())
        mock_stdio_context = Mock()
        mock_stdio_context.__aenter__ = AsyncMock(return_value=mock_streams)
        mock_stdio_context.__aexit__ = AsyncMock(return_value=None)
        mock_stdio_server.return_value = mock_stdio_context

        # Mock server run
        with patch("server.server") as mock_server:
            mock_server.run = AsyncMock()
            mock_server.create_initialization_options = Mock(return_value={})

            # Call main function
            await server.main()

            # Verify all the logging statements were called (lines 583-594)
            expected_log_calls = [
                "Successfully connected to AWS Bedrock",
                "Starting Recursive Companion MCP server",
            ]

            actual_log_calls = [call.args[0] for call in mock_logger.info.call_args_list]

            for expected in expected_log_calls:
                assert any(
                    expected in actual for actual in actual_log_calls
                ), f"Missing log: {expected}"

            # Verify the configuration logging with f-strings
            config_calls = [
                call
                for call in actual_log_calls
                if "Configuration:" in call
                or "Using Claude model:" in call
                or "Using embedding model:" in call
            ]
            assert len(config_calls) >= 2, "Missing configuration logging"

    @patch("server.logger")
    @patch("server.BedrockClient")
    async def test_main_function_initialization_failure(self, mock_bedrock_client, mock_logger):
        """Test main function exception handling and logging"""

        # Mock BedrockClient to raise an exception
        mock_bedrock_client.side_effect = Exception("AWS credentials not found")

        with pytest.raises(Exception) as exc_info:
            await server.main()

        assert str(exc_info.value) == "AWS credentials not found"

        # Verify error logging occurred
        mock_logger.error.assert_called_with("Server error: AWS credentials not found")


class TestUnknownToolHandling:
    """Test handling of unknown tool names"""

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        """Test handling of unknown tool names"""
        result = await server.handle_call_tool("unknown_tool", {})

        assert len(result) == 1
        assert result[0].text == "Unknown tool: unknown_tool"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
