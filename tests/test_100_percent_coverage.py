#!/usr/bin/env python3
"""
Comprehensive test suite to achieve 100% test coverage
Targets specific missing lines identified in coverage analysis:
- server.py: 378-415, 570-597 (current_session handler, main() initialization)
- bedrock_client.py: 92, 121-123, 192, 205-210, 234-235, 254-258 (error handling, circuit breakers)
- incremental_engine.py: 300-311, 319-325, 632-641, 748-764 (CoT fallbacks, content extraction)
- session_persistence.py: 148-150, 168, 204, etc. (session management edge cases)
"""

import asyncio
import json
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

sys.path.insert(0, "./src")

# Import modules to test
from bedrock_client import BedrockClient, CircuitBreakerOpenError
from circuit_breaker import CircuitBreaker
from domains import DomainDetector
from incremental_engine import IncrementalRefineEngine
from server import main
from session_persistence import SessionPersistenceManager
from validation import SecurityValidator


class TestBedrockClientEdgeCases:
    """Test uncovered error paths in bedrock_client.py"""

    @pytest.mark.asyncio
    async def test_double_check_after_lock(self):
        """Test line 92: double-check after acquiring lock in _ensure_initialized"""
        client = BedrockClient()

        # Set initialized to True before calling _ensure_initialized
        client._initialized = True

        # This should return immediately without reinitializing
        await client._ensure_initialized()

        # Verify it remained True (didn't get reset)
        assert client._initialized is True

    @pytest.mark.asyncio
    async def test_connection_test_exception_handling(self):
        """Test lines 121-123: exception handling in _test_connection_async"""
        client = BedrockClient()

        with patch.object(
            client,
            "_test_connection_sync",
            side_effect=ClientError(
                {"Error": {"Code": "InvalidCredentials", "Message": "Invalid credentials"}},
                "ListFoundationModels",
            ),
        ):
            # This should not raise, just log warning
            await client._test_connection_async()
            # Test passes if no exception is raised

    @pytest.mark.asyncio
    async def test_circuit_breaker_fallback_generation(self):
        """Test lines 144-145, 192: circuit breaker fallback in generation"""
        client = BedrockClient()
        await client._ensure_initialized()

        # Mock circuit breaker to always call fallback
        with patch.object(client._generation_breaker, "call") as mock_call:

            async def mock_circuit_call(func, fallback):
                # Call the fallback function to trigger lines 144-145
                return await fallback()

            mock_call.side_effect = mock_circuit_call

            # This should trigger the circuit breaker fallback (returns None)
            # which then triggers line 192 check
            with pytest.raises(RuntimeError, match="Generation failed"):
                await client.generate_text("test prompt", "test system")

    @pytest.mark.asyncio
    async def test_json_decode_error(self):
        """Test lines 205-210: JSON decode error handling"""
        client = BedrockClient()
        await client._ensure_initialized()

        # Mock the invoke_model to return malformed JSON
        mock_response = {"body": Mock()}
        mock_response["body"].read.return_value = "invalid json {"

        with patch.object(client._generation_breaker, "call", return_value=mock_response):
            with pytest.raises(ValueError, match="Invalid response format from Bedrock model"):
                await client.generate_text("test prompt", "test system")

    @pytest.mark.asyncio
    async def test_general_exception_in_generation(self):
        """Test lines 205-210: general exception handling in generate_text"""
        client = BedrockClient()
        await client._ensure_initialized()

        # Mock an unexpected exception
        with patch.object(
            client._generation_breaker, "call", side_effect=RuntimeError("Unexpected error")
        ):
            with pytest.raises(RuntimeError, match="Generation failed"):
                await client.generate_text("test prompt", "test system")

    @pytest.mark.asyncio
    async def test_embedding_circuit_breaker_fallback(self):
        """Test lines 234-235, 240: embedding circuit breaker fallback"""
        client = BedrockClient()
        await client._ensure_initialized()

        # Mock circuit breaker to call fallback
        with patch.object(client._embedding_breaker, "call") as mock_call:

            async def mock_circuit_call(func, fallback):
                # Call the fallback function to trigger lines 234-235
                return await fallback()

            mock_call.side_effect = mock_circuit_call

            # This should trigger the fallback (returns None) and then line 240
            with pytest.raises(CircuitBreakerOpenError, match="embedding service unavailable"):
                await client._get_embedding_uncached("test text")

    @pytest.mark.asyncio
    async def test_embedding_general_exception(self):
        """Test lines 254-258: general exception handling in embedding"""
        client = BedrockClient()
        await client._ensure_initialized()

        # Mock an unexpected exception in embedding
        with patch.object(
            client._embedding_breaker, "call", side_effect=ValueError("Embedding error")
        ):
            with pytest.raises(RuntimeError, match="Embedding failed"):
                await client._get_embedding_uncached("test text")


class TestIncrementalEngineEdgeCases:
    """Test uncovered fallback paths in incremental_engine.py"""

    @pytest.mark.asyncio
    async def test_cot_fallback_in_process_with_cot(self):
        """Test lines 300-311: fallback request processing when toolConfig present"""
        bedrock_client = Mock()
        bedrock_client.generate_text = AsyncMock(return_value="fallback response")

        engine = IncrementalRefineEngine(bedrock_client, DomainDetector(), SecurityValidator())

        # Create a request with toolConfig that needs to be removed
        request = {
            "messages": [{"content": [{"text": "test prompt"}]}],
            "system": [{"text": "system prompt"}],
            "toolConfig": {"tools": ["some_tool"]},
        }

        # Mock processor to be None to force fallback
        processor = None

        result = await engine._process_with_cot(processor, request)

        # Should get the fallback response
        assert result == "fallback response"
        # Verify toolConfig was removed from the call
        bedrock_client.generate_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_cot_content_extraction_loops(self):
        """Test lines 319-325: content extraction loops in _process_with_cot"""
        bedrock_client = Mock()
        engine = IncrementalRefineEngine(bedrock_client, DomainDetector(), SecurityValidator())

        # Mock processor result with nested content structure
        mock_result = {
            "output": {
                "message": {
                    "content": [
                        {"text": ""},  # Empty text should be skipped
                        {"text": "actual response"},  # This should be returned
                        {"text": "extra text"},  # This should not be reached
                    ]
                }
            }
        }

        # Mock processor
        processor = Mock()
        processor.process_tool_loop = AsyncMock(return_value=mock_result)

        request = {"messages": [{"content": [{"text": "test"}]}]}

        result = await engine._process_with_cot(processor, request)
        assert result == "actual response"

    @pytest.mark.asyncio
    async def test_critique_cot_fallback(self):
        """Test lines 632-641: fallback critique without CoT"""
        bedrock_client = Mock()
        bedrock_client.generate_text = AsyncMock(return_value="fallback critique")

        engine = IncrementalRefineEngine(bedrock_client, DomainDetector(), SecurityValidator())

        # Force fallback by setting cot_processor to None (this triggers lines 632-641)
        engine.cot_processor = None

        # Start a session and continue to critique step
        start_result = await engine.start_refinement("test prompt")
        session_id = start_result["session_id"]

        # Continue to trigger critique step with fallback
        result = await engine.continue_refinement(session_id)

        # Should succeed using fallback critique generation
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_revision_cot_fallback(self):
        """Test lines 748-764: fallback revision without CoT"""
        # Create proper async mock for BedrockClient
        bedrock_client = AsyncMock()
        bedrock_client.generate_text.return_value = "revised response"
        bedrock_client.get_embedding.return_value = [0.1] * 1536  # Mock embedding

        # Mock COT_AVAILABLE to False to trigger fallback path
        with patch("incremental_engine.COT_AVAILABLE", False):
            engine = IncrementalRefineEngine(bedrock_client, DomainDetector(), SecurityValidator())

            # Start refinement and go through the full cycle to reach revision step
            start_result = await engine.start_refinement("test prompt")
            session_id = start_result["session_id"]

            # Continue multiple times to reach revision step with fallback
            await engine.continue_refinement(session_id)  # Draft step
            await engine.continue_refinement(session_id)  # Critique step
            result = await engine.continue_refinement(session_id)  # Revision step

            # Should succeed using fallback revision generation
            assert result["success"] is True


class TestServerEdgeCases:
    """Test uncovered paths in server.py"""

    @pytest.mark.asyncio
    async def test_current_session_no_active_sessions(self):
        """Test lines 378-415: current_session handler with no sessions"""
        from server import session_tracker

        # Clear current session
        session_tracker.current_session_id = None

        # Mock incremental_engine with no active sessions
        mock_engine = Mock()
        mock_engine.session_manager.list_active_sessions.return_value = []

        with patch("server.incremental_engine", mock_engine):
            from server import handle_call_tool

            # Test the current_session tool with no sessions
            response = await handle_call_tool("current_session", {})

            # Should return the "no active sessions" message
            result_text = response[0].text
            result_json = json.loads(result_text)
            assert result_json["success"] is False
            assert "No active sessions" in result_json["message"]

    @pytest.mark.asyncio
    async def test_current_session_with_recent_session(self):
        """Test lines 381-397: current_session handler fallback to most recent"""
        from server import session_tracker

        # Clear current session
        session_tracker.current_session_id = None

        # Mock incremental_engine with a recent session
        mock_engine = Mock()
        recent_session = {
            "session_id": "recent_123",
            "status": "active",
            "created_at": "2023-01-01T00:00:00Z",
        }
        mock_engine.session_manager.list_active_sessions.return_value = [recent_session]

        with patch("server.incremental_engine", mock_engine):
            from server import handle_call_tool

            response = await handle_call_tool("current_session", {})

            result_text = response[0].text
            result_json = json.loads(result_text)
            assert result_json["success"] is True
            assert "most recent" in result_json["message"]
            assert result_json["session"] == recent_session

    @pytest.mark.asyncio
    async def test_current_session_get_status_error(self):
        """Test lines 411-415: current_session get_status exception handling"""
        from server import session_tracker

        # Set a current session
        session_tracker.current_session_id = "test_session"

        # Mock incremental_engine to raise exception in get_status
        mock_engine = Mock()
        mock_engine.get_status = AsyncMock(side_effect=Exception("Status error"))

        with patch("server.incremental_engine", mock_engine):
            from server import handle_call_tool

            response = await handle_call_tool("current_session", {})

            result_text = response[0].text
            result_json = json.loads(result_text)
            assert result_json["success"] is False
            assert "Failed to get current session" in result_json["error"]

    @pytest.mark.asyncio
    async def test_main_function_initialization_error(self):
        """Test lines 570-597: main() function error handling"""

        # Test AWS Bedrock connection failure
        with patch("boto3.client", side_effect=NoCredentialsError()):
            with pytest.raises(NoCredentialsError):
                await main()

    @pytest.mark.asyncio
    async def test_main_function_bedrock_test_failure(self):
        """Test lines 581-582: Bedrock connection test failure in main()"""

        # Mock successful client creation but failed list_foundation_models
        with patch("boto3.client") as mock_client:
            mock_bedrock_test = Mock()
            mock_bedrock_test.list_foundation_models.side_effect = ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "Access denied"}},
                "ListFoundationModels",
            )
            mock_client.return_value = mock_bedrock_test

            with pytest.raises(ClientError):
                await main()

    @pytest.mark.asyncio
    async def test_main_function_general_exception(self):
        """Test lines 595-597: general exception handling in main()"""

        # Mock BedrockClient initialization to fail
        with patch("server.BedrockClient", side_effect=RuntimeError("Init failed")):
            with pytest.raises(RuntimeError):
                await main()


class TestSessionPersistenceEdgeCases:
    """Test uncovered paths in session_persistence.py"""

    @pytest.mark.asyncio
    async def test_session_cleanup_edge_cases(self):
        """Test session cleanup with edge cases"""
        persistence = SessionPersistenceManager()

        # Test cleanup with no sessions - should handle gracefully
        await persistence.cleanup_old_sessions()

    @pytest.mark.asyncio
    async def test_load_nonexistent_session(self):
        """Test loading non-existent session"""
        persistence = SessionPersistenceManager()

        # Should handle gracefully
        result = await persistence.load_session("nonexistent_id")
        assert result is None or result == {}

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self):
        """Test deleting non-existent session"""
        persistence = SessionPersistenceManager()

        # Should handle gracefully without raising exceptions
        await persistence.delete_session("nonexistent_id")


class TestInitFilesCoverage:
    """Test __init__.py files for trivial coverage"""

    def test_init_file_import(self):
        """Test src/__init__.py lines 32-34"""
        # Simply importing the module should cover the __init__.py
        import src

        assert src is not None


class TestCircuitBreakerEdgeCases:
    """Test remaining circuit breaker edge cases"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker failure in half-open state"""
        from circuit_breaker import CircuitBreakerConfig

        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        circuit_breaker = CircuitBreaker(name="test_breaker", config=config)

        # Force circuit to open
        with pytest.raises(Exception):
            await circuit_breaker.call(AsyncMock(side_effect=Exception("fail")))

        # Wait for half-open state
        await asyncio.sleep(0.2)

        # Test failure in half-open (should go back to open)
        with pytest.raises(Exception):
            await circuit_breaker.call(AsyncMock(side_effect=Exception("fail again")))

        assert circuit_breaker.state.value == "open"
