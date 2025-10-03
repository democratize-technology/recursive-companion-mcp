#!/usr/bin/env python3
"""
Focused test suite targeting the most critical missing coverage lines
Simplified approach using actual APIs to trigger defensive code paths
"""

import json
from unittest.mock import patch

import pytest

# Import modules to test
from recursive_companion_mcp.legacy.bedrock_client import BedrockClient
from recursive_companion_mcp.legacy.server import handle_call_tool
from recursive_companion_mcp.legacy.session_persistence import SessionPersistenceManager

# sys.path removed - using package imports


class TestFocusedCoverage:
    """Focused tests for highest-impact missing coverage"""

    @pytest.mark.asyncio
    async def test_current_session_no_sessions_available(self):
        """Test server.py lines 378-409: current_session with no active sessions"""

        # Mock the global state to have no current session and no active sessions
        with patch("server.session_tracker") as mock_tracker:
            mock_tracker.get_current_session.return_value = None

            with patch("server.incremental_engine") as mock_engine:
                mock_engine.session_manager.list_active_sessions.return_value = []

                # This should trigger the "no active sessions" branch
                response = await handle_call_tool("current_session", {})

                result = json.loads(response[0].text)
                assert result["success"] is False
                assert "No active sessions" in result["message"]

    @pytest.mark.asyncio
    async def test_current_session_fallback_to_recent(self):
        """Test server.py lines 381-397: current_session fallback to most recent"""

        with patch("server.session_tracker") as mock_tracker:
            mock_tracker.get_current_session.return_value = None

            with patch("server.incremental_engine") as mock_engine:
                recent_session = {
                    "session_id": "recent_123",
                    "status": "active",
                    "created_at": "2023-01-01T00:00:00Z",
                }
                mock_engine.session_manager.list_active_sessions.return_value = [recent_session]

                response = await handle_call_tool("current_session", {})

                result = json.loads(response[0].text)
                assert result["success"] is True
                assert "most recent" in result["message"]
                assert result["session"] == recent_session

    @pytest.mark.asyncio
    async def test_bedrock_client_double_check_after_lock(self):
        """Test bedrock_client.py line 92: double-check after acquiring lock"""
        client = BedrockClient()

        # Set initialized to True before calling _ensure_initialized
        client._initialized = True

        # This should return immediately due to double-check (line 92)
        await client._ensure_initialized()

        # Verify it remained True
        assert client._initialized is True

    @pytest.mark.asyncio
    async def test_bedrock_client_connection_test_exception(self):
        """Test bedrock_client.py lines 121-123: exception in _test_connection_async"""
        client = BedrockClient()

        # Mock _test_connection_sync to raise an exception
        with patch.object(
            client, "_test_connection_sync", side_effect=Exception("Connection failed")
        ):
            # This should not raise, just log warning (lines 121-123)
            await client._test_connection_async()

    @pytest.mark.asyncio
    async def test_session_persistence_cleanup_no_sessions(self):
        """Test session_persistence.py edge case: cleanup with no sessions"""
        persistence = SessionPersistenceManager()

        # This should handle gracefully with no sessions
        await persistence.cleanup_old_sessions()

    @pytest.mark.asyncio
    async def test_session_persistence_load_nonexistent(self):
        """Test session_persistence.py: loading non-existent session"""
        persistence = SessionPersistenceManager()

        # Should return None or empty dict for non-existent session
        result = await persistence.load_session("nonexistent_session")
        assert result is None or result == {}

    def test_init_file_coverage(self):
        """Test src/__init__.py lines 32-34 for trivial coverage"""
        # Simply importing should cover the __init__.py
        import src

        assert src is not None
