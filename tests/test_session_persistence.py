"""
Tests for session persistence functionality.
"""

import asyncio
import json
import os
import pytest
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from session_persistence import SessionPersistenceManager, SessionSnapshot


class TestSessionPersistenceManager:
    """Test session persistence manager"""

    @pytest.fixture
    async def temp_storage(self):
        """Create temporary storage directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    async def persistence_manager(self, temp_storage):
        """Create persistence manager with temp storage"""
        manager = SessionPersistenceManager(storage_path=temp_storage)
        return manager

    @pytest.mark.asyncio
    async def test_initialization(self, temp_storage):
        """Test persistence manager initialization"""
        manager = SessionPersistenceManager(storage_path=temp_storage)
        
        assert manager.storage_path == Path(temp_storage)
        assert manager.storage_path.exists()
        assert manager.storage_path.is_dir()

    @pytest.mark.asyncio
    async def test_default_storage_path(self):
        """Test default storage path creation"""
        with patch.object(Path, 'mkdir') as mock_mkdir:
            manager = SessionPersistenceManager()
            
            expected_path = Path.home() / ".recursive-companion-mcp" / "sessions"
            assert manager.storage_path == expected_path
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @pytest.mark.asyncio
    async def test_save_and_load_session(self, persistence_manager):
        """Test saving and loading a session"""
        session_data = {
            "session_id": "test-123",
            "prompt": "Test prompt",
            "domain": "technical",
            "status": "DRAFTING",
            "current_iteration": 2,
            "max_iterations": 5,
            "convergence_threshold": 0.95,
            "current_draft": "Current draft content",
            "previous_draft": "Previous draft",
            "critiques": ["Critique 1", "Critique 2"],
            "convergence_score": 0.85,
            "iterations_history": [{"iteration": 1}, {"iteration": 2}],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "error_message": None,
            "metadata": {"key": "value"},
        }
        
        # Save session
        success = await persistence_manager.save_session(session_data)
        assert success is True
        
        # Verify file was created
        file_path = persistence_manager._get_session_file_path("test-123")
        assert file_path.exists()
        
        # Load session
        loaded_data = await persistence_manager.load_session("test-123")
        assert loaded_data is not None
        assert loaded_data["session_id"] == "test-123"
        assert loaded_data["prompt"] == "Test prompt"
        assert loaded_data["current_draft"] == "Current draft content"
        assert loaded_data["convergence_score"] == 0.85
        assert "last_saved" in loaded_data

    @pytest.mark.asyncio
    async def test_load_nonexistent_session(self, persistence_manager):
        """Test loading a session that doesn't exist"""
        result = await persistence_manager.load_session("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_session(self, persistence_manager):
        """Test deleting a session"""
        session_data = {
            "session_id": "delete-test",
            "prompt": "Test",
            "status": "DRAFTING",
        }
        
        # Save session
        await persistence_manager.save_session(session_data)
        file_path = persistence_manager._get_session_file_path("delete-test")
        assert file_path.exists()
        
        # Delete session
        success = await persistence_manager.delete_session("delete-test")
        assert success is True
        assert not file_path.exists()
        
        # Try to load deleted session
        result = await persistence_manager.load_session("delete-test")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, persistence_manager):
        """Test deleting a session that doesn't exist"""
        success = await persistence_manager.delete_session("nonexistent")
        assert success is True  # Should succeed even if file doesn't exist

    @pytest.mark.asyncio
    async def test_list_sessions(self, persistence_manager):
        """Test listing all sessions"""
        # Create multiple sessions
        for i in range(3):
            session_data = {
                "session_id": f"session-{i}",
                "prompt": f"Test {i}",
                "status": "DRAFTING",
            }
            await persistence_manager.save_session(session_data)
            await asyncio.sleep(0.01)  # Small delay to ensure different timestamps
        
        # List sessions
        sessions = await persistence_manager.list_sessions()
        
        assert len(sessions) == 3
        assert all("session_id" in s for s in sessions)
        assert all("file_path" in s for s in sessions)
        assert all("size_bytes" in s for s in sessions)
        assert all("modified_time" in s for s in sessions)
        
        # Should be sorted by modification time (newest first)
        assert sessions[0]["session_id"] == "session-2"
        assert sessions[1]["session_id"] == "session-1"
        assert sessions[2]["session_id"] == "session-0"

    @pytest.mark.asyncio
    async def test_cleanup_old_sessions(self, persistence_manager):
        """Test cleanup of old sessions"""
        # Create sessions with different ages
        old_session = {
            "session_id": "old-session",
            "prompt": "Old",
            "status": "CONVERGED",
        }
        await persistence_manager.save_session(old_session)
        
        # Manually set old modification time
        old_path = persistence_manager._get_session_file_path("old-session")
        old_time = time.time() - (8 * 86400)  # 8 days ago
        os.utime(old_path, (old_time, old_time))
        
        # Create recent session
        recent_session = {
            "session_id": "recent-session",
            "prompt": "Recent",
            "status": "DRAFTING",
        }
        await persistence_manager.save_session(recent_session)
        
        # Cleanup sessions older than 7 days
        await persistence_manager.cleanup_old_sessions(max_age_seconds=7 * 86400)
        
        # Old session should be deleted
        assert not old_path.exists()
        assert await persistence_manager.load_session("old-session") is None
        
        # Recent session should still exist
        recent_path = persistence_manager._get_session_file_path("recent-session")
        assert recent_path.exists()
        assert await persistence_manager.load_session("recent-session") is not None

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, persistence_manager):
        """Test concurrent saves to the same session"""
        session_id = "concurrent-test"
        
        async def save_session(iteration):
            session_data = {
                "session_id": session_id,
                "prompt": "Test",
                "current_iteration": iteration,
                "status": "DRAFTING",
            }
            return await persistence_manager.save_session(session_data)
        
        # Save concurrently
        tasks = [save_session(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert all(results)  # All saves should succeed
        
        # Load and check final state
        loaded = await persistence_manager.load_session(session_id)
        assert loaded is not None
        assert loaded["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_session_snapshot(self, persistence_manager):
        """Test creating and saving session snapshots"""
        session_data = {
            "session_id": "snapshot-test",
            "state": "REVISING",
            "iteration": 3,
            "draft": "Current draft",
            "critiques": ["Critique 1"],
            "revisions": ["Revision 1", "Revision 2"],
            "domain": "technical",
            "convergence_score": 0.92,
            "model_config": {"temperature": 0.7},
        }
        
        # Create snapshot
        snapshot = await persistence_manager.create_snapshot(session_data)
        
        assert isinstance(snapshot, SessionSnapshot)
        assert snapshot.session_id == "snapshot-test"
        assert snapshot.state == "REVISING"
        assert snapshot.iteration == 3
        assert snapshot.draft == "Current draft"
        assert len(snapshot.critiques) == 1
        assert len(snapshot.revisions) == 2
        
        # Save snapshot
        success = await persistence_manager.save_snapshot(snapshot)
        assert success is True
        
        # Verify snapshot file exists
        snapshot_dir = persistence_manager.storage_path / "snapshots"
        assert snapshot_dir.exists()
        snapshot_files = list(snapshot_dir.glob("snapshot_snapshot-test_*.json"))
        assert len(snapshot_files) == 1

    @pytest.mark.asyncio
    async def test_make_serializable(self, persistence_manager):
        """Test making complex objects serializable"""
        # Test various data types
        test_data = {
            "string": "test",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, "two", 3.0],
            "dict": {"nested": "value"},
            "datetime": datetime.utcnow(),  # Should be converted to string
            "custom_object": Mock(),  # Should be converted to string
        }
        
        serializable = persistence_manager._make_serializable(test_data)
        
        assert serializable["string"] == "test"
        assert serializable["int"] == 42
        assert serializable["float"] == 3.14
        assert serializable["bool"] is True
        assert serializable["none"] is None
        assert serializable["list"] == [1, "two", 3.0]
        assert serializable["dict"] == {"nested": "value"}
        assert isinstance(serializable["datetime"], str)
        assert isinstance(serializable["custom_object"], str)
        
        # Should be JSON serializable
        json_str = json.dumps(serializable)
        assert json_str is not None

    @pytest.mark.asyncio
    async def test_get_session_size(self, persistence_manager):
        """Test getting session file size"""
        session_data = {
            "session_id": "size-test",
            "prompt": "Test" * 100,  # Make it larger
            "status": "DRAFTING",
            "large_field": "x" * 1000,
        }
        
        await persistence_manager.save_session(session_data)
        
        size = await persistence_manager.get_session_size("size-test")
        assert size is not None
        assert size > 1000  # Should be at least 1KB
        
        # Nonexistent session
        size = await persistence_manager.get_session_size("nonexistent")
        assert size is None

    @pytest.mark.asyncio
    async def test_error_handling_corrupt_file(self, persistence_manager):
        """Test handling of corrupt session files"""
        session_id = "corrupt-test"
        file_path = persistence_manager._get_session_file_path(session_id)
        
        # Create corrupt file
        file_path.write_text("This is not valid JSON{]}")
        
        # Try to load corrupt session
        result = await persistence_manager.load_session(session_id)
        assert result is None  # Should return None instead of crashing

    @pytest.mark.asyncio
    async def test_atomic_writes(self, persistence_manager):
        """Test that writes are atomic using temp files"""
        session_id = "atomic-test"
        
        # Mock to verify temp file usage
        original_write = persistence_manager._write_session_file
        temp_file_used = False
        
        def mock_write(path, data):
            nonlocal temp_file_used
            if path.suffix == ".tmp":
                temp_file_used = True
            return original_write(path, data)
        
        with patch.object(persistence_manager, '_write_session_file', mock_write):
            session_data = {
                "session_id": session_id,
                "prompt": "Test",
                "status": "DRAFTING",
            }
            await persistence_manager.save_session(session_data)
        
        assert temp_file_used  # Verify temp file was used

    @pytest.mark.asyncio
    async def test_session_without_id(self, persistence_manager):
        """Test saving session without session_id fails gracefully"""
        session_data = {
            "prompt": "Test",
            "status": "DRAFTING",
            # Missing session_id
        }
        
        success = await persistence_manager.save_session(session_data)
        assert success is False

    @pytest.mark.asyncio
    async def test_storage_permissions_error(self, temp_storage):
        """Test handling of storage permission errors"""
        # Create read-only directory
        readonly_dir = Path(temp_storage) / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)
        
        try:
            manager = SessionPersistenceManager(storage_path=str(readonly_dir / "sessions"))
            
            session_data = {
                "session_id": "permission-test",
                "prompt": "Test",
                "status": "DRAFTING",
            }
            
            # Should handle permission error gracefully
            success = await manager.save_session(session_data)
            # Might succeed or fail depending on OS, but shouldn't crash
            assert isinstance(success, bool)
            
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)