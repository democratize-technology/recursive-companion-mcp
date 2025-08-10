"""
Session persistence layer for storing and recovering refinement sessions.
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class SessionSnapshot:
    """Snapshot of a session at a point in time."""

    session_id: str
    state: str
    iteration: int
    draft: Optional[str]
    critiques: List[Dict[str, Any]]
    revisions: List[str]
    domain: Optional[str]
    timestamp: float
    metadata: Dict[str, Any]


class SessionPersistenceManager:
    """
    Manages session persistence to disk or database.

    This implementation uses file-based storage for simplicity,
    but can be extended to use Redis, SQLite, or other backends.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize persistence manager.

        Args:
            storage_path: Directory path for storing sessions
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            # Default to user's home directory
            self.storage_path = Path.home() / ".recursive-companion-mcp" / "sessions"

        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Session write lock to prevent concurrent writes
        self._write_locks: Dict[str, asyncio.Lock] = {}

        logger.info(f"Session persistence initialized at: {self.storage_path}")

    def _get_session_file_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        # Use JSON for human-readable storage
        return self.storage_path / f"session_{session_id}.json"

    def _get_write_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create write lock for a session."""
        if session_id not in self._write_locks:
            self._write_locks[session_id] = asyncio.Lock()
        return self._write_locks[session_id]

    async def save_session(self, session_data: Dict[str, Any]) -> bool:
        """
        Save session data to persistent storage.

        Args:
            session_data: Complete session data to save

        Returns:
            True if saved successfully
        """
        session_id = session_data.get("session_id")
        if not session_id:
            logger.error("Cannot save session without session_id")
            return False

        async with self._get_write_lock(session_id):
            try:
                file_path = self._get_session_file_path(session_id)

                # Add timestamp to track last save
                session_data["last_saved"] = time.time()

                # Convert to JSON-serializable format
                serializable_data = self._make_serializable(session_data)

                # Write atomically using temp file
                temp_path = file_path.with_suffix(".tmp")

                # Run file I/O in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self._write_session_file, temp_path, serializable_data
                )

                # Atomic rename
                await loop.run_in_executor(None, temp_path.rename, file_path)

                logger.debug(f"Session {session_id} saved successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to save session {session_id}: {e}")
                return False

    def _write_session_file(self, path: Path, data: Dict[str, Any]):
        """Synchronous file write for executor."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    async def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session data from persistent storage.

        Args:
            session_id: ID of session to load

        Returns:
            Session data if found, None otherwise
        """
        try:
            file_path = self._get_session_file_path(session_id)

            if not file_path.exists():
                logger.debug(f"Session {session_id} not found in storage")
                return None

            # Run file I/O in executor
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self._read_session_file, file_path)

            logger.debug(f"Session {session_id} loaded successfully")
            return data

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def _read_session_file(self, path: Path) -> Dict[str, Any]:
        """Synchronous file read for executor."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete session from persistent storage.

        Args:
            session_id: ID of session to delete

        Returns:
            True if deleted successfully
        """
        async with self._get_write_lock(session_id):
            try:
                file_path = self._get_session_file_path(session_id)

                if file_path.exists():
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, file_path.unlink)
                    logger.debug(f"Session {session_id} deleted")

                # Clean up write lock
                if session_id in self._write_locks:
                    del self._write_locks[session_id]

                return True

            except Exception as e:
                logger.error(f"Failed to delete session {session_id}: {e}")
                return False

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all persisted sessions with metadata.

        Returns:
            List of session summaries
        """
        try:
            sessions = []

            # Get all session files
            loop = asyncio.get_event_loop()
            files = await loop.run_in_executor(None, self._list_session_files)

            for file_path in files:
                try:
                    # Extract session ID from filename
                    session_id = file_path.stem.replace("session_", "")

                    # Get file stats
                    stats = file_path.stat()

                    # Try to load minimal info without parsing full file
                    sessions.append(
                        {
                            "session_id": session_id,
                            "file_path": str(file_path),
                            "size_bytes": stats.st_size,
                            "modified_time": stats.st_mtime,
                            "modified_ago": time.time() - stats.st_mtime,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Error reading session file {file_path}: {e}")

            # Sort by modification time (newest first)
            sessions.sort(key=lambda x: x["modified_time"], reverse=True)

            return sessions

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    def _list_session_files(self) -> List[Path]:
        """List all session files in storage directory."""
        return list(self.storage_path.glob("session_*.json"))

    async def cleanup_old_sessions(self, max_age_seconds: int = 86400 * 7):
        """
        Clean up old sessions older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds (default 7 days)
        """
        try:
            current_time = time.time()
            sessions = await self.list_sessions()

            for session_info in sessions:
                if session_info["modified_ago"] > max_age_seconds:
                    session_id = session_info["session_id"]
                    logger.info(f"Cleaning up old session: {session_id}")
                    await self.delete_session(session_id)

        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")

    async def create_snapshot(self, session_data: Dict[str, Any]) -> SessionSnapshot:
        """
        Create a snapshot of current session state.

        Args:
            session_data: Current session data

        Returns:
            SessionSnapshot object
        """
        return SessionSnapshot(
            session_id=session_data.get("session_id", ""),
            state=session_data.get("state", ""),
            iteration=session_data.get("iteration", 0),
            draft=session_data.get("draft"),
            critiques=session_data.get("critiques", []),
            revisions=session_data.get("revisions", []),
            domain=session_data.get("domain"),
            timestamp=time.time(),
            metadata={
                "convergence_score": session_data.get("convergence_score"),
                "model_config": session_data.get("model_config", {}),
            },
        )

    async def save_snapshot(self, snapshot: SessionSnapshot) -> bool:
        """
        Save a session snapshot.

        Args:
            snapshot: SessionSnapshot to save

        Returns:
            True if saved successfully
        """
        try:
            snapshot_path = self.storage_path / "snapshots"
            snapshot_path.mkdir(exist_ok=True)

            filename = f"snapshot_{snapshot.session_id}_{int(snapshot.timestamp)}.json"
            file_path = snapshot_path / filename

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._write_snapshot_file, file_path, asdict(snapshot))

            logger.debug(f"Snapshot saved: {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return False

    def _write_snapshot_file(self, path: Path, data: Dict[str, Any]):
        """Write snapshot file synchronously."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert other types to string
            return str(obj)

    async def get_session_size(self, session_id: str) -> Optional[int]:
        """
        Get the size of a persisted session in bytes.

        Args:
            session_id: Session ID

        Returns:
            Size in bytes if session exists
        """
        try:
            file_path = self._get_session_file_path(session_id)
            if file_path.exists():
                return file_path.stat().st_size
            return None
        except Exception as e:
            logger.error(f"Failed to get session size: {e}")
            return None


# Global persistence manager instance
persistence_manager = SessionPersistenceManager()
