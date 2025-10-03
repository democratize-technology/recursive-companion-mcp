"""
Tests for Incremental Refinement Engine
"""

# Import components from our modules
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

# sys.path removed - using package imports
from recursive_companion_mcp.legacy.bedrock_client import BedrockClient
from recursive_companion_mcp.legacy.config import config
from recursive_companion_mcp.legacy.domains import DomainDetector
from recursive_companion_mcp.legacy.incremental_engine import (
    IncrementalRefineEngine,
    RefinementSession,
    RefinementStatus,
    SessionManager,
)
from recursive_companion_mcp.legacy.validation import SecurityValidator

MAX_PROMPT_LENGTH = config.max_prompt_length


class TestSessionManager:
    """Test session management functionality"""

    async def test_create_session(self):
        """Test creating a new session"""
        manager = SessionManager()
        session = await manager.create_session(
            prompt="Test prompt",
            domain="technical",
            config={"max_iterations": 5, "convergence_threshold": 0.95},
        )

        assert session.session_id is not None
        assert session.prompt == "Test prompt"
        assert session.domain == "technical"
        assert session.status == RefinementStatus.INITIALIZING
        assert session.max_iterations == 5
        assert session.convergence_threshold == 0.95

    async def test_get_session(self):
        """Test retrieving a session by ID"""
        manager = SessionManager()
        session = await manager.create_session("Test", "general", {})

        retrieved = await manager.get_session(session.session_id)
        assert retrieved == session

        # Non-existent session
        assert await manager.get_session("fake-id") is None

    async def test_update_session(self):
        """Test updating session attributes"""
        manager = SessionManager()
        session = await manager.create_session("Test", "general", {})

        await manager.update_session(
            session.session_id,
            status=RefinementStatus.DRAFTING,
            current_draft="Draft content",
        )

        updated = await manager.get_session(session.session_id)
        assert updated.status == RefinementStatus.DRAFTING
        assert updated.current_draft == "Draft content"

    async def test_cleanup_old_sessions(self):
        """Test cleanup of old sessions"""
        manager = SessionManager()
        # Disable persistence for this test
        manager._persistence_enabled = False

        # Create sessions
        session1 = await manager.create_session("Test1", "general", {})
        session2 = await manager.create_session("Test2", "general", {})

        # Mock old creation time for session1
        from datetime import timedelta

        session1.created_at = datetime.utcnow() - timedelta(minutes=40)
        manager.sessions[session1.session_id] = session1

        # Cleanup sessions older than 30 minutes
        removed = await manager.cleanup_old_sessions(max_age_minutes=30)

        assert removed == 1
        assert await manager.get_session(session1.session_id) is None
        assert await manager.get_session(session2.session_id) is not None


class TestIncrementalRefineEngine:
    """Test the incremental refinement engine"""

    @pytest.fixture
    def mock_bedrock(self):
        """Create mock Bedrock client"""
        client = Mock(spec=BedrockClient)
        client.generate_text = AsyncMock()
        client.get_embedding = AsyncMock()
        client._executor = Mock()
        return client

    @pytest.fixture
    def mock_domain_detector(self):
        """Create mock domain detector"""
        detector = Mock(spec=DomainDetector)
        detector.detect_domain = Mock(return_value="technical")
        return detector

    @pytest.fixture
    def mock_validator(self):
        """Create mock security validator"""
        validator = Mock(spec=SecurityValidator)
        validator.validate_prompt = Mock(return_value=(True, "Valid"))
        return validator

    @pytest.fixture
    def engine(self, mock_bedrock, mock_domain_detector, mock_validator):
        """Create engine with mocked dependencies"""
        return IncrementalRefineEngine(mock_bedrock, mock_domain_detector, mock_validator)

    @pytest.mark.asyncio
    async def test_start_refinement(self, engine):
        """Test starting a new refinement session"""
        result = await engine.start_refinement("Test prompt for refinement")

        assert result["success"] is True
        assert "session_id" in result
        assert result["status"] == "started"
        assert result["domain"] == "technical"

    @pytest.mark.asyncio
    async def test_start_refinement_invalid_prompt(self, engine):
        """Test handling of invalid prompt"""
        engine.validator.validate_prompt.return_value = (False, "Prompt too short")

        result = await engine.start_refinement("Hi")

        assert result["success"] is False
        assert "Invalid prompt" in result["error"]
        assert "_ai_suggestion" in result

    @pytest.mark.asyncio
    async def test_continue_refinement_draft(self, engine):
        """Test draft generation step"""
        # Start a session
        start_result = await engine.start_refinement("Test prompt")
        session_id = start_result["session_id"]

        # Mock draft generation
        engine.bedrock.generate_text.return_value = "Generated draft content"

        # Continue refinement (should do draft)
        result = await engine.continue_refinement(session_id)

        assert result["success"] is True
        assert result["status"] == "draft_complete"
        assert "draft_preview" in result
        assert result["continue_needed"] is True

    @pytest.mark.asyncio
    async def test_continue_refinement_critique(self, engine):
        """Test critique generation step"""
        # Start and setup session
        start_result = await engine.start_refinement("Test prompt")
        session_id = start_result["session_id"]

        # Manually set session to critiquing state
        session = await engine.session_manager.get_session(session_id)
        session.status = RefinementStatus.CRITIQUING
        session.current_draft = "Draft content"

        # Mock critique generation
        engine.bedrock.generate_text.side_effect = [
            "Critique 1: needs improvement",
            "Critique 2: clarity issues",
        ]

        # Continue refinement (should do critique)
        result = await engine.continue_refinement(session_id)

        assert result["success"] is True
        assert result["status"] == "critiques_complete"
        assert result["critique_count"] == 2
        assert result["continue_needed"] is True

    @pytest.mark.asyncio
    async def test_continue_refinement_converge(self, engine):
        """Test convergence detection"""
        # Start and setup session
        start_result = await engine.start_refinement("Test prompt")
        session_id = start_result["session_id"]

        # Manually set session to revising state
        session = await engine.session_manager.get_session(session_id)
        session.status = RefinementStatus.REVISING
        session.current_draft = "Draft content"
        session.critiques = ["Critique 1", "Critique 2"]

        # Mock revision and embeddings for high similarity
        engine.bedrock.generate_text.return_value = "Improved revision"

        # Mock embeddings with high similarity
        vec1 = np.random.randn(100).tolist()
        vec2 = vec1.copy()  # Very similar
        engine.bedrock.get_embedding.side_effect = [vec2, vec1]

        # Continue refinement (should converge)
        result = await engine.continue_refinement(session_id)

        assert result["success"] is True
        assert result["status"] == "converged"
        assert "final_answer" in result
        assert result["continue_needed"] is False

    @pytest.mark.asyncio
    async def test_abort_refinement(self, engine):
        """Test aborting a refinement session"""
        # Start a session
        start_result = await engine.start_refinement("Test prompt")
        session_id = start_result["session_id"]

        # Set some content
        session = await engine.session_manager.get_session(session_id)
        session.current_draft = "Partial content"

        # Abort refinement
        result = await engine.abort_refinement(session_id)

        assert result["success"] is True
        assert result["final_answer"] == "Partial content"
        assert result["reason"] == "User requested abort"

        # Verify session status
        session = await engine.session_manager.get_session(session_id)
        assert session.status == RefinementStatus.ABORTED

    @pytest.mark.asyncio
    async def test_session_not_found(self, engine):
        """Test handling of non-existent session"""
        result = await engine.continue_refinement("fake-session-id")

        assert result["success"] is False
        assert "Session not found" in result["error"]
        assert "_ai_suggestion" in result

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, engine):
        """Test handling when max iterations are reached"""
        # Start a session
        start_result = await engine.start_refinement("Test prompt")
        session_id = start_result["session_id"]

        # Set session to max iterations
        session = await engine.session_manager.get_session(session_id)
        session.current_iteration = session.max_iterations
        session.current_draft = "Final draft"

        # Continue should return completion
        result = await engine.continue_refinement(session_id)

        assert result["success"] is True
        assert result["status"] == "completed"
        assert "Maximum iterations reached" in result["message"]


class TestRefinementSession:
    """Test the RefinementSession dataclass"""

    def test_session_to_dict(self):
        """Test conversion to dictionary"""
        session = RefinementSession(
            session_id="test-123",
            prompt="Test prompt",
            domain="technical",
            status=RefinementStatus.DRAFTING,
            current_iteration=2,
            max_iterations=5,
            convergence_threshold=0.95,
            current_draft="Draft content here",
        )

        data = session.to_dict()

        assert data["session_id"] == "test-123"
        assert data["prompt"] == "Test prompt"
        assert data["domain"] == "technical"
        assert data["status"] == "drafting"
        assert data["current_iteration"] == 2
        assert "draft_preview" in data

    def test_session_metadata(self):
        """Test session metadata handling"""
        session = RefinementSession(
            session_id="test-123",
            prompt="Test",
            domain="general",
            status=RefinementStatus.INITIALIZING,
            current_iteration=0,
            max_iterations=5,
            convergence_threshold=0.95,
            metadata={"custom": "value"},
        )

        assert session.metadata["custom"] == "value"
        data = session.to_dict()
        assert data["metadata"]["custom"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
