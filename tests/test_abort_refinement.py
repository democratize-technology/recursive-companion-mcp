"""
Tests for abort refinement functionality
"""

import pytest
from unittest.mock import Mock, AsyncMock

import sys

sys.path.insert(0, "./src")
from incremental_engine import (
    IncrementalRefineEngine,
    RefinementSession,
    RefinementStatus,
)
from domains import DomainDetector
from validation import SecurityValidator


class TestAbortRefinementComplete:
    """Complete tests for abort refinement"""

    @pytest.fixture
    def engine(self):
        """Create engine with mocked dependencies"""
        mock_bedrock = Mock()
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        return IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)

    @pytest.mark.asyncio
    async def test_abort_with_no_content(self, engine):
        """Test aborting session with no content generated"""
        # Create session with no drafts
        session = engine.session_manager.create_session("Test prompt", "technical", {})
        engine.session_manager.update_session(
            session.session_id,
            status=RefinementStatus.INITIALIZING,
            current_draft=None,
            previous_draft=None,
        )

        result = await engine.abort_refinement(session.session_id)

        assert result["success"] is True
        assert result["message"] == "Refinement aborted"
        assert result["final_answer"] == "No content generated yet"
        assert result["iterations_completed"] == 0
        assert result["reason"] == "User requested abort"

    @pytest.mark.asyncio
    async def test_abort_with_previous_draft_only(self, engine):
        """Test aborting when only previous draft exists"""
        session = engine.session_manager.create_session("Test prompt", "technical", {})
        engine.session_manager.update_session(
            session.session_id,
            status=RefinementStatus.CRITIQUING,
            current_draft=None,
            previous_draft="Previous draft content",
            current_iteration=2,
        )

        result = await engine.abort_refinement(session.session_id)

        assert result["success"] is True
        assert result["final_answer"] == "Previous draft content"
        assert result["iterations_completed"] == 2

    @pytest.mark.asyncio
    async def test_abort_with_current_draft(self, engine):
        """Test aborting with current draft available"""
        session = engine.session_manager.create_session("Test prompt", "technical", {})
        engine.session_manager.update_session(
            session.session_id,
            status=RefinementStatus.REVISING,
            current_draft="Current draft content",
            previous_draft="Previous draft content",
            current_iteration=3,
            convergence_score=0.75,
        )

        result = await engine.abort_refinement(session.session_id)

        assert result["success"] is True
        assert result["final_answer"] == "Current draft content"
        assert result["iterations_completed"] == 3
        assert result["convergence_score"] == 0.75

        # Verify session status was updated
        updated = engine.session_manager.get_session(session.session_id)
        assert updated.status == RefinementStatus.ABORTED

    @pytest.mark.asyncio
    async def test_abort_nonexistent_session_detailed(self, engine):
        """Test aborting non-existent session with active sessions"""
        # Create some active sessions
        engine.session_manager.create_session("Test1", "general", {})
        engine.session_manager.create_session("Test2", "technical", {})

        result = await engine.abort_refinement("fake-session-id")

        assert result["success"] is False
        assert "Session not found" in result["error"]
        assert result["_ai_context"]["requested_session"] == "fake-session-id"
        assert len(result["_ai_context"]["active_sessions"]) > 0
        assert "_ai_suggestion" in result
        assert "_human_action" in result


class TestProgressFormatting:
    """Test progress formatting with different statuses"""

    @pytest.fixture
    def engine(self):
        """Create engine with mocked dependencies"""
        mock_bedrock = Mock()
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        return IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)

    def test_format_progress_drafting(self, engine):
        """Test progress formatting during drafting"""
        session = RefinementSession(
            session_id="test",
            prompt="Test",
            domain="technical",
            status=RefinementStatus.DRAFTING,
            current_iteration=0,
            max_iterations=5,
            convergence_threshold=0.95,
            convergence_score=0.0,
        )

        progress = engine._format_progress(session)

        assert progress["step"] == "1/11"  # 1 + (2 * 5) = 11 total steps
        assert progress["percent"] == 9  # 1/11 ≈ 9%
        assert progress["iteration"] == "0/5"
        assert progress["status_emoji"] == "📝"

    def test_format_progress_critiquing(self, engine):
        """Test progress formatting during critiquing"""
        session = RefinementSession(
            session_id="test",
            prompt="Test",
            domain="technical",
            status=RefinementStatus.CRITIQUING,
            current_iteration=2,
            max_iterations=5,
            convergence_threshold=0.95,
            convergence_score=0.65,
        )

        progress = engine._format_progress(session)

        # Step calculation: 2 + (2 * (2-1)) = 4
        assert progress["step"] == "4/11"
        assert progress["percent"] == 36  # 4/11 ≈ 36%
        assert progress["iteration"] == "2/5"
        assert "65" in progress["convergence"] or "0.65" in progress["convergence"]
        assert progress["status_emoji"] == "🔍"

    def test_format_progress_revising(self, engine):
        """Test progress formatting during revising"""
        session = RefinementSession(
            session_id="test",
            prompt="Test",
            domain="technical",
            status=RefinementStatus.REVISING,
            current_iteration=3,
            max_iterations=5,
            convergence_threshold=0.95,
            convergence_score=0.82,
        )

        progress = engine._format_progress(session)

        # Step calculation: 3 + (2 * (3-1)) = 7
        assert progress["step"] == "7/11"
        assert progress["percent"] == 64  # 7/11 ≈ 64%
        assert progress["iteration"] == "3/5"
        assert "82" in progress["convergence"] or "0.82" in progress["convergence"]
        assert progress["status_emoji"] == "✏️"

    def test_format_progress_converged(self, engine):
        """Test progress formatting when converged"""
        session = RefinementSession(
            session_id="test",
            prompt="Test",
            domain="technical",
            status=RefinementStatus.CONVERGED,
            current_iteration=4,
            max_iterations=10,
            convergence_threshold=0.95,
            convergence_score=0.98,
        )

        progress = engine._format_progress(session)

        assert progress["iteration"] == "4/10"
        assert "98" in progress["convergence"] or "0.98" in progress["convergence"]
        assert progress["status_emoji"] == "✅"
        assert progress["current_action"] == "Refinement complete - convergence achieved"


class TestStatusHelpers:
    """Test status helper methods"""

    @pytest.fixture
    def engine(self):
        """Create engine with mocked dependencies"""
        mock_bedrock = Mock()
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        return IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)

    def test_all_status_emojis(self, engine):
        """Test all status emojis are defined"""
        statuses_and_emojis = [
            (RefinementStatus.INITIALIZING, "🚀"),
            (RefinementStatus.DRAFTING, "📝"),
            (RefinementStatus.CRITIQUING, "🔍"),
            (RefinementStatus.REVISING, "✏️"),
            (RefinementStatus.CONVERGED, "✅"),
            (RefinementStatus.ERROR, "❌"),
            (RefinementStatus.ABORTED, "🛑"),
            (RefinementStatus.TIMEOUT, "⏱️"),
        ]

        for status, expected_emoji in statuses_and_emojis:
            assert engine._get_status_emoji(status) == expected_emoji

    def test_all_action_descriptions(self, engine):
        """Test all action descriptions are defined"""
        statuses_and_descriptions = [
            (RefinementStatus.INITIALIZING, "Starting refinement process"),
            (RefinementStatus.DRAFTING, "Creating initial draft"),
            (RefinementStatus.CRITIQUING, "Analyzing draft for improvements"),
            (RefinementStatus.REVISING, "Incorporating feedback"),
            (RefinementStatus.CONVERGED, "Refinement complete - convergence achieved"),
            (RefinementStatus.ERROR, "Error occurred during refinement"),
            (RefinementStatus.ABORTED, "Refinement aborted by user"),
            (RefinementStatus.TIMEOUT, "Maximum iterations reached"),
        ]

        for status, expected_desc in statuses_and_descriptions:
            assert engine._get_action_description(status) == expected_desc
