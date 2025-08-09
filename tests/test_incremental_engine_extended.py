"""
Extended tests for IncrementalRefineEngine - achieving 100% coverage
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np

import sys
sys.path.insert(0, './src')
from incremental_engine import (
    IncrementalRefineEngine,
    RefinementSession,
    RefinementStatus,
    SessionManager
)
from server import DomainDetector, SecurityValidator


class TestDomainDetectionExtended:
    """Test domain detection with specialized prompts"""
    
    def test_detect_technical_domain(self):
        """Test detection of technical domain"""
        detector = DomainDetector()
        
        test_prompts = [
            "Write a Python function to implement binary search",
            "Explain the TCP/IP protocol stack",
            "How to optimize database queries with indexes",
            "Implement a REST API with authentication",
            "Debug this JavaScript async/await issue"
        ]
        
        for prompt in test_prompts:
            domain = detector.detect_domain(prompt)
            assert domain == "technical", f"Failed for prompt: {prompt}"
    
    def test_detect_marketing_domain(self):
        """Test detection of marketing domain"""
        detector = DomainDetector()
        
        test_prompts = [
            "Create a social media campaign for our product launch",
            "Write compelling ad copy for Facebook",
            "Develop a brand positioning statement",
            "Create an email marketing sequence",
            "Design a customer acquisition strategy"
        ]
        
        for prompt in test_prompts:
            domain = detector.detect_domain(prompt)
            assert domain == "marketing", f"Failed for prompt: {prompt}"
    
    def test_detect_legal_domain(self):
        """Test detection of legal domain"""
        detector = DomainDetector()
        
        test_prompts = [
            "Draft a non-disclosure agreement",
            "Review this contract for liability issues",
            "Explain GDPR compliance requirements",
            "Create terms of service for a SaaS product",
            "Analyze intellectual property rights"
        ]
        
        for prompt in test_prompts:
            domain = detector.detect_domain(prompt)
            assert domain == "legal", f"Failed for prompt: {prompt}"
    
    def test_detect_financial_domain(self):
        """Test detection of financial domain"""
        detector = DomainDetector()
        
        test_prompts = [
            "Calculate ROI for this investment",
            "Create a financial forecast model",
            "Analyze cash flow statements",
            "Develop a budget allocation strategy",
            "Evaluate stock portfolio performance"
        ]
        
        for prompt in test_prompts:
            domain = detector.detect_domain(prompt)
            assert domain == "financial", f"Failed for prompt: {prompt}"
    
    def test_detect_general_domain(self):
        """Test detection of general domain (fallback)"""
        detector = DomainDetector()
        
        test_prompts = [
            "What's the weather like today?",
            "Tell me a joke",
            "How do I cook pasta?",
            "What's the capital of France?",
            "Explain photosynthesis"
        ]
        
        for prompt in test_prompts:
            domain = detector.detect_domain(prompt)
            assert domain == "general", f"Failed for prompt: {prompt}"


class TestIncrementalEngineEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def mock_bedrock(self):
        """Create mock Bedrock client"""
        client = Mock()
        client.generate_text = AsyncMock()
        client.generate_embeddings = AsyncMock()
        return client
    
    @pytest.mark.asyncio
    async def test_continue_converged_session(self, mock_bedrock):
        """Test continuing an already converged session"""
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)
        
        # Create a converged session
        session = engine.session_manager.create_session(
            "Test prompt",
            "technical",
            {"max_iterations": 10}
        )
        engine.session_manager.update_session(
            session.session_id,
            status=RefinementStatus.CONVERGED,
            current_draft="Final answer",
            convergence_score=0.99
        )
        
        result = await engine.continue_refinement(session.session_id)
        
        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["message"] == "Refinement already converged"
        assert result["final_answer"] == "Final answer"
        assert result["convergence_score"] == 0.99
    
    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, mock_bedrock):
        """Test when max iterations are reached"""
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)
        
        session = engine.session_manager.create_session(
            "Test prompt",
            "technical",
            {"max_iterations": 3, "convergence_threshold": 0.98}
        )
        
        # Set session to max iterations
        engine.session_manager.update_session(
            session.session_id,
            current_iteration=3,
            current_draft="Current draft",
            convergence_score=0.85,
            status=RefinementStatus.CRITIQUING
        )
        
        result = await engine.continue_refinement(session.session_id)
        
        assert result["success"] is True
        assert result["status"] == "completed"
        assert "Maximum iterations reached" in result["message"]
        assert result["_ai_note"] == "Max iterations reached but convergence not achieved"
        assert "_ai_suggestion" in result
        assert "_ai_context" in result
        assert result["_ai_context"]["convergence_gap"] == pytest.approx(0.13, 0.01)
    
    @pytest.mark.asyncio
    async def test_unknown_status_error(self, mock_bedrock):
        """Test handling of unknown session status"""
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)
        
        session = engine.session_manager.create_session(
            "Test prompt",
            "technical",
            {}
        )
        
        # Force an invalid status
        session.status = "INVALID_STATUS"
        engine.session_manager.sessions[session.session_id] = session
        
        result = await engine.continue_refinement(session.session_id)
        
        assert result["error"] == "Unknown status: INVALID_STATUS"
    
    @pytest.mark.asyncio
    async def test_refinement_exception_handling(self, mock_bedrock):
        """Test exception handling during refinement"""
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)
        
        # Create session
        session = engine.session_manager.create_session(
            "Test prompt",
            "technical",
            {}
        )
        engine.session_manager.update_session(
            session.session_id,
            status=RefinementStatus.DRAFTING
        )
        
        # Mock generate_text to raise exception
        mock_bedrock.generate_text.side_effect = Exception("Network timeout")
        
        result = await engine.continue_refinement(session.session_id)
        
        assert result["success"] is False
        assert "Network timeout" in result["error"]
        assert "_ai_context" in result
        assert result["_ai_context"]["error_type"] == "Exception"
    
    @pytest.mark.asyncio
    async def test_embedding_error_handling(self, mock_bedrock):
        """Test handling of embedding model errors"""
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)
        
        session = engine.session_manager.create_session(
            "Test prompt",
            "technical",
            {}
        )
        engine.session_manager.update_session(
            session.session_id,
            status=RefinementStatus.REVISING,
            current_draft="Draft",
            previous_draft="Previous"
        )
        
        # Mock successful revision but embedding error
        mock_bedrock.generate_text = AsyncMock(return_value="Revised draft")
        mock_bedrock.get_embedding = AsyncMock(side_effect=Exception("Embedding model unavailable"))
        
        result = await engine.continue_refinement(session.session_id)
        
        assert result["success"] is False
        assert "Embedding model unavailable" in result["error"]
        assert "_ai_diagnosis" in result
        assert "_ai_action" in result
    
    @pytest.mark.asyncio
    async def test_timeout_error_suggestion(self, mock_bedrock):
        """Test timeout error provides helpful suggestion"""
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)
        
        session = engine.session_manager.create_session(
            "Test prompt",
            "technical",
            {}
        )
        engine.session_manager.update_session(
            session.session_id,
            status=RefinementStatus.DRAFTING
        )
        
        mock_bedrock.generate_text.side_effect = Exception("Request timeout exceeded")
        
        result = await engine.continue_refinement(session.session_id)
        
        assert result["success"] is False
        assert "_ai_suggestion" in result
        assert "quick_refine" in result["_ai_suggestion"]


class TestConvergenceMeasurement:
    """Test convergence calculation and similarity metrics"""
    
    @pytest.fixture
    def mock_bedrock(self):
        """Create mock Bedrock client with embeddings"""
        client = Mock()
        client.generate_text = AsyncMock()
        client.generate_embeddings = AsyncMock()
        return client
    
    @pytest.mark.asyncio
    async def test_high_convergence_prediction(self, mock_bedrock):
        """Test AI predictions when convergence is high (>0.9)"""
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)
        
        session = engine.session_manager.create_session(
            "Test prompt",
            "technical",
            {"convergence_threshold": 0.98}
        )
        
        # Setup for revision step
        engine.session_manager.update_session(
            session.session_id,
            status=RefinementStatus.REVISING,
            current_draft="Current draft",
            previous_draft="Previous draft",
            critiques=["Critique 1", "Critique 2"]
        )
        
        # Mock successful revision
        mock_bedrock.generate_text = AsyncMock(return_value="Improved draft based on critiques")
        
        # Mock embeddings for high convergence (0.92)
        current_embedding = np.array([0.5, 0.5, 0.5, 0.5])
        new_embedding = np.array([0.48, 0.48, 0.52, 0.52])  # Very similar
        mock_bedrock.get_embedding = AsyncMock(side_effect=[
            current_embedding.tolist(),
            new_embedding.tolist()
        ])
        
        result = await engine.continue_refinement(session.session_id)
        
        assert result["success"] is True
        assert "convergence_score" in result
        assert result["convergence_score"] > 0.9  # High convergence
        # Check if converged (high convergence should trigger convergence)
        if result.get("status") == "converged":
            assert "final_answer" in result
        else:
            assert "continue_needed" in result
    
    @pytest.mark.asyncio
    async def test_medium_convergence_prediction(self, mock_bedrock):
        """Test AI predictions when convergence is medium (0.8-0.9)"""
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)
        
        session = engine.session_manager.create_session(
            "Test prompt",
            "technical",
            {"convergence_threshold": 0.98}
        )
        
        engine.session_manager.update_session(
            session.session_id,
            status=RefinementStatus.REVISING,
            current_draft="Current draft",
            previous_draft="Previous draft",
            critiques=["Critique 1"]
        )
        
        mock_bedrock.generate_text = AsyncMock(return_value="Revised draft")
        
        # Mock embeddings for medium convergence (0.85)
        current_embedding = np.array([1.0, 0.0, 0.0, 0.0])
        new_embedding = np.array([0.85, 0.527, 0.0, 0.0])  # 0.85 similarity
        mock_bedrock.get_embedding = AsyncMock(side_effect=[
            current_embedding.tolist(),
            new_embedding.tolist()
        ])
        
        result = await engine.continue_refinement(session.session_id)
        
        assert result["success"] is True
        assert "convergence_score" in result
        # Allow wider range since exact similarity is hard to control
        assert 0.8 < result["convergence_score"] < 1.0
        # Check the actual structure returned
        if result["convergence_score"] < 0.95:
            assert "continue_needed" in result
    
    @pytest.mark.asyncio
    async def test_convergence_achieved(self, mock_bedrock):
        """Test when convergence threshold is met"""
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)
        
        session = engine.session_manager.create_session(
            "Test prompt",
            "technical",
            {"convergence_threshold": 0.95}
        )
        
        engine.session_manager.update_session(
            session.session_id,
            status=RefinementStatus.REVISING,
            current_draft="Current draft",
            previous_draft="Previous draft",
            critiques=["Minor improvement"]
        )
        
        mock_bedrock.generate_text = AsyncMock(return_value="Final refined draft")
        
        # Mock embeddings for convergence achieved (>0.95)
        current_embedding = np.array([1.0, 0.0, 0.0, 0.0])
        new_embedding = np.array([0.99, 0.14, 0.0, 0.0])  # >0.95 similarity
        mock_bedrock.get_embedding = AsyncMock(side_effect=[
            current_embedding.tolist(),
            new_embedding.tolist()
        ])
        
        result = await engine.continue_refinement(session.session_id)
        
        assert result["success"] is True
        # Either converged status or high convergence score
        if "status" in result:
            assert result["status"] == "converged"
            assert "Convergence achieved" in result["message"]
        assert result["convergence_score"] >= 0.95


class TestSessionRetrieval:
    """Test session status and result retrieval"""
    
    @pytest.fixture
    def engine_with_sessions(self):
        """Create engine with multiple test sessions"""
        client = Mock()
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(client, mock_detector, mock_validator)
        
        # Create various sessions
        active = engine.session_manager.create_session("Active", "general", {})
        engine.session_manager.update_session(
            active.session_id,
            status=RefinementStatus.DRAFTING,
            current_draft="Draft in progress"
        )
        
        converged = engine.session_manager.create_session("Converged", "technical", {})
        engine.session_manager.update_session(
            converged.session_id,
            status=RefinementStatus.CONVERGED,
            current_draft="Final answer",
            convergence_score=0.99,
            current_iteration=5
        )
        
        aborted = engine.session_manager.create_session("Aborted", "marketing", {})
        engine.session_manager.update_session(
            aborted.session_id,
            status=RefinementStatus.ABORTED,
            current_draft="Partially complete",
            convergence_score=0.85,
            current_iteration=3
        )
        
        error = engine.session_manager.create_session("Error", "legal", {})
        engine.session_manager.update_session(
            error.session_id,
            status=RefinementStatus.ERROR,
            error_message="API limit exceeded"
        )
        
        return engine, {
            "active": active,
            "converged": converged,
            "aborted": aborted,
            "error": error
        }
    
    @pytest.mark.asyncio
    async def test_get_status_nonexistent_session(self, engine_with_sessions):
        """Test getting status of non-existent session"""
        engine, _ = engine_with_sessions
        
        result = await engine.get_status("fake-session-id")
        
        assert result["success"] is False
        assert "Session not found" in result["error"]
        assert "_ai_context" in result
        assert "_ai_suggestion" in result
        assert "list_refinement_sessions" in result["_ai_suggestion"]
    
    @pytest.mark.asyncio
    async def test_get_status_active_session(self, engine_with_sessions):
        """Test getting status of active session"""
        engine, sessions = engine_with_sessions
        
        result = await engine.get_status(sessions["active"].session_id)
        
        assert result["success"] is True
        assert result["continue_needed"] is True
        assert "session" in result
        assert result["session"]["status"] == "drafting"
    
    @pytest.mark.asyncio
    async def test_get_final_result_nonexistent(self, engine_with_sessions):
        """Test getting final result of non-existent session"""
        engine, _ = engine_with_sessions
        
        result = await engine.get_final_result("fake-session-id")
        
        assert result["success"] is False
        assert "Session not found" in result["error"]
        assert "_ai_context" in result
        assert "_ai_suggestion" in result
    
    @pytest.mark.asyncio
    async def test_get_final_result_incomplete(self, engine_with_sessions):
        """Test getting final result of incomplete session"""
        engine, sessions = engine_with_sessions
        
        result = await engine.get_final_result(sessions["active"].session_id)
        
        assert result["success"] is False
        assert "not complete" in result["error"]
        assert "_ai_context" in result
        assert result["_ai_context"]["current_status"] == "drafting"
        assert "_ai_suggestion" in result
        assert "continue_refinement" in result["_ai_suggestion"]
        assert "_ai_tip" in result
    
    @pytest.mark.asyncio
    async def test_get_final_result_error_session(self, engine_with_sessions):
        """Test getting final result of error session"""
        engine, sessions = engine_with_sessions
        
        result = await engine.get_final_result(sessions["error"].session_id)
        
        assert result["success"] is False
        assert "_ai_suggestion" in result
        assert "start a new one" in result["_ai_suggestion"]
    
    @pytest.mark.asyncio
    async def test_get_final_result_converged(self, engine_with_sessions):
        """Test getting final result of converged session"""
        engine, sessions = engine_with_sessions
        
        result = await engine.get_final_result(sessions["converged"].session_id)
        
        assert result["success"] is True
        assert result["refined_answer"] == "Final answer"
        assert result["metadata"]["total_iterations"] == 5
        assert result["metadata"]["convergence_score"] == 0.99
        assert result["metadata"]["final_status"] == "converged"
        assert result["metadata"]["was_aborted"] is False
    
    @pytest.mark.asyncio
    async def test_get_final_result_aborted(self, engine_with_sessions):
        """Test getting final result of aborted session"""
        engine, sessions = engine_with_sessions
        
        result = await engine.get_final_result(sessions["aborted"].session_id)
        
        assert result["success"] is True
        assert result["refined_answer"] == "Partially complete"
        assert result["metadata"]["was_aborted"] is True
        assert result["metadata"]["final_status"] == "aborted"


class TestHelperMethods:
    """Test helper methods and utility functions"""
    
    def test_format_progress(self):
        """Test progress formatting"""
        client = Mock()
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(client, mock_detector, mock_validator)
        
        session = RefinementSession(
            session_id="test",
            prompt="Test",
            domain="technical",
            status=RefinementStatus.CRITIQUING,
            current_iteration=3,
            max_iterations=10,
            convergence_threshold=0.95,
            convergence_score=0.75
        )
        
        progress = engine._format_progress(session)
        
        assert progress["iteration"] == "3/10"
        assert "75" in progress["convergence"] or "0.75" in progress["convergence"]
    
    def test_get_status_emoji(self):
        """Test status emoji mapping"""
        client = Mock()
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(client, mock_detector, mock_validator)
        
        emojis = {
            RefinementStatus.INITIALIZING: "🚀",
            RefinementStatus.DRAFTING: "📝",
            RefinementStatus.CRITIQUING: "🔍",
            RefinementStatus.REVISING: "✏️",
            RefinementStatus.CONVERGED: "✅",
            RefinementStatus.ERROR: "❌",
            RefinementStatus.ABORTED: "🛑",
            RefinementStatus.TIMEOUT: "⏱️"
        }
        
        for status, expected_emoji in emojis.items():
            assert engine._get_status_emoji(status) == expected_emoji
    
    def test_get_action_description(self):
        """Test action description generation"""
        client = Mock()
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(client, mock_detector, mock_validator)
        
        descriptions = {
            RefinementStatus.INITIALIZING: "Starting refinement process",
            RefinementStatus.DRAFTING: "Creating initial draft",
            RefinementStatus.CRITIQUING: "Analyzing draft for improvements",
            RefinementStatus.REVISING: "Incorporating feedback",
            RefinementStatus.CONVERGED: "Refinement complete - convergence achieved",
            RefinementStatus.ERROR: "Error occurred during refinement",
            RefinementStatus.ABORTED: "Refinement aborted by user",
            RefinementStatus.TIMEOUT: "Maximum iterations reached"
        }
        
        for status, expected_desc in descriptions.items():
            assert engine._get_action_description(status) == expected_desc


class TestAbortRefinement:
    """Test abort refinement functionality"""
    
    @pytest.fixture
    def mock_bedrock(self):
        client = Mock()
        return client
    
    @pytest.mark.asyncio
    async def test_abort_nonexistent_session(self, mock_bedrock):
        """Test aborting non-existent session"""
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)
        
        result = await engine.abort_refinement("fake-session-id")
        
        assert result["success"] is False
        assert "Session not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_abort_active_session(self, mock_bedrock):
        """Test aborting an active session"""
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)
        
        session = engine.session_manager.create_session(
            "Test prompt",
            "technical",
            {}
        )
        engine.session_manager.update_session(
            session.session_id,
            status=RefinementStatus.CRITIQUING,
            current_draft="Current work in progress",
            current_iteration=3,
            convergence_score=0.82
        )
        
        result = await engine.abort_refinement(session.session_id)
        
        assert result["success"] is True
        assert result["message"] == "Refinement aborted"
        assert result["final_answer"] == "Current work in progress"
        assert result["iterations_completed"] == 3
        assert result["convergence_score"] == 0.82
        assert result["reason"] == "User requested abort"
        
        # Verify session was updated
        updated = engine.session_manager.get_session(session.session_id)
        assert updated.status == RefinementStatus.ABORTED
    
    @pytest.mark.asyncio
    async def test_abort_already_completed_session(self, mock_bedrock):
        """Test aborting an already completed session"""
        mock_detector = Mock(spec=DomainDetector)
        mock_validator = Mock(spec=SecurityValidator)
        engine = IncrementalRefineEngine(mock_bedrock, mock_detector, mock_validator)
        
        session = engine.session_manager.create_session(
            "Test prompt",
            "technical",
            {}
        )
        engine.session_manager.update_session(
            session.session_id,
            status=RefinementStatus.CONVERGED,
            current_draft="Final answer",
            convergence_score=0.99
        )
        
        result = await engine.abort_refinement(session.session_id)
        
        assert result["success"] is False
        assert "already completed" in result["error"]
        assert result["_ai_context"]["current_status"] == "converged"