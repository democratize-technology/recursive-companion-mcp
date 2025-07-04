"""
Session-based Refinement Engine for Recursive Companion MCP
Implements incremental refinement to avoid timeouts and show progress
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum


class RefinementStatus(Enum):
    """Status of a refinement session"""
    INITIALIZING = "initializing"
    DRAFTING = "drafting"
    CRITIQUING = "critiquing"
    REVISING = "revising"
    CONVERGED = "converged"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class RefinementSession:
    """Represents an active refinement session"""
    session_id: str
    prompt: str
    domain: str
    status: RefinementStatus
    current_iteration: int
    max_iterations: int
    convergence_threshold: float
    current_draft: str = ""
    previous_draft: str = ""
    critiques: list = field(default_factory=list)
    convergence_score: float = 0.0
    iterations_history: list = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for JSON serialization"""
        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "domain": self.domain,
            "status": self.status.value,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "convergence_score": round(self.convergence_score, 4),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error_message": self.error_message,
            "metadata": self.metadata,
            "draft_preview": self.current_draft[:200] + "..." if len(self.current_draft) > 200 else self.current_draft,
            "iterations_completed": len(self.iterations_history)
        }


class SessionManager:
    """Manages refinement sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, RefinementSession] = {}
        self._cleanup_task = None
        
    def create_session(self, prompt: str, domain: str, config: Dict[str, Any]) -> RefinementSession:
        """Create a new refinement session"""
        session_id = str(uuid.uuid4())
        session = RefinementSession(
            session_id=session_id,
            prompt=prompt,
            domain=domain,
            status=RefinementStatus.INITIALIZING,
            current_iteration=0,
            max_iterations=config.get('max_iterations', 5),
            convergence_threshold=config.get('convergence_threshold', 0.95),
            metadata=config
        )
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[RefinementSession]:
        """Get a session by ID"""
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, **updates) -> Optional[RefinementSession]:
        """Update a session"""
        session = self.sessions.get(session_id)
        if session:
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            session.updated_at = datetime.utcnow()
        return session
    
    def list_active_sessions(self) -> list:
        """List all active sessions"""
        return [
            {
                "session_id": s.session_id,
                "status": s.status.value,
                "domain": s.domain,
                "iteration": s.current_iteration,
                "created_at": s.created_at.isoformat(),
                "prompt_preview": s.prompt[:50] + "..." if len(s.prompt) > 50 else s.prompt
            }
            for s in self.sessions.values()
        ]
    
    def cleanup_old_sessions(self, max_age_minutes: int = 30):
        """Remove sessions older than max_age_minutes"""
        now = datetime.utcnow()
        to_remove = []
        for session_id, session in self.sessions.items():
            age = (now - session.created_at).total_seconds() / 60
            if age > max_age_minutes:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.sessions[session_id]
        
        return len(to_remove)


class IncrementalRefineEngine:
    """Refinement engine that operates incrementally"""
    
    def __init__(self, bedrock_client, domain_detector, validator):
        self.bedrock = bedrock_client
        self.domain_detector = domain_detector
        self.validator = validator
        self.session_manager = SessionManager()
        
    async def start_refinement(self, prompt: str, domain: str = "auto", config: Optional[Dict] = None) -> Dict[str, Any]:
        """Start a new refinement session - returns immediately"""
        # Validate input
        is_valid, validation_msg = self.validator.validate_prompt(prompt)
        if not is_valid:
            return {
                "success": False,
                "error": f"Invalid prompt: {validation_msg}"
            }
        
        # Auto-detect domain if needed
        if domain == "auto":
            domain = self.domain_detector.detect_domain(prompt)
        
        # Create session
        config = config or {}
        session = self.session_manager.create_session(prompt, domain, config)
        
        # Start with drafting status
        self.session_manager.update_session(
            session.session_id,
            status=RefinementStatus.DRAFTING
        )
        
        return {
            "success": True,
            "session_id": session.session_id,
            "status": "started",
            "domain": domain,
            "message": "Refinement session started. Use continue_refinement to proceed.",
            "next_action": "continue_refinement"
        }
    
    async def continue_refinement(self, session_id: str) -> Dict[str, Any]:
        """Continue refinement for one iteration - returns quickly"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return {
                "success": False,
                "error": "Session not found"
            }
        
        try:
            # Check if already converged or at max iterations
            if session.status == RefinementStatus.CONVERGED:
                return {
                    "success": True,
                    "status": "completed",
                    "message": "Refinement already converged",
                    "final_answer": session.current_draft,
                    "convergence_score": session.convergence_score,
                    "total_iterations": session.current_iteration
                }
            
            if session.current_iteration >= session.max_iterations:
                self.session_manager.update_session(
                    session_id,
                    status=RefinementStatus.CONVERGED
                )
                return {
                    "success": True,
                    "status": "completed",
                    "message": "Maximum iterations reached",
                    "final_answer": session.current_draft,
                    "convergence_score": session.convergence_score,
                    "total_iterations": session.current_iteration
                }
            
            # Perform one step based on current status
            if session.status == RefinementStatus.DRAFTING:
                result = await self._do_draft_step(session)
            elif session.status == RefinementStatus.CRITIQUING:
                result = await self._do_critique_step(session)
            elif session.status == RefinementStatus.REVISING:
                result = await self._do_revise_step(session)
            else:
                result = {"error": f"Unknown status: {session.status}"}
            
            return result
            
        except Exception as e:
            self.session_manager.update_session(
                session_id,
                status=RefinementStatus.ERROR,
                error_message=str(e)
            )
            return {
                "success": False,
                "error": f"Refinement error: {str(e)}",
                "status": "error"
            }

    
    async def _do_draft_step(self, session: RefinementSession) -> Dict[str, Any]:
        """Generate initial draft"""
        system_prompt = self._get_domain_system_prompt(session.domain)
        draft_prompt = f"Please provide a comprehensive response to the following:\n\n{session.prompt}"
        
        draft = await self.bedrock.generate_text(draft_prompt, system_prompt)
        
        # Update session
        self.session_manager.update_session(
            session.session_id,
            current_draft=draft,
            current_iteration=1,
            status=RefinementStatus.CRITIQUING
        )
        
        # Add to iteration history
        session.iterations_history.append({
            "iteration": 1,
            "type": "draft",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "success": True,
            "status": "draft_complete",
            "iteration": 1,
            "message": "Initial draft generated. Ready for critiques.",
            "draft_preview": draft[:300] + "..." if len(draft) > 300 else draft,
            "next_action": "continue_refinement",
            "continue_needed": True
        }
    
    async def _do_critique_step(self, session: RefinementSession) -> Dict[str, Any]:
        """Generate critiques"""
        critique_prompts = [
            f"Critically analyze this response for accuracy and completeness:\n\nOriginal question: {session.prompt}\n\nResponse: {session.current_draft}\n\nProvide specific improvements.",
            f"Evaluate this response for clarity and structure:\n\nOriginal question: {session.prompt}\n\nResponse: {session.current_draft}\n\nSuggest how to make it clearer."
        ]
        
        # Generate critiques in parallel
        # Use Haiku for faster critiques if available
        import os
        critique_model = os.getenv("CRITIQUE_MODEL_ID", os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"))
        
        critique_tasks = [
            self.bedrock.generate_text(prompt, temperature=0.8, model_override=critique_model)
            for prompt in critique_prompts[:2]  # Use 2 for speed
        ]
        
        critiques = await asyncio.gather(*critique_tasks, return_exceptions=True)
        valid_critiques = [c for c in critiques if isinstance(c, str)]
        
        # Update session
        self.session_manager.update_session(
            session.session_id,
            critiques=valid_critiques,
            status=RefinementStatus.REVISING
        )
        
        return {
            "success": True,
            "status": "critiques_complete",
            "iteration": session.current_iteration,
            "message": f"Generated {len(valid_critiques)} critiques. Ready to revise.",
            "critique_count": len(valid_critiques),
            "next_action": "continue_refinement",
            "continue_needed": True
        }

    
    async def _do_revise_step(self, session: RefinementSession) -> Dict[str, Any]:
        """Synthesize revision and check convergence"""
        system_prompt = self._get_domain_system_prompt(session.domain)
        critique_summary = "\n\n".join([f"Critique {i+1}: {c}" for i, c in enumerate(session.critiques)])
        
        revision_prompt = f"""Given the original question, current response, and critiques, create an improved version.

Original question: {session.prompt}

Current response: {session.current_draft}

Critiques:
{critique_summary}

Create an improved response that addresses these critiques while maintaining accuracy and clarity."""
        
        revision = await self.bedrock.generate_text(revision_prompt, system_prompt, temperature=0.6)
        
        # Calculate convergence
        current_embedding = await self.bedrock.get_embedding(revision)
        previous_embedding = await self.bedrock.get_embedding(session.current_draft)
        convergence_score = self._cosine_similarity(previous_embedding, current_embedding)
        
        # Update session
        self.session_manager.update_session(
            session.session_id,
            previous_draft=session.current_draft,
            current_draft=revision,
            convergence_score=convergence_score,
            current_iteration=session.current_iteration + 1
        )
        
        # Add to iteration history
        session.iterations_history.append({
            "iteration": session.current_iteration,
            "type": "revision",
            "convergence_score": convergence_score,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Check if converged
        if convergence_score >= session.convergence_threshold:
            self.session_manager.update_session(
                session.session_id,
                status=RefinementStatus.CONVERGED
            )
            
            return {
                "success": True,
                "status": "converged",
                "message": f"Refinement converged at iteration {session.current_iteration}!",
                "final_answer": revision,
                "convergence_score": round(convergence_score, 4),
                "total_iterations": session.current_iteration,
                "continue_needed": False
            }
        else:
            # Continue refining
            self.session_manager.update_session(
                session.session_id,
                status=RefinementStatus.CRITIQUING
            )
            
            return {
                "success": True,
                "status": "revision_complete",
                "iteration": session.current_iteration,
                "message": f"Revision complete. Convergence: {round(convergence_score, 4)}",
                "convergence_score": round(convergence_score, 4),
                "draft_preview": revision[:300] + "..." if len(revision) > 300 else revision,
                "next_action": "continue_refinement",
                "continue_needed": True
            }
    
    async def get_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a refinement session"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return {
                "success": False,
                "error": "Session not found"
            }
        
        return {
            "success": True,
            "session": session.to_dict(),
            "continue_needed": session.status not in [RefinementStatus.CONVERGED, RefinementStatus.ERROR]
        }
    
    async def get_final_result(self, session_id: str) -> Dict[str, Any]:
        """Get the final refined result"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return {
                "success": False,
                "error": "Session not found"
            }
        
        if session.status != RefinementStatus.CONVERGED:
            return {
                "success": False,
                "error": f"Refinement not complete. Current status: {session.status.value}"
            }
        
        return {
            "success": True,
            "refined_answer": session.current_draft,
            "metadata": {
                "domain": session.domain,
                "total_iterations": session.current_iteration,
                "convergence_score": session.convergence_score,
                "session_id": session.session_id,
                "duration_seconds": (session.updated_at - session.created_at).total_seconds()
            },
            "thinking_history": session.iterations_history
        }
    
    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2))
    
    def _get_domain_system_prompt(self, domain: str) -> str:
        """Get domain-specific system prompt"""
        prompts = {
            "technical": "You are a technical expert. Focus on accuracy, best practices, and clear technical explanations.",
            "marketing": "You are a marketing strategist. Focus on audience engagement, brand messaging, and ROI.",
            "strategy": "You are a strategic advisor. Focus on long-term vision, competitive advantage, and actionable insights.",
            "legal": "You are a legal expert. Focus on compliance, risk mitigation, and precise legal language.",
            "financial": "You are a financial analyst. Focus on quantitative analysis, financial metrics, and data-driven insights.",
            "general": "You are a helpful assistant. Provide clear, accurate, and well-structured responses."
        }
        return prompts.get(domain, prompts["general"])
