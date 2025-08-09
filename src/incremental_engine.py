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
from domains import get_domain_system_prompt


class RefinementStatus(Enum):
    """Status of a refinement session"""
    INITIALIZING = "initializing"
    DRAFTING = "drafting"
    CRITIQUING = "critiquing"
    REVISING = "revising"
    CONVERGED = "converged"
    ERROR = "error"
    TIMEOUT = "timeout"
    ABORTED = "aborted"


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
        is_valid, validation_msg = self.validator.validate_prompt(prompt)
        if not is_valid:
            return {
                "success": False,
                "error": f"Invalid prompt: {validation_msg}",
                "_ai_context": {
                    "validation_rule": validation_msg,
                    "prompt_length": len(prompt) if prompt else 0,
                    "min_length": 10,  # From validator
                    "max_length": 10000  # From validator
                },
                "_ai_suggestion": "Ensure prompt is between 10 and 10,000 characters",
                "_human_action": "Provide a more detailed prompt"
            }
        
        if domain == "auto":
            domain = self.domain_detector.detect_domain(prompt)
        
        config = config or {}
        session = self.session_manager.create_session(prompt, domain, config)
        
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
            active_sessions = self.session_manager.list_active_sessions()
            return {
                "success": False,
                "error": "Session not found",
                "_ai_context": {
                    "requested_session": session_id,
                    "active_session_count": len(active_sessions),
                    "available_sessions": active_sessions[:3] if active_sessions else []
                },
                "_ai_suggestion": "Check list_refinement_sessions for valid session IDs",
                "_ai_recovery": "Start a new session with start_refinement",
                "_human_action": "Use a valid session ID or start a new refinement"
            }
        
        try:
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
                    status=RefinementStatus.TIMEOUT
                )
                return {
                    "success": True,
                    "status": "completed",
                    "message": "Maximum iterations reached",
                    "final_answer": session.current_draft,
                    "convergence_score": session.convergence_score,
                    "total_iterations": session.current_iteration,
                    "_ai_note": "Max iterations reached but convergence not achieved",
                    "_ai_suggestion": "Consider higher max_iterations or lower convergence_threshold",
                    "_ai_context": {
                        "convergence_gap": session.convergence_threshold - session.convergence_score,
                        "likely_iterations_needed": 2 if session.convergence_score > 0.9 else 3
                    }
                }
            
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
            
            # Provide AI-helpful error context
            error_response = {
                "success": False,
                "error": f"Refinement error: {str(e)}",
                "status": "error",
                "_ai_context": {
                    "session_status": session.status.value if session else "unknown",
                    "iteration": session.current_iteration if session else 0,
                    "error_type": type(e).__name__
                }
            }
            
            # Add specific hints based on error type
            if "timeout" in str(e).lower():
                error_response["_ai_suggestion"] = "Use quick_refine with longer max_wait for this prompt"
            elif "embedding" in str(e).lower():
                error_response["_ai_diagnosis"] = "Embedding model issue - check AWS Bedrock access"
                error_response["_ai_action"] = "Verify Titan embedding model is enabled in your region"
            
            return error_response

    
    async def _do_draft_step(self, session: RefinementSession) -> Dict[str, Any]:
        """Generate initial draft"""
        system_prompt = self._get_domain_system_prompt(session.domain)
        draft_prompt = f"Please provide a comprehensive response to the following:\n\n{session.prompt}"
        
        draft = await self.bedrock.generate_text(draft_prompt, system_prompt)
        
        self.session_manager.update_session(
            session.session_id,
            current_draft=draft,
            current_iteration=1,
            status=RefinementStatus.CRITIQUING
        )
        
        session.iterations_history.append({
            "iteration": 1,
            "type": "draft",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "success": True,
            "status": "draft_complete",
            "iteration": 1,
            "progress": self._format_progress(session),
            "message": f"{self._get_status_emoji(RefinementStatus.DRAFTING)} Initial draft generated. Ready for critiques.",
            "draft_preview": draft[:300] + "..." if len(draft) > 300 else draft,
            "next_action": "continue_refinement",
            "continue_needed": True,
            "_ai_performance": {
                "draft_generation_model": self._get_model_name(),
                "tip": "First iteration is always the slowest - subsequent ones are faster"
            }
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
        
        self.session_manager.update_session(
            session.session_id,
            critiques=valid_critiques,
            status=RefinementStatus.REVISING
        )
        
        return {
            "success": True,
            "status": "critiques_complete",
            "iteration": session.current_iteration,
            "progress": self._format_progress(session),
            "message": f"{self._get_status_emoji(RefinementStatus.CRITIQUING)} Generated {len(valid_critiques)} critiques. Ready to revise.",
            "critique_count": len(valid_critiques),
            "critique_preview": valid_critiques[0][:100] + "..." if valid_critiques else None,
            "next_action": "continue_refinement",
            "continue_needed": True,
            "_ai_performance": {
                "critique_model": critique_model,
                "parallel_critiques": len(critique_prompts),
                "tip": "Using Claude Haiku for critiques can reduce iteration time by ~50%",
                "recommendation": "Set CRITIQUE_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0 in .env"
            }
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
        
        self.session_manager.update_session(
            session.session_id,
            previous_draft=session.current_draft,
            current_draft=revision,
            convergence_score=convergence_score,
            current_iteration=session.current_iteration + 1
        )
        
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
                "progress": self._format_progress(session),
                "message": f"{self._get_status_emoji(RefinementStatus.CONVERGED)} Refinement converged at iteration {session.current_iteration}!",
                "final_answer": revision,
                "convergence_score": round(convergence_score, 4),
                "total_iterations": session.current_iteration,
                "continue_needed": False,
                "_ai_insight": {
                    "convergence_threshold": session.convergence_threshold,
                    "final_score": round(convergence_score, 4),
                    "quality_note": "Higher convergence = more polished but potentially less creative",
                    "typical_range": "0.92-0.96 is usually optimal for most use cases"
                }
            }
        else:
            # Continue refining
            self.session_manager.update_session(
                session.session_id,
                status=RefinementStatus.CRITIQUING
            )
            
            # Prepare AI insights based on convergence
            ai_prediction = {}
            if convergence_score > 0.9:
                ai_prediction = {
                    "_ai_prediction": "Likely to converge in 1-2 more iterations",
                    "_ai_suggestion": "Consider abort_refinement if current quality is sufficient"
                }
            elif convergence_score > 0.8:
                ai_prediction = {
                    "_ai_prediction": "Making good progress, 2-3 iterations likely needed",
                    "_ai_pattern": "Typical convergence acceleration happens around 0.85"
                }
            
            response = {
                "success": True,
                "status": "revision_complete",
                "iteration": session.current_iteration,
                "progress": self._format_progress(session),
                "message": f"{self._get_status_emoji(RefinementStatus.REVISING)} Revision complete. Convergence: {round(convergence_score, 4)}",
                "convergence_score": round(convergence_score, 4),
                "draft_preview": revision[:300] + "..." if len(revision) > 300 else revision,
                "next_action": "continue_refinement",
                "continue_needed": True
            }
            response.update(ai_prediction)
            return response
    
    async def get_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a refinement session"""
        session = self.session_manager.get_session(session_id)
        if not session:
            active_sessions = self.session_manager.list_active_sessions()
            return {
                "success": False,
                "error": "Session not found",
                "_ai_context": {
                    "requested_session": session_id,
                    "active_sessions": active_sessions[:3] if active_sessions else []
                },
                "_ai_suggestion": "Use list_refinement_sessions to see valid sessions",
                "_human_action": "Check session ID or start a new refinement"
            }
        
        return {
            "success": True,
            "session": session.to_dict(),
            "progress": self._format_progress(session),
            "message": f"{self._get_status_emoji(session.status)} {self._get_action_description(session.status)}",
            "continue_needed": session.status not in [RefinementStatus.CONVERGED, RefinementStatus.ERROR, RefinementStatus.ABORTED, RefinementStatus.TIMEOUT]
        }
    
    async def get_final_result(self, session_id: str) -> Dict[str, Any]:
        """Get the final refined result"""
        session = self.session_manager.get_session(session_id)
        if not session:
            active_sessions = self.session_manager.list_active_sessions()
            return {
                "success": False,
                "error": "Session not found",
                "_ai_context": {
                    "requested_session": session_id,
                    "active_sessions": active_sessions[:3] if active_sessions else []
                },
                "_ai_suggestion": "Use list_refinement_sessions to find your session",
                "_human_action": "Verify session ID or check if session has expired"
            }
        
        if session.status not in [RefinementStatus.CONVERGED, RefinementStatus.ABORTED]:
            return {
                "success": False,
                "error": f"Refinement not complete. Current status: {session.status.value}",
                "_ai_context": {
                    "current_status": session.status.value,
                    "current_iteration": session.current_iteration,
                    "convergence_score": session.convergence_score,
                    "was_aborted": session.status == RefinementStatus.ABORTED
                },
                "_ai_suggestion": "Use continue_refinement to proceed" if session.status not in [RefinementStatus.ERROR, RefinementStatus.TIMEOUT] else "Session ended, start a new one",
                "_ai_tip": f"Currently at iteration {session.current_iteration}, convergence at {session.convergence_score:.1%}",
                "_human_action": "Continue the refinement process or use abort_refinement to get current best result"
            }
        
        return {
            "success": True,
            "refined_answer": session.current_draft,
            "metadata": {
                "domain": session.domain,
                "total_iterations": session.current_iteration,
                "convergence_score": session.convergence_score,
                "session_id": session.session_id,
                "duration_seconds": (session.updated_at - session.created_at).total_seconds(),
                "final_status": session.status.value,
                "was_aborted": session.status == RefinementStatus.ABORTED
            },
            "thinking_history": session.iterations_history
        }
    
    async def abort_refinement(self, session_id: str) -> Dict[str, Any]:
        """Stop refinement and return best result so far"""
        session = self.session_manager.get_session(session_id)
        if not session:
            active_sessions = self.session_manager.list_active_sessions()
            return {
                "success": False, 
                "error": "Session not found",
                "_ai_context": {
                    "requested_session": session_id,
                    "active_sessions": active_sessions[:3] if active_sessions else []
                },
                "_ai_suggestion": "Can't abort a non-existent session",
                "_human_action": "Use list_refinement_sessions to find valid sessions"
            }
        
        # Check if session is already completed
        if session.status in [RefinementStatus.CONVERGED, RefinementStatus.ERROR, RefinementStatus.TIMEOUT]:
            return {
                "success": False,
                "error": f"Session already completed with status: {session.status.value}",
                "_ai_context": {
                    "current_status": session.status.value,
                    "iterations_completed": session.current_iteration,
                    "convergence_score": session.convergence_score
                },
                "_ai_suggestion": "Cannot abort a completed session",
                "_human_action": "Use get_final_result to retrieve the completed result"
            }
        
        # Mark as aborted
        self.session_manager.update_session(
            session_id,
            status=RefinementStatus.ABORTED
        )
        
        return {
            "success": True,
            "message": "Refinement aborted",
            "final_answer": session.current_draft or session.previous_draft or "No content generated yet",
            "iterations_completed": session.current_iteration,
            "convergence_score": session.convergence_score,
            "reason": "User requested abort"
        }
    
    def _format_progress(self, session: RefinementSession) -> Dict[str, Any]:
        """Create detailed progress information"""
        # Estimate steps: draft(1) + (critique(1) + revise(1)) * iterations
        estimated_total_steps = 1 + (2 * session.max_iterations)
        current_step = 1 + (2 * session.current_iteration)
        
        if session.status == RefinementStatus.DRAFTING:
            current_step = 1
        elif session.status == RefinementStatus.CRITIQUING:
            current_step = 2 + (2 * (session.current_iteration - 1))
        elif session.status == RefinementStatus.REVISING:
            current_step = 3 + (2 * (session.current_iteration - 1))
        
        return {
            "step": f"{current_step}/{estimated_total_steps}",
            "percent": round((current_step / estimated_total_steps) * 100),
            "current_action": self._get_action_description(session.status),
            "iteration": f"{session.current_iteration}/{session.max_iterations}",
            "convergence": f"{session.convergence_score:.1%}",
            "status_emoji": self._get_status_emoji(session.status)
        }
    
    def _get_action_description(self, status: RefinementStatus) -> str:
        """Human-friendly action descriptions"""
        descriptions = {
            RefinementStatus.INITIALIZING: "Starting refinement process",
            RefinementStatus.DRAFTING: "Creating initial draft",
            RefinementStatus.CRITIQUING: "Analyzing draft for improvements",
            RefinementStatus.REVISING: "Incorporating feedback",
            RefinementStatus.CONVERGED: "Refinement complete - convergence achieved",
            RefinementStatus.ABORTED: "Refinement aborted by user",
            RefinementStatus.TIMEOUT: "Maximum iterations reached",
            RefinementStatus.ERROR: "Error occurred during refinement"
        }
        return descriptions.get(status, "Processing")
    
    def _get_status_emoji(self, status: RefinementStatus) -> str:
        """Fun status indicators"""
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
        return emojis.get(status, "⏳")
    
    def _get_model_name(self) -> str:
        """Get current model name for performance hints"""
        import os
        return os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
    
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
        return get_domain_system_prompt(domain)
