#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2025 Recursive Companion Contributors
# Based on work by Hank Besser (https://github.com/hankbesser/recursive-companion)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Session-based Refinement Engine for Recursive Companion MCP
Implements incremental refinement to avoid timeouts and show progress
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from .configuration_manager import ConfigurationManager
from .convergence import ConvergenceDetector
from .cot_enhancement import create_cot_enhancer

# Use internal chain-of-thought implementation for security
from .internal_cot import TOOL_SPECS, AsyncChainOfThoughtProcessor

# Extracted utility modules
from .progress_tracker import ProgressTracker
from .refinement_types import RefinementSession, RefinementStatus
from .session_persistence import persistence_manager

COT_AVAILABLE = True

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages refinement sessions with persistence support"""

    def __init__(self):
        self.sessions: dict[str, RefinementSession] = {}
        self._cleanup_task = None
        self._persistence_enabled = True
        self._autosave_interval = 30  # seconds

    async def create_session(
        self, prompt: str, domain: str, config: dict[str, Any]
    ) -> RefinementSession:
        """Create a new refinement session with persistence"""
        session_id = str(uuid.uuid4())
        session = RefinementSession(
            session_id=session_id,
            prompt=prompt,
            domain=domain,
            status=RefinementStatus.INITIALIZING,
            current_iteration=0,
            max_iterations=config.get("max_iterations", 5),
            convergence_threshold=config.get("convergence_threshold", 0.95),
            metadata=config,
        )
        self.sessions[session_id] = session

        # Persist the new session
        if self._persistence_enabled:
            await self._persist_session(session)

        return session

    async def get_session(self, session_id: str) -> RefinementSession | None:
        """Get a session by ID, loading from persistence if needed"""
        # Check in-memory sessions first
        if session_id in self.sessions:
            return self.sessions[session_id]

        # Try to load from persistence
        if self._persistence_enabled:
            session_data = await persistence_manager.load_session(session_id)
            if session_data:
                session = self._reconstruct_session(session_data)
                if session:
                    self.sessions[session_id] = session
                    logger.info(f"Session {session_id} restored from persistence")
                    return session

        return None

    async def _persist_session(self, session: RefinementSession) -> bool:
        """Persist a session to storage"""
        try:
            session_data = self._serialize_session(session)
            return await persistence_manager.save_session(session_data)
        except Exception as e:
            logger.error(f"Failed to persist session {session.session_id}: {e}")
            return False

    def _serialize_session(self, session: RefinementSession) -> dict[str, Any]:
        """Serialize session for persistence"""
        return {
            "session_id": session.session_id,
            "prompt": session.prompt,
            "domain": session.domain,
            "status": session.status.value,
            "current_iteration": session.current_iteration,
            "max_iterations": session.max_iterations,
            "convergence_threshold": session.convergence_threshold,
            "current_draft": session.current_draft,
            "previous_draft": session.previous_draft,
            "critiques": session.critiques,
            "convergence_score": session.convergence_score,
            "iterations_history": session.iterations_history,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "error_message": session.error_message,
            "metadata": session.metadata,
        }

    def _reconstruct_session(self, data: dict[str, Any]) -> RefinementSession | None:
        """Reconstruct a session from persisted data"""
        try:
            return RefinementSession(
                session_id=data["session_id"],
                prompt=data["prompt"],
                domain=data["domain"],
                status=RefinementStatus(data["status"]),
                current_iteration=data["current_iteration"],
                max_iterations=data["max_iterations"],
                convergence_threshold=data["convergence_threshold"],
                current_draft=data.get("current_draft", ""),
                previous_draft=data.get("previous_draft", ""),
                critiques=data.get("critiques", []),
                convergence_score=data.get("convergence_score", 0.0),
                iterations_history=data.get("iterations_history", []),
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                error_message=data.get("error_message"),
                metadata=data.get("metadata", {}),
            )
        except Exception as e:
            logger.error(f"Failed to reconstruct session: {e}")
            return None

    async def update_session(self, session_id: str, **updates) -> RefinementSession | None:
        """Update a session with persistence"""
        session = await self.get_session(session_id)
        if session:
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            session.updated_at = datetime.utcnow()

            # Persist updated session
            if self._persistence_enabled:
                await self._persist_session(session)
        return session

    async def list_active_sessions(self) -> list:
        """List all active sessions, including persisted ones"""
        # First get persisted sessions
        if self._persistence_enabled:
            persisted = await persistence_manager.list_sessions()
            for session_info in persisted[:10]:  # Load max 10 most recent
                session_id = session_info["session_id"]
                if session_id not in self.sessions:
                    # Try to load the session
                    await self.get_session(session_id)

        return [
            {
                "session_id": s.session_id,
                "status": s.status.value,
                "domain": s.domain,
                "iteration": s.current_iteration,
                "created_at": s.created_at.isoformat(),
                "prompt_preview": s.prompt[:50] + "..." if len(s.prompt) > 50 else s.prompt,
            }
            for s in self.sessions.values()
        ]

    async def cleanup_old_sessions(self, max_age_minutes: int = 30):
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
        self.convergence_detector = ConvergenceDetector()

        # Initialize CoT processor placeholder
        self.cot_processor = None

        # Initialize CoT enhancer for prompt improvement
        self.cot_enhancer = create_cot_enhancer(enabled=True)

        # Initialize Chain of Thought availability check
        if not COT_AVAILABLE:
            logger.warning(
                "chain-of-thought-tool not available. Install with: pip install chain-of-thought-tool"
            )

        logger.info(
            f"Chain of Thought enhancement: {'available' if COT_AVAILABLE else 'not available'}"
        )
        if COT_AVAILABLE:
            logger.info("Chain of Thought will enhance draft, critique, and synthesis steps")

        # Log enhancer initialization
        if self.cot_enhancer.cot_available:
            logger.info(
                "CoT enhancer initialized - prompts will include structured thinking patterns"
            )
        else:
            logger.info("CoT enhancer using fallback mode - basic structured prompts only")

    def get_cot_tools(self) -> list[dict[str, Any]]:
        """Get Chain of Thought tool specifications for Bedrock."""
        if not COT_AVAILABLE:
            return []
        return TOOL_SPECS

    async def _process_with_cot(self, processor, request) -> str:
        """Process a request with Chain of Thought reasoning."""
        try:
            if not COT_AVAILABLE or processor is None:
                # Fallback to basic generation without tools
                basic_request = request.copy()
                if "toolConfig" in basic_request:
                    del basic_request["toolConfig"]

                # Extract the prompt from the request
                messages = basic_request.get("messages", [])
                if messages:
                    prompt = messages[0].get("content", [{}])[0].get("text", "")
                    system_prompt = basic_request.get("system", [{}])[0].get("text", "")
                    return await self.bedrock.generate_text(prompt, system_prompt)

                return "No response generated"

            # Use CoT processor for enhanced reasoning
            result = await processor.process_tool_loop(
                bedrock_client=self.bedrock.bedrock_client, initial_request=request
            )

            # Extract the final response text
            if "output" in result and "message" in result["output"]:
                content = result["output"]["message"].get("content", [])
                for item in content:
                    if item.get("text"):
                        return item["text"]

            return "No response generated"
        except Exception as e:
            logger.error(f"CoT processing error: {e}")
            # Fallback to basic generation
            basic_request = request.copy()
            if "toolConfig" in basic_request:
                del basic_request["toolConfig"]

            messages = basic_request.get("messages", [])
            if messages:
                prompt = messages[0].get("content", [{}])[0].get("text", "")
                system_prompt = basic_request.get("system", [{}])[0].get("text", "")
                return await self.bedrock.generate_text(prompt, system_prompt)

            return "Error processing request"

    async def start_refinement(
        self, prompt: str, domain: str = "auto", config: dict | None = None
    ) -> dict[str, Any]:
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
                    "max_length": 10000,  # From validator
                },
                "_ai_suggestion": "Ensure prompt is between 10 and 10,000 characters",
                "_human_action": "Provide a more detailed prompt",
            }

        if domain == "auto":
            domain = self.domain_detector.detect_domain(prompt)

        config = config or {}
        session = await self.session_manager.create_session(prompt, domain, config)

        await self.session_manager.update_session(
            session.session_id, status=RefinementStatus.DRAFTING
        )

        return {
            "success": True,
            "session_id": session.session_id,
            "status": "started",
            "domain": domain,
            "message": "Refinement session started. Use continue_refinement to proceed.",
            "next_action": "continue_refinement",
        }

    async def continue_refinement(self, session_id: str) -> dict[str, Any]:
        """Continue refinement for one iteration - returns quickly"""
        session = await self.session_manager.get_session(session_id)
        if not session:
            active_sessions = await self.session_manager.list_active_sessions()
            return {
                "success": False,
                "error": "Session not found",
                "_ai_context": {
                    "requested_session": session_id,
                    "active_session_count": len(active_sessions),
                    "available_sessions": active_sessions[:3] if active_sessions else [],
                },
                "_ai_suggestion": "Check list_refinement_sessions for valid session IDs",
                "_ai_recovery": "Start a new session with start_refinement",
                "_human_action": "Use a valid session ID or start a new refinement",
            }

        try:
            if session.status == RefinementStatus.CONVERGED:
                return {
                    "success": True,
                    "status": "completed",
                    "message": "Refinement already converged",
                    "final_answer": session.current_draft,
                    "convergence_score": session.convergence_score,
                    "total_iterations": session.current_iteration,
                }

            if session.current_iteration >= session.max_iterations:
                await self.session_manager.update_session(
                    session_id, status=RefinementStatus.TIMEOUT
                )
                return {
                    "success": True,
                    "status": "completed",
                    "message": "Maximum iterations reached",
                    "final_answer": session.current_draft,
                    "convergence_score": session.convergence_score,
                    "total_iterations": session.current_iteration,
                    "_ai_note": "Max iterations reached but convergence not achieved",
                    "_ai_suggestion": (
                        "Consider higher max_iterations or lower convergence_threshold"
                    ),
                    "_ai_context": {
                        "convergence_gap": session.convergence_threshold
                        - session.convergence_score,
                        "likely_iterations_needed": 2 if session.convergence_score > 0.9 else 3,
                    },
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
            await self.session_manager.update_session(
                session_id, status=RefinementStatus.ERROR, error_message=str(e)
            )

            # Provide AI-helpful error context
            error_response = {
                "success": False,
                "error": f"Refinement error: {str(e)}",
                "status": "error",
                "_ai_context": {
                    "session_status": session.status.value if session else "unknown",
                    "iteration": session.current_iteration if session else 0,
                    "error_type": type(e).__name__,
                },
            }

            # Add specific hints based on error type
            if "timeout" in str(e).lower():
                error_response["_ai_suggestion"] = (
                    "Use quick_refine with longer max_wait for this prompt"
                )
            elif "embedding" in str(e).lower():
                error_response["_ai_diagnosis"] = "Embedding model issue - check AWS Bedrock access"
                error_response["_ai_action"] = (
                    "Verify Titan embedding model is enabled in your region"
                )

            return error_response

    async def _do_draft_step(self, session: RefinementSession) -> dict[str, Any]:
        """Generate initial draft with CoT enhancement"""
        system_prompt = self._get_domain_system_prompt(session.domain)

        # Enhance the initial prompt using the CoT enhancer
        enhanced_user_prompt = self.cot_enhancer.enhance_initial_refinement_prompt(
            session.prompt, session.domain
        )

        # Create CoT processor for this draft step
        if COT_AVAILABLE and self.cot_processor is not None:
            processor = AsyncChainOfThoughtProcessor(conversation_id=f"draft-{session.session_id}")

            # Enhance system prompt with CoT instructions
            enhanced_system_prompt = f"""{system_prompt}

You have access to Chain of Thought tools to structure your reasoning:
- Use chain_of_thought_step to work through your response systematically
- Start with Problem Definition stage to understand the task
- Move through Analysis to break down the requirements
- Use Synthesis stage to plan your approach
- End with Conclusion stage to provide your final response
- Set next_step_needed=false when you're ready to give the final draft

Provide a comprehensive, well-structured response to the user's request."""

            # Prepare messages for draft generation with enhanced prompt
            messages = [
                {
                    "role": "user",
                    "content": [{"text": enhanced_user_prompt}],
                }
            ]

            request = {
                "modelId": self._get_model_name(),
                "messages": messages,
                "system": [{"text": enhanced_system_prompt}],
                "toolConfig": {"tools": self.get_cot_tools()},
                "inferenceConfig": {
                    "temperature": 0.7,
                    "maxTokens": 4000,
                },
            }

            draft = await self._process_with_cot(processor, request)
        else:
            # Fallback to basic generation without CoT tools, but still use enhanced prompt
            draft = await self.bedrock.generate_text(enhanced_user_prompt, system_prompt)

        # Log CoT enhancement details
        logger.info(f"Generating draft with CoT enhancement (available: {COT_AVAILABLE})")
        if COT_AVAILABLE:
            logger.debug("Draft generated using Chain of Thought reasoning")

        await self.session_manager.update_session(
            session.session_id,
            current_draft=draft,
            current_iteration=1,
            status=RefinementStatus.CRITIQUING,
        )

        session.iterations_history.append(
            {
                "iteration": 1,
                "type": "draft",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        return {
            "success": True,
            "status": "draft_complete",
            "iteration": 1,
            "progress": self._format_progress(session),
            "message": (
                f"{self._get_status_emoji(RefinementStatus.DRAFTING)} "
                "Initial draft generated. Ready for critiques."
            ),
            "draft_preview": draft[:300] + "..." if len(draft) > 300 else draft,
            "next_action": "continue_refinement",
            "continue_needed": True,
            "_ai_performance": {
                "draft_generation_model": self._get_model_name(),
                "tip": "First iteration is always the slowest - subsequent ones are faster",
            },
        }

    async def _do_critique_step(self, session: RefinementSession) -> dict[str, Any]:
        """Generate critiques with CoT enhancement"""
        # Get critique model for faster parallel critiques
        import os

        critique_model = os.getenv(
            "CRITIQUE_MODEL_ID",
            os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"),
        )

        # Define critique types
        critique_types = [
            (
                "accuracy and completeness",
                "Critically analyze this response for accuracy and completeness. Provide specific improvements.",
            ),
            (
                "clarity and structure",
                "Evaluate this response for clarity and structure. Suggest how to make it clearer.",
            ),
        ]

        critique_tasks = []

        for i, (focus, base_prompt) in enumerate(critique_types):
            if COT_AVAILABLE and self.cot_processor is not None:
                processor = AsyncChainOfThoughtProcessor(
                    conversation_id=f"critique-{session.session_id}-{i}"
                )

                enhanced_system_prompt = f"""You are a critical reviewer focused on {focus}.

You have access to Chain of Thought tools to structure your analysis:
- Use chain_of_thought_step to work through your critique systematically
- Start with Problem Definition stage to understand what to analyze
- Move through Analysis to examine the response against your focus area
- Use Synthesis stage to formulate specific improvements
- End with Conclusion stage to provide your final critique
- Set next_step_needed=false when you're ready to give the final critique

Provide specific, actionable feedback for improvement."""

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": f"""Please critically review this response with focus on {focus}:

Original Question: {session.prompt}

Response to Review:
{session.current_draft}

Use chain_of_thought_step to analyze this systematically and provide specific, actionable improvements."""
                            }
                        ],
                    }
                ]

                request = {
                    "modelId": critique_model,
                    "messages": messages,
                    "system": [{"text": enhanced_system_prompt}],
                    "toolConfig": {"tools": self.get_cot_tools()},
                    "inferenceConfig": {
                        "temperature": 0.8,
                        "maxTokens": 3000,
                    },
                }

                task = self._process_with_cot(processor, request)
            else:
                # Fallback to basic critique without CoT
                critique_prompt = f"""{base_prompt}

Original Question: {session.prompt}

Response to Review:
{session.current_draft}

Provide specific, actionable feedback for improvement."""

                task = self.bedrock.generate_text(
                    critique_prompt, temperature=0.8, model_override=critique_model
                )

            critique_tasks.append(task)

        # Log CoT enhancement for critiques
        logger.info(
            f"Generating {len(critique_tasks)} critiques with CoT enhancement (available: {COT_AVAILABLE})"
        )
        if COT_AVAILABLE:
            logger.debug("Critiques generated using Chain of Thought reasoning")

        # Generate critiques in parallel
        critiques = await asyncio.gather(*critique_tasks, return_exceptions=True)
        valid_critiques = [c for c in critiques if isinstance(c, str)]

        await self.session_manager.update_session(
            session.session_id,
            critiques=valid_critiques,
            status=RefinementStatus.REVISING,
        )

        return {
            "success": True,
            "status": "critiques_complete",
            "iteration": session.current_iteration,
            "progress": self._format_progress(session),
            "message": (
                f"{self._get_status_emoji(RefinementStatus.CRITIQUING)} "
                f"Generated {len(valid_critiques)} critiques. Ready to revise."
            ),
            "critique_count": len(valid_critiques),
            "critique_preview": valid_critiques[0][:100] + "..." if valid_critiques else None,
            "next_action": "continue_refinement",
            "continue_needed": True,
            "_ai_performance": {
                "critique_model": critique_model,
                "parallel_critiques": len(critique_tasks),
                "tip": "Using Claude Haiku for critiques can reduce iteration time by ~50%",
                "recommendation": (
                    "Set CRITIQUE_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0 in .env"
                ),
            },
        }

    async def _do_revise_step(self, session: RefinementSession) -> dict[str, Any]:
        """Synthesize revision with CoT enhancement and check convergence"""
        system_prompt = self._get_domain_system_prompt(session.domain)

        # Prepare iteration data for CoT enhancement
        iteration_data = {
            "current_draft": session.current_draft,
            "previous_draft": session.previous_draft,
            "critiques": session.critiques,
            "convergence_score": session.convergence_score,
            "iteration_number": session.current_iteration,
            "domain_type": session.domain,
        }

        # Enhance the revision prompt using the CoT enhancer
        enhanced_user_prompt = self.cot_enhancer.enhance_iteration_prompt(iteration_data)

        # Create CoT processor for synthesis step
        if COT_AVAILABLE and self.cot_processor is not None:
            processor = AsyncChainOfThoughtProcessor(
                conversation_id=f"revise-{session.session_id}-{session.current_iteration}"
            )

            # Enhance system prompt with CoT instructions
            enhanced_system_prompt = f"""{system_prompt}

You have access to Chain of Thought tools to structure your synthesis:
- Use chain_of_thought_step to work through your revision systematically
- Start with Problem Definition stage to understand what needs improvement
- Move through Analysis to examine the critiques and current draft
- Use Synthesis stage to integrate feedback and plan improvements
- End with Conclusion stage to provide your final revised response
- Set next_step_needed=false when you're ready to give the final revision

Create an improved response that addresses the critiques while maintaining accuracy and clarity."""

            messages = [
                {
                    "role": "user",
                    "content": [{"text": enhanced_user_prompt}],
                }
            ]

            request = {
                "modelId": self._get_model_name(),
                "messages": messages,
                "system": [{"text": enhanced_system_prompt}],
                "toolConfig": {"tools": self.get_cot_tools()},
                "inferenceConfig": {
                    "temperature": 0.6,
                    "maxTokens": 4000,
                },
            }

            revision = await self._process_with_cot(processor, request)
        else:
            # Fallback to basic synthesis without CoT tools, but still use enhanced prompt
            revision = await self.bedrock.generate_text(
                enhanced_user_prompt, system_prompt, temperature=0.6
            )

        # Log CoT enhancement for synthesis
        logger.info(
            f"Generating synthesis revision with CoT enhancement (available: {COT_AVAILABLE})"
        )
        if COT_AVAILABLE:
            logger.debug(
                f"Synthesis completed with {len(session.critiques)} critiques using Chain of Thought reasoning"
            )

        # Calculate convergence
        current_embedding = await self.bedrock.get_embedding(revision)
        previous_embedding = await self.bedrock.get_embedding(session.current_draft)
        convergence_score = self.convergence_detector.cosine_similarity(
            previous_embedding, current_embedding
        )

        await self.session_manager.update_session(
            session.session_id,
            previous_draft=session.current_draft,
            current_draft=revision,
            convergence_score=convergence_score,
            current_iteration=session.current_iteration + 1,
        )

        session.iterations_history.append(
            {
                "iteration": session.current_iteration,
                "type": "revision",
                "convergence_score": convergence_score,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Enhanced convergence decision with CoT reasoning
        # Use CoT enhancer to provide structured convergence decision-making
        enhanced_prompt = self.cot_enhancer.enhance_convergence_decision_prompt(
            current=revision,
            previous=session.current_draft,
            similarity_score=convergence_score,
            threshold=session.convergence_threshold,
            iteration_count=session.current_iteration,
        )

        # Log the enhanced convergence reasoning (for debugging/transparency)
        logger.debug(
            f"Convergence decision enhanced with CoT reasoning for session {session.session_id}: {enhanced_prompt[:100]}..."
        )

        # Check if converged (maintain existing logic as fallback)
        if convergence_score >= session.convergence_threshold:
            await self.session_manager.update_session(
                session.session_id, status=RefinementStatus.CONVERGED
            )

            return {
                "success": True,
                "status": "converged",
                "progress": self._format_progress(session),
                "message": (
                    f"{self._get_status_emoji(RefinementStatus.CONVERGED)} "
                    f"Refinement converged at iteration {session.current_iteration}!"
                ),
                "final_answer": revision,
                "convergence_score": round(convergence_score, 4),
                "total_iterations": session.current_iteration,
                "continue_needed": False,
                "_ai_insight": {
                    "convergence_threshold": session.convergence_threshold,
                    "final_score": round(convergence_score, 4),
                    "quality_note": (
                        "Higher convergence = more polished but potentially less creative"
                    ),
                    "typical_range": "0.92-0.96 is usually optimal for most use cases",
                },
            }
        else:
            # Continue refining
            await self.session_manager.update_session(
                session.session_id, status=RefinementStatus.CRITIQUING
            )

            # Prepare AI insights based on convergence
            ai_prediction = {}
            if convergence_score > 0.9:
                ai_prediction = {
                    "_ai_prediction": "Likely to converge in 1-2 more iterations",
                    "_ai_suggestion": "Consider abort_refinement if current quality is sufficient",
                }
            elif convergence_score > 0.8:
                ai_prediction = {
                    "_ai_prediction": "Making good progress, 2-3 iterations likely needed",
                    "_ai_pattern": "Typical convergence acceleration happens around 0.85",
                }

            response = {
                "success": True,
                "status": "revision_complete",
                "iteration": session.current_iteration,
                "progress": self._format_progress(session),
                "message": (
                    f"{self._get_status_emoji(RefinementStatus.REVISING)} "
                    f"Revision complete. Convergence: {round(convergence_score, 4)}"
                ),
                "convergence_score": round(convergence_score, 4),
                "draft_preview": revision[:300] + "..." if len(revision) > 300 else revision,
                "next_action": "continue_refinement",
                "continue_needed": True,
            }
            response.update(ai_prediction)
            return response

    async def get_status(self, session_id: str) -> dict[str, Any]:
        """Get current status of a refinement session"""
        session = await self.session_manager.get_session(session_id)
        if not session:
            active_sessions = await self.session_manager.list_active_sessions()
            return {
                "success": False,
                "error": "Session not found",
                "_ai_context": {
                    "requested_session": session_id,
                    "active_sessions": active_sessions[:3] if active_sessions else [],
                },
                "_ai_suggestion": "Use list_refinement_sessions to see valid sessions",
                "_human_action": "Check session ID or start a new refinement",
            }

        return {
            "success": True,
            "session": session.to_dict(),
            "progress": self._format_progress(session),
            "message": (
                f"{self._get_status_emoji(session.status)} "
                f"{self._get_action_description(session.status)}"
            ),
            "continue_needed": session.status
            not in [
                RefinementStatus.CONVERGED,
                RefinementStatus.ERROR,
                RefinementStatus.ABORTED,
                RefinementStatus.TIMEOUT,
            ],
        }

    async def get_final_result(self, session_id: str) -> dict[str, Any]:
        """Get the final refined result"""
        session = await self.session_manager.get_session(session_id)
        if not session:
            active_sessions = await self.session_manager.list_active_sessions()
            return {
                "success": False,
                "error": "Session not found",
                "_ai_context": {
                    "requested_session": session_id,
                    "active_sessions": active_sessions[:3] if active_sessions else [],
                },
                "_ai_suggestion": "Use list_refinement_sessions to find your session",
                "_human_action": "Verify session ID or check if session has expired",
            }

        if session.status not in [RefinementStatus.CONVERGED, RefinementStatus.ABORTED]:
            return {
                "success": False,
                "error": f"Refinement not complete. Current status: {session.status.value}",
                "_ai_context": {
                    "current_status": session.status.value,
                    "current_iteration": session.current_iteration,
                    "convergence_score": session.convergence_score,
                    "was_aborted": session.status == RefinementStatus.ABORTED,
                },
                "_ai_suggestion": (
                    "Use continue_refinement to proceed"
                    if session.status not in [RefinementStatus.ERROR, RefinementStatus.TIMEOUT]
                    else "Session ended, start a new one"
                ),
                "_ai_tip": (
                    f"Currently at iteration {session.current_iteration}, "
                    f"convergence at {session.convergence_score:.1%}"
                ),
                "_human_action": (
                    "Continue the refinement process or use abort_refinement "
                    "to get current best result"
                ),
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
                "was_aborted": session.status == RefinementStatus.ABORTED,
            },
            "thinking_history": session.iterations_history,
        }

    async def abort_refinement(self, session_id: str) -> dict[str, Any]:
        """Stop refinement and return best result so far"""
        session = await self.session_manager.get_session(session_id)
        if not session:
            active_sessions = await self.session_manager.list_active_sessions()
            return {
                "success": False,
                "error": "Session not found",
                "_ai_context": {
                    "requested_session": session_id,
                    "active_sessions": active_sessions[:3] if active_sessions else [],
                },
                "_ai_suggestion": "Can't abort a non-existent session",
                "_human_action": "Use list_refinement_sessions to find valid sessions",
            }

        # Check if session is already completed
        if session.status in [
            RefinementStatus.CONVERGED,
            RefinementStatus.ERROR,
            RefinementStatus.TIMEOUT,
        ]:
            return {
                "success": False,
                "error": f"Session already completed with status: {session.status.value}",
                "_ai_context": {
                    "current_status": session.status.value,
                    "iterations_completed": session.current_iteration,
                    "convergence_score": session.convergence_score,
                },
                "_ai_suggestion": "Cannot abort a completed session",
                "_human_action": "Use get_final_result to retrieve the completed result",
            }

        # Mark as aborted
        await self.session_manager.update_session(session_id, status=RefinementStatus.ABORTED)

        return {
            "success": True,
            "message": "Refinement aborted",
            "final_answer": session.current_draft
            or session.previous_draft
            or "No content generated yet",
            "iterations_completed": session.current_iteration,
            "convergence_score": session.convergence_score,
            "reason": "User requested abort",
        }

    def _format_progress(self, session: RefinementSession) -> dict[str, Any]:
        """Create detailed progress information"""
        return ProgressTracker.format_progress(session)

    def _get_action_description(self, status: RefinementStatus) -> str:
        """Human-friendly action descriptions"""
        return ProgressTracker.get_action_description(status)

    def _get_status_emoji(self, status: RefinementStatus) -> str:
        """Fun status indicators"""
        return ProgressTracker.get_status_emoji(status)

    def _get_model_name(self) -> str:
        """Get current model name for performance hints"""
        return ConfigurationManager.get_model_name()

    def _get_domain_system_prompt(self, domain: str) -> str:
        """Get domain-specific system prompt"""
        return ConfigurationManager.get_domain_system_prompt(domain)
