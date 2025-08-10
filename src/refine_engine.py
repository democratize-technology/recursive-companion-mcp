"""
Core refinement engine implementing the Draft → Critique → Revise → Converge pattern.
"""
import asyncio
import logging
import time
from typing import List

from bedrock_client import BedrockClient
from config import config
from domains import DomainDetector, get_domain_system_prompt
from session_manager import RefinementIteration, RefinementResult
from validation import SecurityValidator

logger = logging.getLogger(__name__)


class RefineEngine:
    """Implements the Draft → Critique → Revise → Converge refinement pattern."""
    
    def __init__(self, bedrock_client: BedrockClient):
        self.bedrock = bedrock_client
        self.domain_detector = DomainDetector()
        self.validator = SecurityValidator()
        
    async def _generate_draft(self, prompt: str, domain: str) -> str:
        """Generate initial draft response."""
        system_prompt = get_domain_system_prompt(domain)
        draft_prompt = f"Please provide a comprehensive response to the following:\n\n{prompt}"
        
        return await self.bedrock.generate_text(draft_prompt, system_prompt)
        
    async def _generate_critiques(self, prompt: str, draft: str, domain: str) -> List[str]:
        """Generate multiple critiques in parallel."""
        critique_prompts = [
            f"Critically analyze this response for accuracy and completeness:\n\nOriginal question: {prompt}\n\nResponse: {draft}\n\nProvide specific improvements.",
            f"Evaluate this response for clarity and structure:\n\nOriginal question: {prompt}\n\nResponse: {draft}\n\nSuggest how to make it clearer.",
            f"Review this response for {domain} best practices:\n\nOriginal question: {prompt}\n\nResponse: {draft}\n\nIdentify areas for domain-specific improvement."
        ]
        
        # Generate critiques in parallel for performance
        critique_tasks = [
            self.bedrock.generate_text(
                critique_prompt, 
                temperature=0.8, 
                model_override=config.critique_model_id
            )
            for critique_prompt in critique_prompts[:config.parallel_critiques]
        ]
        
        critiques = await asyncio.gather(*critique_tasks, return_exceptions=True)
        
        # Filter out any failed critiques
        valid_critiques = [c for c in critiques if isinstance(c, str)]
        
        if not valid_critiques:
            logger.warning("All critique generations failed, using fallback")
            return ["Please improve the accuracy and clarity of the response."]
            
        return valid_critiques

    async def _synthesize_revision(self, prompt: str, draft: str, critiques: List[str], domain: str) -> str:
        """Synthesize critiques into an improved revision."""
        system_prompt = get_domain_system_prompt(domain)
        
        critique_summary = "\n\n".join([f"Critique {i+1}: {c}" for i, c in enumerate(critiques)])
        
        revision_prompt = f"""Given the original question, current response, and critiques, create an improved version.

Original question: {prompt}

Current response: {draft}

Critiques:
{critique_summary}

Create an improved response that addresses these critiques while maintaining accuracy and clarity."""
        
        return await self.bedrock.generate_text(revision_prompt, system_prompt, temperature=0.6)
        
    async def refine(self, prompt: str, domain: str = "auto") -> RefinementResult:
        """Main refinement loop implementing Draft → Critique → Revise → Converge."""
        start_time = time.time()
        
        # Validate input
        is_valid, validation_msg = self.validator.validate_prompt(prompt)
        if not is_valid:
            raise ValueError(f"Invalid prompt: {validation_msg}")
            
        # Auto-detect domain if needed
        if domain == "auto":
            domain = self.domain_detector.detect_domain(prompt)
            logger.info(f"Auto-detected domain: {domain}")

        iterations = []
        current_response = ""
        previous_embedding = None
        convergence_achieved = False
        
        try:
            for iteration_num in range(1, config.max_iterations + 1):
                logger.info(f"Starting iteration {iteration_num}")
                
                # Generate draft (or use previous revision)
                if iteration_num == 1:
                    draft = await self._generate_draft(prompt, domain)
                else:
                    draft = current_response
                    
                # Generate critiques in parallel
                critiques = await self._generate_critiques(prompt, draft, domain)
                
                # Synthesize revision
                revision = await self._synthesize_revision(prompt, draft, critiques, domain)
                current_response = revision
                
                # Calculate convergence
                current_embedding = await self.bedrock.get_embedding(revision)
                
                if previous_embedding is not None:
                    convergence_score = self.bedrock.calculate_cosine_similarity(
                        previous_embedding, 
                        current_embedding
                    )
                    logger.info(f"Convergence score: {convergence_score}")
                    
                    if convergence_score >= config.convergence_threshold:
                        convergence_achieved = True
                        logger.info(f"Convergence achieved at iteration {iteration_num}")
                else:
                    convergence_score = 0.0
                    
                # Record iteration
                iterations.append(RefinementIteration(
                    iteration_number=iteration_num,
                    draft=draft,
                    critiques=critiques,
                    revision=revision,
                    convergence_score=convergence_score
                ))
                
                # Check for convergence
                if convergence_achieved:
                    break
                    
                previous_embedding = current_embedding
                
        except asyncio.TimeoutError:
            logger.error("Refinement timeout exceeded")
            raise TimeoutError(f"Refinement exceeded {config.request_timeout} seconds")
        except Exception as e:
            logger.error(f"Refinement error: {e}")
            raise
            
        execution_time = time.time() - start_time
        
        return RefinementResult(
            final_answer=current_response,
            domain=domain,
            iterations=iterations,
            total_iterations=len(iterations),
            convergence_achieved=convergence_achieved,
            execution_time=execution_time,
            metadata={
                "model": config.bedrock_model_id,
                "embedding_model": config.embedding_model_id,
                "convergence_threshold": config.convergence_threshold,
                "max_iterations": config.max_iterations,
                "parallel_critiques": config.parallel_critiques
            }
        )