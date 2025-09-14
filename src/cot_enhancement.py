"""Chain of Thought enhancement for recursive refinement sessions.

This module provides structured thinking capabilities for iterative content
refinement with convergence reasoning patterns.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ChainOfThoughtEnhancer:
    """Enhances refinement prompts with CoT structured thinking patterns."""

    def __init__(self, enabled: bool = True):
        """Initialize the CoT enhancer.

        Args:
            enabled: Whether CoT enhancement is active
        """
        self.enabled = enabled

        # Use internal chain-of-thought implementation for security
        try:
            import internal_cot  # noqa: F401

            self.cot_available = True
            logger.info(
                "Internal Chain of Thought tools loaded successfully for refinement enhancement"
            )
        except ImportError as e:
            self.cot_available = False
            logger.warning(f"Internal Chain of Thought tools not available for enhancement: {e}")

    def enhance_initial_refinement_prompt(self, content: str, domain_type: str) -> str:
        """Enhance initial refinement prompt with structured thinking.

        Args:
            content: The content to be refined
            domain_type: The detected domain (technical, marketing, legal, etc.)

        Returns:
            Enhanced prompt with CoT structure for initial refinement
        """
        if not self.enabled or not self.cot_available:
            return self._create_initial_fallback_prompt(content, domain_type)

        # Create domain-aware structured thinking prompt
        cot_prompt = f"""Let me think through this {domain_type} content refinement systematically:

**Step 1 - Content Analysis:**
What type of {domain_type} content is this? What are its core goals and constraints?

**Step 2 - Domain Requirements:**
What are the key quality criteria for {domain_type} content?
- Technical: accuracy, clarity, completeness, maintainability
- Marketing: persuasiveness, audience appeal, call-to-action clarity
- Legal: precision, compliance, risk mitigation
- General: readability, structure, purpose alignment

**Step 3 - Improvement Strategy:**
What specific aspects need refinement to meet {domain_type} standards?
How can I enhance this while preserving the original intent?

**Step 4 - Refinement Planning:**
What systematic approach will yield the best refined version?

---

Original content to refine:
{content}

Create an improved version that excels in {domain_type} quality criteria while maintaining the core message."""

        return cot_prompt

    def enhance_iteration_prompt(self, iteration_data: dict[str, Any]) -> str:
        """Enhance iteration prompts with convergence context.

        Args:
            iteration_data: Dictionary containing iteration context:
                - current_draft: The current refined version
                - previous_draft: The previous version
                - critiques: List of critiques from this iteration
                - convergence_score: Current convergence score (0.0-1.0)
                - iteration_number: Current iteration count
                - domain_type: Content domain

        Returns:
            Enhanced prompt with convergence reasoning
        """
        if not self.enabled or not self.cot_available:
            return self._create_iteration_fallback_prompt(iteration_data)

        current_draft = iteration_data.get("current_draft", "")
        previous_draft = iteration_data.get("previous_draft", "")
        critiques = iteration_data.get("critiques", [])
        convergence_score = iteration_data.get("convergence_score", 0.0)
        iteration_num = iteration_data.get("iteration_number", 0)
        domain_type = iteration_data.get("domain_type", "general")

        # Analyze convergence context for reasoning
        convergence_context = self._analyze_convergence_context(convergence_score, iteration_num)

        critique_summary = self._summarize_critiques(critiques)

        cot_prompt = f"""Let me think through this refinement iteration systematically:

**Step 1 - Convergence Analysis:**
Current convergence score: {convergence_score:.3f} (iteration {iteration_num})
{convergence_context}

**Step 2 - Critique Integration:**
Key feedback from this iteration:
{critique_summary}

**Step 3 - Change Assessment:**
What meaningful improvements can I make without over-engineering?
How do I balance refinement with convergence efficiency?

**Step 4 - Strategic Refinement:**
What changes will have the highest impact on {domain_type} content quality?
Should I make incremental tweaks or more substantial improvements?

**Step 5 - Convergence Prediction:**
Will these changes likely lead to convergence, or do we need more iterations?
What's the optimal level of refinement for this content?

---

Previous version:
{previous_draft}

Current version:
{current_draft}

Create the next iteration by thoughtfully addressing the critiques while considering convergence dynamics."""

        return cot_prompt

    def enhance_convergence_decision_prompt(
        self,
        current: str,
        previous: str,
        similarity_score: float,
        threshold: float,
        iteration_count: int,
    ) -> str:
        """Enhance convergence decision-making with structured reasoning.

        Args:
            current: Current iteration content
            previous: Previous iteration content
            similarity_score: Computed similarity score
            threshold: Convergence threshold
            iteration_count: Number of iterations completed

        Returns:
            Enhanced prompt for convergence reasoning
        """
        if not self.enabled or not self.cot_available:
            return self._create_convergence_fallback_prompt(
                similarity_score, threshold, iteration_count
            )

        # Determine convergence context
        converged = similarity_score >= threshold
        convergence_gap = threshold - similarity_score

        cot_prompt = f"""Let me analyze this convergence decision systematically:

**Step 1 - Quantitative Analysis:**
Similarity score: {similarity_score:.4f} vs threshold: {threshold:.4f}
Gap to convergence: {convergence_gap:.4f}
Iterations completed: {iteration_count}

**Step 2 - Qualitative Assessment:**
Are the changes between versions meaningful improvements or just rephrasing?
Is the content reaching its quality potential or could it be better?

**Step 3 - Refinement Economics:**
Cost: Additional iterations require more time and resources
Benefit: Each iteration potentially improves quality
Risk: Over-refinement can introduce errors or reduce naturalness

**Step 4 - Convergence Recommendation:**
Based on both metrics and content quality, should we:
a) Stop here - content has reached good convergence
b) Continue - meaningful improvements still possible
c) Override threshold - quality demands exceed convergence metrics

**Step 5 - Final Decision:**
Provide a reasoned recommendation with supporting rationale.

---

Previous iteration:
{previous[:500]}...

Current iteration:
{current[:500]}...

Analyze whether these represent meaningful convergence or if further refinement would be valuable."""

        return cot_prompt

    def _create_initial_fallback_prompt(self, content: str, domain_type: str) -> str:
        """Create fallback prompt for initial refinement when CoT unavailable."""
        return f"""Refine this {domain_type} content to improve its quality:

Consider:
- {domain_type}-specific quality criteria
- Clarity and structure
- Purpose alignment
- Audience effectiveness

Original content:
{content}

Provide an improved version."""

    def _create_iteration_fallback_prompt(self, iteration_data: dict[str, Any]) -> str:
        """Create fallback prompt for iterations when CoT unavailable."""
        current_draft = iteration_data.get("current_draft", "")
        critiques = iteration_data.get("critiques", [])

        critique_summary = self._summarize_critiques(critiques)

        return f"""Refine this content based on the feedback:

Critiques to address:
{critique_summary}

Current version:
{current_draft}

Create an improved version addressing these points."""

    def _create_convergence_fallback_prompt(
        self, similarity_score: float, threshold: float, iteration_count: int
    ) -> str:
        """Create fallback prompt for convergence decisions when CoT unavailable."""
        converged = similarity_score >= threshold

        return f"""Convergence Analysis:
Score: {similarity_score:.4f} vs Threshold: {threshold:.4f}
Iterations: {iteration_count}
Status: {'Converged' if converged else 'Continue refining'}

Should refinement continue? Consider quality vs. efficiency."""

    def _analyze_convergence_context(self, score: float, iteration: int) -> str:
        """Analyze convergence context for reasoning enhancement."""
        if score >= 0.95:
            return "High convergence - content is stabilizing rapidly"
        elif score >= 0.85:
            return "Moderate convergence - making steady progress toward stability"
        elif score >= 0.70:
            return "Low convergence - significant changes still happening"
        elif iteration <= 2:
            return "Early iterations - expect lower convergence as content develops"
        else:
            return "Low convergence despite multiple iterations - may need strategy change"

    def _summarize_critiques(self, critiques: list) -> str:
        """Summarize critiques for inclusion in prompts."""
        if not critiques:
            return "No specific critiques provided"

        # Extract focus areas from critiques
        if isinstance(critiques[0], dict):
            summaries = []
            for i, critique in enumerate(critiques):
                focus = critique.get("focus", f"Area {i+1}")
                content = critique.get("content", str(critique))[:200]
                summaries.append(f"- {focus}: {content}...")
            return "\n".join(summaries)
        else:
            # Handle simple string critiques
            return "\n".join(f"- {str(critique)[:200]}..." for critique in critiques[:3])

    def should_use_cot(self, complexity_score: float = 0.6) -> bool:
        """Determine if CoT should be used based on complexity.

        Args:
            complexity_score: Estimated refinement complexity (0.0-1.0)

        Returns:
            True if CoT should be used
        """
        if not self.enabled or not self.cot_available:
            return False

        # Use CoT for moderately complex refinements and above
        return complexity_score > 0.5

    def estimate_refinement_complexity(
        self, content_length: int, domain_type: str, iteration_count: int
    ) -> float:
        """Estimate refinement complexity to determine CoT usage.

        Args:
            content_length: Length of content being refined
            domain_type: Content domain (affects complexity)
            iteration_count: Current iteration number

        Returns:
            Complexity score (0.0-1.0)
        """
        # Base complexity from content length
        length_factor = min(content_length / 1000, 1.0)  # 0-1 based on length up to 1000 chars

        # Domain complexity multipliers
        domain_complexity = {
            "technical": 0.8,
            "legal": 0.9,
            "marketing": 0.6,
            "general": 0.5,
            "academic": 0.7,
            "creative": 0.4,
        }

        domain_factor = domain_complexity.get(domain_type, 0.5)

        # Iteration complexity (early iterations are more complex)
        iteration_factor = max(0.3, 1.0 - (iteration_count * 0.2))

        # Combined complexity score
        complexity = length_factor * 0.4 + domain_factor * 0.4 + iteration_factor * 0.2

        return min(complexity, 1.0)


# Factory function for creating configured enhancer
def create_cot_enhancer(enabled: bool = True) -> ChainOfThoughtEnhancer:
    """Create a configured Chain of Thought enhancer for refinement.

    Args:
        enabled: Whether to enable CoT enhancement

    Returns:
        Configured ChainOfThoughtEnhancer instance
    """
    enhancer = ChainOfThoughtEnhancer(enabled)

    logger.info(
        f"Created refinement CoT enhancer: enabled={enabled}, available={enhancer.cot_available}"
    )

    return enhancer
