"""
Chain of Thought (CoT) enhancement module for recursive-companion-mcp.
Provides structured thinking prompts to improve LLM reasoning quality.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CoTMode(Enum):
    """Different modes of Chain of Thought prompting"""
    
    BASIC = "basic"
    STRUCTURED = "structured"
    DOMAIN_SPECIFIC = "domain_specific"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"


@dataclass
class CoTConfig:
    """Configuration for Chain of Thought enhancement"""
    
    enabled: bool = True
    mode: CoTMode = CoTMode.STRUCTURED
    max_thinking_steps: int = 5
    include_metacognition: bool = True
    domain_context: Optional[str] = None


class CoTEnhancer:
    """
    Chain of Thought enhancement for LLM calls in recursive refinement.
    Adds structured thinking prompts to improve reasoning quality.
    """
    
    def __init__(self, config: Optional[CoTConfig] = None):
        self.config = config or CoTConfig()
        self._cot_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[CoTMode, str]:
        """Load CoT prompt templates for different modes"""
        return {
            CoTMode.BASIC: """
Let's think step by step:
1. What is the core question or task?
2. What information do I have?
3. What approach should I take?
4. What could go wrong?

""",
            
            CoTMode.STRUCTURED: """
Let me approach this systematically:

**Analysis Phase:**
1. Understanding: What exactly is being asked?
2. Context: What information is available?
3. Constraints: What limitations should I consider?

**Planning Phase:**
4. Approach: What method will work best?
5. Steps: How should I break this down?
6. Validation: How will I check my work?

**Execution Phase:**
Now I'll work through this step by step:

""",
            
            CoTMode.DOMAIN_SPECIFIC: """
Domain-specific thinking for {domain}:

**Domain Context:** {domain_context}

**Structured Analysis:**
1. Domain-specific considerations
2. Best practices in this field
3. Potential pitfalls to avoid
4. Quality criteria for this domain

**Reasoning Process:**

""",
            
            CoTMode.CRITIQUE: """
Let me critically analyze this with structured thinking:

**Initial Assessment:**
1. What claims or conclusions are made?
2. What evidence supports them?
3. What assumptions are implicit?

**Critical Analysis:**
4. Where might this be weak or incomplete?
5. What alternative perspectives exist?
6. What would strengthen this response?

**Structured Critique:**

""",
            
            CoTMode.SYNTHESIS: """
Let me synthesize this information thoughtfully:

**Input Analysis:**
1. What are the key elements to combine?
2. What patterns or themes emerge?
3. What contradictions need resolution?

**Synthesis Strategy:**
4. How can I best integrate these elements?
5. What structure will be most clear?
6. What insights emerge from combination?

**Integrated Response:**

"""
        }
    
    def enhance_prompt(
        self,
        original_prompt: str,
        mode: Optional[CoTMode] = None,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enhance a prompt with Chain of Thought reasoning structure
        
        Args:
            original_prompt: The original prompt to enhance
            mode: CoT mode to use (overrides config default)
            domain: Domain context for domain-specific CoT
            context: Additional context for enhancement
        
        Returns:
            Enhanced prompt with CoT structure
        """
        if not self.config.enabled:
            return original_prompt
        
        # Determine mode
        cot_mode = mode or self.config.mode
        
        # Get template
        template = self._cot_templates.get(cot_mode, self._cot_templates[CoTMode.BASIC])
        
        # Format domain-specific template if needed
        if cot_mode == CoTMode.DOMAIN_SPECIFIC and domain:
            domain_context = self.config.domain_context or f"Working in {domain} domain"
            template = template.format(domain=domain, domain_context=domain_context)
        
        # Add metacognition if enabled
        metacognition = ""
        if self.config.include_metacognition:
            metacognition = self._get_metacognition_prompt(cot_mode)
        
        # Construct enhanced prompt
        enhanced = f"""{template}{metacognition}**Original Request:**
{original_prompt}

**Structured Response:**"""
        
        logger.debug(f"Enhanced prompt with {cot_mode.value} CoT (length: {len(enhanced)})")
        return enhanced
    
    def _get_metacognition_prompt(self, mode: CoTMode) -> str:
        """Get metacognitive prompting based on mode"""
        metacognition_prompts = {
            CoTMode.BASIC: "Before responding, I'll reflect on my thinking process.\n\n",
            CoTMode.STRUCTURED: "I'll monitor my reasoning quality throughout this process.\n\n",
            CoTMode.DOMAIN_SPECIFIC: "I'll apply domain expertise while staying aware of my reasoning.\n\n",
            CoTMode.CRITIQUE: "I'll maintain objectivity and intellectual honesty in my analysis.\n\n",
            CoTMode.SYNTHESIS: "I'll ensure my synthesis is coherent and well-reasoned.\n\n"
        }
        return metacognition_prompts.get(mode, "")
    
    def enhance_draft_prompt(
        self,
        prompt: str,
        domain: str = "general"
    ) -> str:
        """
        Enhance initial draft generation with appropriate CoT
        
        Args:
            prompt: Original user prompt
            domain: Domain context
        
        Returns:
            Enhanced prompt for better initial drafts
        """
        return self.enhance_prompt(
            prompt,
            mode=CoTMode.DOMAIN_SPECIFIC if domain != "general" else CoTMode.STRUCTURED,
            domain=domain
        )
    
    def enhance_critique_prompt(
        self,
        original_prompt: str,
        draft_content: str,
        critique_type: str = "general"
    ) -> str:
        """
        Enhance critique generation with critical thinking structure
        
        Args:
            original_prompt: Original user request
            draft_content: Content to critique
            critique_type: Type of critique (accuracy, clarity, etc.)
        
        Returns:
            Enhanced critique prompt
        """
        context_prompt = f"""
Original Question: {original_prompt}

Content to Analyze: {draft_content}

Focus: {critique_type} analysis
"""
        
        return self.enhance_prompt(
            context_prompt,
            mode=CoTMode.CRITIQUE
        )
    
    def enhance_synthesis_prompt(
        self,
        original_prompt: str,
        draft_content: str,
        critiques: List[str]
    ) -> str:
        """
        Enhance synthesis/revision with integration thinking
        
        Args:
            original_prompt: Original user request
            draft_content: Current draft
            critiques: List of critiques to integrate
        
        Returns:
            Enhanced synthesis prompt
        """
        critique_summary = "\n\n".join([
            f"Critique {i+1}: {critique}" 
            for i, critique in enumerate(critiques)
        ])
        
        synthesis_context = f"""
Original Question: {original_prompt}

Current Response: {draft_content}

Critiques to Address:
{critique_summary}

Task: Create an improved response that addresses these critiques while maintaining accuracy and clarity.
"""
        
        return self.enhance_prompt(
            synthesis_context,
            mode=CoTMode.SYNTHESIS
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CoT enhancement statistics"""
        return {
            "enabled": self.config.enabled,
            "mode": self.config.mode.value,
            "max_thinking_steps": self.config.max_thinking_steps,
            "include_metacognition": self.config.include_metacognition,
            "available_modes": [mode.value for mode in CoTMode]
        }


# Utility functions for easy integration
def create_cot_enhancer(
    enabled: bool = True,
    mode: str = "structured",
    domain_context: Optional[str] = None
) -> CoTEnhancer:
    """
    Factory function to create CoT enhancer with common settings
    
    Args:
        enabled: Whether to enable CoT enhancement
        mode: CoT mode ("basic", "structured", "domain_specific", etc.)
        domain_context: Context for domain-specific mode
    
    Returns:
        Configured CoT enhancer
    """
    try:
        cot_mode = CoTMode(mode)
    except ValueError:
        logger.warning(f"Unknown CoT mode '{mode}', using 'structured'")
        cot_mode = CoTMode.STRUCTURED
    
    config = CoTConfig(
        enabled=enabled,
        mode=cot_mode,
        domain_context=domain_context
    )
    
    return CoTEnhancer(config)


# Decorator for adding CoT to any LLM call
def with_cot(
    mode: str = "structured",
    enabled: bool = True
):
    """
    Decorator to add CoT enhancement to any LLM generation function
    
    Usage:
        @with_cot(mode="critique")
        async def generate_critique(prompt):
            return await llm.generate(prompt)
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if not enabled:
                return await func(*args, **kwargs)
            
            # Find prompt in args or kwargs
            prompt = None
            if args:
                prompt = args[0]
            elif 'prompt' in kwargs:
                prompt = kwargs['prompt']
            
            if prompt:
                enhancer = create_cot_enhancer(enabled=enabled, mode=mode)
                enhanced_prompt = enhancer.enhance_prompt(prompt)
                
                # Replace prompt in args or kwargs
                if args:
                    args = (enhanced_prompt,) + args[1:]
                else:
                    kwargs['prompt'] = enhanced_prompt
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator