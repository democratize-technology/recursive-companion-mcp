#!/usr/bin/env python3
"""
Recursive Companion MCP Server - AWS Bedrock Edition
Based on Hank Besser's recursive-companion: https://github.com/hankbesser/recursive-companion
Implements iterative refinement through Draft → Critique → Revise → Converge cycles
"""
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from functools import lru_cache
import hashlib

import boto3
import numpy as np
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from incremental_engine import IncrementalRefineEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("recursive-companion")

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
CLAUDE_MODEL = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
CRITIQUE_MODEL = os.getenv("CRITIQUE_MODEL_ID", CLAUDE_MODEL)  # Can use Haiku for faster critiques
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.98"))
PARALLEL_CRITIQUES = int(os.getenv("PARALLEL_CRITIQUES", "3"))

# Security configurations
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "10000"))
MIN_PROMPT_LENGTH = int(os.getenv("MIN_PROMPT_LENGTH", "10"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes

# Domain configurations
DOMAIN_KEYWORDS = {
    "technical": ["code", "algorithm", "api", "debug", "performance", "architecture", "system", "database", "security"],
    "marketing": ["campaign", "audience", "brand", "roi", "conversion", "engagement", "strategy", "market"],
    "strategy": ["goal", "objective", "plan", "roadmap", "vision", "competitive", "analysis", "swot"],
    "legal": ["contract", "compliance", "regulation", "liability", "agreement", "terms", "privacy", "gdpr"],
    "financial": ["revenue", "cost", "budget", "forecast", "investment", "profit", "cash flow", "valuation"]
}

@dataclass
class RefinementIteration:
    """Represents a single iteration in the refinement process"""
    iteration_number: int
    draft: str
    critiques: List[str]
    revision: str
    convergence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
@dataclass
class RefinementResult:
    """Complete result of the refinement process"""
    final_answer: str
    domain: str
    iterations: List[RefinementIteration]
    total_iterations: int
    convergence_achieved: bool
    execution_time: float
    metadata: Dict[str, Any]

class SecurityValidator:
    """Handles input validation and security checks"""
    
    @staticmethod
    def validate_prompt(prompt: str) -> Tuple[bool, str]:
        """Validate prompt for security and constraints"""
        if not prompt or len(prompt.strip()) < MIN_PROMPT_LENGTH:
            return False, f"Prompt too short (minimum {MIN_PROMPT_LENGTH} characters)"
            
        if len(prompt) > MAX_PROMPT_LENGTH:
            return False, f"Prompt too long (maximum {MAX_PROMPT_LENGTH} characters)"
            
        # Check for potential injection patterns
        dangerous_patterns = [
            r"ignore\s+previous\s+instructions",
            r"system\s+prompt",
            r"<\s*script",
            r"javascript:",
            r"eval\s*\(",
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return False, "Potentially dangerous content detected"
                
        return True, "Valid"

class DomainDetector:
    """Detects the appropriate domain for a given prompt"""
    
    @staticmethod
    def detect_domain(prompt: str) -> str:
        """Auto-detect domain based on keywords and patterns"""
        prompt_lower = prompt.lower()
        domain_scores = {}
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > 0:
                domain_scores[domain] = score
                
        if not domain_scores:
            return "general"
            
        # Return domain with highest score
        return max(domain_scores, key=domain_scores.get)

class BedrockClient:
    """Wrapper for AWS Bedrock operations"""
    
    def __init__(self):
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=AWS_REGION
        )
        self._embedding_cache = {}
        
    async def generate_text(self, prompt: str, system_prompt: str = "", temperature: float = 0.7, model_override: str = None) -> str:
        """Generate text using Claude via Bedrock"""
        try:
            model = model_override or CLAUDE_MODEL
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            if system_prompt:
                body["system"] = system_prompt
                
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.bedrock_runtime.invoke_model(
                    modelId=model,
                    body=json.dumps(body)
                )
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
            
        except Exception as e:
            logger.error(f"Bedrock generation error: {e}")
            raise

    @lru_cache(maxsize=1000)
    def _get_embedding_cached(self, text_hash: str) -> List[float]:
        """Cache wrapper for embeddings"""
        return self._get_embedding_uncached(text_hash)
        
    def _get_embedding_uncached(self, text: str) -> List[float]:
        """Get text embedding using Titan"""
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=EMBEDDING_MODEL,
                body=json.dumps({"inputText": text})
            )
            response_body = json.loads(response['body'].read())
            return response_body['embedding']
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise
            
    async def get_embedding(self, text: str) -> List[float]:
        """Get text embedding with caching"""
        # Create hash for caching
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._get_embedding_cached(text_hash) 
            if text_hash in self._embedding_cache 
            else self._get_embedding_uncached(text)
        )

class RefineEngine:
    """Implements the Draft → Critique → Revise → Converge refinement pattern"""
    
    def __init__(self, bedrock_client: BedrockClient):
        self.bedrock = bedrock_client
        self.domain_detector = DomainDetector()
        self.validator = SecurityValidator()
        
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
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

    async def _generate_draft(self, prompt: str, domain: str) -> str:
        """Generate initial draft response"""
        system_prompt = self._get_domain_system_prompt(domain)
        draft_prompt = f"Please provide a comprehensive response to the following:\n\n{prompt}"
        
        return await self.bedrock.generate_text(draft_prompt, system_prompt)
        
    async def _generate_critiques(self, prompt: str, draft: str, domain: str) -> List[str]:
        """Generate multiple critiques in parallel"""
        critique_prompts = [
            f"Critically analyze this response for accuracy and completeness:\n\nOriginal question: {prompt}\n\nResponse: {draft}\n\nProvide specific improvements.",
            f"Evaluate this response for clarity and structure:\n\nOriginal question: {prompt}\n\nResponse: {draft}\n\nSuggest how to make it clearer.",
            f"Review this response for {domain} best practices:\n\nOriginal question: {prompt}\n\nResponse: {draft}\n\nIdentify areas for domain-specific improvement."
        ]
        
        # Generate critiques in parallel for performance
        critique_tasks = [
            self.bedrock.generate_text(critique_prompt, temperature=0.8, model_override=CRITIQUE_MODEL)
            for critique_prompt in critique_prompts[:PARALLEL_CRITIQUES]
        ]
        
        critiques = await asyncio.gather(*critique_tasks, return_exceptions=True)
        
        # Filter out any failed critiques
        valid_critiques = [c for c in critiques if isinstance(c, str)]
        
        if not valid_critiques:
            logger.warning("All critique generations failed, using fallback")
            return ["Please improve the accuracy and clarity of the response."]
            
        return valid_critiques

    async def _synthesize_revision(self, prompt: str, draft: str, critiques: List[str], domain: str) -> str:
        """Synthesize critiques into an improved revision"""
        system_prompt = self._get_domain_system_prompt(domain)
        
        critique_summary = "\n\n".join([f"Critique {i+1}: {c}" for i, c in enumerate(critiques)])
        
        revision_prompt = f"""Given the original question, current response, and critiques, create an improved version.

Original question: {prompt}

Current response: {draft}

Critiques:
{critique_summary}

Create an improved response that addresses these critiques while maintaining accuracy and clarity."""
        
        return await self.bedrock.generate_text(revision_prompt, system_prompt, temperature=0.6)
        
    async def refine(self, prompt: str, domain: str = "auto") -> RefinementResult:
        """Main refinement loop implementing Draft → Critique → Revise → Converge"""
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
            for iteration_num in range(1, MAX_ITERATIONS + 1):
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
                    convergence_score = self._cosine_similarity(previous_embedding, current_embedding)
                    logger.info(f"Convergence score: {convergence_score}")
                    
                    if convergence_score >= CONVERGENCE_THRESHOLD:
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
            raise TimeoutError(f"Refinement exceeded {REQUEST_TIMEOUT} seconds")
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
                "model": CLAUDE_MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "convergence_threshold": CONVERGENCE_THRESHOLD,
                "max_iterations": MAX_ITERATIONS,
                "parallel_critiques": PARALLEL_CRITIQUES
            }
        )

# Initialize global instances
bedrock_client = None
refine_engine = None
incremental_engine = None

# Tool handlers
@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="start_refinement",
            description=(
                "Start a new incremental refinement session. Returns immediately with a session ID. "
                "Use continue_refinement to proceed step by step."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The question or task to refine"
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["auto", "technical", "marketing", "strategy", "legal", "financial", "general"],
                        "default": "auto"
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="continue_refinement",
            description=(
                "Continue an active refinement session by one step. "
                "Each call performs one action: draft, critique, or revise."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The refinement session ID"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="get_refinement_status",
            description="Get the current status of a refinement session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The refinement session ID"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="get_final_result",
            description="Get the final refined answer once convergence is achieved.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The refinement session ID"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="list_refinement_sessions",
            description="List all active refinement sessions.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    global refine_engine, incremental_engine
    
    # Incremental refinement tools
    if name == "start_refinement":
        try:
            if not incremental_engine:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Incremental engine not initialized",
                        "success": False
                    }, indent=2)
                )]
            
            prompt = arguments.get('prompt', '')
            domain = arguments.get('domain', 'auto')
            
            result = await incremental_engine.start_refinement(prompt, domain)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            logger.error(f"Start refinement error: {e}")
            return [TextContent(type="text", text=json.dumps({
                "error": f"Failed to start refinement: {str(e)}",
                "success": False
            }, indent=2))]
    
    elif name == "continue_refinement":
        try:
            if not incremental_engine:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Incremental engine not initialized",
                        "success": False
                    }, indent=2)
                )]
            
            session_id = arguments.get('session_id', '')
            
            result = await incremental_engine.continue_refinement(session_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            logger.error(f"Continue refinement error: {e}")
            return [TextContent(type="text", text=json.dumps({
                "error": f"Failed to continue refinement: {str(e)}",
                "success": False
            }, indent=2))]
    
    elif name == "get_refinement_status":
        try:
            if not incremental_engine:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Incremental engine not initialized",
                        "success": False
                    }, indent=2)
                )]
            
            session_id = arguments.get('session_id', '')
            
            result = await incremental_engine.get_status(session_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            logger.error(f"Get status error: {e}")
            return [TextContent(type="text", text=json.dumps({
                "error": f"Failed to get status: {str(e)}",
                "success": False
            }, indent=2))]
    
    elif name == "get_final_result":
        try:
            if not incremental_engine:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Incremental engine not initialized",
                        "success": False
                    }, indent=2)
                )]
            
            session_id = arguments.get('session_id', '')
            
            result = await incremental_engine.get_final_result(session_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            logger.error(f"Get final result error: {e}")
            return [TextContent(type="text", text=json.dumps({
                "error": f"Failed to get final result: {str(e)}",
                "success": False
            }, indent=2))]
    
    elif name == "list_refinement_sessions":
        try:
            if not incremental_engine:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "Incremental engine not initialized",
                        "success": False
                    }, indent=2)
                )]
            
            sessions = incremental_engine.session_manager.list_active_sessions()
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "sessions": sessions,
                "count": len(sessions)
            }, indent=2))]
            
        except Exception as e:
            logger.error(f"List sessions error: {e}")
            return [TextContent(type="text", text=json.dumps({
                "error": f"Failed to list sessions: {str(e)}",
                "success": False
            }, indent=2))]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Main entry point"""
    global bedrock_client, refine_engine, incremental_engine
    
    try:
        # Initialize Bedrock client
        bedrock_client = BedrockClient()
        refine_engine = RefineEngine(bedrock_client)
        
        # Initialize incremental engine
        incremental_engine = IncrementalRefineEngine(
            bedrock_client,
            DomainDetector(),
            SecurityValidator()
        )
        
        # Test Bedrock connection
        bedrock_test = boto3.client(
            service_name='bedrock',
            region_name=AWS_REGION
        )
        bedrock_test.list_foundation_models()
        logger.info("Successfully connected to AWS Bedrock")
        logger.info(f"Using Claude model: {CLAUDE_MODEL}")
        logger.info(f"Using embedding model: {EMBEDDING_MODEL}")
        
        logger.info("Starting Recursive Companion MCP server")
        logger.info(f"Configuration: max_iterations={MAX_ITERATIONS}, convergence_threshold={CONVERGENCE_THRESHOLD}")
        
        async with stdio_server() as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
