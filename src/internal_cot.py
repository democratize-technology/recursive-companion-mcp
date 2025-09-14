"""Internal Chain of Thought implementation.

This module provides an internal implementation of chain-of-thought reasoning
to replace the external chain-of-thought-tool dependency and eliminate supply chain risk.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Tool specifications for AWS Bedrock Chain of Thought reasoning
TOOL_SPECS = [
    {
        "toolSpec": {
            "name": "chain_of_thought_step",
            "description": "Structure your reasoning using systematic chain of thought steps",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "stage": {
                            "type": "string",
                            "enum": [
                                "Problem Definition",
                                "Research",
                                "Analysis",
                                "Synthesis",
                                "Conclusion",
                            ],
                            "description": "Current reasoning stage",
                        },
                        "thought": {
                            "type": "string",
                            "description": "Your reasoning at this stage",
                        },
                        "next_step_needed": {
                            "type": "boolean",
                            "description": "Whether another reasoning step is needed after this one",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence level in this reasoning step (0.0 to 1.0)",
                        },
                    },
                    "required": ["stage", "thought", "next_step_needed"],
                }
            },
        }
    }
]


class AsyncChainOfThoughtProcessor:
    """Internal Chain of Thought processor for structured reasoning.

    This class provides the same interface as the external chain-of-thought-tool
    but with an internal implementation to eliminate supply chain risk.
    """

    def __init__(self, conversation_id: str):
        """Initialize the CoT processor.

        Args:
            conversation_id: Unique identifier for this conversation/session
        """
        self.conversation_id = conversation_id
        self.reasoning_steps: list[dict[str, Any]] = []
        logger.debug(f"Initialized internal CoT processor for conversation: {conversation_id}")

    def add_reasoning_step(self, stage: str, thought: str, confidence: float = 0.8) -> None:
        """Add a reasoning step to the chain.

        Args:
            stage: The reasoning stage
            thought: The actual reasoning content
            confidence: Confidence level in this step
        """
        step = {
            "stage": stage,
            "thought": thought,
            "confidence": confidence,
            "timestamp": self._get_timestamp(),
        }
        self.reasoning_steps.append(step)
        logger.debug(f"Added reasoning step: {stage}")

    def get_reasoning_chain(self) -> list[dict[str, Any]]:
        """Get the complete reasoning chain."""
        return self.reasoning_steps.copy()

    def clear_reasoning_chain(self) -> None:
        """Clear the current reasoning chain."""
        self.reasoning_steps.clear()
        logger.debug(f"Cleared reasoning chain for conversation: {self.conversation_id}")

    def get_summary(self) -> str:
        """Get a summary of the reasoning chain."""
        if not self.reasoning_steps:
            return "No reasoning steps recorded."

        summary_parts = []
        for i, step in enumerate(self.reasoning_steps, 1):
            summary_parts.append(
                f"{i}. {step['stage']}: {step['thought'][:100]}..."
                if len(step["thought"]) > 100
                else f"{i}. {step['stage']}: {step['thought']}"
            )

        return "\n".join(summary_parts)

    async def process_tool_loop(self, bedrock_client, initial_request):
        """Process a tool-enabled request using AWS Bedrock with Chain of Thought reasoning.

        This method provides the same interface as the external chain-of-thought-tool
        for compatibility with existing code.

        Args:
            bedrock_client: AWS Bedrock client instance
            initial_request: Request dictionary containing messages and toolConfig

        Returns:
            Dictionary with format expected by incremental_engine.py:
            {
                "output": {
                    "message": {
                        "content": [{"text": "response text"}]
                    }
                }
            }
        """
        try:
            # Use the bedrock client to process the request with tool support
            logger.debug(f"Processing CoT request for conversation: {self.conversation_id}")

            # Call bedrock converse with tool support
            response = await bedrock_client.converse(**initial_request)

            # Extract the response content
            response_message = response.get("output", {}).get("message", {})
            content = response_message.get("content", [])

            # Log the reasoning steps if available
            for item in content:
                if "toolUse" in item:
                    tool_use = item["toolUse"]
                    if tool_use.get("name") == "chain_of_thought_step":
                        input_data = tool_use.get("input", {})
                        self.add_reasoning_step(
                            stage=input_data.get("stage", "Unknown"),
                            thought=input_data.get("thought", ""),
                            confidence=input_data.get("confidence", 0.8),
                        )

            # Return in expected format
            return {"output": {"message": response_message}}

        except Exception as e:
            logger.error(f"Error in CoT process_tool_loop: {e}")
            # Return fallback response in expected format
            return {
                "output": {
                    "message": {
                        "content": [{"text": f"Error in chain of thought processing: {str(e)}"}]
                    }
                }
            }

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime

        return datetime.utcnow().isoformat()


# Factory function for creating processors (maintains compatibility)
def create_processor(conversation_id: str) -> AsyncChainOfThoughtProcessor:
    """Create a new Chain of Thought processor.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        Configured AsyncChainOfThoughtProcessor instance
    """
    return AsyncChainOfThoughtProcessor(conversation_id)


# Compatibility function to check if CoT is available
def is_available() -> bool:
    """Check if Chain of Thought processing is available.

    Returns:
        True (always available with internal implementation)
    """
    return True


# Export the main components needed by the incremental engine
__all__ = ["TOOL_SPECS", "AsyncChainOfThoughtProcessor", "create_processor", "is_available"]
