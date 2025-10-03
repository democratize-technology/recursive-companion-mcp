"""Tests for internal chain-of-thought implementation."""

from unittest.mock import AsyncMock

import pytest

from recursive_companion_mcp.core.internal_cot import (
    TOOL_SPECS,
    AsyncChainOfThoughtProcessor,
    create_processor,
    is_available,
)


class TestInternalCoTModule:
    """Test the internal chain-of-thought module functions."""

    def test_is_available(self):
        """Test that is_available always returns True for internal implementation."""
        assert is_available() is True

    def test_create_processor(self):
        """Test the processor factory function."""
        processor = create_processor("test-conversation")
        assert isinstance(processor, AsyncChainOfThoughtProcessor)
        assert processor.conversation_id == "test-conversation"

    def test_tool_specs_structure(self):
        """Test that TOOL_SPECS has the correct structure for AWS Bedrock."""
        assert isinstance(TOOL_SPECS, list)
        assert len(TOOL_SPECS) == 1

        tool_spec = TOOL_SPECS[0]
        assert "toolSpec" in tool_spec

        tool_def = tool_spec["toolSpec"]
        assert tool_def["name"] == "chain_of_thought_step"
        assert "description" in tool_def
        assert "inputSchema" in tool_def

        schema = tool_def["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "properties" in schema

        properties = schema["properties"]
        assert "stage" in properties
        assert "thought" in properties
        assert "next_step_needed" in properties
        assert "confidence" in properties

        # Check required fields
        assert schema["required"] == ["stage", "thought", "next_step_needed"]


class TestAsyncChainOfThoughtProcessor:
    """Test the AsyncChainOfThoughtProcessor class."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = AsyncChainOfThoughtProcessor("test-id")
        assert processor.conversation_id == "test-id"
        assert processor.reasoning_steps == []

    def test_add_reasoning_step(self):
        """Test adding reasoning steps."""
        processor = AsyncChainOfThoughtProcessor("test-id")

        processor.add_reasoning_step("Analysis", "Testing reasoning", 0.9)

        assert len(processor.reasoning_steps) == 1
        step = processor.reasoning_steps[0]
        assert step["stage"] == "Analysis"
        assert step["thought"] == "Testing reasoning"
        assert step["confidence"] == 0.9
        assert "timestamp" in step

    def test_get_reasoning_chain(self):
        """Test getting the reasoning chain returns a copy."""
        processor = AsyncChainOfThoughtProcessor("test-id")

        processor.add_reasoning_step("Analysis", "Test step", 0.8)
        chain = processor.get_reasoning_chain()

        # Should be a copy
        assert chain == processor.reasoning_steps
        assert chain is not processor.reasoning_steps

    def test_clear_reasoning_chain(self):
        """Test clearing the reasoning chain."""
        processor = AsyncChainOfThoughtProcessor("test-id")

        processor.add_reasoning_step("Analysis", "Test step", 0.8)
        assert len(processor.reasoning_steps) == 1

        processor.clear_reasoning_chain()
        assert len(processor.reasoning_steps) == 0

    def test_get_summary_empty(self):
        """Test get_summary with no steps."""
        processor = AsyncChainOfThoughtProcessor("test-id")
        summary = processor.get_summary()
        assert summary == "No reasoning steps recorded."

    def test_get_summary_with_steps(self):
        """Test get_summary with reasoning steps."""
        processor = AsyncChainOfThoughtProcessor("test-id")

        processor.add_reasoning_step("Analysis", "Short thought", 0.8)
        processor.add_reasoning_step("Synthesis", "A very long thought " * 10, 0.9)

        summary = processor.get_summary()
        lines = summary.split("\n")
        assert len(lines) == 2
        assert "1. Analysis: Short thought" in lines[0]
        assert "2. Synthesis:" in lines[1]
        assert "..." in lines[1]  # Should be truncated

    @pytest.mark.asyncio
    async def test_process_tool_loop_success(self):
        """Test successful process_tool_loop execution."""
        processor = AsyncChainOfThoughtProcessor("test-id")

        # Mock bedrock client
        mock_client = AsyncMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "Test response"}]}}
        }

        request = {
            "messages": [{"role": "user", "content": [{"text": "test"}]}],
            "toolConfig": {"tools": TOOL_SPECS},
        }

        result = await processor.process_tool_loop(mock_client, request)

        # Verify call was made
        mock_client.converse.assert_called_once_with(**request)

        # Verify response structure
        assert "output" in result
        assert "message" in result["output"]
        assert result["output"]["message"]["content"] == [{"text": "Test response"}]

    @pytest.mark.asyncio
    async def test_process_tool_loop_with_tool_use(self):
        """Test process_tool_loop with chain-of-thought tool usage."""
        processor = AsyncChainOfThoughtProcessor("test-id")

        # Mock bedrock client with tool usage
        mock_client = AsyncMock()
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "chain_of_thought_step",
                                "input": {
                                    "stage": "Analysis",
                                    "thought": "Analyzing the problem",
                                    "confidence": 0.85,
                                    "next_step_needed": True,
                                },
                            }
                        },
                        {"text": "Final response"},
                    ]
                }
            }
        }

        request = {"messages": [], "toolConfig": {"tools": TOOL_SPECS}}

        result = await processor.process_tool_loop(mock_client, request)

        # Verify reasoning step was recorded
        assert len(processor.reasoning_steps) == 1
        step = processor.reasoning_steps[0]
        assert step["stage"] == "Analysis"
        assert step["thought"] == "Analyzing the problem"
        assert step["confidence"] == 0.85

        # Verify response structure
        assert "output" in result
        assert "message" in result["output"]

    @pytest.mark.asyncio
    async def test_process_tool_loop_exception_handling(self):
        """Test process_tool_loop exception handling."""
        processor = AsyncChainOfThoughtProcessor("test-id")

        # Mock bedrock client that raises exception
        mock_client = AsyncMock()
        mock_client.converse.side_effect = Exception("Bedrock error")

        request = {"messages": []}

        result = await processor.process_tool_loop(mock_client, request)

        # Should return error response in expected format
        assert "output" in result
        assert "message" in result["output"]
        assert "content" in result["output"]["message"]

        content = result["output"]["message"]["content"]
        assert len(content) == 1
        assert "Error in chain of thought processing" in content[0]["text"]

    def test_get_timestamp(self):
        """Test timestamp generation."""
        timestamp = AsyncChainOfThoughtProcessor._get_timestamp()

        # Should be ISO format string
        assert isinstance(timestamp, str)
        assert "T" in timestamp  # ISO format contains 'T'
        assert timestamp.endswith("Z") or "+" in timestamp or "-" in timestamp
