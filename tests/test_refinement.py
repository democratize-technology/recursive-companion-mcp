"""
Tests for Recursive Companion MCP - Bedrock Edition
"""
import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np
import boto3
from moto import mock_bedrock

from src.server import (
    RecursiveCompanionMCP, 
    Domain, 
    RefinementRequest,
    RefinementHistory,
    MAX_PROMPT_LENGTH,
    RATE_LIMIT_MAX_CALLS
)

@pytest.fixture
def server():
    """Create a test server instance"""
    with patch.dict('os.environ', {
        'AWS_REGION': 'us-east-1',
        'BEDROCK_MODEL_ID': 'anthropic.claude-3-sonnet-20240229-v1:0'
    }):
        server = RecursiveCompanionMCP()
        # Mock the Bedrock clients
        server.bedrock_client = MagicMock()
        server.bedrock_runtime = MagicMock()
        return server

@pytest.fixture
def mock_bedrock_response():
    """Mock Bedrock response for Claude"""
    return {
        'body': MagicMock(read=lambda: json.dumps({
            'content': [{'text': 'Test response from Bedrock Claude'}]
        }).encode('utf-8'))
    }

@pytest.fixture
def mock_embedding_response():
    """Mock Bedrock response for Titan embeddings"""
    return {
        'body': MagicMock(read=lambda: json.dumps({
            'embedding': np.random.randn(1536).tolist()
        }).encode('utf-8'))
    }

class TestRefinementRequest:
    """Test input validation"""
    
    def test_valid_request(self):
        req = RefinementRequest(prompt="Test prompt")
        assert req.prompt == "Test prompt"
        assert req.domain == "auto"
        assert req.convergence_threshold == 0.98
    
    def test_empty_prompt_rejected(self):
        with pytest.raises(ValueError):
            RefinementRequest(prompt="")

class TestDomainDetection:
    """Test domain auto-detection"""
    
    @pytest.mark.asyncio
    async def test_technical_domain_detection(self, server):
        domain = server._detect_domain("How do I optimize this algorithm for better performance?")
        assert domain == Domain.TECHNICAL
    
    @pytest.mark.asyncio
    async def test_marketing_domain_detection(self, server):
        domain = server._detect_domain("Create a campaign to increase brand engagement")
        assert domain == Domain.MARKETING
    
    @pytest.mark.asyncio
    async def test_general_domain_fallback(self, server):
        domain = server._detect_domain("What is the weather today?")
        assert domain == Domain.GENERAL

if __name__ == "__main__":
    pytest.main([__file__, "-v"])