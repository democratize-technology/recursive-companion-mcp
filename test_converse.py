#!/usr/bin/env python3
"""
Test script for converse API with refinement tools
"""

import asyncio
import boto3
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
CLAUDE_MODEL = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")


async def test_converse_with_tools():
    """Test the converse API with refinement tools"""
    
    # Initialize bedrock client
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name=AWS_REGION
    )
    
    # Define refinement tools
    tools = [
        {
            "toolSpec": {
                "name": "identify_weakness",
                "description": "Analyze text and identify a specific weakness or area for improvement",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "text_segment": {
                                "type": "string",
                                "description": "The portion of text to analyze"
                            },
                            "weakness_type": {
                                "type": "string",
                                "enum": ["clarity", "accuracy", "completeness", "coherence", "depth"],
                                "description": "Type of weakness to look for"
                            }
                        },
                        "required": ["text_segment", "weakness_type"]
                    }
                }
            }
        }
    ]
    
    # Test message
    test_text = """Microservices architecture is a way to build software applications 
    as a collection of small services. Each service runs in its own process 
    and communicates through APIs."""
    
    messages = [
        {
            "role": "user",
            "content": [{"text": f"Analyze this text for weaknesses using the identify_weakness tool:\n\n{test_text}"}]
        }
    ]
    
    try:
        print("Testing converse API with tools...")
        print(f"Model: {CLAUDE_MODEL}")
        print(f"Region: {AWS_REGION}")
        print("-" * 50)
        
        # Call converse API
        response = bedrock_runtime.converse(
            modelId=CLAUDE_MODEL,
            messages=messages,
            toolConfig={
                "tools": tools,
                "toolChoice": {"auto": {}}
            },
            inferenceConfig={
                "temperature": 0.3,
                "maxTokens": 1000
            }
        )
        
        print("\nResponse received!")
        print("-" * 50)
        
        # Parse response
        message = response['output']['message']
        
        print("\nContent blocks:")
        for i, content_block in enumerate(message['content']):
            print(f"\nBlock {i+1}:")
            if 'text' in content_block:
                print(f"  Type: Text")
                print(f"  Content: {content_block['text'][:200]}...")
            elif 'toolUse' in content_block:
                tool_use = content_block['toolUse']
                print(f"  Type: Tool Use")
                print(f"  Tool: {tool_use['name']}")
                print(f"  Input: {json.dumps(tool_use['input'], indent=2)}")
        
        print("\n" + "=" * 50)
        print("✅ Converse API with tools is working!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response}")


if __name__ == "__main__":
    asyncio.run(test_converse_with_tools())
