#!/usr/bin/env python3
"""
Test the tool-based refinement functionality
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from server import BedrockClient, DomainDetector, SecurityValidator
from incremental_engine import IncrementalRefineEngine


async def test_tool_refinement():
    """Test the refinement with tools"""
    
    print("🧪 Testing Tool-Based Refinement")
    print("=" * 50)
    
    # Initialize components
    bedrock = BedrockClient()
    domain_detector = DomainDetector()
    validator = SecurityValidator()
    
    engine = IncrementalRefineEngine(bedrock, domain_detector, validator)
    
    # Start a refinement session
    prompt = "Explain the benefits of test-driven development"
    print(f"\n📝 Starting refinement for: '{prompt}'")
    
    result = await engine.start_refinement(prompt, domain="technical")
    if not result['success']:
        print(f"❌ Failed to start: {result}")
        return
        
    session_id = result['session_id']
    print(f"✅ Session started: {session_id}")
    
    # Continue through the refinement steps
    step_count = 0
    while step_count < 10:  # Safety limit
        step_count += 1
        print(f"\n🔄 Step {step_count}...")
        
        result = await engine.continue_refinement(session_id)
        
        if not result['success']:
            print(f"❌ Error: {result}")
            break
            
        print(f"   Status: {result.get('status')}")
        print(f"   Progress: {result.get('progress', {}).get('percent', 0)}%")
        
        # Show tool usage if available
        if 'tool_calls' in result:
            print(f"   🛠️  Tools used: {result['tool_calls']}")
            
        if '_ai_performance' in result and 'tools_used' in result['_ai_performance']:
            print(f"   🔧 Tool details: {result['_ai_performance']['tools_used']}")
            
        # Check if we're done
        if not result.get('continue_needed', True):
            print(f"\n✅ Refinement complete!")
            if 'final_answer' in result:
                print(f"\n📄 Final answer preview:")
                print(result['final_answer'][:500] + "...")
            break
    
    print("\n" + "=" * 50)
    print("🎉 Test complete!")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(test_tool_refinement())
