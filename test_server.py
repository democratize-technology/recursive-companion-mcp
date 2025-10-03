#!/usr/bin/env python3
"""Quick test script for the FastMCP server"""

import asyncio
import sys


async def test_server():
    """Test server initialization"""
    try:
        from recursive_companion_mcp.core import mcp

        print("✓ Server imported")

        # List tools
        tools = await mcp.list_tools()
        print(f"✓ Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:60]}...")

        return True
    except Exception as e:
        print(f"✗ Server test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_server())
    sys.exit(0 if success else 1)
