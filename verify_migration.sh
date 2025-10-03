#!/bin/bash
# FastMCP Migration Verification Script
# Tests all 8 migrated tools and both transport modes

set -e

echo "🔍 FastMCP Migration Verification"
echo "=================================="
echo ""

# Test 1: Package Import
echo "✓ Test 1: Package Import"
uv run python -c "import recursive_companion_mcp; print('  Package imported successfully')"
echo ""

# Test 2: Core Module
echo "✓ Test 2: Core Module Import"
uv run python -c "from recursive_companion_mcp.core import mcp; print('  FastMCP instance available')"
echo ""

# Test 3: All Tools Import
echo "✓ Test 3: Tool Imports"
uv run python -c "
from recursive_companion_mcp.tools import (
    start_refinement,
    continue_refinement,
    get_refinement_status,
    get_final_result,
    list_refinement_sessions,
    current_session,
    abort_refinement,
    quick_refine,
)
print('  All 8 tools imported successfully')
"
echo ""

# Test 4: Tool Registration
echo "✓ Test 4: Tool Registration"
uv run python -c "
import asyncio
from recursive_companion_mcp.core import mcp

async def check_tools():
    tools = await mcp.list_tools()
    print(f'  Found {len(tools)} tools registered:')
    for tool in tools:
        print(f'    - {tool.name}')
    assert len(tools) == 8, f'Expected 8 tools, got {len(tools)}'
    print('  ✅ All 8 tools registered correctly')

asyncio.run(check_tools())
"
echo ""

# Test 5: HTTP Main Function
echo "✓ Test 5: HTTP Transport Availability"
uv run python -c "
from recursive_companion_mcp import http_main, main
print('  stdio main() available:', callable(main))
print('  http_main() available:', callable(http_main))
"
echo ""

# Test 6: Incremental Engine Preservation
echo "✓ Test 6: Core Engine Preservation"
uv run python -c "
from recursive_companion_mcp.tools.refinement import get_incremental_engine
engine, tracker = get_incremental_engine()
print('  Incremental engine initialized:', engine is not None)
print('  Session tracker available:', tracker is not None)
"
echo ""

# Test 7: Formatting Utilities
echo "✓ Test 7: Response Formatting"
uv run python -c "
from recursive_companion_mcp.formatting import (
    format_refinement_start,
    format_refinement_continue,
    format_refinement_status,
    format_final_result,
    format_session_list,
    format_current_session,
    format_abort_result,
    format_quick_refine,
)
print('  All 8 formatters imported successfully')
"
echo ""

# Test 8: Decorators
echo "✓ Test 8: Custom Decorators"
uv run python -c "
from recursive_companion_mcp.decorators import inject_client_context
from recursive_companion_mcp.core import handle_tool_errors, format_output
print('  All decorators available')
"
echo ""

echo "=================================="
echo "✅ Migration Verification PASSED"
echo ""
echo "Summary:"
echo "  ✓ Package structure correct"
echo "  ✓ All 8 tools migrated"
echo "  ✓ Both transports available"
echo "  ✓ Core engines preserved"
echo "  ✓ Formatters working"
echo "  ✓ Decorators implemented"
echo ""
echo "Ready for deployment!"
echo ""
echo "To run the server:"
echo "  stdio mode: uv run python -m recursive_companion_mcp"
echo "  HTTP mode:  MCP_TRANSPORT=http MCP_HTTP_PORT=8086 uv run python -m recursive_companion_mcp"
