#!/bin/bash
# Wrapper script for Recursive Companion MCP that finds uv dynamically

# Find uv in PATH
UV_PATH=$(which uv)

if [ -z "$UV_PATH" ]; then
    echo "Error: uv not found in PATH" >&2
    exit 1
fi

# Change to the project directory
cd "$(dirname "$0")"

# Run the server using uv (FastMCP module entry point)
exec "$UV_PATH" run python -m recursive_companion_mcp
