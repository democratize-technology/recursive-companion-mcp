#!/bin/bash
export MCP_TRANSPORT=http
export MCP_HTTP_PORT=8086
uv run python -m recursive_companion_mcp
