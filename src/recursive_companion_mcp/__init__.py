#!/usr/bin/env python3
"""
Recursive Companion MCP Server
Iterative refinement through Draft → Critique → Revise → Converge cycles

CRITICAL: This server uses stdio transport for MCP protocol communication.
- stdout is reserved for MCP JSON-RPC messages
- All logging/debug output must go to stderr or files
- Never logger.info() to stdout in MCP server code
"""

import asyncio
import logging
import os
import sys

# Configure logging to stderr only - NEVER stdout in MCP servers
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from .core.server import get_mcp_server  # noqa: E402

# Import tools at module level so they register with mcp instance
from .tools import (  # noqa: F401, E402
    abort_refinement,
    continue_refinement,
    current_session,
    get_final_result,
    get_refinement_status,
    list_refinement_sessions,
    quick_refine,
    start_refinement,
)

__all__ = ["main", "create_server"]


def create_server():
    """Create and return the MCP server instance.

    Returns:
        The configured MCP server instance with all tools registered.
    """
    # Tools are already imported at module level and registered with mcp instance
    return get_mcp_server()


def main() -> None:
    """Run the MCP server with stdio transport (default)"""
    logger.info("Starting Recursive Companion MCP server (stdio)")

    # Test AWS Bedrock connection
    try:
        import boto3

        from .config import config

        bedrock_test = boto3.client(service_name="bedrock", region_name=config.aws_region)
        bedrock_test.list_foundation_models()
        logger.info("Successfully connected to AWS Bedrock")
        logger.info(f"Using Claude model: {config.bedrock_model_id}")
        logger.info(f"Using embedding model: {config.embedding_model_id}")
    except Exception as e:
        logger.warning(f"AWS Bedrock connection test failed (continuing): {e}")

    # Tools are already imported at module level
    try:
        server = get_mcp_server()
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


def http_main(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the MCP server with HTTP transport using FastMCP's native support.

    Args:
        host: Host to bind to (default: 127.0.0.1 for localhost only)
        port: Port to bind to (default: 8080)
    """

    logger.info(f"Starting Recursive Companion MCP server (HTTP) on {host}:{port}")

    # Test AWS Bedrock connection
    try:
        import boto3

        from .config import config

        bedrock_test = boto3.client(service_name="bedrock", region_name=config.aws_region)
        bedrock_test.list_foundation_models()
        logger.info("Successfully connected to AWS Bedrock")
    except Exception as e:
        logger.warning(f"AWS Bedrock connection test failed (continuing): {e}")

    try:
        server = get_mcp_server(host=host, port=port)
        # Use run_streamable_http_async for HTTP transport
        asyncio.run(server.run_streamable_http_async())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception:
        logger.exception("Server error")
        raise


if __name__ == "__main__":
    # Check if HTTP mode requested
    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    if transport == "http":
        host = os.environ.get("MCP_HTTP_HOST", "127.0.0.1")
        port = int(os.environ.get("MCP_HTTP_PORT", "8080"))
        http_main(host=host, port=port)
    else:
        main()
