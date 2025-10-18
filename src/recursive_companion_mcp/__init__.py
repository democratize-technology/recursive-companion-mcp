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

__all__ = ["create_server", "main"]


def create_server():
    """Create and return the MCP server instance.

    Returns:
        The configured MCP server instance with all tools registered.
    """
    # Import tools LAZILY (only when creating server) to avoid circular import deadlock
    # Tools register with the mcp instance via decorators when imported
    from .tools import (  # noqa: F401
        abort_refinement,
        continue_refinement,
        current_session,
        get_final_result,
        get_refinement_status,
        list_refinement_sessions,
        quick_refine,
        start_refinement,
    )

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

    # Import tools and create server
    try:
        server = create_server()  # Uses lazy tool import
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
        # Import tools lazily before creating server
        from .tools import (  # noqa: F401
            abort_refinement,
            continue_refinement,
            current_session,
            get_final_result,
            get_refinement_status,
            list_refinement_sessions,
            quick_refine,
            start_refinement,
        )

        server = get_mcp_server(host=host, port=port)
        # Use run_streamable_http_async for HTTP transport
        asyncio.run(server.run_streamable_http_async())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception:
        logger.exception("Server error")
        raise


def streamable_http_main(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the MCP server with Streamable HTTP transport for enterprise scalability.

    Args:
        host: Host to bind to (default: 127.0.0.1 for localhost only)
        port: Port to bind to (default: 8080)
    """
    import uvicorn

    logger.info(f"Starting Recursive Companion MCP server (Streamable HTTP) on {host}:{port}")

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
        # Create server factory for stateless operation
        from .core.server import create_server_factory
        from .transport import StreamableHTTPTransport

        server_factory = create_server_factory()

        # Create streamable HTTP transport
        transport = StreamableHTTPTransport(
            mcp_server_factory=server_factory,
            host=host,
            port=port,
            enable_json_response=True,
            analytics_mode=os.environ.get("ANALYTICS_MODE", "false").lower() == "true",
        )

        # Create and run Starlette app
        app = transport.create_app()

        # Add session ID to response headers if analytics mode is enabled
        if transport.analytics_mode:

            @app.middleware("http")
            async def add_session_header(request, call_next):
                response = await call_next(request)
                # Session ID is added to request headers during session creation
                session_id = getattr(request.state, "session_id", None)
                if session_id:
                    response.headers["Mcp-Session-Id"] = session_id
                return response

        # Run with uvicorn
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="warning",  # Reduce uvicorn logging, let our logger handle it
            access_log=False,
        )

        server = uvicorn.Server(config)
        logger.info(f"Streamable HTTP transport ready on http://{host}:{port}/mcp")
        server.run()

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
    elif transport == "streamable_http":
        host = os.environ.get("MCP_HTTP_HOST", "127.0.0.1")
        port = int(os.environ.get("MCP_HTTP_PORT", "8080"))
        streamable_http_main(host=host, port=port)
    else:
        main()
