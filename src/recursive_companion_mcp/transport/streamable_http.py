"""
Streamable HTTP transport for Recursive Companion MCP server.

Implements stateless HTTP JSON transport following HuggingFace's pattern.
Supports session management, browser detection, and enterprise scalability.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

logger = logging.getLogger(__name__)


class StreamableHTTPTransport:
    """
    Stateless HTTP JSON transport implementation.

    Creates a new server AND transport instance for each request to ensure complete isolation.
    Supports session management via Mcp-Session-Id headers for enterprise scenarios.
    """

    def __init__(
        self,
        mcp_server_factory,
        host: str = "127.0.0.1",
        port: int = 8080,
        enable_json_response: bool = True,
        analytics_mode: bool = False,
    ):
        """
        Initialize streamable HTTP transport.

        Args:
            mcp_server_factory: Factory function to create MCP server instances
            host: Host to bind to
            port: Port to bind to
            enable_json_response: Whether to enable direct JSON responses
            analytics_mode: Whether to enable analytics session tracking
        """
        self.mcp_server_factory = mcp_server_factory
        self.host = host
        self.port = port
        self.enable_json_response = enable_json_response
        self.analytics_mode = analytics_mode

        # Analytics session tracking
        self.analytics_sessions: dict[str, dict[str, Any]] = {}

        # Metrics tracking
        self.metrics = {"requests_handled": 0, "sessions_created": 0, "errors": 0}

    def create_app(self) -> Starlette:
        """Create Starlette application with HTTP routes."""
        routes = [
            Route("/mcp", self.handle_mcp_request, methods=["POST"]),
            Route("/mcp", self.handle_mcp_get, methods=["GET"]),
            (
                Route("/mcp", self.handle_mcp_delete, methods=["DELETE"])
                if self.analytics_mode
                else None
            ),
            Route("/health", self.handle_health, methods=["GET"]),
            Route(
                "/.well-known/oauth-protected-resource", self.handle_oauth_metadata, methods=["GET"]
            ),
        ]

        # Filter out None routes (when analytics_mode is False)
        routes = [route for route in routes if route is not None]

        return Starlette(routes=routes, on_startup=[self.on_startup])

    async def on_startup(self):
        """Called when the application starts."""
        logger.info(f"Streamable HTTP transport initialized on {self.host}:{self.port}")
        if self.analytics_mode:
            logger.info("Analytics mode enabled for session tracking")

    async def handle_health(self, request: Request) -> JSONResponse:
        """Health check endpoint for load balancers."""
        return JSONResponse(
            {
                "status": "healthy",
                "service": "recursive-companion",
                "transport": "streamable-http",
                "active_sessions": len(self.analytics_sessions) if self.analytics_mode else 0,
            }
        )

    async def handle_oauth_metadata(self, request: Request) -> JSONResponse:
        """OAuth 2.0 Protected Resource Metadata endpoint."""
        server_url = os.environ.get("MCP_SERVER_URL")
        issuer_url = os.environ.get("OAUTH_ISSUER_URL")

        if not server_url:
            return JSONResponse({"error": "Server URL not configured"}, status_code=500)

        metadata = {
            "resource": server_url,
            "authorization_servers": [issuer_url] if issuer_url else [],
            "scopes_supported": ["openid", "profile", "email"],
            "bearer_methods_supported": ["header"],
            "resource_documentation": f"{server_url}/docs",
        }

        return JSONResponse(metadata)

    async def handle_mcp_get(self, request: Request) -> Response:
        """
        Handle GET requests to /mcp endpoint.

        Serves welcome page for browsers or returns 405 for API clients.
        """
        user_agent = request.headers.get("user-agent", "").lower()

        # Check if this is a browser request (more permissive detection)
        is_browser = any(
            browser in user_agent.lower()
            for browser in ["mozilla", "chrome", "safari", "edge", "firefox", "opera"]
        ) or user_agent.startswith("Mozilla/")

        # Check for strict compliance mode
        strict_compliance = os.environ.get("MCP_STRICT_COMPLIANCE") == "true"

        if strict_compliance or not is_browser:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": "Method not allowed. Use POST for stateless JSON-RPC requests.",
                        "data": {
                            "allowed_methods": ["POST"],
                            "endpoint": "/mcp",
                            "content_type": "application/json",
                        },
                    },
                },
                status_code=405,
            )

        # Serve welcome page for browsers
        welcome_html = self._generate_welcome_page()
        return Response(welcome_html, media_type="text/html")

    async def handle_mcp_delete(self, request: Request) -> JSONResponse:
        """
        Handle DELETE requests to /mcp endpoint.

        Only available in analytics mode for session cleanup.
        """
        if not self.analytics_mode:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": "Method not allowed",
                        "data": "DELETE requests only supported in analytics mode",
                    },
                },
                status_code=405,
            )

        headers = dict(request.headers)
        session_id = headers.get("mcp-session-id")

        if not session_id:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request",
                        "data": "Session ID required for DELETE requests",
                    },
                },
                status_code=400,
            )

        if session_id in self.analytics_sessions:
            # Get session info before deletion for logging
            self.analytics_sessions[session_id]
            del self.analytics_sessions[session_id]

            logger.info(f"Deleted analytics session: {session_id}")

            return JSONResponse(
                {"jsonrpc": "2.0", "result": {"deleted": True, "session_id": session_id}}
            )
        else:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32001,
                        "message": "Session not found",
                        "data": f"Session {session_id} does not exist",
                    },
                },
                status_code=404,
            )

    async def handle_mcp_request(self, request: Request) -> JSONResponse:
        """
        Handle POST requests to /mcp endpoint.

        Creates a new server instance for each request to ensure stateless operation.
        """
        start_time = asyncio.get_event_loop().time()
        self.metrics["requests_handled"] += 1

        try:
            # Parse request body
            body = await request.json()

            # Extract headers and session info
            headers = dict(request.headers)
            session_id = headers.get("mcp-session-id")

            # Validate JSON-RPC format
            if not self._is_valid_jsonrpc(body):
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request",
                            "data": "Missing or invalid JSON-RPC 2.0 structure",
                        },
                    },
                    status_code=400,
                )

            # Handle session management in analytics mode
            response_headers = {}
            if self.analytics_mode:
                session_result, session_id = self._handle_analytics_session(
                    body, session_id, headers
                )
                if session_result:
                    return session_result
                if session_id:
                    response_headers["Mcp-Session-Id"] = session_id

            # Check if this is a notification (no 'id' field)
            is_notification = "id" not in body

            if is_notification:
                # Handle notifications asynchronously
                asyncio.create_task(self._handle_notification(body, headers))
                return JSONResponse({"jsonrpc": "2.0", "result": None})

            # Create new server instance for this request
            server = self.mcp_server_factory()

            # Create compatible transport handler for this request
            transport_handler = _StreamableRequestHandler(
                server,
                {
                    "enableJsonResponse": self.enable_json_response,
                },
            )

            try:
                # Handle the request using our compatible transport
                result = await transport_handler.handle_request(request, body)

                # Track successful request
                duration = asyncio.get_event_loop().time() - start_time
                logger.debug(f"Request completed in {duration:.3f}s")

                # Add response headers if needed
                if response_headers and hasattr(result, "headers"):
                    for key, value in response_headers.items():
                        result.headers[key] = value

                return result

            finally:
                # Clean up server and handler
                # FastMCP doesn't have a close method, so skip if not available
                if hasattr(server, "close"):
                    await server.close()

        except json.JSONDecodeError:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error",
                        "data": "Invalid JSON in request body",
                    },
                },
                status_code=400,
            )

        except Exception as e:
            self.metrics["errors"] += 1
            logger.exception(f"Error handling MCP request: {e}")

            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": "Internal error", "data": str(e)},
                },
                status_code=500,
            )

    def _handle_analytics_session(
        self, body: dict[str, Any], session_id: str | None, headers: dict[str, str]
    ) -> tuple[JSONResponse | None, str | None]:
        """
        Handle analytics session management.

        Returns:
            Tuple of (JSONResponse or None, session_id or None)
            - JSONResponse if session handling fails, None otherwise
            - session_id if session was created/validated, None otherwise
        """
        method = body.get("method")

        if method == "initialize":
            # Create new session
            session_id = str(uuid.uuid4())
            self.analytics_sessions[session_id] = {
                "id": session_id,
                "created_at": asyncio.get_event_loop().time(),
                "last_activity": asyncio.get_event_loop().time(),
                "request_count": 1,
                "client_info": body.get("params", {}).get("clientInfo", {}),
                "capabilities": body.get("params", {}).get("capabilities", {}),
            }
            self.metrics["sessions_created"] += 1

            logger.debug(f"Created analytics session: {session_id}")
            return None, session_id

        elif session_id:
            # Resume existing session
            if session_id in self.analytics_sessions:
                self.analytics_sessions[session_id][
                    "last_activity"
                ] = asyncio.get_event_loop().time()
                self.analytics_sessions[session_id]["request_count"] += 1
                logger.debug(f"Resumed analytics session: {session_id}")
                return None, session_id
            else:
                # Session not found
                return (
                    JSONResponse(
                        {
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32001,
                                "message": "Session not found",
                                "data": f"Session {session_id} does not exist or has expired",
                            },
                        },
                        status_code=404,
                    ),
                    None,
                )
        else:
            # No session ID provided for non-initialize request
            return (
                JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request",
                            "data": "Session ID required for non-initialize requests in analytics mode",
                        },
                    },
                    status_code=400,
                ),
                None,
            )

    async def _handle_notification(self, body: dict[str, Any], headers: dict[str, str]):
        """Handle JSON-RPC notifications asynchronously."""
        method = body.get("method")
        logger.debug(f"Handling notification: {method}")

        # Notifications don't require responses, just log them
        # In a full implementation, you might want to process certain notifications
        # like progress updates or status changes

    def _is_valid_jsonrpc(self, body: Any) -> bool:
        """Check if request body is valid JSON-RPC 2.0."""
        if not isinstance(body, dict):
            return False

        # Must have jsonrpc version
        if body.get("jsonrpc") != "2.0":
            return False

        # Must have method
        if "method" not in body:
            return False

        # Method must be string
        if not isinstance(body.get("method"), str):
            return False

        return True

    def _generate_welcome_page(self) -> str:
        """Generate HTML welcome page for browsers."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recursive Companion MCP Server</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
            background: #f8f9fa;
        }
        .container {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 0.5rem;
        }
        .status {
            background: #e8f5e8;
            border: 1px solid #4caf50;
            color: #2e7d32;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        .endpoint {
            background: #f5f5f5;
            border: 1px solid #ddd;
            padding: 1rem;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', monospace;
            margin: 1rem 0;
        }
        .method {
            color: #e91e63;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Recursive Companion MCP Server</h1>

        <div class="status">
            ✅ Server is running and healthy
        </div>

        <h2>Streamable HTTP Transport</h2>
        <p>This MCP server provides iterative refinement through Draft → Critique → Revise → Converge cycles.</p>

        <h3>API Endpoint</h3>
        <div class="endpoint">
            <span class="method">POST</span> /mcp
            <br><br>
            Content-Type: application/json
            <br>
            Accept: application/json, text/event-stream
        </div>

        <h3>Session Management</h3>
        <p>This server supports session tracking via the <code>Mcp-Session-Id</code> header for enterprise scenarios.</p>

        <h3>JSON-RPC 2.0 API</h3>
        <p>Send JSON-RPC 2.0 requests to the <code>/mcp</code> endpoint:</p>

        <div class="endpoint">
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "params": {
    "clientInfo": {
      "name": "test-client",
      "version": "1.0.0"
    }
  },
  "id": 1
}
        </div>

        <h3>Available Tools</h3>
        <ul>
            <li><strong>start_refinement</strong> - Begin iterative refinement process</li>
            <li><strong>continue_refinement</strong> - Continue refinement cycles</li>
            <li><strong>get_final_result</strong> - Get converged result</li>
            <li><strong>quick_refine</strong> - Quick single-pass refinement</li>
            <li><strong>abort_refinement</strong> - Abort current refinement</li>
        </ul>

        <p><em>This is a stateless HTTP transport implementation following enterprise patterns.</em></p>
    </div>
</body>
</html>
        """

    def get_metrics(self) -> dict[str, Any]:
        """Get transport metrics."""
        return {
            **self.metrics,
            "active_sessions": len(self.analytics_sessions) if self.analytics_mode else 0,
            "transport_type": "streamable-http",
            "analytics_mode": self.analytics_mode,
        }

    async def cleanup(self):
        """Clean up resources."""
        if self.analytics_mode:
            session_count = len(self.analytics_sessions)
            self.analytics_sessions.clear()
            logger.info(f"Cleaned up {session_count} analytics sessions")

        logger.info("Streamable HTTP transport cleanup complete")


class _StreamableRequestHandler:
    """
    Compatible request handler for streamable HTTP transport.

    Provides similar functionality to MCP SDK's StreamableHTTPServerTransport
    but implemented to work with FastMCP servers.
    """

    def __init__(self, server, config: dict[str, Any]):
        """
        Initialize request handler.

        Args:
            server: FastMCP server instance
            config: Configuration options
        """
        self.server = server
        self.config = config
        self.enable_json_response = config.get("enableJsonResponse", True)

    async def handle_request(self, request: Request, body: dict[str, Any]) -> JSONResponse:
        """
        Handle JSON-RPC request using FastMCP server.

        Args:
            request: Starlette request object
            body: Parsed JSON-RPC request body

        Returns:
            JSONResponse with JSON-RPC result or error
        """
        try:
            # Extract method and params
            method = body.get("method")
            params = body.get("params", {})
            request_id = body.get("id")

            if not method:
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request",
                            "data": "Missing 'method' field",
                        },
                    },
                    status_code=400,
                )

            # Route request to appropriate handler
            if method == "initialize":
                result = await self._handle_initialize(params, request_id)
            elif method == "tools/list":
                result = await self._handle_tools_list(params, request_id)
            elif method == "tools/call":
                result = await self._handle_tools_call(params, request_id)
            elif method == "prompts/list":
                result = await self._handle_prompts_list(params, request_id)
            elif method == "prompts/get":
                result = await self._handle_prompts_get(params, request_id)
            elif method == "resources/list":
                result = await self._handle_resources_list(params, request_id)
            elif method == "resources/read":
                result = await self._handle_resources_read(params, request_id)
            else:
                result = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": "Method not found",
                        "data": f"Method '{method}' not supported",
                    },
                }

            return JSONResponse(result)

        except Exception as e:
            logger.exception(f"Error in request handler: {e}")
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": "Internal error", "data": str(e)},
                },
                status_code=500,
            )

    async def _handle_initialize(self, params: dict[str, Any], request_id: Any) -> dict[str, Any]:
        """Handle initialize method."""
        try:
            # Get server capabilities
            capabilities = getattr(self.server, "capabilities", {})

            # Return initialization result
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": capabilities,
                    "serverInfo": {"name": "recursive-companion", "version": "1.0.0"},
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32002, "message": "Initialization failed", "data": str(e)},
            }

    async def _handle_tools_list(self, params: dict[str, Any], request_id: Any) -> dict[str, Any]:
        """Handle tools/list method."""
        try:
            # Get tools from server
            tools = []
            if hasattr(self.server, "list_tools"):
                tools_result = await self.server.list_tools()
                tools = tools_result.get("tools", [])

            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools}}
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32002, "message": "Failed to list tools", "data": str(e)},
            }

    async def _handle_tools_call(self, params: dict[str, Any], request_id: Any) -> dict[str, Any]:
        """Handle tools/call method."""
        try:
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if not tool_name:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params",
                        "data": "Missing 'name' parameter",
                    },
                }

            # Call the tool
            if hasattr(self.server, "call_tool"):
                result = await self.server.call_tool(tool_name, arguments)
                return {"jsonrpc": "2.0", "id": request_id, "result": result}
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": "Tool calling not supported",
                        "data": "Server does not implement call_tool method",
                    },
                }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32002, "message": "Tool execution failed", "data": str(e)},
            }

    async def _handle_prompts_list(self, params: dict[str, Any], request_id: Any) -> dict[str, Any]:
        """Handle prompts/list method."""
        return {"jsonrpc": "2.0", "id": request_id, "result": {"prompts": []}}

    async def _handle_prompts_get(self, params: dict[str, Any], request_id: Any) -> dict[str, Any]:
        """Handle prompts/get method."""
        prompt_name = params.get("name")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32601,
                "message": "Prompt not found",
                "data": f"Prompt '{prompt_name}' not found",
            },
        }

    async def _handle_resources_list(
        self, params: dict[str, Any], request_id: Any
    ) -> dict[str, Any]:
        """Handle resources/list method."""
        return {"jsonrpc": "2.0", "id": request_id, "result": {"resources": []}}

    async def _handle_resources_read(
        self, params: dict[str, Any], request_id: Any
    ) -> dict[str, Any]:
        """Handle resources/read method."""
        uri = params.get("uri")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32601,
                "message": "Resource not found",
                "data": f"Resource '{uri}' not found",
            },
        }
