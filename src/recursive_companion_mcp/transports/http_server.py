"""HTTP server for Recursive Companion MCP.

Implements MCP Spec 2025-06-18 Streamable HTTP transport.
Self-contained implementation with no external dependencies.
"""

import json
import logging
import secrets
from typing import Any

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def create_http_app():
    """Create Starlette HTTP app for MCP server.

    Returns:
        Starlette application instance
    """
    from ..core import mcp

    # Security validator
    validator = SecurityValidator()

    # JSON-RPC handler
    JSONRPCHandler()

    async def handle_mcp_request(request: Request) -> Response:
        """Handle MCP requests via HTTP."""
        # Handle CORS preflight
        if request.method == "OPTIONS":
            return _cors_preflight(request, validator)

        # Validate origin
        origin = request.headers.get("origin", "")
        if origin and not validator.validate_origin(origin):
            logger.warning(f"Invalid origin rejected: {origin}")
            return JSONResponse(
                {"error": "Invalid origin"},
                status_code=403,
                headers=validator.get_cors_headers(origin),
            )

        # Validate content type
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32000, "message": "Content-Type must be application/json"},
                },
                status_code=400,
                headers=validator.get_cors_headers(origin),
            )

        # Parse request body
        try:
            body = await request.json()
        except Exception as e:
            logger.error(f"JSON parse error: {e}")
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                },
                status_code=400,
                headers=validator.get_cors_headers(origin),
            )

        # Handle request
        try:
            # Check if streaming requested
            accept = request.headers.get("accept", "")
            if "text/event-stream" in accept and _should_stream(body):
                return await _handle_streaming(body, mcp, validator, origin)
            else:
                return await _handle_json(body, mcp, validator, origin)
        except Exception as e:
            logger.error(f"Request handling error: {e}", exc_info=True)
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": body.get("id") if isinstance(body, dict) else None,
                    "error": {"code": -32603, "message": "Internal error"},
                },
                status_code=500,
                headers=validator.get_cors_headers(origin),
            )

    async def health_check(request: Request) -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse(
            {
                "status": "ok",
                "transport": "streamable-http",
                "server": "recursive-companion-mcp",
            }
        )

    # Create app
    app = Starlette(
        routes=[
            Route("/mcp", handle_mcp_request, methods=["POST", "OPTIONS"]),
            Route("/health", health_check, methods=["GET"]),
        ]
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    return app


def _should_stream(body: Any) -> bool:
    """Check if request should stream."""
    if isinstance(body, dict):
        params = body.get("params", {})
        if isinstance(params, dict):
            args = params.get("arguments", {})
            return args.get("stream", False)
    return False


async def _handle_json(body: Any, mcp, validator: "SecurityValidator", origin: str) -> JSONResponse:
    """Handle JSON response."""
    # Simple pass-through to FastMCP
    # FastMCP handles tool execution internally
    try:
        # For now, return not implemented
        # Full implementation would integrate with FastMCP's internal tool routing
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": body.get("id") if isinstance(body, dict) else None,
                "error": {
                    "code": -32601,
                    "message": "HTTP transport not yet fully implemented - use stdio",
                },
            },
            headers=validator.get_cors_headers(origin),
        )
    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": body.get("id") if isinstance(body, dict) else None,
                "error": {"code": -32603, "message": str(e)},
            },
            status_code=500,
            headers=validator.get_cors_headers(origin),
        )


async def _handle_streaming(
    body: Any, mcp, validator: "SecurityValidator", origin: str
) -> StreamingResponse:
    """Handle SSE streaming response."""

    async def stream_events():
        # Placeholder for streaming implementation
        request_id = body.get("id") if isinstance(body, dict) else None
        yield f"data: {json.dumps({'jsonrpc': '2.0', 'id': request_id, 'error': {'code': -32601, 'message': 'Streaming not yet implemented'}})}\n\n"

    headers = validator.get_cors_headers(origin)
    headers.update(
        {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

    return StreamingResponse(stream_events(), media_type="text/event-stream", headers=headers)


def _cors_preflight(request: Request, validator: "SecurityValidator") -> Response:
    """Handle CORS preflight."""
    origin = request.headers.get("origin", "")
    headers = validator.get_cors_headers(origin)
    if not headers:
        return Response(status_code=403)
    return Response(status_code=200, headers=headers)


class SecurityValidator:
    """Security validation for HTTP transport."""

    def __init__(self):
        self.allowed_origins = {
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        }

    def validate_origin(self, origin: str) -> bool:
        """Validate origin header."""
        if not origin:
            return False
        return origin in self.allowed_origins

    def get_cors_headers(self, origin: str) -> dict:
        """Get CORS headers if origin valid."""
        if not self.validate_origin(origin):
            return {}
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Accept, Origin",
            "Access-Control-Max-Age": "86400",
        }

    @staticmethod
    def generate_session_id() -> str:
        """Generate secure session ID."""
        return secrets.token_urlsafe(32)


class JSONRPCHandler:
    """Handle JSON-RPC 2.0 messages."""

    @staticmethod
    def create_error_response(request_id: Any, code: int, message: str) -> dict:
        """Create error response."""
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    @staticmethod
    def create_success_response(request_id: Any, result: Any) -> dict:
        """Create success response."""
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
