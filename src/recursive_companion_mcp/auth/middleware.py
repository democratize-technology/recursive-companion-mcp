"""
Transport-aware authentication middleware for FastMCP.

Implements OAuth 2.1 authentication for HTTP transport while
preserving stdio transport backward compatibility.

Key Features:
- Transport detection (HTTP vs stdio)
- 401 responses with WWW-Authenticate headers
- Request state injection for downstream handlers
- MCP protocol compliance (no stdout pollution)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from starlette.datastructures import State
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

if TYPE_CHECKING:
    from starlette.requests import Request

    from . import AuthProvider

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Transport-aware authentication middleware.

    Only applies authentication to HTTP transport requests.
    Stdio transport bypasses auth (preserves backward compatibility).

    Architecture:
    - Detects transport type from request context
    - Validates credentials using configured AuthProvider
    - Injects user_context into request.state for tools
    - Returns 401 with WWW-Authenticate on auth failures
    - Passes through if auth disabled (NoAuthProvider)

    Attributes:
        auth_provider: Authentication provider instance (OAuth21Provider or NoAuthProvider)
    """

    def __init__(self, app, auth_provider: AuthProvider) -> None:  # type: ignore[no-untyped-def]
        """Initialize middleware with auth provider.

        Args:
            app: Starlette/FastAPI application
            auth_provider: Authentication provider instance
        """
        super().__init__(app)
        self.auth_provider = auth_provider

        logger.info(
            f"AuthMiddleware initialized: "
            f"auth_enabled={auth_provider.is_enabled()}, "
            f"provider={auth_provider.__class__.__name__}"
        )

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        """Process request with transport-aware authentication.

        Flow:
        1. Detect if this is HTTP transport (skip stdio)
        2. Extract user context using auth provider
        3. If auth enabled but no valid context → 401
        4. Inject user_context into request.state
        5. Continue request processing

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            Response from downstream handler or 401 if auth fails
        """
        # Skip auth for non-HTTP requests (stdio transport)
        # MCP stdio uses JSON-RPC over stdin/stdout, not HTTP
        if not hasattr(request, "headers"):
            logger.debug("Non-HTTP request detected (stdio transport), skipping auth")
            return await call_next(request)

        # Extract user context (returns None if disabled or invalid)
        user_context = self.auth_provider.get_user_context(request)

        # If auth is enabled but no valid user context → 401
        if self.auth_provider.is_enabled() and not user_context:
            www_auth_header = self.auth_provider.get_www_authenticate_header()

            logger.warning(
                f"Unauthorized request: path={request.url.path}, "
                f"client={request.client.host if request.client else 'unknown'}"
            )

            if www_auth_header:
                # OAuth 2.1 mode: return WWW-Authenticate header per RFC9728
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": www_auth_header},
                    content=b"Unauthorized",
                    media_type="text/plain",
                )
            # Generic 401 (shouldn't happen with proper provider)
            return Response(
                status_code=401,
                content=b"Unauthorized",
                media_type="text/plain",
            )

        # Inject user_context into request state for downstream tools
        # Tools can access via request.state.user_context if available
        if not hasattr(request, "state"):
            # Create state object if missing (defensive)
            request.state = State()

        request.state.user_context = user_context

        # Log authentication status for monitoring
        if user_context:
            logger.info(
                f"Authenticated request: user={user_context.user_id}, tier={user_context.tier}, path={request.url.path}"
            )
        else:
            logger.debug(f"Unauthenticated request (auth disabled): path={request.url.path}")

        # Continue request processing
        return await call_next(request)
