"""
Generic authentication provider interface for MCP servers.

Supports OAuth 2.1 from any compliant authorization server
(Cognito, Auth0, Okta, custom, etc.) via environment configuration.

Architecture:
- AuthProvider Protocol: Interface for all auth implementations
- NoAuthProvider: Default mode for local development (no authentication)
- OAuth21Provider: Generic OAuth 2.1 with JWT validation
- Factory: get_auth_provider() selects based on AUTH_PROVIDER env var
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Protocol

from .models import UserContext
from .oauth21 import OAuth21Provider

__all__ = [
    "AuthProvider",
    "NoAuthProvider",
    "OAuth21Provider",
    "UserContext",
    "get_auth_provider",
]

logger = logging.getLogger(__name__)


class AuthProvider(Protocol):
    """Authentication provider interface.

    Implementations validate tokens from OAuth 2.1 servers or
    disable authentication entirely for local development.

    Methods:
        get_user_context: Extract user context from HTTP request
        is_enabled: Whether authentication is enabled
        get_www_authenticate_header: WWW-Authenticate header for 401 responses
    """

    def get_user_context(self, request: Any) -> Optional[UserContext]:
        """Extract user context from request.

        Args:
            request: HTTP request object (Starlette/FastAPI Request)

        Returns:
            UserContext if request is authenticated and valid, None otherwise

        Note:
            Returns None if:
            - Authentication is disabled (NoAuthProvider)
            - No credentials in request
            - Token validation fails
        """
        ...

    def is_enabled(self) -> bool:
        """Whether authentication is enabled.

        Returns:
            True if provider enforces authentication, False otherwise
        """
        ...

    def get_www_authenticate_header(self) -> Optional[str]:
        """Get WWW-Authenticate header value for 401 responses.

        Per RFC9728, OAuth 2.1 protected resources must return
        WWW-Authenticate headers to guide clients to authorization servers.

        Returns:
            RFC9728-compliant WWW-Authenticate header value if OAuth enabled,
            None if authentication is disabled
        """
        return None


class NoAuthProvider:
    """No authentication provider for local development.

    This is the default when AUTH_PROVIDER is not set or set to "none".
    All requests are allowed without authentication.

    Use for:
    - Local development
    - Internal testing
    - Environments where auth is handled upstream (e.g., ALB)
    """

    def get_user_context(self, request: Any) -> Optional[UserContext]:
        """Allow all requests without authentication.

        Args:
            request: Ignored in no-auth mode

        Returns:
            None (no user context in no-auth mode)
        """
        return None

    def is_enabled(self) -> bool:
        """Authentication is disabled in no-auth mode.

        Returns:
            False
        """
        return False

    def get_www_authenticate_header(self) -> Optional[str]:
        """No WWW-Authenticate header in no-auth mode.

        Returns:
            None
        """
        return None


def get_auth_provider() -> AuthProvider:
    """Factory function to get auth provider based on environment.

    Selects authentication provider based on AUTH_PROVIDER environment variable:
    - "none" or unset: NoAuthProvider (default - no authentication)
    - "oauth21": OAuth21Provider (generic OAuth 2.1)

    Future providers:
    - "alb-cognito": AWS ALB with Cognito (header-based)
    - "auth0": Auth0-specific optimizations

    Returns:
        AuthProvider instance configured based on environment

    Examples:
        >>> # Local development (default)
        >>> provider = get_auth_provider()  # NoAuthProvider

        >>> # OAuth 2.1 mode
        >>> os.environ["AUTH_PROVIDER"] = "oauth21"
        >>> provider = get_auth_provider()  # OAuth21Provider
    """
    provider_type = os.environ.get("AUTH_PROVIDER", "none").lower()

    if provider_type == "none":
        logger.info("Auth: disabled (NoAuthProvider)")
        return NoAuthProvider()

    elif provider_type == "oauth21":
        logger.info("Auth: OAuth 2.1 enabled")
        return OAuth21Provider()

    else:
        logger.warning(f"Unknown AUTH_PROVIDER '{provider_type}', using NoAuthProvider. Valid options: none, oauth21")
        return NoAuthProvider()
