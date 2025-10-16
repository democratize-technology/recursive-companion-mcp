"""
Generic OAuth 2.1 provider with JWT validation.

Works with ANY OAuth 2.1 compliant authorization server including:
- AWS Cognito
- Auth0
- Okta
- Azure AD
- Custom OAuth servers

Configuration via environment variables - provider-agnostic design.

Architecture Improvements:
- TTL-based JWKS caching for key rotation detection
- Rate limiting integration to prevent DoS attacks
- RFC9728 compliant WWW-Authenticate headers
- RFC8707 resource claim validation for token binding
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import jwt
from cachetools import TTLCache  # type: ignore[import-untyped]
from jwt import PyJWKClient

from .models import UserContext

logger = logging.getLogger(__name__)


class OAuth21Provider:
    """Generic OAuth 2.1 authentication provider.

    Validates JWT bearer tokens from any OAuth 2.1 compliant server.
    Implements comprehensive security controls:
    - JWKS signature validation with TTL caching
    - Standard claim validation (exp, iss, aud)
    - Resource claim validation (RFC8707)
    - Rate limiting to prevent DoS attacks
    - Key rotation detection via cache TTL

    Required environment variables:
        MCP_SERVER_URL: Canonical URI of this MCP server
        OAUTH_AUDIENCE: Expected audience claim in tokens

    And ONE of:
        OAUTH_ISSUER_URL: Direct issuer URL (for Auth0, Okta, etc.)
        OR
        USER_POOL_ID + AWS_REGION: Auto-construct Cognito issuer URL

    Optional:
        OAUTH_JWKS_CACHE_TTL: JWKS cache duration in seconds (default: 3600)
        OAUTH_JWKS_MAX_KEYS: Maximum cached signing keys (default: 16)

    Examples:
        # Auth0
        OAUTH_ISSUER_URL=https://yourcompany.auth0.com
        OAUTH_AUDIENCE=https://api.yourcompany.com

        # Okta
        OAUTH_ISSUER_URL=https://yourcompany.okta.com/oauth2/default
        OAUTH_AUDIENCE=api://your-app

        # Cognito (issuer URL auto-constructed)
        USER_POOL_ID=us-east-1_XXXXX
        AWS_REGION=us-east-1
        OAUTH_AUDIENCE=your-cognito-client-id
    """

    def __init__(self) -> None:
        """Initialize OAuth 2.1 provider with environment-based configuration.

        Raises:
            ValueError: If required environment variables are missing
        """
        self.server_url = os.environ.get("MCP_SERVER_URL")
        self.audience = os.environ.get("OAUTH_AUDIENCE")

        # Support both direct issuer URL and Cognito-style config
        self.issuer_url = os.environ.get("OAUTH_ISSUER_URL")

        if not self.issuer_url:
            # Auto-construct for Cognito if USER_POOL_ID provided
            user_pool_id = os.environ.get("USER_POOL_ID")
            region = os.environ.get("AWS_REGION", "us-east-1")

            if user_pool_id:
                self.issuer_url = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"
                logger.info("Cognito mode: constructed issuer URL from USER_POOL_ID")

        # Validate required config
        if not self.server_url:
            raise ValueError("MCP_SERVER_URL must be set for OAuth21Provider")
        if not self.issuer_url:
            raise ValueError(
                "Either OAUTH_ISSUER_URL or USER_POOL_ID must be set for OAuth21Provider"
            )
        if not self.audience:
            raise ValueError("OAUTH_AUDIENCE must be set for OAuth21Provider")

        # JWKS configuration with TTL for key rotation detection
        self.jwks_cache_ttl = int(os.environ.get("OAUTH_JWKS_CACHE_TTL", "3600"))
        self.jwks_max_keys = int(os.environ.get("OAUTH_JWKS_MAX_KEYS", "16"))
        self.jwks_url = f"{self.issuer_url}/.well-known/jwks.json"

        # TTL cache for JWKS clients - replaces lru_cache for key rotation detection
        # Cache expires after TTL, forcing re-fetch to detect rotated keys
        self._jwks_cache: TTLCache = TTLCache(maxsize=1, ttl=self.jwks_cache_ttl)

        # Rate limiting cache for token validation attempts (prevents DoS)
        # Track validation attempts per client IP to prevent JWKS endpoint abuse
        self._rate_limit_cache: TTLCache = TTLCache(maxsize=10000, ttl=60)

        logger.info(
            f"OAuth21Provider initialized:\n"
            f"  Server: {self.server_url}\n"
            f"  Issuer: {self.issuer_url}\n"
            f"  Audience: {self.audience}\n"
            f"  JWKS Cache TTL: {self.jwks_cache_ttl}s\n"
            f"  JWKS Max Keys: {self.jwks_max_keys}"
        )

    def get_user_context(self, request: Any) -> UserContext | None:
        """Validate bearer token and extract user context.

        Performs comprehensive validation:
        1. Rate limiting check (prevent DoS)
        2. Bearer token extraction
        3. JWT signature validation with JWKS
        4. Standard claims (exp, iss, aud)
        5. Resource claim (RFC8707) if present

        Args:
            request: HTTP request object with headers (Starlette/FastAPI Request)

        Returns:
            UserContext if token is valid, None otherwise

        Note:
            Returns None for any validation failure to prevent information leakage.
            All failures are logged to stderr for security monitoring.
        """
        try:
            # Rate limiting: Prevent DoS via expensive JWKS fetches
            client_ip = self._get_client_ip(request)
            if not self._check_rate_limit(client_ip):
                logger.warning(f"Rate limit exceeded for client IP: {client_ip}")
                return None

            # Extract bearer token
            auth_header = self._get_header(request, "authorization")
            if not auth_header:
                logger.debug("No Authorization header")
                return None

            if not auth_header.startswith("Bearer "):
                logger.warning("Authorization header doesn't start with 'Bearer '")
                return None

            token = auth_header[7:]  # Remove "Bearer " prefix

            # Validate token
            claims = self._validate_token(token)
            if not claims:
                return None

            # Extract user context from claims
            return UserContext(
                user_id=claims.get("sub", ""),
                email=claims.get("email"),
                tier=claims.get("tier", "free"),
                tenant_id=claims.get("tenant_id"),
                scopes=claims.get("scope", "").split() if claims.get("scope") else [],
                token_expires_at=(
                    datetime.fromtimestamp(claims["exp"], tz=timezone.utc)
                    if "exp" in claims
                    else None
                ),
                metadata={
                    "iss": claims.get("iss"),
                    "aud": claims.get("aud"),
                    "iat": claims.get("iat"),
                },
            )

        except Exception as e:
            # Never expose internal errors to clients
            logger.error(f"Error extracting user context: {e}", exc_info=True)
            return None

    def is_enabled(self) -> bool:
        """Authentication is enabled for OAuth 2.1 mode.

        Returns:
            True
        """
        return True

    def get_www_authenticate_header(self) -> str:
        """Generate WWW-Authenticate header per RFC9728 Section 5.1.

        This tells MCP clients where to find authorization server metadata
        for automatic OAuth flow initiation.

        Returns:
            RFC9728-compliant WWW-Authenticate header value

        Example:
            Bearer realm="https://mcp.example.com",
                   as_uri="https://auth.example.com/.well-known/openid-configuration"
        """
        return f'Bearer realm="{self.server_url}", as_uri="{self.issuer_url}/.well-known/openid-configuration"'

    def _validate_token(self, token: str) -> dict[str, Any] | None:
        """Validate JWT token with comprehensive security checks.

        Validates:
        - Signature using JWKS from authorization server
        - Expiration (exp claim)
        - Issuer matches expected issuer (iss claim)
        - Audience includes this server (aud claim)
        - Resource claim if present (RFC8707)

        Args:
            token: JWT token string

        Returns:
            Decoded claims dict if valid, None otherwise

        Note:
            Returns None for any validation failure. Failures are logged
            to stderr for security monitoring.
        """
        try:
            # Get JWKS client (with TTL-based caching)
            jwks_client = self._get_jwks_client()

            # Get signing key from token
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            # Decode and validate all claims
            # Type cast: jwt.decode() returns Any in PyJWT's type stubs
            claims: dict[str, Any] = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience=self.audience,
                issuer=self.issuer_url,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_aud": True,
                    "verify_iss": True,
                },
            )

            # Additional validation: check 'resource' claim if present
            # Per RFC8707, tokens should be bound to specific resources
            if "resource" in claims:
                if self.server_url not in claims["resource"]:
                    logger.warning(
                        f"Token resource claim doesn't include this server: {claims['resource']}"
                    )
                    return None

            logger.info(f"Token validated successfully for user: {claims.get('sub')}")
            return claims

        except jwt.ExpiredSignatureError:
            logger.warning("Token validation failed: token expired")
            return None
        except jwt.InvalidAudienceError:
            logger.warning(f"Token validation failed: invalid audience (expected {self.audience})")
            return None
        except jwt.InvalidIssuerError:
            logger.warning(f"Token validation failed: invalid issuer (expected {self.issuer_url})")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token validation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error validating token: {e}", exc_info=True)
            return None

    def _get_jwks_client(self) -> PyJWKClient:
        """Get PyJWKClient for JWKS fetching with TTL-based caching.

        Uses TTL cache instead of lru_cache to enable key rotation detection.
        When cache expires, fresh JWKS is fetched to detect rotated signing keys.

        Returns:
            PyJWKClient instance (cached with TTL)

        Raises:
            Exception: If JWKS client initialization fails
        """
        # Check cache first
        if "jwks_client" in self._jwks_cache:
            return self._jwks_cache["jwks_client"]

        # Create new client and cache it
        try:
            logger.debug(f"Initializing JWKS client for {self.jwks_url}")
            client = PyJWKClient(
                self.jwks_url,
                cache_keys=True,
                max_cached_keys=self.jwks_max_keys,
            )
            self._jwks_cache["jwks_client"] = client
            return client
        except Exception as e:
            logger.error(f"Failed to initialize JWKS client: {e}", exc_info=True)
            raise

    def _check_rate_limit(self, client_id: str, limit: int = 60) -> bool:
        """Check if client has exceeded rate limit for token validation.

        Prevents DoS attacks via expensive JWKS fetches by limiting
        validation attempts per client per minute.

        Args:
            client_id: Client identifier (typically IP address)
            limit: Maximum attempts per minute (default: 60)

        Returns:
            True if within rate limit, False if exceeded
        """
        current_count = self._rate_limit_cache.get(client_id, 0)

        if current_count >= limit:
            return False

        # Increment counter
        self._rate_limit_cache[client_id] = current_count + 1
        return True

    def _get_client_ip(self, request: Any) -> str:
        """Extract client IP address from request.

        Checks X-Forwarded-For header first (for proxy scenarios),
        falls back to direct client address.

        Args:
            request: HTTP request object

        Returns:
            Client IP address string
        """
        # Check X-Forwarded-For for proxy scenarios
        forwarded = self._get_header(request, "x-forwarded-for")
        if forwarded:
            # Take first IP in chain
            return forwarded.split(",")[0].strip()

        # Fall back to direct client
        if hasattr(request, "client") and request.client:
            # Type cast: request.client.host returns Any in Starlette's type stubs
            return str(request.client.host)

        return "unknown"

    def _get_header(self, request: Any, header_name: str) -> str | None:
        """Get header from request (works with FastAPI/Starlette).

        Args:
            request: HTTP request object
            header_name: Name of header to retrieve (case-insensitive)

        Returns:
            Header value if present, None otherwise
        """
        if hasattr(request, "headers"):
            # Type cast: headers.get() returns Any in Starlette's type stubs
            value = request.headers.get(header_name)
            return str(value) if value is not None else None
        if hasattr(request, "get"):
            # Type cast: dict-like get() returns Any
            value = request.get(header_name)
            return str(value) if value is not None else None
        return None


def get_protected_resource_metadata(server_url: str, issuer_url: str) -> dict[str, Any]:
    """Generate Protected Resource Metadata per RFC9728.

    This should be served at /.well-known/oauth-protected-resource
    to allow MCP clients to discover the authorization server.

    Args:
        server_url: Canonical URI of this MCP server
        issuer_url: Authorization server issuer URL

    Returns:
        RFC9728-compliant metadata dictionary

    Example:
        {
            "resource": "https://mcp.example.com",
            "authorization_servers": ["https://auth.example.com"],
            "bearer_methods_supported": ["header"],
            "resource_signing_alg_values_supported": ["RS256", "ES256"]
        }
    """
    return {
        "resource": server_url,
        "authorization_servers": [issuer_url],
        "bearer_methods_supported": ["header"],
        "resource_signing_alg_values_supported": ["RS256", "ES256"],
    }
