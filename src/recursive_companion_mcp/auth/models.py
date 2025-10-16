"""
Data models for authentication module.

Separated from __init__.py to avoid circular imports between
the main auth module and provider implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class UserContext:
    """User context extracted from authenticated request.

    Contains identity, authorization, and metadata extracted from
    validated OAuth 2.1 tokens.

    Attributes:
        user_id: Unique user identifier (from 'sub' claim)
        email: User email address (from 'email' claim)
        tier: Service tier for quota/rate limiting ('free', 'pro', 'enterprise')
        tenant_id: Multi-tenant organization identifier
        scopes: OAuth scopes granted to this token
        token_expires_at: Token expiration timestamp (from 'exp' claim)
        metadata: Additional claims and provider-specific data
    """

    user_id: str
    email: Optional[str] = None
    tier: str = "free"
    tenant_id: Optional[str] = None
    scopes: list[str] = field(default_factory=list)
    token_expires_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)
