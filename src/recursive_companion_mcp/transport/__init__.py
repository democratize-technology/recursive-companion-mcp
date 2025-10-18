"""
Transport layer implementations for Recursive Companion MCP server.

Provides different transport mechanisms including:
- Streamable HTTP transport for enterprise scalability
- Session management and browser detection
- JSON-RPC 2.0 error handling
"""

from .streamable_http import StreamableHTTPTransport

__all__ = ["StreamableHTTPTransport"]
