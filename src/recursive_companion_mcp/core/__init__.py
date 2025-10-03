"""Core server module with FastMCP instance and decorators"""

from .server import format_output, handle_tool_errors, mcp

__all__ = ["mcp", "handle_tool_errors", "format_output"]
