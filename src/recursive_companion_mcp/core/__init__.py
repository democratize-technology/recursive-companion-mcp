"""Core server module with FastMCP instance and decorators"""

# Import decorators and helpers - DO NOT import mcp here to avoid circular import
from .server import format_output, handle_tool_errors

__all__ = ["handle_tool_errors", "format_output"]
