"""
CLI entry point for Recursive Companion MCP server
"""

import os

if __name__ == "__main__":
    from . import http_main, main, streamable_http_main

    # Check if HTTP mode requested
    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    if transport == "http":
        host = os.environ.get("MCP_HTTP_HOST", "127.0.0.1")
        port = int(os.environ.get("MCP_HTTP_PORT", "8087"))
        http_main(host=host, port=port)
    elif transport == "streamable_http":
        host = os.environ.get("MCP_HTTP_HOST", "127.0.0.1")
        port = int(os.environ.get("MCP_HTTP_PORT", "8087"))
        streamable_http_main(host=host, port=port)
    else:
        main()
