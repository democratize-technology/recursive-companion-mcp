# Streamable HTTP Transport

This document describes the Streamable HTTP transport implementation for Recursive Companion MCP server.

## Overview

The Streamable HTTP transport provides enterprise-grade scalability and session management capabilities beyond the standard HTTP transport. It follows the stateless HTTP JSON pattern used by HuggingFace and other large-scale MCP deployments.

## Key Features

### 1. Stateless Operation
- Each HTTP request creates a new server instance
- Complete isolation between requests
- No in-memory state between requests
- Suitable for load-balanced deployments

### 2. Session Management
- Session tracking via `Mcp-Session-Id` headers
- Analytics mode for session metrics
- Session lifecycle management (create, resume, delete)
- DELETE `/mcp` endpoint for session cleanup

### 3. Enterprise Features
- JSON-RPC 2.0 compliant error handling
- Browser detection and friendly GET responses
- Health check endpoints for load balancers
- OAuth 2.0 Protected Resource Metadata endpoint
- Metrics and analytics tracking

### 4. Transport Compatibility
- Supports both `Accept: application/json` and `text/event-stream`
- Graceful degradation for different client types
- Strict compliance mode for API-only access

## Usage

### Basic Usage

```bash
# Start with streamable HTTP transport
MCP_TRANSPORT=streamable_http MCP_HTTP_HOST=0.0.0.0 MCP_HTTP_PORT=8080 \
uv run python -m recursive_companion_mcp
```

### Analytics Mode

```bash
# Enable analytics session tracking
ANALYTICS_MODE=true MCP_TRANSPORT=streamable_http \
uv run python -m recursive_companion_mcp
```

### Session Management

```bash
# Initialize session (creates new session ID)
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
      "clientInfo": {"name": "test-client", "version": "1.0.0"}
    },
    "id": 1
  }'

# Subsequent requests include session ID
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: <session-id-from-response>" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 2
  }'

# Delete session
curl -X DELETE http://localhost:8080/mcp \
  -H "Mcp-Session-Id: <session-id>"
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TRANSPORT` | `stdio` | Set to `streamable_http` to enable |
| `MCP_HTTP_HOST` | `127.0.0.1` | Host to bind to |
| `MCP_HTTP_PORT` | `8087` | Port to bind to |
| `ANALYTICS_MODE` | `false` | Enable session analytics |
| `MCP_STRICT_COMPLIANCE` | `false` | Reject GET requests in strict mode |
| `MCP_SERVER_URL` | - | Server URL for OAuth metadata |
| `OAUTH_ISSUER_URL` | - | OAuth issuer URL |

### OAuth 2.0 Integration

When OAuth is enabled (`AUTH_PROVIDER=oauth21`), the transport provides:

- Protected Resource Metadata at `/.well-known/oauth-protected-resource`
- `WWW-Authenticate` headers on 401 responses
- Resource documentation links

## Endpoints

### `/mcp` (POST)
Main JSON-RPC 2.0 endpoint for MCP operations.

### `/mcp` (GET)
Serves welcome page for browsers, returns 405 for API clients (unless in strict compliance mode).

### `/health` (GET)
Health check endpoint for load balancers and monitoring:
```json
{
  "status": "healthy",
  "service": "recursive-companion",
  "transport": "streamable_http",
  "active_sessions": 5
}
```

### `/.well-known/oauth-protected-resource` (GET)
OAuth 2.0 Protected Resource Metadata (RFC 9728).

### `/mcp` (DELETE)
Delete analytics session (analytics mode only).

## Error Handling

The transport implements comprehensive JSON-RPC 2.0 error handling:

| Error Code | Meaning | When Used |
|------------|---------|-----------|
| -32700 | Parse error | Invalid JSON in request |
| -32600 | Invalid Request | Missing/invalid JSON-RPC structure |
| -32601 | Method not found | Unsupported JSON-RPC method |
| -32602 | Invalid params | Missing required parameters |
| -32603 | Internal error | Server-side errors |
| -32001 | Session not found | Invalid session ID (analytics mode) |
| -32002 | Server error | MCP server initialization/operation errors |

## Browser Support

The transport provides a user-friendly welcome page for browser requests to `/mcp`. This includes:

- Server status and capabilities
- API documentation
- Example requests
- Available tools list

In strict compliance mode (`MCP_STRICT_COMPLIANCE=true`), GET requests return 405 Method Not Allowed.

## Metrics and Analytics

In analytics mode, the transport tracks:

- Request count and duration
- Session creation and lifecycle
- Error rates and types
- Active session count

Access metrics via the transport's `get_metrics()` method or health endpoint.

## Security Considerations

1. **Stateless Operation**: No sensitive data is stored between requests
2. **Session IDs**: Random UUIDs, not predictable sequences
3. **HTTPS**: Production deployments should use HTTPS
4. **CORS**: Configure appropriate CORS headers for cross-origin requests
5. **Rate Limiting**: Consider rate limiting for production deployments

## Comparison with Standard HTTP Transport

| Feature | Standard HTTP | Streamable HTTP |
|---------|---------------|-----------------|
| Server Lifecycle | Long-running | Per-request |
| State Management | In-memory | Session headers |
| Scalability | Limited | High (load balancer friendly) |
| Session Analytics | Basic | Advanced |
| Enterprise Features | Basic | Comprehensive |
| MCP Protocol Compatibility | Full | Full |
| Deployment Complexity | Simple | Moderate |

## Troubleshooting

### Common Issues

1. **Session Not Found (404)**
   - Session expired or invalid
   - Analytics mode not enabled
   - Missing `Mcp-Session-Id` header

2. **Method Not Allowed (405)**
   - GET request to `/mcp` in strict compliance mode
   - Use POST for JSON-RPC requests

3. **Internal Error (500)**
   - Check server logs for detailed error information
   - Verify AWS Bedrock configuration
   - Ensure all dependencies are installed

### Debug Logging

Enable debug logging:
```bash
MCP_TRANSPORT=streamable_http LOG_LEVEL=DEBUG \
uv run python -m recursive_companion_mcp
```

### Health Monitoring

Monitor the health endpoint:
```bash
curl http://localhost:8080/health
```

## Development

### Adding New Features

1. **New Endpoints**: Add routes in `StreamableHTTPTransport.create_app()`
2. **Session Features**: Extend analytics session handling
3. **Metrics**: Add to the metrics dictionary
4. **Error Codes**: Follow JSON-RPC 2.0 specifications

### Testing

Run integration tests:
```bash
# Test streamable transport
MCP_TRANSPORT=streamable_http uv run pytest tests/test_streamable_http.py

# Test session management
ANALYTICS_MODE=true uv run pytest tests/test_analytics_sessions.py
```

## Performance

### Benchmarks

- Request overhead: ~5-10ms per request (server creation)
- Memory usage: ~10-20MB per concurrent request
- Session tracking: ~1KB per active session
- Throughput: 1000+ requests/second (depends on hardware)

### Optimization Tips

1. **Connection Pooling**: Use HTTP/1.1 keep-alive
2. **Session Cleanup**: Regular cleanup of expired sessions
3. **Load Balancing**: Multiple instances behind load balancer
4. **Caching**: Cache static responses (health endpoint, etc.)
