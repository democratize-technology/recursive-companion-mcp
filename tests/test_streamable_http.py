"""
Integration tests for Streamable HTTP transport.

Tests the enterprise-grade streamable HTTP transport implementation
following HuggingFace's stateless HTTP JSON pattern.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from recursive_companion_mcp.transport.streamable_http import StreamableHTTPTransport


class TestStreamableHTTPTransport:
    """Test suite for StreamableHTTPTransport."""

    @pytest.fixture
    def mock_server_factory(self):
        """Mock MCP server factory."""

        def factory():
            server = AsyncMock()
            server.list_tools.return_value = {
                "tools": [
                    {
                        "name": "start_refinement",
                        "description": "Start iterative refinement process",
                    }
                ]
            }
            server.call_tool.return_value = {"content": [{"type": "text", "text": "Test response"}]}
            server.close = AsyncMock()
            # Mock capabilities attribute
            server.capabilities = {"tools": {}, "prompts": {}, "resources": {}}
            return server

        return factory

    @pytest.fixture
    def transport(self, mock_server_factory):
        """Create transport instance for testing."""
        return StreamableHTTPTransport(
            mcp_server_factory=mock_server_factory,
            host="127.0.0.1",
            port=8080,
            enable_json_response=True,
            analytics_mode=False,
        )

    @pytest.fixture
    def analytics_transport(self, mock_server_factory):
        """Create transport with analytics mode enabled."""
        return StreamableHTTPTransport(
            mcp_server_factory=mock_server_factory,
            host="127.0.0.1",
            port=8080,
            enable_json_response=True,
            analytics_mode=True,
        )

    def test_health_endpoint(self, transport):
        """Test health check endpoint."""
        app = transport.create_app()
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "recursive-companion"
        assert data["transport"] == "streamable-http"
        assert data["active_sessions"] == 0

    def test_oauth_metadata_endpoint(self, transport):
        """Test OAuth metadata endpoint."""
        with patch.dict("os.environ", {"MCP_SERVER_URL": "https://test.example.com"}):
            app = transport.create_app()
            client = TestClient(app)

            response = client.get("/.well-known/oauth-protected-resource")
            assert response.status_code == 200

            data = response.json()
            assert data["resource"] == "https://test.example.com"
            assert "scopes_supported" in data
            assert "bearer_methods_supported" in data

    def test_welcome_page_get_request(self, transport):
        """Test GET request to /mcp returns welcome page."""
        app = transport.create_app()
        client = TestClient(app)

        response = client.get("/mcp", headers={"User-Agent": "Mozilla/5.0 (Test Browser)"})
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Recursive Companion MCP Server" in response.text

    def test_strict_compliance_get_request(self, transport):
        """Test GET request in strict compliance mode."""
        with patch.dict("os.environ", {"MCP_STRICT_COMPLIANCE": "true"}):
            app = transport.create_app()
            client = TestClient(app)

            response = client.get("/mcp")
            assert response.status_code == 405

            data = response.json()
            assert data["error"]["code"] == -32601
            assert "Method not allowed" in data["error"]["message"]

    def test_initialize_request(self, transport):
        """Test JSON-RPC initialize request."""
        app = transport.create_app()
        client = TestClient(app)

        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {"clientInfo": {"name": "test-client", "version": "1.0.0"}},
            "id": 1,
        }

        response = client.post("/mcp", json=init_request)
        assert response.status_code == 200

        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert "result" in data
        assert data["result"]["protocolVersion"] == "2024-11-05"
        assert data["result"]["serverInfo"]["name"] == "recursive-companion"

    def test_tools_list_request(self, transport):
        """Test tools/list request."""
        app = transport.create_app()
        client = TestClient(app)

        tools_request = {"jsonrpc": "2.0", "method": "tools/list", "id": 2}

        response = client.post("/mcp", json=tools_request)
        assert response.status_code == 200

        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 2
        assert "result" in data
        assert "tools" in data["result"]
        assert len(data["result"]["tools"]) > 0

    def test_tools_call_request(self, transport):
        """Test tools/call request."""
        app = transport.create_app()
        client = TestClient(app)

        tool_call_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "start_refinement",
                "arguments": {"topic": "Test topic", "style": "analytical"},
            },
            "id": 3,
        }

        response = client.post("/mcp", json=tool_call_request)
        assert response.status_code == 200

        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 3
        assert "result" in data

    def test_invalid_json_request(self, transport):
        """Test handling of invalid JSON."""
        app = transport.create_app()
        client = TestClient(app)

        response = client.post(
            "/mcp", data="invalid json", headers={"content-type": "application/json"}
        )
        assert response.status_code == 400

        data = response.json()
        assert data["error"]["code"] == -32700
        assert "Parse error" in data["error"]["message"]

    def test_invalid_jsonrpc_request(self, transport):
        """Test handling of invalid JSON-RPC structure."""
        app = transport.create_app()
        client = TestClient(app)

        # Missing jsonrpc version
        invalid_request = {"method": "initialize", "id": 1}

        response = client.post("/mcp", json=invalid_request)
        assert response.status_code == 400

        data = response.json()
        assert data["error"]["code"] == -32600
        assert "Invalid Request" in data["error"]["message"]

    def test_method_not_found(self, transport):
        """Test handling of unknown method."""
        app = transport.create_app()
        client = TestClient(app)

        unknown_method_request = {"jsonrpc": "2.0", "method": "unknown_method", "id": 4}

        response = client.post("/mcp", json=unknown_method_request)
        assert response.status_code == 200

        data = response.json()
        assert data["error"]["code"] == -32601
        assert "Method not found" in data["error"]["message"]

    def test_notification_handling(self, transport):
        """Test handling of JSON-RPC notifications (no id field)."""
        app = transport.create_app()
        client = TestClient(app)

        notification_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            # No "id" field - this is a notification
        }

        response = client.post("/mcp", json=notification_request)
        assert response.status_code == 200  # Notifications return 200 with null result

        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["result"] is None

    def test_analytics_session_creation(self, analytics_transport):
        """Test session creation in analytics mode."""
        app = analytics_transport.create_app()
        client = TestClient(app)

        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {"clientInfo": {"name": "test-client", "version": "1.0.0"}},
            "id": 1,
        }

        response = client.post("/mcp", json=init_request)
        assert response.status_code == 200

        # Check for session ID header
        assert "mcp-session-id" in response.headers
        session_id = response.headers["mcp-session-id"]
        assert len(session_id) > 0

        # Verify session exists in transport
        assert session_id in analytics_transport.analytics_sessions
        session_data = analytics_transport.analytics_sessions[session_id]
        assert session_data["client_info"]["name"] == "test-client"
        assert session_data["request_count"] == 1

    def test_analytics_session_resume(self, analytics_transport):
        """Test session resumption in analytics mode."""
        app = analytics_transport.create_app()
        client = TestClient(app)

        # First, create a session
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {"clientInfo": {"name": "test-client", "version": "1.0.0"}},
            "id": 1,
        }

        init_response = client.post("/mcp", json=init_request)
        session_id = init_response.headers["mcp-session-id"]

        # Now make a request with the session ID
        tools_request = {"jsonrpc": "2.0", "method": "tools/list", "id": 2}

        response = client.post("/mcp", json=tools_request, headers={"mcp-session-id": session_id})
        assert response.status_code == 200

        # Verify session was updated
        session_data = analytics_transport.analytics_sessions[session_id]
        assert session_data["request_count"] == 2
        assert session_data["last_activity"] > session_data["created_at"]

    def test_analytics_session_not_found(self, analytics_transport):
        """Test handling of invalid session ID."""
        app = analytics_transport.create_app()
        client = TestClient(app)

        # Make request with invalid session ID
        tools_request = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}

        response = client.post(
            "/mcp", json=tools_request, headers={"mcp-session-id": str(uuid.uuid4())}  # Random UUID
        )
        assert response.status_code == 404

        data = response.json()
        assert data["error"]["code"] == -32001
        assert "Session not found" in data["error"]["message"]

    def test_analytics_session_delete(self, analytics_transport):
        """Test session deletion in analytics mode."""
        app = analytics_transport.create_app()
        client = TestClient(app)

        # First, create a session
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {"clientInfo": {"name": "test-client", "version": "1.0.0"}},
            "id": 1,
        }

        init_response = client.post("/mcp", json=init_request)
        session_id = init_response.headers["mcp-session-id"]

        # Delete the session
        response = client.delete("/mcp", headers={"mcp-session-id": session_id})
        assert response.status_code == 200

        data = response.json()
        assert data["result"]["deleted"] is True

        # Verify session is gone
        assert session_id not in analytics_transport.analytics_sessions

    def test_metrics_tracking(self, transport):
        """Test metrics tracking functionality."""
        app = transport.create_app()
        client = TestClient(app)

        # Make some requests
        for i in range(3):
            request = {"jsonrpc": "2.0", "method": "tools/list", "id": i}
            client.post("/mcp", json=request)

        metrics = transport.get_metrics()
        assert metrics["requests_handled"] == 3
        assert metrics["transport_type"] == "streamable-http"
        assert metrics["analytics_mode"] is False

    def test_server_factory_called_per_request(self, mock_server_factory):
        """Test that server factory is called for each request."""
        # Wrap the factory to track calls
        call_count = 0
        original_factory = mock_server_factory

        def tracking_factory():
            nonlocal call_count
            call_count += 1
            return original_factory()

        transport = StreamableHTTPTransport(
            mcp_server_factory=tracking_factory, host="127.0.0.1", port=8080
        )

        app = transport.create_app()
        client = TestClient(app)

        # Make multiple requests
        for i in range(3):
            request = {"jsonrpc": "2.0", "method": "tools/list", "id": i}
            client.post("/mcp", json=request)

        # Verify factory was called for each request
        assert call_count == 3

    @patch("recursive_companion_mcp.transport.streamable_http.logger")
    def test_error_handling_and_logging(self, mock_logger, transport):
        """Test error handling and logging."""

        # Create a server factory that raises an exception
        def failing_factory():
            server = AsyncMock()
            # Mock capabilities attribute to prevent initialization errors
            server.capabilities = {"tools": {}, "prompts": {}, "resources": {}}
            # Make the server factory itself fail
            raise Exception("Test error")

        transport = StreamableHTTPTransport(
            mcp_server_factory=failing_factory, host="127.0.0.1", port=8080
        )

        app = transport.create_app()
        client = TestClient(app)

        request = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}

        response = client.post("/mcp", json=request)
        assert response.status_code == 500

        data = response.json()
        assert data["error"]["code"] == -32603
        assert "Internal error" in data["error"]["message"]

        # Verify error was logged
        mock_logger.exception.assert_called_once()

    def test_accept_headers_handling(self, transport):
        """Test handling of different Accept headers."""
        app = transport.create_app()
        client = TestClient(app)

        request = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}

        # Test with application/json
        response = client.post("/mcp", json=request, headers={"Accept": "application/json"})
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

        # Test with text/event-stream
        response = client.post("/mcp", json=request, headers={"Accept": "text/event-stream"})
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

    def test_browser_detection_get_request(self, transport):
        """Test browser detection for GET requests."""
        app = transport.create_app()

        # Test with browser user agent
        with TestClient(app) as client:
            response = client.get(
                "/mcp",
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

        # Test with non-browser user agent
        with TestClient(app) as client:
            response = client.get("/mcp", headers={"User-Agent": "curl/7.68.0"})
            assert response.status_code == 405

    def test_concurrent_requests(self, transport):
        """Test handling of concurrent requests."""
        app = transport.create_app()
        client = TestClient(app)

        import threading

        results = []

        def make_request(request_id):
            request = {"jsonrpc": "2.0", "method": "tools/list", "id": request_id}
            response = client.post("/mcp", json=request)
            results.append(response.status_code)

        # Make concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)
