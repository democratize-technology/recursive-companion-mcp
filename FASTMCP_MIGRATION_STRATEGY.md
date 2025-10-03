# Recursive Companion MCP - FastMCP 2.0 Migration Strategy

**Document Version:** 1.0
**Target Architecture:** FastMCP 2.0 with HTTP Transport Support
**Reference Implementation:** devil-advocate-mcp
**Migration Risk Level:** Medium - Moderate refactoring with preserved functionality

---

## Executive Summary

This document outlines a comprehensive, low-risk migration strategy for converting recursive-companion-mcp from the legacy MCP SDK to FastMCP 2.0. The migration enables HTTP transport integration while preserving all existing refinement functionality, session management, and convergence logic.

**Key Benefits:**
- HTTP transport support for web-based integrations
- Improved error handling and validation patterns
- Better separation of concerns
- Modernized tool registration
- Aligned with platform standards (matches devil-advocate-mcp architecture)

**Migration Duration:** 2-3 days for full implementation and testing
**Risk Mitigation:** Phased approach with comprehensive testing at each stage

---

## 1. Current Architecture Analysis

### 1.1 Current MCP SDK Implementation

**File:** `src/server.py` (658 lines)

**Current Pattern:**
```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("recursive-companion")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [Tool(...), ...]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "start_refinement":
        # Implementation
    elif name == "continue_refinement":
        # Implementation
    # ... 8 total tools

async def main():
    async with stdio_server() as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())
```

**Key Components:**
1. **Monolithic Handler:** Single `handle_call_tool()` function with 400+ lines
2. **Manual Tool Registration:** Tools defined as dictionaries in `handle_list_tools()`
3. **Text-based Returns:** All tools return `list[TextContent]` with JSON strings
4. **Session Management:** Custom `SessionTracker` class (separate from incremental engine)
5. **Error Handling:** Try-catch blocks in each tool branch
6. **Incremental Engine:** Core refinement logic in `incremental_engine.py` (973 lines)

### 1.2 Current Tools Inventory

| Tool Name | Purpose | Current Implementation | Lines of Code |
|-----------|---------|----------------------|---------------|
| `start_refinement` | Initialize refinement session | Lines 205-236 | 32 |
| `continue_refinement` | Execute one refinement step | Lines 238-304 | 67 |
| `get_refinement_status` | Get session status | Lines 306-337 | 32 |
| `get_final_result` | Retrieve converged result | Lines 339-373 | 35 |
| `list_refinement_sessions` | List active sessions | Lines 375-415 | 41 |
| `current_session` | Get most recent session | Lines 417-466 | 50 |
| `abort_refinement` | Stop and return best result | Lines 468-515 | 48 |
| `quick_refine` | Auto-continue until complete | Lines 517-614 | 98 |

**Total:** 8 tools, ~400 lines in handler

### 1.3 Core Dependencies (Preserved)

**Critical Components to Maintain:**
- `bedrock_client.py` - AWS Bedrock integration
- `incremental_engine.py` - Refinement logic and convergence
- `session_manager.py` - Session persistence (enhanced)
- `convergence.py` - Cosine similarity detection
- `domains.py` - Domain detection
- `validation.py` - Security validation
- `config.py` - Configuration management

**Chain of Thought Integration:**
- `internal_cot.py` - Chain of thought processor
- `cot_enhancement.py` - Prompt enhancement

---

## 2. FastMCP Reference Architecture (devil-advocate-mcp)

### 2.1 FastMCP Pattern

**File:** `src/devil_advocate_mcp/core/server.py`

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("devil-advocate")

# In tool files (e.g., tools/adversarial_analysis.py):
from ..core import mcp, handle_tool_errors, format_output
from ..decorators import inject_client_context

@mcp.tool(description="...")
@format_output
@handle_tool_errors
@inject_client_context
async def start_adversarial_analysis(
    idea: str,
    context: str,
    stakes: str,
    perspectives: Optional[List[str]] = None,
    model_backend: str = "bedrock",
    model_name: Optional[str] = None,
    client_id: str = "default",
) -> str:  # Returns formatted string, not dict
    # Implementation
    return formatted_result
```

**Key Patterns:**
1. **FastMCP Instance:** Single `mcp = FastMCP(name)` instance
2. **Decorator-based Tools:** `@mcp.tool()` decorator for auto-registration
3. **Separated Tool Files:** Each tool in `tools/` directory
4. **String Returns:** Tools return formatted strings, not dicts
5. **Decorator Stack:** Error handling, formatting, client injection via decorators
6. **HTTP Transport Support:** Built-in via FastMCP 2.0

### 2.2 Directory Structure

```
devil-advocate-mcp/
├── src/devil_advocate_mcp/
│   ├── __init__.py              # Module exports, main() and http_main()
│   ├── core/
│   │   ├── __init__.py
│   │   └── server.py            # FastMCP instance, decorators
│   ├── tools/
│   │   ├── __init__.py          # Tool exports
│   │   ├── adversarial_analysis.py
│   │   ├── challenge_assumptions.py
│   │   ├── premortem.py
│   │   └── ...
│   ├── config.py                # Configuration
│   ├── session_manager.py       # Session management
│   ├── orchestrator.py          # LLM orchestration
│   ├── validation.py            # Input validation
│   ├── decorators.py            # Custom decorators
│   ├── models.py                # Pydantic models
│   └── transports/
│       ├── __init__.py
│       └── http.py              # HTTP transport (FastMCP 2.0)
```

### 2.3 HTTP Transport Integration

**File:** `src/devil_advocate_mcp/__init__.py`

```python
def http_main(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the MCP server with HTTP transport."""
    from .transports import create_http_app

    app = create_http_app()

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    if transport == "http":
        http_main(port=int(os.environ.get("MCP_HTTP_PORT", "8080")))
    else:
        main()  # stdio transport
```

---

## 3. Migration Strategy - Phased Approach

### Phase 1: Foundation Setup (Day 1, Morning)

**Objective:** Create new directory structure without breaking existing code

**Actions:**

1. **Create New Directory Structure**
   ```bash
   mkdir -p src/recursive_companion_mcp/core
   mkdir -p src/recursive_companion_mcp/tools
   mkdir -p src/recursive_companion_mcp/transports
   ```

2. **Create FastMCP Core Module**
   - New file: `src/recursive_companion_mcp/core/server.py`
   - Copy FastMCP instance pattern from devil-advocate-mcp
   - Copy decorators: `handle_tool_errors`, `format_output`, `auto_retry`, `circuit_breaker`
   - Adapt error types to recursive-companion context

3. **Create Package Initialization**
   - New file: `src/recursive_companion_mcp/__init__.py`
   - Implement `main()` for stdio transport
   - Implement `http_main()` for HTTP transport
   - Copy transport switching logic from devil-advocate-mcp

4. **Dependency Updates**
   - Update `pyproject.toml` with FastMCP >= 2.12.0
   - Add `uvicorn` for HTTP transport
   - Add `starlette` for HTTP support

**Testing Phase 1:**
```bash
# Verify structure
uv run python -c "import recursive_companion_mcp; print('Import successful')"

# Run tests (existing tests should still work with old server.py)
uv run pytest tests/ -v
```

**Risk Level:** Low - No changes to existing functionality

---

### Phase 2: Tool Migration (Day 1, Afternoon)

**Objective:** Convert each tool from monolithic handler to individual decorated functions

#### 2.1 Tool Migration Pattern

**Before (Old MCP SDK):**
```python
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "start_refinement":
        try:
            prompt = arguments.get("prompt", "")
            domain = arguments.get("domain", "auto")
            result = await incremental_engine.start_refinement(prompt, domain)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            logger.error(f"Error: {e}")
            return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
```

**After (FastMCP):**
```python
# File: src/recursive_companion_mcp/tools/refinement.py
from ..core import mcp, handle_tool_errors, format_output
from ..decorators import inject_client_context

@mcp.tool(description="Start a new incremental refinement session...")
@format_output
@handle_tool_errors
@inject_client_context
async def start_refinement(
    prompt: str,
    domain: str = "auto",
    client_id: str = "default",
) -> str:
    """Start a new refinement session - returns immediately"""
    result = await incremental_engine.start_refinement(prompt, domain)
    return format_refinement_response(result)
```

#### 2.2 Tool-by-Tool Migration Order

**Priority 1 (Core Functionality):**
1. `start_refinement` → `tools/refinement.py`
2. `continue_refinement` → `tools/refinement.py`
3. `get_refinement_status` → `tools/refinement.py`

**Priority 2 (Retrieval & Management):**
4. `get_final_result` → `tools/results.py`
5. `list_refinement_sessions` → `tools/sessions.py`
6. `current_session` → `tools/sessions.py`

**Priority 3 (Advanced Features):**
7. `abort_refinement` → `tools/control.py`
8. `quick_refine` → `tools/convenience.py`

#### 2.3 Formatting Functions

Create formatting helpers to convert dict results to formatted strings:

**File:** `src/recursive_companion_mcp/formatting.py`

```python
def format_refinement_response(result: dict) -> str:
    """Format refinement response for LLM consumption"""
    if not result.get("success"):
        return f"❌ **Error**: {result.get('error', 'Unknown error')}"

    status = result.get("status", "unknown")
    session_id = result.get("session_id", "N/A")

    output = f"""✅ **Refinement Session Started**

**Session ID:** `{session_id}`
**Status:** {status}
**Domain:** {result.get('domain', 'auto')}

**Next Action:** Use `continue_refinement` to proceed with refinement.

*Session ID: {session_id}*
"""
    return output
```

#### 2.4 Tool File Structure

**File:** `src/recursive_companion_mcp/tools/__init__.py`
```python
from .refinement import start_refinement, continue_refinement, get_refinement_status
from .results import get_final_result
from .sessions import list_refinement_sessions, current_session
from .control import abort_refinement
from .convenience import quick_refine

__all__ = [
    "start_refinement",
    "continue_refinement",
    "get_refinement_status",
    "get_final_result",
    "list_refinement_sessions",
    "current_session",
    "abort_refinement",
    "quick_refine",
]
```

**Testing Phase 2:**
```bash
# Test individual tool imports
uv run python -c "from recursive_companion_mcp.tools import start_refinement; print('Tools imported')"

# Run integration tests for migrated tools
uv run pytest tests/test_tools.py -v

# Test stdio mode
DEBUG=1 ./run.sh
```

**Risk Level:** Medium - Requires careful mapping of tool logic

---

### Phase 3: Session Manager Integration (Day 2, Morning)

**Objective:** Integrate existing SessionManager with FastMCP patterns

#### 3.1 Session Manager Enhancements

**Current:** `src/session_manager.py` - Simple session tracker
**Target:** Enhanced session manager similar to devil-advocate-mcp

**Changes Required:**

1. **Add Client ID Support**
   ```python
   class SessionTracker:
       def __init__(self):
           self.current_sessions: dict[str, str | None] = {}  # client_id -> session_id
           self.session_history: dict[str, list[dict[str, Any]]] = {}  # client_id -> history
   ```

2. **Add TTL and Cleanup**
   ```python
   def cleanup_old_sessions(self, max_age_minutes: int = 60):
       """Remove sessions older than max_age_minutes"""
   ```

3. **Thread-Safety**
   ```python
   import threading

   class SessionTracker:
       def __init__(self):
           self._lock = threading.Lock()
   ```

**Testing Phase 3:**
```bash
# Unit tests for session manager
uv run pytest tests/test_session_manager.py -v

# Integration tests with tools
uv run pytest tests/test_session_integration.py -v
```

**Risk Level:** Low - Additive changes only

---

### Phase 4: HTTP Transport Integration (Day 2, Afternoon)

**Objective:** Enable HTTP transport support

#### 4.1 Create HTTP Transport Module

**File:** `src/recursive_companion_mcp/transports/__init__.py`
```python
from .http import create_http_app

__all__ = ["create_http_app"]
```

**File:** `src/recursive_companion_mcp/transports/http.py`
```python
"""HTTP transport for recursive-companion MCP server"""

import logging
from typing import Any

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse

from ..core import mcp

logger = logging.getLogger(__name__)

async def mcp_endpoint(request: Any) -> JSONResponse:
    """Handle MCP protocol requests over HTTP"""
    try:
        data = await request.json()

        # Process MCP request
        result = await mcp.handle_request(data)

        return JSONResponse(result)
    except Exception as e:
        logger.error(f"HTTP request error: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

def create_http_app() -> Starlette:
    """Create HTTP application for MCP server"""
    routes = [
        Route("/mcp", mcp_endpoint, methods=["POST"]),
        Route("/health", lambda r: JSONResponse({"status": "healthy"}), methods=["GET"]),
    ]

    return Starlette(debug=False, routes=routes)
```

#### 4.2 Update Main Entry Points

**File:** `src/recursive_companion_mcp/__init__.py`
```python
def http_main(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the MCP server with HTTP transport"""
    import uvicorn
    from .transports import create_http_app

    logger.info(f"Starting Recursive Companion MCP server (HTTP) on {host}:{port}")

    app = create_http_app()
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    import os

    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    if transport == "http":
        port = int(os.environ.get("MCP_HTTP_PORT", "8080"))
        http_main(port=port)
    else:
        main()
```

**Testing Phase 4:**
```bash
# Test HTTP mode
MCP_TRANSPORT=http MCP_HTTP_PORT=8081 uv run python -m recursive_companion_mcp &

# Test HTTP endpoint
curl -X POST http://localhost:8081/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/list"}'

# Test health endpoint
curl http://localhost:8081/health

# Kill background server
pkill -f "recursive_companion_mcp"
```

**Risk Level:** Medium - New functionality, requires HTTP testing

---

### Phase 5: Incremental Engine Preservation (Day 3, Morning)

**Objective:** Ensure incremental_engine.py works with new architecture

#### 5.1 Interface Verification

**No changes required to:**
- `IncrementalRefineEngine` class
- Convergence detection logic
- CoT integration
- Domain detection
- Bedrock client integration

**Adapter Pattern (if needed):**
```python
# In tools/refinement.py

class IncrementalEngineAdapter:
    """Adapter to bridge legacy incremental engine with FastMCP"""

    def __init__(self, engine: IncrementalRefineEngine):
        self.engine = engine

    async def start_refinement_formatted(
        self, prompt: str, domain: str = "auto"
    ) -> str:
        """Start refinement and return formatted response"""
        result = await self.engine.start_refinement(prompt, domain)
        return format_refinement_response(result)
```

**Testing Phase 5:**
```bash
# Integration tests with incremental engine
uv run pytest tests/test_incremental_engine.py -v

# End-to-end refinement tests
uv run pytest tests/test_e2e_refinement.py -v
```

**Risk Level:** Low - Minimal changes to core logic

---

### Phase 6: Testing & Validation (Day 3, Afternoon)

**Objective:** Comprehensive testing across all migration layers

#### 6.1 Test Categories

**1. Unit Tests (Existing + New)**
```bash
uv run pytest tests/test_tools.py -v          # Individual tool tests
uv run pytest tests/test_session_manager.py -v # Session management
uv run pytest tests/test_formatting.py -v      # Formatting functions
uv run pytest tests/test_decorators.py -v      # Decorator functionality
```

**2. Integration Tests**
```bash
uv run pytest tests/test_integration.py -v     # End-to-end workflows
uv run pytest tests/test_http_transport.py -v  # HTTP mode
uv run pytest tests/test_stdio_transport.py -v # stdio mode
```

**3. Regression Tests**
```bash
# Compare outputs between old and new implementation
uv run pytest tests/test_regression.py -v
```

**4. Performance Tests**
```bash
# Ensure no performance degradation
uv run pytest tests/test_performance.py -v
```

#### 6.2 Manual Testing Checklist

- [ ] Start refinement session (stdio mode)
- [ ] Continue refinement until convergence
- [ ] Get final result
- [ ] List sessions
- [ ] Current session retrieval
- [ ] Abort refinement
- [ ] Quick refine
- [ ] HTTP mode: all tools
- [ ] HTTP mode: error handling
- [ ] Concurrent sessions
- [ ] Session cleanup
- [ ] Error recovery

**Testing Phase 6:**
```bash
# Full test suite
uv run pytest tests/ -v --cov=src/recursive_companion_mcp --cov-report=term-missing

# Coverage should be >= 85%
```

**Risk Level:** Low - Testing phase only

---

## 4. Tool Mapping Reference

### 4.1 Complete Tool Migration Map

| Old Tool | New Module | New Function | Changes Required |
|----------|-----------|--------------|------------------|
| `start_refinement` | `tools/refinement.py` | `start_refinement()` | Add formatting, remove JSON wrapping |
| `continue_refinement` | `tools/refinement.py` | `continue_refinement()` | Add formatting, improve error messages |
| `get_refinement_status` | `tools/refinement.py` | `get_refinement_status()` | Add formatting, enhance status display |
| `get_final_result` | `tools/results.py` | `get_final_result()` | Add formatting, improve result presentation |
| `list_refinement_sessions` | `tools/sessions.py` | `list_refinement_sessions()` | Add client_id support, improve formatting |
| `current_session` | `tools/sessions.py` | `current_session()` | Add client_id support, enhance display |
| `abort_refinement` | `tools/control.py` | `abort_refinement()` | Add confirmation, improve messaging |
| `quick_refine` | `tools/convenience.py` | `quick_refine()` | Preserve auto-continue logic, add progress updates |

### 4.2 Parameter Changes

**No parameter changes required** - All existing tool parameters remain the same for backward compatibility.

**New optional parameters:**
- `client_id: str = "default"` - Auto-injected by decorator, used for multi-client scenarios

---

## 5. Risk Assessment & Mitigation

### 5.1 Risk Matrix

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| Session state loss during migration | Low | High | Phased migration, preserve session_manager.py |
| Incremental engine breakage | Low | Critical | No changes to core logic, adapter pattern if needed |
| Tool behavior changes | Medium | Medium | Comprehensive regression testing |
| HTTP transport failures | Medium | Low | Fallback to stdio, separate testing phase |
| Performance degradation | Low | Medium | Benchmark before/after, optimize if needed |
| CoT integration issues | Low | High | Preserve existing CoT enhancement code |
| Convergence algorithm changes | Low | Critical | No changes to convergence.py |
| Bedrock client errors | Low | High | Preserve bedrock_client.py unchanged |

### 5.2 Rollback Plan

**If migration fails:**

1. **Keep old server.py as `server_legacy.py`**
   ```bash
   cp src/server.py src/server_legacy.py
   ```

2. **Create rollback script:**
   ```bash
   #!/bin/bash
   # rollback.sh
   rm -rf src/recursive_companion_mcp/
   cp src/server_legacy.py src/server.py
   git checkout pyproject.toml
   uv sync
   ```

3. **Git branch strategy:**
   ```bash
   # Work in feature branch
   git checkout -b feature/fastmcp-migration

   # Can always revert to main
   git checkout main
   ```

### 5.3 Success Criteria

**Migration is successful when:**

- [ ] All 8 tools work in stdio mode
- [ ] All 8 tools work in HTTP mode
- [ ] Test coverage >= 85%
- [ ] No performance regression (< 10% slower)
- [ ] Convergence algorithm unchanged
- [ ] Session management improved (TTL, cleanup)
- [ ] Error messages more helpful
- [ ] Documentation updated
- [ ] CI/CD pipeline passes

---

## 6. File Structure Changes

### 6.1 Before Migration

```
recursive-companion-mcp/
├── src/
│   ├── server.py                    # Monolithic server (658 lines)
│   ├── bedrock_client.py
│   ├── incremental_engine.py
│   ├── session_manager.py
│   ├── convergence.py
│   ├── domains.py
│   ├── validation.py
│   ├── config.py
│   ├── cot_enhancement.py
│   ├── internal_cot.py
│   └── ... (other modules)
├── tests/
│   └── test_server.py
└── pyproject.toml
```

### 6.2 After Migration

```
recursive-companion-mcp/
├── src/recursive_companion_mcp/
│   ├── __init__.py                  # main(), http_main()
│   ├── __main__.py                  # CLI entry point
│   ├── core/
│   │   ├── __init__.py
│   │   └── server.py                # FastMCP instance, decorators
│   ├── tools/
│   │   ├── __init__.py              # Tool exports
│   │   ├── refinement.py            # start, continue, get_status
│   │   ├── results.py               # get_final_result
│   │   ├── sessions.py              # list_sessions, current_session
│   │   ├── control.py               # abort_refinement
│   │   └── convenience.py           # quick_refine
│   ├── transports/
│   │   ├── __init__.py
│   │   └── http.py                  # HTTP transport
│   ├── formatting.py                # Response formatters
│   ├── decorators.py                # Custom decorators
│   ├── bedrock_client.py            # (Preserved)
│   ├── incremental_engine.py        # (Preserved)
│   ├── session_manager.py           # (Enhanced)
│   ├── convergence.py               # (Preserved)
│   ├── domains.py                   # (Preserved)
│   ├── validation.py                # (Preserved)
│   ├── config.py                    # (Preserved)
│   ├── cot_enhancement.py           # (Preserved)
│   ├── internal_cot.py              # (Preserved)
│   └── ... (other modules)
├── tests/
│   ├── test_tools.py                # Tool-specific tests
│   ├── test_session_manager.py      # Session tests
│   ├── test_formatting.py           # Formatting tests
│   ├── test_http_transport.py       # HTTP tests
│   ├── test_integration.py          # E2E tests
│   └── test_regression.py           # Regression tests
└── pyproject.toml                   # Updated dependencies
```

**Key Changes:**
- Package structure: `src/` → `src/recursive_companion_mcp/`
- Monolithic `server.py` → Modular `core/` + `tools/`
- New `transports/` directory
- New `formatting.py` for response formatting
- Enhanced testing structure

---

## 7. Testing Strategy

### 7.1 Test Coverage Goals

**Target:** 85% overall coverage

**Critical Paths (100% coverage required):**
- Tool functions (all 8 tools)
- Session manager
- Error handling decorators
- HTTP transport
- Formatting functions

**Important Paths (90% coverage):**
- Incremental engine integration
- Configuration
- Validation

**Optional Paths (70% coverage):**
- CoT enhancement
- Utility functions

### 7.2 Test Types

**1. Unit Tests**
```python
# tests/test_tools.py
@pytest.mark.asyncio
async def test_start_refinement():
    """Test start_refinement tool"""
    result = await start_refinement(
        prompt="Test prompt",
        domain="technical",
        client_id="test-client"
    )

    assert "Session ID" in result
    assert "✅" in result  # Success indicator
```

**2. Integration Tests**
```python
# tests/test_integration.py
@pytest.mark.asyncio
async def test_full_refinement_cycle():
    """Test complete refinement workflow"""
    # Start
    start_result = await start_refinement(
        prompt="Explain Python decorators",
        domain="technical"
    )
    session_id = extract_session_id(start_result)

    # Continue until convergence
    for _ in range(10):
        continue_result = await continue_refinement(session_id=session_id)
        if "converged" in continue_result.lower():
            break

    # Get final result
    final_result = await get_final_result(session_id=session_id)
    assert "refined_answer" in final_result.lower()
```

**3. Regression Tests**
```python
# tests/test_regression.py
@pytest.mark.asyncio
async def test_convergence_algorithm_unchanged():
    """Ensure convergence algorithm produces same results"""
    # Run same prompt with same settings
    # Compare convergence scores between old and new
    pass
```

**4. HTTP Transport Tests**
```python
# tests/test_http_transport.py
@pytest.mark.asyncio
async def test_http_tool_invocation():
    """Test tool invocation via HTTP"""
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8081/mcp",
            json={
                "method": "tools/call",
                "params": {
                    "name": "start_refinement",
                    "arguments": {
                        "prompt": "Test",
                        "domain": "auto"
                    }
                }
            }
        )

        assert response.status_code == 200
        assert "session_id" in response.text
```

### 7.3 Test Execution

**Development:**
```bash
# Quick tests during development
uv run pytest tests/test_tools.py -v -k test_start_refinement

# Watch mode
uv run pytest-watch
```

**Pre-commit:**
```bash
# Full suite with coverage
uv run pytest tests/ -v --cov=src/recursive_companion_mcp --cov-report=html

# Open coverage report
open htmlcov/index.html
```

**CI/CD:**
```bash
# GitHub Actions
uv run pytest tests/ -v --cov=src/recursive_companion_mcp --cov-report=xml
```

---

## 8. Configuration Changes

### 8.1 pyproject.toml Updates

**Add FastMCP dependency:**
```toml
dependencies = [
    "fastmcp>=2.12.0,<3.0.0",  # NEW
    "pydantic>=2.11.0,<3.0.0",
    "boto3>=1.39.0,<2.0.0",
    # ... existing dependencies
    "uvicorn>=0.27.0,<1.0.0",  # NEW - for HTTP transport
    "starlette>=0.47.3",       # NEW - for HTTP transport
]
```

**Update package structure:**
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/recursive_companion_mcp"]  # Updated path
```

### 8.2 Environment Variables

**New environment variables:**
```bash
# Transport selection
MCP_TRANSPORT=stdio|http  # Default: stdio

# HTTP transport config (if MCP_TRANSPORT=http)
MCP_HTTP_PORT=8080        # Default: 8080
MCP_HTTP_HOST=127.0.0.1   # Default: 127.0.0.1

# Existing variables (preserved)
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
CONVERGENCE_THRESHOLD=0.98
MAX_ITERATIONS=10
```

### 8.3 Claude Desktop Configuration

**stdio mode (default):**
```json
{
  "mcpServers": {
    "recursive-companion": {
      "command": "uv",
      "args": ["run", "python", "-m", "recursive_companion_mcp"],
      "env": {
        "AWS_REGION": "us-east-1",
        "BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1:0"
      }
    }
  }
}
```

**HTTP mode:**
```json
{
  "mcpServers": {
    "recursive-companion": {
      "command": "uv",
      "args": ["run", "python", "-m", "recursive_companion_mcp"],
      "env": {
        "MCP_TRANSPORT": "http",
        "MCP_HTTP_PORT": "8081",
        "AWS_REGION": "us-east-1"
      }
    }
  }
}
```

---

## 9. Documentation Updates Required

### 9.1 Files to Update

1. **README.md**
   - Update installation instructions
   - Add HTTP transport usage
   - Update configuration examples

2. **CLAUDE.md**
   - Update architecture overview
   - Add new file structure
   - Update development commands

3. **CONFIGURATION.md** (new)
   - Comprehensive configuration guide
   - Environment variables
   - Transport modes

4. **DEVELOPMENT.md** (new)
   - Developer setup guide
   - Tool development patterns
   - Testing guidelines

5. **MIGRATION.md** (this document)
   - Keep as historical reference
   - Update with lessons learned

### 9.2 Code Comments

**Add comprehensive docstrings:**
```python
@mcp.tool(description="...")
async def start_refinement(
    prompt: str,
    domain: str = "auto",
    client_id: str = "default",
) -> str:
    """
    Start a new incremental refinement session.

    This tool initializes a refinement session that uses Draft → Critique → Revise
    cycles to iteratively improve responses through self-critique and convergence
    measurement.

    Args:
        prompt: The question or task to refine (10-10,000 characters)
        domain: Domain for specialized prompts (auto|technical|marketing|strategy|
                legal|financial|general). Default: auto-detect
        client_id: Client identifier for session isolation (auto-injected)

    Returns:
        Formatted string with session ID and next action instructions

    Example:
        >>> result = await start_refinement(
        ...     prompt="Explain Python decorators in detail",
        ...     domain="technical"
        ... )
        >>> # Returns: "✅ Refinement Session Started\n\nSession ID: abc-123..."

    Raises:
        ValidationError: If prompt is invalid (too short/long, malicious content)
        LLMError: If AWS Bedrock is unavailable

    See Also:
        - continue_refinement: Execute refinement steps
        - get_refinement_status: Check progress
        - abort_refinement: Stop early and get best result
    """
```

---

## 10. Post-Migration Checklist

### 10.1 Code Quality

- [ ] All tools migrated and tested
- [ ] Test coverage >= 85%
- [ ] No regressions in functionality
- [ ] Performance benchmarked (< 10% degradation)
- [ ] Code formatted with ruff
- [ ] Type hints verified with mypy
- [ ] Security scan passed (bandit)
- [ ] Dependency audit passed (pip-audit)

### 10.2 Documentation

- [ ] README.md updated
- [ ] CLAUDE.md updated
- [ ] API documentation complete
- [ ] Configuration guide written
- [ ] Development guide written
- [ ] Migration lessons documented
- [ ] Changelog updated

### 10.3 Testing

- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Regression tests passing
- [ ] HTTP transport tests passing
- [ ] Manual testing checklist complete
- [ ] Performance tests passing
- [ ] Load testing (if applicable)

### 10.4 Deployment

- [ ] CI/CD pipeline updated
- [ ] Claude Desktop config tested
- [ ] HTTP mode tested in production-like environment
- [ ] Rollback plan tested
- [ ] Monitoring configured
- [ ] Error tracking configured

### 10.5 Cleanup

- [ ] Old `src/server.py` removed (keep as `server_legacy.py` backup)
- [ ] Unused imports removed
- [ ] Debug code removed
- [ ] Commented-out code removed
- [ ] Temporary files removed
- [ ] Git history clean

---

## 11. Implementation Timeline

### Day 1

**Morning (3-4 hours):**
- Phase 1: Foundation Setup (2 hours)
- Phase 2: Tool Migration (start) (2 hours)

**Afternoon (3-4 hours):**
- Phase 2: Tool Migration (continue) (3 hours)
- Testing of migrated tools (1 hour)

**Deliverables:**
- New directory structure
- FastMCP core module
- 4-5 tools migrated
- Basic tests passing

### Day 2

**Morning (3-4 hours):**
- Phase 2: Tool Migration (complete) (1 hour)
- Phase 3: Session Manager Integration (2 hours)
- Testing (1 hour)

**Afternoon (3-4 hours):**
- Phase 4: HTTP Transport Integration (3 hours)
- HTTP testing (1 hour)

**Deliverables:**
- All 8 tools migrated
- Enhanced session manager
- HTTP transport working
- HTTP tests passing

### Day 3

**Morning (3-4 hours):**
- Phase 5: Incremental Engine Preservation (1 hour)
- Phase 6: Testing & Validation (start) (3 hours)

**Afternoon (3-4 hours):**
- Phase 6: Testing & Validation (complete) (2 hours)
- Documentation updates (2 hours)

**Deliverables:**
- All tests passing
- Coverage >= 85%
- Documentation complete
- Migration complete

**Total Estimated Time:** 20-24 hours (2.5-3 days)

---

## 12. Success Metrics

### 12.1 Functional Metrics

- **Tool Coverage:** 8/8 tools migrated and working
- **Transport Support:** stdio and HTTP both functional
- **Test Coverage:** >= 85% overall
- **Regression:** 0 behavioral changes in core refinement logic
- **Error Handling:** Improved error messages with context

### 12.2 Performance Metrics

- **Startup Time:** < 1 second (no degradation)
- **Tool Latency:** < 5% increase from decorator overhead
- **Memory Usage:** < 10% increase
- **Convergence Speed:** Unchanged (same algorithm)

### 12.3 Quality Metrics

- **Code Duplication:** Reduced (shared decorators)
- **Maintainability:** Improved (modular structure)
- **Testability:** Improved (isolated tools)
- **Documentation:** Comprehensive and up-to-date

---

## 13. Lessons Learned & Best Practices

### 13.1 Migration Best Practices

1. **Phased Approach Works:** Breaking migration into 6 phases reduced risk
2. **Keep Old Code as Backup:** `server_legacy.py` provides safety net
3. **Test Each Phase:** Catch issues early before compounding
4. **Preserve Core Logic:** Don't change algorithms during migration
5. **Document Everything:** Future maintainers will thank you

### 13.2 FastMCP Patterns to Follow

1. **Decorator Stack Order:** `@mcp.tool() → @format_output → @handle_tool_errors → @inject_client_context`
2. **Return Strings, Not Dicts:** Let formatters handle structure
3. **Centralize Error Handling:** Use decorators, not try-catch in every tool
4. **Separate Concerns:** Tools in `tools/`, logic in core modules
5. **Transport Agnostic:** Same code works for stdio and HTTP

### 13.3 Common Pitfalls to Avoid

1. **Don't Change Algorithms:** Migration is NOT refactoring
2. **Don't Skip Testing:** Each phase needs validation
3. **Don't Rush HTTP:** Test stdio thoroughly first
4. **Don't Forget Documentation:** Code without docs is incomplete
5. **Don't Ignore Performance:** Benchmark before and after

---

## 14. Appendix

### A. Reference Implementation Files

**Key devil-advocate-mcp files to study:**
- `src/devil_advocate_mcp/__init__.py` - Entry points
- `src/devil_advocate_mcp/core/server.py` - FastMCP instance
- `src/devil_advocate_mcp/tools/adversarial_analysis.py` - Tool pattern
- `src/devil_advocate_mcp/transports/http.py` - HTTP transport
- `src/devil_advocate_mcp/formatting.py` - Response formatting

### B. Migration Commands Cheat Sheet

```bash
# Setup
mkdir -p src/recursive_companion_mcp/{core,tools,transports}
touch src/recursive_companion_mcp/{__init__.py,__main__.py}
touch src/recursive_companion_mcp/core/{__init__.py,server.py}

# Development
uv run python -m recursive_companion_mcp  # stdio mode
MCP_TRANSPORT=http uv run python -m recursive_companion_mcp  # HTTP mode
DEBUG=1 ./run.sh  # Debug mode

# Testing
uv run pytest tests/ -v  # All tests
uv run pytest tests/test_tools.py -v -k test_start_refinement  # Single test
uv run pytest --cov=src/recursive_companion_mcp --cov-report=html  # Coverage

# Quality
uv run ruff check src/  # Linting
uv run ruff format src/  # Formatting
uv run mypy src/recursive_companion_mcp/  # Type checking
uv run bandit -r src/  # Security scan
```

### C. Troubleshooting Guide

**Issue: Import errors after migration**
```bash
# Solution: Verify package structure
uv run python -c "import recursive_companion_mcp; print(recursive_companion_mcp.__file__)"
```

**Issue: Tools not registering**
```bash
# Solution: Check tool imports in __init__.py
uv run python -c "from recursive_companion_mcp.tools import *; print(dir())"
```

**Issue: HTTP mode not starting**
```bash
# Solution: Check dependencies
uv run pip list | grep -E "uvicorn|starlette|fastmcp"
```

**Issue: Tests failing**
```bash
# Solution: Run with verbose output
uv run pytest tests/ -vv --tb=short
```

---

## 15. Conclusion

This migration strategy provides a comprehensive, low-risk path to modernizing recursive-companion-mcp with FastMCP 2.0. The phased approach ensures that:

1. **Core functionality is preserved** - No changes to refinement algorithm
2. **Testing is thorough** - 85%+ coverage with regression testing
3. **Rollback is possible** - Clear rollback plan and backup files
4. **Documentation is complete** - All changes documented
5. **HTTP transport is enabled** - Modern integration capabilities

**Key Takeaway:** Migration is about **enabling new capabilities** (HTTP transport) while **preserving existing excellence** (convergence detection, CoT integration, domain optimization).

**Next Steps:**
1. Review this strategy with stakeholders
2. Set up development environment
3. Begin Phase 1: Foundation Setup
4. Follow phased approach with testing at each stage
5. Document lessons learned for future migrations

---

**Document Metadata:**
- **Version:** 1.0
- **Date:** 2025-10-03
- **Author:** Architect Agent
- **Review Status:** Pending stakeholder approval
- **Estimated Effort:** 20-24 hours (2.5-3 days)
- **Risk Level:** Medium (mitigated by phased approach)
