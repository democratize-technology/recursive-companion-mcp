# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

**Running the server:**
```bash
# Stdio transport (default - for Claude Desktop)
uv run python -m recursive_companion_mcp

# HTTP transport (for network access)
export MCP_TRANSPORT=http
export MCP_HTTP_HOST=127.0.0.1  # Optional, defaults to 127.0.0.1
export MCP_HTTP_PORT=8080       # Optional, defaults to 8080
uv run python -m recursive_companion_mcp
```

### Transport Modes

**Stdio Transport (default - for Claude Desktop):**
```bash
uv run python -m recursive_companion_mcp
```

**HTTP Transport (for network access):**
```bash
export MCP_TRANSPORT=http
export MCP_HTTP_HOST=127.0.0.1  # Optional, defaults to 127.0.0.1
export MCP_HTTP_PORT=8080       # Optional, defaults to 8080
uv run python -m recursive_companion_mcp
```

Or in a single command:
```bash
MCP_TRANSPORT=http MCP_HTTP_PORT=8080 uv run python -m recursive_companion_mcp
```

**Running tests:**
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_server.py

# Run with coverage
uv run pytest --cov=src tests/
```

**Development setup:**
```bash
# Install/sync dependencies
uv sync

# Format code (line-length 100)
uv run black src/ tests/

# Lint code
uv run flake8 src/ tests/
```

## Architecture Overview

This MCP server implements iterative refinement through self-critique cycles, inspired by Hank Besser's recursive-companion. The system uses a **Draft → Critique → Revise → Converge** pattern with incremental processing to avoid timeouts.

### Core Components

1. **MCP Server (`src/server.py`)**
   - Handles MCP protocol communication
   - Manages session lifecycle and tracking
   - Provides AI-friendly error handling with diagnostic hints
   - Tools: `start_refinement`, `continue_refinement`, `get_final_result`, `quick_refine`

2. **Incremental Engine (`src/incremental_engine.py`)**
   - Session-based refinement management
   - Implements convergence measurement using cosine similarity
   - Handles domain detection and specialized prompts
   - States: INITIALIZING → DRAFTING → CRITIQUING → REVISING → CONVERGED

3. **AWS Bedrock Integration**
   - Primary LLM: Claude models for generation
   - Critique LLM: Can use faster models (e.g., Haiku) for parallel critiques
   - Embeddings: Amazon Titan for convergence measurement

### Key Design Patterns

- **Session Management**: Auto-tracks current session ID to reduce manual management
- **Incremental Processing**: Each refinement step returns immediately to prevent timeouts
- **Parallel Critiques**: Multiple critique perspectives generated concurrently
- **Domain Optimization**: Auto-detects domain (technical/marketing/legal/financial) and adjusts prompts
- **Convergence Tracking**: Mathematical similarity measurement to determine when refinement is complete

### Environment Configuration

Critical environment variables (set in `.env` or Claude Desktop config):
- `AWS_REGION`: AWS region for Bedrock (default: us-east-1)
- `BEDROCK_MODEL_ID`: Main generation model
- `CRITIQUE_MODEL_ID`: Model for critiques (use Haiku for 50% speed improvement)
- `CONVERGENCE_THRESHOLD`: Similarity threshold 0.90-0.99 (default: 0.98)
- `PARALLEL_CRITIQUES`: Number of parallel critiques (default: 3)
- `MAX_ITERATIONS`: Maximum refinement iterations (default: 10)

### Error Handling Philosophy

The server provides AI-assistant-friendly error responses with:
- `_ai_diagnosis`: What went wrong
- `_ai_actions`: Steps the AI can take
- `_ai_suggestion`: Alternative approaches
- `_human_action`: What the human user should do

### Testing Strategy

- Unit tests for core refinement logic
- Integration tests for MCP server endpoints
- Mock AWS Bedrock responses using moto
- Test domain detection and convergence measurement
