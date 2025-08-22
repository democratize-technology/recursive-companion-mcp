# Recursive Companion MCP

[![CI](https://github.com/thinkerz-ai/recursive-companion-mcp/workflows/CI/badge.svg)](https://github.com/thinkerz-ai/recursive-companion-mcp/actions)
[![codecov](https://codecov.io/gh/thinkerz-ai/recursive-companion-mcp/branch/main/graph/badge.svg)](https://codecov.io/gh/thinkerz-ai/recursive-companion-mcp)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io/)

An MCP (Model Context Protocol) server that implements iterative refinement through self-critique cycles. Inspired by [Hank Besser's recursive-companion](https://github.com/hankbesser/recursive-companion), this implementation adds incremental processing to avoid timeouts and enable progress visibility.

## Features

- **Incremental Refinement**: Avoids timeouts by breaking refinement into discrete steps
- **Mathematical Convergence**: Uses cosine similarity to measure when refinement is complete
- **Domain-Specific Optimization**: Auto-detects and optimizes for technical, marketing, strategy, legal, and financial domains
- **Progress Visibility**: Each step returns immediately, allowing UI updates
- **Parallel Sessions**: Support for multiple concurrent refinement sessions
- **Auto Session Tracking**: No manual session ID management needed
- **AI-Friendly Error Handling**: Actionable diagnostics and recovery hints for AI assistants

## How It Works

The refinement process follows a **Draft → Critique → Revise → Converge** pattern:

1. **Draft**: Generate initial response
2. **Critique**: Create multiple parallel critiques (using faster models)
3. **Revise**: Synthesize critiques into improved version
4. **Converge**: Measure similarity and repeat until threshold reached

For detailed architecture diagrams and system design documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Installation

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- AWS Account with Bedrock access
- Claude Desktop app

### Setup

1. Clone the repository:
```bash
git clone https://github.com/thinkerz-ai/recursive-companion-mcp.git
cd recursive-companion-mcp
```

2. Install dependencies:
```bash
uv sync
```

3. Configure AWS credentials as environment variables or through AWS CLI

4. Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

**Basic Configuration:**
```json
{
  "mcpServers": {
    "recursive-companion": {
      "command": "/path/to/recursive-companion-mcp/run.sh",
      "env": {
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "your-access-key",
        "AWS_SECRET_ACCESS_KEY": "your-secret-key"
      }
    }
  }
}
```

**Optimized Configuration (Recommended):**
```json
{
  "mcpServers": {
    "recursive-companion": {
      "command": "/path/to/recursive-companion-mcp/run.sh",
      "env": {
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "your-access-key", 
        "AWS_SECRET_ACCESS_KEY": "your-secret-key",
        "BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1:0",
        "CRITIQUE_MODEL_ID": "anthropic.claude-3-haiku-20240307-v1:0",
        "CONVERGENCE_THRESHOLD": "0.95",
        "PARALLEL_CRITIQUES": "2",
        "MAX_ITERATIONS": "5",
        "REQUEST_TIMEOUT": "600"
      }
    }
  }
}
```

**Performance Tips:**
- Use Haiku for `CRITIQUE_MODEL_ID` for 50% speed improvement
- Lower `CONVERGENCE_THRESHOLD` to 0.95 for faster convergence
- Reduce `PARALLEL_CRITIQUES` to 2 for better resource usage
- See [API_EXAMPLES.md](API_EXAMPLES.md) for more configuration examples

## Usage

The tool provides several MCP endpoints for iterative refinement:

### Quick Start Examples

**Simple refinement (auto-complete):**
```bash
quick_refine(prompt="Explain the key principles of secure API design", max_wait=60)
```

**Step-by-step refinement (full control):**
```bash
# Start session
start_refinement(prompt="Design a microservices architecture for e-commerce", domain="technical")

# Continue iteratively
continue_refinement()  # Draft phase
continue_refinement()  # Critique phase  
continue_refinement()  # Revision phase

# Get final result
get_final_result()
```

**Session management:**
```bash
current_session()           # Check active session
list_refinement_sessions()  # List all sessions
abort_refinement()          # Stop and get best result so far
```

### Complete API Reference

For comprehensive examples with realistic scenarios, error handling patterns, and integration workflows, see **[API_EXAMPLES.md](API_EXAMPLES.md)**.

### Available Tools
- `start_refinement` - Begin new refinement session with domain detection
- `continue_refinement` - Advance session through draft→critique→revise cycles  
- `get_final_result` - Retrieve completed refinement
- `get_refinement_status` - Check progress without advancing
- `current_session` - Get active session info (no ID needed)
- `list_refinement_sessions` - See all active sessions
- `abort_refinement` - Stop refinement, return best version so far
- `quick_refine` - Auto-complete simple refinements (under 60s)

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `BEDROCK_MODEL_ID` | anthropic.claude-3-sonnet-20240229-v1:0 | Main generation model |
| `CRITIQUE_MODEL_ID` | Same as BEDROCK_MODEL_ID | Model for critiques (use Haiku for speed) |
| `CONVERGENCE_THRESHOLD` | 0.98 | Similarity threshold for convergence (0.90-0.99) |
| `PARALLEL_CRITIQUES` | 3 | Number of parallel critiques per iteration |
| `MAX_ITERATIONS` | 10 | Maximum refinement iterations |
| `REQUEST_TIMEOUT` | 300 | Timeout in seconds |

## Performance

With optimized settings:
- Each iteration: 60-90 seconds
- Typical convergence: 2-3 iterations
- Total time: 2-4 minutes (distributed across multiple calls)

Using Haiku for critiques reduces iteration time by ~50%.

## AI-Friendly Features

This tool includes special features for AI assistants using it:

- **Auto Session Tracking**: The `current_session_id` is automatically maintained
- **Smart Error Messages**: Errors include `_ai_` prefixed fields with actionable diagnostics
- **Performance Hints**: Responses include optimization tips and predictions
- **Progress Predictions**: Convergence tracking includes estimates of remaining iterations

Example AI-helpful error response:
```json
{
  "success": false,
  "error": "No session_id provided and no current session",
  "_ai_context": {
    "current_session_id": null,
    "active_session_count": 2,
    "recent_sessions": [...]
  },
  "_ai_suggestion": "Use start_refinement to create a new session",
  "_human_action": "Start a new refinement session first"
}
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Claude    │────▶│  MCP Server  │────▶│   Bedrock   │
│  Desktop    │◀────│              │◀────│   Claude    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Session    │
                    │   Manager    │
                    └──────────────┘
```

## Development

### Running tests
```bash
uv run pytest tests/
```

### Local testing
```bash
uv run python test_incremental.py
```

### Automation Infrastructure

This project includes comprehensive automation for OSS readiness:

- **🤖 Dependabot**: Automated dependency updates with intelligent grouping
- **🚀 Semantic Release**: Automated versioning and releases based on conventional commits
- **🔒 Security Monitoring**: Multi-tool security scanning (Safety, Bandit, CodeQL, Trivy)
- **✅ Quality Gates**: Automated testing, coverage, linting, and type checking
- **📦 Dependency Management**: Advanced dependency health monitoring and updates

#### Automation Commands

```bash
# Verify automation setup
uv run python scripts/setup_check.py

# Validate workflow configurations
uv run python scripts/validate_workflows.py

# Manual release (if needed)
uv run semantic-release version --noop  # dry run
uv run semantic-release version --minor  # actual release
```

#### Development Workflow

1. **Feature Development**: Work on feature branches
2. **Pull Requests**: Quality gates run automatically
3. **Code Review**: Automated security and quality feedback
4. **Merge to develop**: Beta releases created automatically
5. **Merge to main**: Production releases created automatically

See [AUTOMATION.md](AUTOMATION.md) for complete automation documentation.

## Attribution

This project is inspired by [recursive-companion](https://github.com/hankbesser/recursive-companion) by Hank Besser. The original implementation provided the conceptual Draft → Critique → Revise → Converge pattern. This MCP version adds:

- Session-based incremental processing to avoid timeouts
- AWS Bedrock integration for Claude and Titan embeddings
- Domain auto-detection and specialized prompts
- Mathematical convergence measurement
- Support for different models for critiques vs generation

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original concept: [Hank Besser's recursive-companion](https://github.com/hankbesser/recursive-companion)
- Built for the [Model Context Protocol](https://github.com/anthropics/mcp)
- Uses AWS Bedrock for LLM access
- Inspired by iterative refinement patterns in AI reasoning
