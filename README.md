# Recursive Companion MCP

An MCP (Model Context Protocol) server that implements iterative refinement through self-critique cycles. Inspired by the sequential-thinking pattern, this tool enables AI models to improve their responses through multiple rounds of critique and revision.

## Features

- **Incremental Refinement**: Avoids timeouts by breaking refinement into discrete steps
- **Mathematical Convergence**: Uses cosine similarity to measure when refinement is complete
- **Domain-Specific Optimization**: Auto-detects and optimizes for technical, marketing, strategy, legal, and financial domains
- **Progress Visibility**: Each step returns immediately, allowing UI updates
- **Parallel Sessions**: Support for multiple concurrent refinement sessions

## How It Works

The refinement process follows a **Draft → Critique → Revise → Converge** pattern:

1. **Draft**: Generate initial response
2. **Critique**: Create multiple parallel critiques (using faster models)
3. **Revise**: Synthesize critiques into improved version
4. **Converge**: Measure similarity and repeat until threshold reached

## Installation

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- AWS Account with Bedrock access
- Claude Desktop app

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/recursive-companion-mcp.git
cd recursive-companion-mcp
```

2. Install dependencies:
```bash
uv sync
```

3. Configure AWS credentials as environment variables or through AWS CLI

4. Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "recursive-companion": {
      "command": "/path/to/recursive-companion-mcp/run_server.sh",
      "env": {
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "your-key",
        "AWS_SECRET_ACCESS_KEY": "your-secret",
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

## Usage

The tool provides several MCP endpoints:

### Start a refinement session
```
Use start_refinement to refine: "Explain the key principles of secure API design"
```

### Continue refinement step by step
```
Use continue_refinement with session_id "abc123..."
```

### Get final result
```
Use get_final_result with session_id "abc123..."
```

### Other tools
- `get_refinement_status` - Check progress without advancing
- `list_refinement_sessions` - See all active sessions

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

## Attribution

This project is inspired by the sequential-thinking pattern demonstrated in various MCP implementations. The incremental approach was developed to solve timeout issues while maintaining refinement quality.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for the [Model Context Protocol](https://github.com/anthropics/mcp)
- Uses AWS Bedrock for LLM access
- Inspired by iterative refinement patterns in AI reasoning
