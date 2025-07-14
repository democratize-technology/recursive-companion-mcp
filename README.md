# Recursive Companion MCP

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

## Tool-Based Refinement (NEW)

The latest version supports **structured refinement using the AWS Bedrock Converse API** with custom tools:

### Refinement Tools

1. **identify_weakness**: Analyzes text for specific weaknesses (clarity, accuracy, completeness, coherence, depth)
2. **propose_revision**: Suggests targeted improvements for identified issues
3. **measure_improvement**: Compares versions to quantify progress
4. **check_convergence**: Determines if further refinement would add value

### Benefits

- **Structured Reasoning**: Tool calls make the refinement logic explicit and trackable
- **Better Convergence Detection**: Combines traditional cosine similarity with tool-based assessment
- **Detailed Insights**: Each tool call provides specific feedback about what needs improvement
- **Future-Proof**: Ready for advanced model features like exposed reasoning traces

The tool-based approach is automatically used when available, with fallback to traditional methods for compatibility.

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
