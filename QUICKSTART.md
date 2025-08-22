# Quick Start - Claude Desktop Integration

## 1. Run Setup Script

```bash
cd /path/to/recursive-companion-mcp

# First time setup with uv:
uv sync

# Or run the setup script:
./setup.sh

# Verify the environment:
uv run python check_env.py
```

## ✅ Server Status

The server is now working! It currently returns placeholder responses while we implement the full refinement logic.

To test in Claude Desktop:
1. Update your config with the configuration below
2. Restart Claude Desktop
3. Ask me to "use the refine_answer tool to explain quantum computing"

## 2. Configure Claude Desktop

Add to `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "recursive-companion": {
      "command": "uv",
      "args": ["run", "python", "src/server.py"],
      "cwd": "/path/to/recursive-companion-mcp",
      "env": {
        "AWS_REGION": "us-east-1",
        "BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1:0"
      }
    }
  }
}
```

**Alternative - Direct Python path (simpler):**
```json
{
  "mcpServers": {
    "recursive-companion": {
      "command": "/path/to/recursive-companion-mcp/.venv/bin/python",
      "args": ["/path/to/recursive-companion-mcp/src/server.py"],
      "env": {
        "AWS_REGION": "us-east-1",
        "BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1:0"
      }
    }
  }
}
```

**Note**: If Claude Desktop can't find `uv`, you can use the full path `/usr/local/bin/uv` or use `python3` instead.

**Alternative configuration using system Python:**
```json
{
  "mcpServers": {
    "recursive-companion": {
      "command": "/path/to/recursive-companion-mcp/.venv/bin/python",
      "args": ["src/server.py"],
      "cwd": "/path/to/recursive-companion-mcp",
      "env": {
        "AWS_REGION": "us-east-1",
        "BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1:0"
      }
    }
  }
}
```

## 3. Restart Claude Desktop

After saving the config, restart Claude Desktop.

## 4. Test It Out

In Claude Desktop, try:
- "Use the refine_answer tool to explain quantum computing for executives"
- "Use refine_answer with marketing domain to create a product launch announcement"
- "Apply refine_answer to draft a technical specification for a REST API"

## Troubleshooting

### "ModuleNotFoundError: No module named 'boto3'" 
This means the dependencies aren't being found. Make sure:
1. You've run `uv sync` in the project directory
2. Your config includes the `"cwd"` parameter pointing to the project directory
3. The path in `"cwd"` is the absolute path to the project root

### "spawn python ENOENT" or "spawn uv ENOENT" error
- If using `uv`: Make sure uv is installed: `which uv`
- If using `python3`: Make sure Python 3 is installed: `which python3`
- Update your Claude Desktop config to use the correct command

### AWS Bedrock Access Denied
If you see "AWS Bedrock initialization failed":
1. Check that your AWS account has Bedrock access enabled
2. Go to AWS Console → Bedrock → Model access
3. Request access to Claude and Titan models
4. Wait for approval (usually instant for Claude, may take time for others)

### Tool doesn't appear
1. Check AWS credentials: `aws sts get-caller-identity`
2. Check Claude Desktop logs: `tail -f ~/.config/claude/logs/*.log`
3. Try restarting Claude Desktop after configuration changes

## Available Models

You can change the model in the config:
- Claude 3 Opus: `anthropic.claude-3-opus-20240229-v1:0`
- Claude 3 Sonnet: `anthropic.claude-3-sonnet-20240229-v1:0` (default)
- Claude 3 Haiku: `anthropic.claude-3-haiku-20240307-v1:0`
