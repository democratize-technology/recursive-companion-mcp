#!/bin/bash
# Setup script for Recursive Companion MCP

echo "🚀 Setting up Recursive Companion MCP..."

# Check if uv is installed
if command -v uv &> /dev/null; then
    echo "✓ Found uv"
    echo "📦 Installing dependencies with uv..."
    uv sync --dev
else
    echo "⚠️  uv not found, using standard pip with pyproject.toml"

    # Check Python version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✓ Found Python $python_version"

    # Create virtual environment
    echo "📦 Creating virtual environment..."
    python3 -m venv venv

    # Activate virtual environment
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate

    # Upgrade pip
    echo "📦 Upgrading pip..."
    pip install --upgrade pip

    # Install project in development mode
    echo "📦 Installing project and dependencies..."
    pip install -e ".[dev]"
fi

# Copy environment file
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
else
    echo "✓ .env file already exists"
fi

# Check AWS credentials
echo "🔍 Checking AWS credentials..."
if aws sts get-caller-identity &>/dev/null; then
    echo "✓ AWS credentials configured"
    region=$(aws configure get region || echo "us-east-1")
    echo "  Region: $region"
else
    echo "⚠️  AWS credentials not configured"
    echo "  Run 'aws configure' to set up your credentials"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
if command -v uv &> /dev/null; then
    echo "1. Configure AWS if needed: aws configure"
    echo "2. Edit .env file if you want to change models"
    echo "3. Run the server: uv run src/server.py"
else
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Configure AWS if needed: aws configure"
    echo "3. Edit .env file if you want to change models"
    echo "4. Run the server: python src/server.py"
fi
echo ""
echo "For Claude Desktop integration, add this to your config:"
echo "  ~/.config/claude/claude_desktop_config.json"
echo ""
echo "Happy refining! 🎉"
