# Contributing to Recursive Companion MCP

Thank you for your interest in contributing to Recursive Companion MCP! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/recursive-companion-mcp.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Submit a pull request

## Development Setup

1. Install [uv](https://github.com/astral-sh/uv) if you haven't already
2. Install dependencies: `uv sync`
3. Set up pre-commit hooks: `uv run pre-commit install`

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Keep line length under 100 characters

## Testing

- Write tests for new features
- Ensure all tests pass: `uv run pytest`
- Maintain or improve code coverage

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the CHANGELOG.md with your changes
3. Ensure all tests pass
4. Request review from maintainers

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include as much detail as possible:
  - Steps to reproduce
  - Expected behavior
  - Actual behavior
  - Environment details

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Assume good intentions

## Questions?

Feel free to open an issue for any questions about contributing!
