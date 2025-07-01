#!/bin/bash
# Script to add GitHub remote and push to origin

echo "🚀 Recursive Companion MCP - GitHub Setup"
echo "========================================"
echo ""
echo "This script will help you push to your GitHub repository."
echo ""
echo "Please enter your GitHub username:"
read github_username

echo ""
echo "Please enter your repository name (default: recursive-companion-mcp):"
read repo_name
repo_name=${repo_name:-recursive-companion-mcp}

# Add the remote
git remote add origin "https://github.com/${github_username}/${repo_name}.git"

echo ""
echo "Remote added. To push your code, run:"
echo "  git push -u origin main"
echo ""
echo "Make sure you've created the repository on GitHub first!"
echo "https://github.com/new"
