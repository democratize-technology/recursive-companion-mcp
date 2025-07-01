#!/usr/bin/env python3
"""
Quick verification script for Recursive Companion MCP
"""
import sys
import subprocess
import json

print("🔍 Verifying Recursive Companion MCP installation...\n")

# Check Python version
python_version = sys.version_info
print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version < (3, 8):
    print("  ⚠️  Python 3.8+ required")
    sys.exit(1)

# Check required packages
required_packages = [
    "mcp",
    "boto3", 
    "numpy",
    "scipy",
    "pydantic"
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"✓ {package} installed")
    except ImportError:
        print(f"✗ {package} missing")
        missing_packages.append(package)

if missing_packages:
    print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
    print("  Run: pip install -r requirements.txt")
    sys.exit(1)

# Check AWS credentials
print("\n🔍 Checking AWS configuration...")
try:
    result = subprocess.run(
        ["aws", "sts", "get-caller-identity"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        identity = json.loads(result.stdout)
        print(f"✓ AWS credentials configured")
        print(f"  Account: {identity.get('Account', 'Unknown')}")
        print(f"  User: {identity.get('Arn', 'Unknown').split('/')[-1]}")
    else:
        print("⚠️  AWS credentials not configured")
        print("  Run: aws configure")
except Exception as e:
    print(f"⚠️  Could not check AWS credentials: {e}")

print("\n✅ Basic verification complete!")
print("\nTo test the server:")
print("  python src/server.py")
print("\nFor Claude Desktop, add the configuration from QUICKSTART.md")