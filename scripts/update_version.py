#!/usr/bin/env python3
"""
Version update script for semantic release.
Updates the version in pyproject.toml during automated releases.
"""

import sys
import tomllib
from pathlib import Path

import tomli_w


def update_version(new_version: str) -> None:
    """Update version in pyproject.toml file."""
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")

    # Read current pyproject.toml
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    # Update version
    old_version = data.get("project", {}).get("version", "unknown")
    data["project"]["version"] = new_version

    # Write back to file
    with open(pyproject_path, "wb") as f:
        tomli_w.dump(data, f)

    print(f"✅ Updated version from {old_version} to {new_version}")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <new_version>")
        sys.exit(1)

    new_version = sys.argv[1]

    # Validate version format (basic check)
    if not new_version.replace(".", "").replace("-", "").replace("+", "").isalnum():
        print(f"❌ Invalid version format: {new_version}")
        sys.exit(1)

    try:
        update_version(new_version)
        print(f"🎉 Version successfully updated to {new_version}")
    except Exception as e:
        print(f"❌ Error updating version: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
