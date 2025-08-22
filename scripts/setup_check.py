#!/usr/bin/env python3
"""
Setup verification script for automation infrastructure.
Checks that all required components are in place for OSS readiness.
"""

from pathlib import Path


def check_file_exists(file_path: Path, description: str) -> tuple[bool, str]:
    """Check if a file exists and return status."""
    if file_path.exists():
        return True, f"✅ {description}"
    else:
        return False, f"❌ {description} - File missing: {file_path}"


def check_directory_exists(dir_path: Path, description: str) -> tuple[bool, str]:
    """Check if a directory exists and return status."""
    if dir_path.exists() and dir_path.is_dir():
        return True, f"✅ {description}"
    else:
        return False, f"❌ {description} - Directory missing: {dir_path}"


def main():
    """Main setup verification function."""
    repo_root = Path(__file__).parent.parent
    print("🔍 Recursive Companion MCP - Setup Verification")
    print("=" * 60)

    checks = []

    # Core workflow files
    print("\n📋 GitHub Actions Workflows:")
    workflow_files = [
        (".github/workflows/ci.yml", "Continuous Integration"),
        (".github/workflows/release.yml", "Automated Release"),
        (".github/workflows/security.yml", "Security Monitoring"),
        (".github/workflows/quality-gate.yml", "Quality Gate"),
        (".github/workflows/dependency-update.yml", "Dependency Updates"),
    ]

    for file_path, description in workflow_files:
        status, message = check_file_exists(repo_root / file_path, description)
        checks.append(status)
        print(f"  {message}")

    # Configuration files
    print("\n⚙️ Configuration Files:")
    config_files = [
        (".github/dependabot.yml", "Dependabot Configuration"),
        (".releaserc.json", "Semantic Release Configuration"),
        ("pyproject.toml", "Python Project Configuration"),
        ("AUTOMATION.md", "Automation Documentation"),
    ]

    for file_path, description in config_files:
        status, message = check_file_exists(repo_root / file_path, description)
        checks.append(status)
        print(f"  {message}")

    # Script files
    print("\n🔧 Automation Scripts:")
    script_files = [
        ("scripts/update_version.py", "Version Update Script"),
        ("scripts/validate_workflows.py", "Workflow Validation Script"),
        ("scripts/setup_check.py", "Setup Verification Script"),
    ]

    for file_path, description in script_files:
        status, message = check_file_exists(repo_root / file_path, description)
        checks.append(status)
        print(f"  {message}")

    # Directories
    print("\n📁 Directory Structure:")
    directories = [
        (".github/workflows", "GitHub Actions Workflows Directory"),
        (".github/ISSUE_TEMPLATE", "Issue Templates Directory"),
        ("scripts", "Automation Scripts Directory"),
        ("src", "Source Code Directory"),
        ("tests", "Test Suite Directory"),
    ]

    for dir_path, description in directories:
        status, message = check_directory_exists(repo_root / dir_path, description)
        checks.append(status)
        print(f"  {message}")

    # Dependency checks
    print("\n📦 Python Dependencies:")
    try:
        import tomllib

        with open(repo_root / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        # Check for release dependencies
        release_deps = (
            pyproject.get("project", {}).get("optional-dependencies", {}).get("release", [])
        )
        if release_deps:
            print(f"  ✅ Release dependencies configured: {len(release_deps)} packages")
            checks.append(True)
        else:
            print("  ❌ Release dependencies not configured")
            checks.append(False)

        # Check for dev dependencies
        dev_deps = pyproject.get("project", {}).get("optional-dependencies", {}).get("dev", [])
        if dev_deps:
            print(f"  ✅ Dev dependencies configured: {len(dev_deps)} packages")
            checks.append(True)
        else:
            print("  ❌ Dev dependencies not configured")
            checks.append(False)

    except Exception as e:
        print(f"  ❌ Error reading pyproject.toml: {e}")
        checks.append(False)

    # Environment checks
    print("\n🌍 Environment Setup:")

    # Check for uv installation
    try:
        import subprocess

        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ UV package manager: {result.stdout.strip()}")
            checks.append(True)
        else:
            print("  ❌ UV package manager not found")
            checks.append(False)
    except FileNotFoundError:
        print("  ❌ UV package manager not installed")
        checks.append(False)

    # Check for git repository
    if (repo_root / ".git").exists():
        print("  ✅ Git repository initialized")
        checks.append(True)
    else:
        print("  ❌ Git repository not initialized")
        checks.append(False)

    # Summary
    print("\n" + "=" * 60)
    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print("🎉 Setup Complete! All automation infrastructure is in place.")
        print(f"✅ {passed}/{total} checks passed")

        print("\n📋 Next Steps:")
        print("1. 🔐 Configure GitHub repository secrets:")
        print("   - PYPI_API_TOKEN (for PyPI publishing)")
        print("   - CODECOV_TOKEN (optional, for coverage)")
        print("")
        print("2. 🛡️ Set up branch protection rules:")
        print("   - Protect 'main' and 'develop' branches")
        print("   - Require status checks to pass")
        print("   - Require up-to-date branches")
        print("")
        print("3. 🧪 Test the setup:")
        print("   - Create a test PR to trigger workflows")
        print("   - Verify all quality gates pass")
        print("   - Test semantic release with a commit")
        print("")
        print("4. 🤖 Monitor automation:")
        print("   - Check Dependabot PRs are created")
        print("   - Verify security scanning reports")
        print("   - Monitor release automation")

        return 0
    else:
        print(f"❌ Setup Incomplete: {passed}/{total} checks passed")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())
