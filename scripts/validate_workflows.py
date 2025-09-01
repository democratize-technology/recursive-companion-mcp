#!/usr/bin/env python3
"""
Workflow validation script to ensure GitHub Actions workflows are properly configured.
"""

import json
from pathlib import Path
from typing import Any

import yaml


def validate_workflow_file(workflow_path: Path) -> dict[str, Any]:
    """Validate a GitHub Actions workflow file."""
    results = {
        "file": str(workflow_path),
        "valid": False,
        "errors": [],
        "warnings": [],
        "info": [],
    }

    try:
        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)

        # Basic structure validation
        # Note: 'on' keyword in YAML is parsed as boolean True, so we check for both
        required_keys = ["name", "jobs"]
        on_key = True if True in workflow else "on" if "on" in workflow else None

        for key in required_keys:
            if key not in workflow:
                results["errors"].append(f"Missing required key: {key}")

        if on_key is None:
            results["errors"].append("Missing required key: on (trigger configuration)")
        else:
            results["info"].append("Trigger configuration found")

        # Analyze trigger configuration (handle YAML 'on' parsing as True)
        triggers = workflow.get(True) or workflow.get("on")
        if triggers and isinstance(triggers, dict):
            if "push" in triggers and "branches" in triggers.get("push", {}):
                results["info"].append(f"Push triggers: {triggers['push']['branches']}")
            if "pull_request" in triggers:
                results["info"].append("Pull request triggers configured")
            if "schedule" in triggers:
                results["info"].append("Scheduled triggers configured")
            if "workflow_dispatch" in triggers:
                results["info"].append("Manual dispatch configured")

        # Job validation
        if "jobs" in workflow:
            job_count = len(workflow["jobs"])
            results["info"].append(f"Jobs defined: {job_count}")

            for job_name, job_config in workflow["jobs"].items():
                if "runs-on" not in job_config:
                    results["errors"].append(f"Job '{job_name}' missing 'runs-on'")

                if "steps" not in job_config:
                    results["warnings"].append(f"Job '{job_name}' has no steps")

        # Security checks
        if "permissions" in workflow:
            results["info"].append("Permissions explicitly defined")
        else:
            results["warnings"].append("No permissions defined - using defaults")

        # Check for secrets usage
        workflow_str = yaml.dump(workflow)
        if "${{ secrets." in workflow_str:
            results["info"].append("Uses repository secrets")

        if not results["errors"]:
            results["valid"] = True

    except yaml.YAMLError as e:
        results["errors"].append(f"YAML parsing error: {e}")
    except Exception as e:
        results["errors"].append(f"Unexpected error: {e}")

    return results


def validate_dependabot_config(config_path: Path) -> dict[str, Any]:
    """Validate Dependabot configuration."""
    results = {
        "file": str(config_path),
        "valid": False,
        "errors": [],
        "warnings": [],
        "info": [],
    }

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check version
        if config.get("version") != 2:
            results["errors"].append("Dependabot config must use version 2")

        # Check updates configuration
        if "updates" not in config:
            results["errors"].append("Missing 'updates' section")
        else:
            updates = config["updates"]
            ecosystems = [update.get("package-ecosystem") for update in updates]
            results["info"].append(f"Configured ecosystems: {ecosystems}")

            for update in updates:
                ecosystem = update.get("package-ecosystem")
                if not ecosystem:
                    results["errors"].append("Update missing package-ecosystem")

                if "directory" not in update:
                    results["errors"].append(f"Update for {ecosystem} missing directory")

                if "schedule" not in update:
                    results["warnings"].append(f"Update for {ecosystem} missing schedule")

        if not results["errors"]:
            results["valid"] = True

    except yaml.YAMLError as e:
        results["errors"].append(f"YAML parsing error: {e}")
    except Exception as e:
        results["errors"].append(f"Unexpected error: {e}")

    return results


def validate_semantic_release_config(config_path: Path) -> dict[str, Any]:
    """Validate semantic release configuration."""
    results = {
        "file": str(config_path),
        "valid": False,
        "errors": [],
        "warnings": [],
        "info": [],
    }

    try:
        with open(config_path) as f:
            config = json.load(f)

        # Check required sections
        if "branches" not in config:
            results["warnings"].append("No branches configuration")
        else:
            branches = config["branches"]
            results["info"].append(f"Configured branches: {branches}")

        if "plugins" not in config:
            results["errors"].append("Missing plugins configuration")
        else:
            plugins = config["plugins"]
            plugin_names = []
            for plugin in plugins:
                if isinstance(plugin, str):
                    plugin_names.append(plugin)
                elif isinstance(plugin, list) and len(plugin) > 0:
                    plugin_names.append(plugin[0])
            results["info"].append(f"Configured plugins: {plugin_names}")

        if not results["errors"]:
            results["valid"] = True

    except json.JSONDecodeError as e:
        results["errors"].append(f"JSON parsing error: {e}")
    except Exception as e:
        results["errors"].append(f"Unexpected error: {e}")

    return results


def main():
    """Main validation function."""
    repo_root = Path(__file__).parent.parent
    print(f"🔍 Validating automation configuration in {repo_root}")
    print("=" * 60)

    all_valid = True

    # Validate GitHub Actions workflows
    workflows_dir = repo_root / ".github" / "workflows"
    if workflows_dir.exists():
        print("\n📋 GitHub Actions Workflows:")
        for workflow_file in workflows_dir.glob("*.yml"):
            result = validate_workflow_file(workflow_file)
            status = "✅" if result["valid"] else "❌"
            print(f"{status} {workflow_file.name}")

            if result["errors"]:
                for error in result["errors"]:
                    print(f"   ❌ {error}")
                all_valid = False

            if result["warnings"]:
                for warning in result["warnings"]:
                    print(f"   ⚠️  {warning}")

            if result["info"]:
                for info in result["info"][:2]:  # Limit info output
                    print(f"   ℹ️  {info}")

    # Validate Dependabot configuration
    dependabot_file = repo_root / ".github" / "dependabot.yml"
    if dependabot_file.exists():
        print("\n🤖 Dependabot Configuration:")
        result = validate_dependabot_config(dependabot_file)
        status = "✅" if result["valid"] else "❌"
        print(f"{status} dependabot.yml")

        if result["errors"]:
            for error in result["errors"]:
                print(f"   ❌ {error}")
            all_valid = False

        if result["warnings"]:
            for warning in result["warnings"]:
                print(f"   ⚠️  {warning}")

        if result["info"]:
            for info in result["info"][:2]:
                print(f"   ℹ️  {info}")

    # Validate semantic release configuration
    semantic_release_file = repo_root / ".releaserc.json"
    if semantic_release_file.exists():
        print("\n🚀 Semantic Release Configuration:")
        result = validate_semantic_release_config(semantic_release_file)
        status = "✅" if result["valid"] else "❌"
        print(f"{status} .releaserc.json")

        if result["errors"]:
            for error in result["errors"]:
                print(f"   ❌ {error}")
            all_valid = False

        if result["warnings"]:
            for warning in result["warnings"]:
                print(f"   ⚠️  {warning}")

        if result["info"]:
            for info in result["info"][:2]:
                print(f"   ℹ️  {info}")

    # Summary
    print("\n" + "=" * 60)
    if all_valid:
        print("🎉 All automation configurations are valid!")
        print("\n📋 Next Steps:")
        print("1. Commit and push these configuration files")
        print("2. Configure required repository secrets (PYPI_API_TOKEN)")
        print("3. Set up branch protection rules")
        print("4. Test workflows with a sample PR")
    else:
        print("❌ Some configurations have errors that need to be fixed.")
        print("Please review the errors above and correct them.")

    return 0 if all_valid else 1


if __name__ == "__main__":
    exit(main())
