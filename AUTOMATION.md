# 🤖 Automation Infrastructure

This document describes the comprehensive automation infrastructure set up for the recursive-companion-mcp project to ensure OSS readiness, security, and maintainability.

## 🔄 Automated Workflows

### 1. Dependabot Configuration (`.github/dependabot.yml`)

Automated dependency updates with intelligent grouping and scheduling:

- **Schedule**: Weekly updates on specific days to avoid conflicts
  - Python dependencies: Mondays at 09:00 UTC
  - GitHub Actions: Tuesdays at 09:00 UTC  
  - Docker: Wednesdays at 09:00 UTC
- **Grouping**: Related dependencies are grouped together:
  - AWS dependencies (boto3, botocore, awscli)
  - Test dependencies (pytest, moto, coverage)
  - Dev tools (black, flake8, mypy)
  - Security dependencies (cryptography, requests, urllib3)
- **Target**: PRs created against `develop` branch
- **Limits**: Controlled PR limits to prevent spam (5 for Python, 3 for Actions)

#### Key Features:
- **Semantic versioning strategy**: Increase version appropriately
- **Rebase strategy**: Automatic rebasing for conflicting PRs
- **Review assignment**: Auto-assigns to maintenance team
- **Security prioritization**: Security updates get higher priority

### 2. Semantic Release (`.github/workflows/release.yml`)

Fully automated release process with semantic versioning:

- **Triggers**: 
  - Push to `main` branch (automatic)
  - Manual dispatch with release type selection
- **Process**:
  1. Run full test suite
  2. Perform security checks
  3. Determine version bump based on conventional commits
  4. Update version in `pyproject.toml`
  5. Generate changelog
  6. Create Git tag and GitHub release
  7. Build Python package
  8. Publish to PyPI

#### Conventional Commit Types:
- `feat:` → Minor version bump
- `fix:` → Patch version bump
- `perf:` → Patch version bump
- `BREAKING CHANGE:` → Major version bump
- `docs:`, `style:`, `test:`, `chore:` → No release

#### Manual Release Types:
- `auto`: Analyze commits automatically (default)
- `patch`: Force patch release
- `minor`: Force minor release
- `major`: Force major release

### 3. Quality Gate (`.github/workflows/quality-gate.yml`)

Comprehensive quality checks for all PRs and pushes:

#### Quality Checks:
- **Testing**: Full test suite execution with coverage analysis
- **Coverage**: Minimum 80% coverage requirement
- **Linting**: Code style and quality checks (flake8, black)
- **Type Checking**: Static type analysis with mypy
- **Security**: Multi-tool security scanning
- **Performance**: Import time benchmarks

#### PR Integration:
- **Status Comments**: Automated quality gate results on PRs
- **Blocking**: Failed checks prevent merging
- **Artifacts**: Detailed reports uploaded for investigation
- **Auto-merge**: Dependabot PRs auto-merge if all checks pass

### 4. Security Monitoring (`.github/workflows/security.yml`)

Multi-layered security scanning and monitoring:

#### Security Tools:
- **Safety**: Known vulnerability database scanning
- **Bandit**: Static security analysis for Python
- **Pip-audit**: PyPI vulnerability scanning
- **CodeQL**: GitHub's semantic code analysis
- **Trivy**: Supply chain security scanning
- **Dependency Review**: License and vulnerability analysis

#### Scheduling:
- **Daily**: Comprehensive security scans at 6 AM UTC
- **PR-triggered**: Security checks on all pull requests
- **Manual**: On-demand security analysis

### 5. Dependency Updates (`.github/workflows/dependency-update.yml`)

Advanced dependency management beyond Dependabot:

#### Features:
- **Audit Phase**: Comprehensive dependency health analysis
- **Update Types**: Configurable update strategies (patch/minor/major)
- **Testing**: Automated testing after updates
- **Security Validation**: Post-update security scanning
- **PR Creation**: Automatic PR with detailed update summary

#### Update Strategies:
- **Patch**: Security and bug fixes only
- **Minor**: Feature updates within compatibility
- **Major**: Breaking changes (manual review required)
- **All**: Latest versions (use with caution)

## 🛠️ Setup Requirements

### GitHub Repository Settings

#### Required Secrets:
```bash
# PyPI publishing (for releases)
PYPI_API_TOKEN=pypi-xxxxx

# Code coverage (optional)
CODECOV_TOKEN=xxxxx
```

#### Branch Protection Rules:
Configure for `main` and `develop` branches:
- ✅ Require status checks to pass
- ✅ Require branches to be up to date
- ✅ Include administrators
- ✅ Restrict pushes to matching branches

#### Required Status Checks:
- `Quality Checks`
- `Security Analysis` 
- `Dependency Review`
- `CodeQL Analysis`

### Local Development Setup

#### Install Release Dependencies:
```bash
# Install semantic release tools
uv sync --extra release

# Configure semantic release (one-time setup)
uv run semantic-release version --noop  # Dry run to test
```

#### Manual Release Process:
```bash
# Check what would be released
uv run semantic-release version --noop

# Force a specific release type
uv run semantic-release version --patch
uv run semantic-release version --minor  
uv run semantic-release version --major

# Full release process
uv run semantic-release version --commit --tag --push --changelog
```

### Team Configuration

#### Repository Teams:
- **recursive-companion-maintainers**: Core maintainers
  - Code review permissions
  - Release approval authority
  - Security incident response

#### Dependabot Settings:
- Update `recursive-companion-maintainers` team in dependabot.yml
- Configure auto-merge policies as needed

## 📊 Monitoring & Metrics

### Quality Metrics Tracking:
- **Coverage Trends**: Track coverage over time
- **Security Alerts**: Monitor vulnerability introductions
- **Dependency Health**: Track outdated/vulnerable dependencies
- **Release Frequency**: Monitor release cadence and automation success

### Workflow Monitoring:
- **Success Rates**: Track automation workflow success rates
- **Performance**: Monitor CI/CD execution times
- **Cost**: Track GitHub Actions minutes usage

## 🚨 Incident Response

### Failed Release Recovery:
1. Check workflow logs for failure cause
2. Fix issues locally and push to `develop`
3. Create hotfix branch if critical
4. Manual release process if automation fails

### Security Alert Response:
1. Review Dependabot/security workflow alerts
2. Assess impact and criticality
3. Apply patches via dependency updates
4. Emergency release if critical vulnerability

### Quality Gate Failures:
1. Review failed checks in PR comments
2. Download artifacts for detailed analysis
3. Fix issues and push updates
4. Re-run workflows if needed

## 🔧 Customization

### Adjusting Thresholds:
Edit workflow files to modify:
- Coverage threshold (currently 80%)
- Security severity levels
- Performance benchmarks
- Update frequencies

### Adding New Checks:
1. Add new tools to `pyproject.toml` dev dependencies
2. Integrate into quality-gate workflow
3. Update status comment generation
4. Add artifact collection

### Workflow Optimization:
- Parallel job execution for faster feedback
- Caching strategies for dependency installation
- Matrix builds for multiple Python versions
- Conditional execution based on changed files

## 📚 Resources

- [Conventional Commits](https://conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [Dependabot Configuration](https://docs.github.com/en/code-security/supply-chain-security/keeping-your-dependencies-updated-automatically/configuration-options-for-dependency-updates)
- [GitHub Actions Security](https://docs.github.com/en/actions/security-guides)
- [Python Security Best Practices](https://bandit.readthedocs.io/en/latest/)

---

This automation infrastructure provides enterprise-grade DevOps practices while maintaining simplicity and reliability. All workflows are designed to fail safely and provide clear feedback for rapid issue resolution.
