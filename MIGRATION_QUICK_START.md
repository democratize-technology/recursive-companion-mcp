# Test Suite Migration - Quick Start Guide

## TL;DR

**Problem**: 91 tests collected, 16 import errors
**Solution**: Create `legacy/` subpackage, move 21 modules, update imports
**Time**: ~2 hours
**Risk**: LOW (no code changes)

---

## Execute Migration (5 Steps)

### Step 1: Create Legacy Subpackage (5 min)

```bash
cd /Users/jeremy/Development/hacks/thinkerz/recursive-companion-mcp

# Create directory
mkdir -p src/recursive_companion_mcp/legacy

# Move all 21 legacy modules
cd src
for f in base_cognitive.py bedrock_client.py circuit_breaker.py config.py \
         configuration_manager.py convergence.py cot_enhancement.py domains.py \
         error_handling.py incremental_engine.py internal_cot.py progress_tracker.py \
         refine_engine.py refinement_types.py security_utils.py session_manager.py \
         session_persistence.py validation.py __init__.py server.py server_legacy.py; do
    git mv "$f" recursive_companion_mcp/legacy/ 2>/dev/null || echo "Skipping $f"
done
```

### Step 2: Create Legacy __init__.py (2 min)

```bash
cat > src/recursive_companion_mcp/legacy/__init__.py << 'EOF'
"""Legacy modules for backward compatibility during FastMCP migration."""

from .bedrock_client import BedrockClient
from .incremental_engine import IncrementalRefineEngine
from .convergence import ConvergenceDetector, ConvergenceConfig, EmbeddingService
from .domains import DomainDetector
from .validation import SecurityValidator
from .session_manager import SessionTracker, RefinementIteration, RefinementResult
from .refine_engine import RefineEngine
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerManager
from .security_utils import CredentialSanitizer
from .error_handling import create_ai_error_response
from .session_persistence import SessionPersistenceManager
from .internal_cot import is_available, create_processor
from .config import config
from .base_cognitive import BaseCognitivePattern
from .cot_enhancement import CoTEnhancement
from .configuration_manager import ConfigurationManager
from .progress_tracker import ProgressTracker
from .refinement_types import RefinementStatus, RefinementSession

__all__ = [
    "BedrockClient", "IncrementalRefineEngine", "ConvergenceDetector",
    "ConvergenceConfig", "EmbeddingService", "DomainDetector",
    "SecurityValidator", "SessionTracker", "RefinementIteration",
    "RefinementResult", "RefineEngine", "CircuitBreaker",
    "CircuitBreakerConfig", "CircuitBreakerManager", "CredentialSanitizer",
    "create_ai_error_response", "SessionPersistenceManager",
    "is_available", "create_processor", "config", "BaseCognitivePattern",
    "CoTEnhancement", "ConfigurationManager", "ProgressTracker",
    "RefinementStatus", "RefinementSession",
]
EOF
```

### Step 3: Update Tool Files (10 min)

Edit `src/recursive_companion_mcp/tools/refinement.py`:

**Find** (around lines 19-31):
```python
import sys
import os

# Add src to path to import existing modules
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from bedrock_client import BedrockClient
from domains import DomainDetector
from validation import SecurityValidator
from incremental_engine import IncrementalRefineEngine
```

**Replace with**:
```python
# Import legacy modules from package structure
from ..legacy import (
    BedrockClient,
    DomainDetector,
    SecurityValidator,
    IncrementalRefineEngine,
    SessionTracker,
)
```

Repeat for:
- `tools/sessions.py`
- `tools/control.py`
- Any other tool files importing legacy modules

### Step 4: Update Test Files (60 min)

For each test file in `tests/`, update imports:

**OLD**:
```python
sys.path.insert(0, "./src")
from bedrock_client import BedrockClient
from incremental_engine import IncrementalRefineEngine
```

**NEW**:
```python
from recursive_companion_mcp.legacy.bedrock_client import BedrockClient
from recursive_companion_mcp.legacy.incremental_engine import IncrementalRefineEngine
```

**Automated approach** (use with caution):
```bash
cd tests

# For each test file
for f in test_*.py; do
    # Remove sys.path lines
    sed -i.bak 's/^sys\.path\.insert.*$/# sys.path removed - using package imports/' "$f"

    # Update common imports (add more as needed)
    sed -i.bak 's/^from bedrock_client import/from recursive_companion_mcp.legacy.bedrock_client import/' "$f"
    sed -i.bak 's/^from incremental_engine import/from recursive_companion_mcp.legacy.incremental_engine import/' "$f"
    sed -i.bak 's/^from convergence import/from recursive_companion_mcp.legacy.convergence import/' "$f"
    sed -i.bak 's/^from circuit_breaker import/from recursive_companion_mcp.legacy.circuit_breaker import/' "$f"
    sed -i.bak 's/^from validation import/from recursive_companion_mcp.legacy.validation import/' "$f"
    sed -i.bak 's/^from domains import/from recursive_companion_mcp.legacy.domains import/' "$f"
    sed -i.bak 's/^from session_manager import/from recursive_companion_mcp.legacy.session_manager import/' "$f"
    sed -i.bak 's/^from refine_engine import/from recursive_companion_mcp.legacy.refine_engine import/' "$f"
    sed -i.bak 's/^from security_utils import/from recursive_companion_mcp.legacy.security_utils import/' "$f"
    sed -i.bak 's/^from error_handling import/from recursive_companion_mcp.legacy.error_handling import/' "$f"
    sed -i.bak 's/^from session_persistence import/from recursive_companion_mcp.legacy.session_persistence import/' "$f"
    sed -i.bak 's/^from internal_cot import/from recursive_companion_mcp.legacy.internal_cot import/' "$f"
    sed -i.bak 's/^from config import/from recursive_companion_mcp.legacy.config import/' "$f"
    sed -i.bak 's/^from base_cognitive import/from recursive_companion_mcp.legacy.base_cognitive import/' "$f"
done

# Review changes before committing
```

### Step 5: Verify Everything Works (10 min)

```bash
cd /Users/jeremy/Development/hacks/thinkerz/recursive-companion-mcp

# Check test collection
uv run pytest tests/ --collect-only
# Expected: collected 91 items / 0 errors

# Run full test suite
uv run pytest tests/ -v

# Verify server starts
uv run python -m recursive_companion_mcp
```

---

## Success Checklist

- [ ] Legacy subpackage created
- [ ] 21 modules moved with git mv
- [ ] Legacy __init__.py created with re-exports
- [ ] Tool files updated (3+ files)
- [ ] Test files updated (22 files)
- [ ] Tests collect without import errors (0 errors)
- [ ] Server starts successfully
- [ ] Git commits made with clear messages

---

## Rollback If Needed

```bash
# Restore everything
git reset --hard HEAD

# Or restore just tests
cd tests && for f in *.bak; do mv "$f" "${f%.bak}"; done
```

---

## Next Steps After Migration

1. **Verify CI/CD**: Push to branch, check tests pass
2. **Document**: Update CLAUDE.md with new structure
3. **Plan Refactoring**: Schedule gradual migration from legacy/ to proper architecture
4. **Monitor**: Watch for any edge cases in production

---

## Full Details

See `TEST_MIGRATION_STRATEGY.md` for comprehensive analysis, risk assessment, and future roadmap.

---

**Quick Questions?**

**Q**: Why `legacy/` instead of proper architecture?
**A**: Minimizes risk. No code changes = no bugs. Enables gradual refactoring.

**Q**: Will tests pass after migration?
**A**: Import errors will be fixed. Pre-existing test failures tracked separately.

**Q**: Can I still use old imports during transition?
**A**: No - legacy modules are MOVED, not copied. Update all imports.

**Q**: How long until we clean up legacy/?
**A**: 3-6 months. Migrate high-frequency modules first, then deprecate.
