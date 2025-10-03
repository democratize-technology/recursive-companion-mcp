# Recursive Companion MCP: Test Suite Migration Strategy

## Executive Summary

**Status**: 91 tests collected, 16 collection errors due to import path issues
**Root Cause**: Tests import from legacy flat `src/` structure; new FastMCP package is `src/recursive_companion_mcp/`
**Recommended Strategy**: **Option C+ (Hybrid with Legacy Subpackage)**
**Estimated Effort**: ~2 hours for complete migration with verification
**Risk Level**: **LOW** - No code changes, only file moves and import updates

---

## 1. Test Suite Inventory

### Test Files Analysis (22 total)

| Test File | Legacy Imports | Primary Focus | Lines |
|-----------|----------------|---------------|-------|
| `test_server.py` | bedrock_client, domains, refine_engine, server, validation | MCP server handlers | ~200 |
| `test_refinement.py` | bedrock_client, domains, incremental_engine, validation | Incremental refinement engine | ~250 |
| `test_convergence.py` | convergence | Convergence detection & embeddings | ~300 |
| `test_bedrock_client_complete_coverage.py` | bedrock_client | AWS Bedrock integration | ~400 |
| `test_extracted_modules.py` | bedrock_client, config, error_handling, refine_engine, session_manager, validation | Core module units | ~150 |
| `test_mcp_handlers.py` | error_handling, incremental_engine, server | MCP protocol handlers | ~200 |
| `test_incremental_engine_extended.py` | incremental_engine | Extended engine scenarios | ~350 |
| `test_incremental_engine_surgical.py` | incremental_engine | Specific coverage gaps | ~200 |
| `test_refine_engine_coverage.py` | refine_engine | Basic refinement logic | ~180 |
| `test_convergence_coverage.py` | convergence | Convergence edge cases | ~250 |
| `test_base_cognitive.py` | base_cognitive | Cognitive patterns | ~150 |
| `test_circuit_breaker.py` | circuit_breaker | Circuit breaker patterns | ~200 |
| `test_circuit_breaker_coverage.py` | circuit_breaker | Circuit breaker edge cases | ~150 |
| `test_session_persistence.py` | session_persistence | Session storage | ~200 |
| `test_security_features.py` | security_utils, validation | Security validation | ~180 |
| `test_security_utils_coverage.py` | security_utils | Sanitization logic | ~120 |
| `test_internal_cot.py` | internal_cot | Chain-of-thought processing | ~180 |
| `test_abort_refinement.py` | incremental_engine | Abort operations | ~100 |
| `test_server_extended.py` | server | Extended server scenarios | ~250 |
| `test_server_edge_cases.py` | server | Server edge cases | ~180 |
| `test_100_percent_coverage.py` | Multiple modules | Coverage completeness | ~300 |
| `test_focused_coverage.py` | Multiple modules | Targeted coverage gaps | ~200 |

**Total Test LoC**: ~4,790 lines of test code

### Import Pattern Analysis

**Current Pattern** (18/22 test files):
```python
sys.path.insert(0, "./src")
from bedrock_client import BedrockClient
from incremental_engine import IncrementalRefineEngine
from convergence import ConvergenceDetector
```

**Working Tests** (4/22 - no legacy imports):
- `test_circuit_breaker.py` - Still fails due to `from circuit_breaker` (needs legacy/ prefix)
- `test_session_persistence.py` - Same issue
- `test_internal_cot.py` - Same issue
- `test_security_utils_coverage.py` - Same issue

**Key Insight**: Even "working" tests fail because they expect modules at top-level, not in package structure.

---

## 2. Legacy Module Inventory & Categorization

### 21 Legacy Modules in `src/`

#### HIGH-FREQUENCY (8+ test files depend on)
1. **bedrock_client.py** (15,380 bytes) - AWS Bedrock integration, caching, embeddings
2. **incremental_engine.py** (40,804 bytes) - Core refinement engine with convergence
3. **convergence.py** (10,393 bytes) - Mathematical convergence detection
4. **domains.py** (5,728 bytes) - Domain detection (technical/marketing/legal/etc.)
5. **validation.py** (2,600 bytes) - Input security validation
6. **session_manager.py** (3,588 bytes) - Session tracking and lifecycle

#### MEDIUM-FREQUENCY (4-7 test files)
7. **refine_engine.py** (8,486 bytes) - Basic refinement logic
8. **circuit_breaker.py** (13,675 bytes) - Fault tolerance patterns
9. **security_utils.py** (9,076 bytes) - Credential sanitization
10. **error_handling.py** (4,567 bytes) - AI-friendly error responses
11. **session_persistence.py** (13,541 bytes) - Session storage
12. **internal_cot.py** (7,456 bytes) - Chain-of-thought processing
13. **config.py** (4,324 bytes) - Configuration management

#### LOW-FREQUENCY (1-3 test files)
14. **base_cognitive.py** (13,023 bytes) - Cognitive patterns base
15. **cot_enhancement.py** (12,453 bytes) - Chain-of-thought enhancements
16. **configuration_manager.py** (2,049 bytes) - Config utilities
17. **progress_tracker.py** (3,480 bytes) - Progress tracking
18. **refinement_types.py** (2,142 bytes) - Type definitions

#### ALREADY MIGRATED
19. **server.py** (24,173 bytes) - Now in `core/server.py` + `tools/` split

#### SUPPORT FILES
20. **__init__.py** (1,506 bytes)
21. **server_legacy.py** (24,173 bytes) - Backup of original server

**Total Legacy Code**: ~206 KB

---

## 3. New Package Structure Analysis

### Current FastMCP Package Layout

```
src/recursive_companion_mcp/
├── __init__.py                    # Main package init with tool imports
├── __main__.py                    # Entry point
├── decorators.py                  # Custom decorators
├── formatting.py                  # Output formatting
├── core/
│   ├── __init__.py
│   └── server.py                  # MCP server instance
├── tools/                         # MCP tools (split from monolithic server.py)
│   ├── __init__.py
│   ├── refinement.py             # start_refinement, continue_refinement, get_status
│   ├── results.py                # get_final_result
│   ├── sessions.py               # list_sessions, current_session
│   ├── control.py                # abort_refinement
│   └── convenience.py            # quick_refine
└── transports/
    ├── __init__.py
    └── http_server.py            # HTTP transport implementation
```

**Key Discovery**: New package tools use **sys.path manipulation** to import legacy modules:

```python
# From tools/refinement.py lines 20-31
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from bedrock_client import BedrockClient
from domains import DomainDetector
from validation import SecurityValidator
from incremental_engine import IncrementalRefineEngine
```

**Status**: Server works in production, tests fail due to import structure mismatch.

---

## 4. Migration Strategy Options

### Option A: Test-Only Import Updates (Quick Fix)
**Approach**: Update test imports to match current sys.path pattern
**Changes**: 18 test files, ~50 import statements
**Pros**:
- Fastest (30 minutes)
- Zero risk to production server
- Tests pass immediately

**Cons**:
- Perpetuates sys.path manipulation anti-pattern
- Technical debt remains
- Future refactoring harder

**Verdict**: ❌ **NOT RECOMMENDED** - Kicks can down road

---

### Option B: Complete Module Migration (Clean Slate)
**Approach**: Move all 21 modules into proper package structure
**Target Structure**:
```
src/recursive_companion_mcp/
├── clients/
│   └── bedrock.py              # bedrock_client.py
├── engines/
│   ├── incremental.py          # incremental_engine.py
│   ├── refine.py               # refine_engine.py
│   └── convergence.py
├── validation/
│   ├── security.py             # validation.py
│   └── sanitizer.py            # security_utils.py
├── session/
│   ├── manager.py              # session_manager.py
│   └── persistence.py
├── utils/
│   ├── domains.py
│   ├── config.py
│   └── error_handling.py
└── cognitive/
    ├── base.py                 # base_cognitive.py
    ├── cot.py                  # internal_cot.py
    └── enhancement.py          # cot_enhancement.py
```

**Changes**: 21 module moves + renames, 3 tool files, 18 test files
**Pros**:
- Clean architecture
- Proper module organization
- No sys.path hacks

**Cons**:
- HIGH RISK - requires changing working production code
- Extensive testing needed (each module move could break things)
- Circular import risks
- Inter-module dependency resolution complex
- Estimated 6-8 hours with high failure risk

**Verdict**: ❌ **NOT RECOMMENDED** - Too risky for immediate needs

---

### Option C: Hybrid Legacy Subpackage (Pragmatic)
**Approach**: Create `legacy/` subpackage, move modules unchanged
**Target Structure**:
```
src/recursive_companion_mcp/
├── legacy/                      # NEW: Legacy module container
│   ├── __init__.py             # Re-exports for backward compatibility
│   ├── bedrock_client.py       # Moved unchanged
│   ├── incremental_engine.py
│   ├── convergence.py
│   └── ... (all 21 modules)
├── core/
├── tools/
└── transports/
```

**Changes**:
- Create `legacy/` directory
- Move 21 modules (git mv for history preservation)
- Update 3 tool files' import paths
- Update 18 test files' import paths
- Add backward-compatible re-exports in `__init__.py`

**Pros**:
- LOW RISK - no code changes, only moves
- Tests pass immediately
- Server keeps working (tools import from `.legacy`)
- Clear technical debt marking
- Foundation for gradual refactoring
- Git history preserved

**Cons**:
- "Legacy" naming admits technical debt
- Still requires future cleanup

**Verdict**: ✅ **RECOMMENDED** - Best risk/reward ratio

---

### Option C+: Hybrid with Re-export Layer (RECOMMENDED)
**Enhancement to Option C**: Add compatibility shim for smooth migration

**Additional Changes**:
```python
# src/recursive_companion_mcp/__init__.py
from .legacy import (
    BedrockClient,
    IncrementalRefineEngine,
    ConvergenceDetector,
    DomainDetector,
    SecurityValidator,
    # ... all legacy exports
)
```

**Benefits**:
- Tests can import from `recursive_companion_mcp.legacy.bedrock_client` OR `recursive_companion_mcp`
- Enables gradual migration (future modules can move to proper structure)
- Backward compatibility guaranteed
- Clear upgrade path documented

**Verdict**: ✅✅ **STRONGLY RECOMMENDED**

---

## 5. Recommended Migration Plan: Option C+

### Phase 1: Create Legacy Subpackage (30 minutes, LOW RISK)

**Actions**:
1. Create directory structure:
   ```bash
   mkdir -p src/recursive_companion_mcp/legacy
   ```

2. Move all 21 legacy modules (preserves git history):
   ```bash
   cd src
   git mv bedrock_client.py recursive_companion_mcp/legacy/
   git mv incremental_engine.py recursive_companion_mcp/legacy/
   git mv convergence.py recursive_companion_mcp/legacy/
   # ... repeat for all 21 modules
   ```

3. Create `src/recursive_companion_mcp/legacy/__init__.py`:
   ```python
   """
   Legacy modules from flat src/ structure.

   These modules are preserved here for backward compatibility during
   the FastMCP migration. Future refactoring will move these into
   proper submodules (clients/, engines/, utils/, etc.).

   Migration tracking: See TEST_MIGRATION_STRATEGY.md
   """

   # Re-export all legacy modules for backward compatibility
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
       "BedrockClient",
       "IncrementalRefineEngine",
       "ConvergenceDetector",
       "ConvergenceConfig",
       "EmbeddingService",
       "DomainDetector",
       "SecurityValidator",
       "SessionTracker",
       "RefinementIteration",
       "RefinementResult",
       "RefineEngine",
       "CircuitBreaker",
       "CircuitBreakerConfig",
       "CircuitBreakerManager",
       "CredentialSanitizer",
       "create_ai_error_response",
       "SessionPersistenceManager",
       "is_available",
       "create_processor",
       "config",
       "BaseCognitivePattern",
       "CoTEnhancement",
       "ConfigurationManager",
       "ProgressTracker",
       "RefinementStatus",
       "RefinementSession",
   ]
   ```

4. Update main package `__init__.py` to re-export legacy modules:
   ```python
   # Add to existing src/recursive_companion_mcp/__init__.py

   # Legacy module re-exports for backward compatibility
   from .legacy import (
       BedrockClient,
       IncrementalRefineEngine,
       ConvergenceDetector,
       DomainDetector,
       SecurityValidator,
       # ... all exports
   )
   ```

**Verification**:
```bash
# Verify modules moved successfully
ls -la src/recursive_companion_mcp/legacy/
# Should show 21 .py files

# Verify imports work
python -c "from recursive_companion_mcp.legacy import BedrockClient; print('✓ Legacy imports work')"
python -c "from recursive_companion_mcp import BedrockClient; print('✓ Re-exports work')"
```

---

### Phase 2: Update Tool Files (15 minutes, LOW RISK)

**Files to Update**: 3 tool files currently using sys.path manipulation

#### 2.1 Update `tools/refinement.py`

**OLD** (lines 19-31):
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

**NEW**:
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

#### 2.2 Update Other Tool Files

Check and update any other tool files with similar patterns:
- `tools/sessions.py`
- `tools/control.py`
- `tools/results.py`
- `tools/convenience.py`

**Verification**:
```bash
# Test server still works
cd /Users/jeremy/Development/hacks/thinkerz/recursive-companion-mcp
uv run python -m recursive_companion_mcp --help
```

---

### Phase 3: Update Test Files (45 minutes, LOW RISK)

**Bulk Update Strategy**: Use find/replace on all 18 test files

#### 3.1 Create Migration Script

```bash
cat > migrate_test_imports.sh << 'EOF'
#!/bin/bash
# Migrate test imports to use legacy subpackage

TEST_DIR="tests"

# Pattern 1: Update sys.path manipulation (remove or comment)
find "$TEST_DIR" -name "*.py" -type f -exec sed -i.bak \
  's/sys\.path\.insert(0, "\.\/src")/# Legacy sys.path removed - using package imports/' {} \;

# Pattern 2: Update imports to use recursive_companion_mcp.legacy
find "$TEST_DIR" -name "*.py" -type f -exec sed -i.bak \
  's/^from bedrock_client import/from recursive_companion_mcp.legacy.bedrock_client import/' {} \;

find "$TEST_DIR" -name "*.py" -type f -exec sed -i.bak \
  's/^from incremental_engine import/from recursive_companion_mcp.legacy.incremental_engine import/' {} \;

find "$TEST_DIR" -name "*.py" -type f -exec sed -i.bak \
  's/^from convergence import/from recursive_companion_mcp.legacy.convergence import/' {} \;

# ... repeat for all legacy modules

echo "✓ Test imports migrated"
echo "Review .bak files and remove if satisfied"
EOF

chmod +x migrate_test_imports.sh
```

#### 3.2 Manual Import Update Pattern

For each test file, update imports from:
```python
sys.path.insert(0, "./src")
from bedrock_client import BedrockClient
from incremental_engine import IncrementalRefineEngine
from convergence import ConvergenceDetector
```

To:
```python
from recursive_companion_mcp.legacy.bedrock_client import BedrockClient
from recursive_companion_mcp.legacy.incremental_engine import IncrementalRefineEngine
from recursive_companion_mcp.legacy.convergence import ConvergenceDetector
```

#### 3.3 Test Files to Update (18 total)

1. ✅ test_server.py
2. ✅ test_refinement.py
3. ✅ test_convergence.py
4. ✅ test_bedrock_client_complete_coverage.py
5. ✅ test_extracted_modules.py
6. ✅ test_mcp_handlers.py
7. ✅ test_incremental_engine_extended.py
8. ✅ test_incremental_engine_surgical.py
9. ✅ test_refine_engine_coverage.py
10. ✅ test_convergence_coverage.py
11. ✅ test_base_cognitive.py
12. ✅ test_circuit_breaker.py
13. ✅ test_circuit_breaker_coverage.py
14. ✅ test_session_persistence.py
15. ✅ test_security_features.py
16. ✅ test_security_utils_coverage.py
17. ✅ test_internal_cot.py
18. ✅ test_abort_refinement.py
19. ✅ test_server_extended.py
20. ✅ test_server_edge_cases.py
21. ✅ test_100_percent_coverage.py
22. ✅ test_focused_coverage.py

**Verification After Each Update**:
```bash
uv run pytest tests/test_circuit_breaker.py -v
# Should collect tests without import errors
```

---

### Phase 4: Full Verification (15 minutes, VERIFY)

**Run Complete Test Suite**:
```bash
cd /Users/jeremy/Development/hacks/thinkerz/recursive-companion-mcp

# Full test collection
uv run pytest tests/ --collect-only

# Expected output:
# collected 91 items / 0 errors  ← GOAL: 0 errors

# Run full test suite
uv run pytest tests/ -v

# Expected: All tests either pass or fail on their merits, not import errors
```

**Success Criteria**:
- ✅ 91 tests collected
- ✅ 0 collection errors
- ✅ All import errors resolved
- ✅ Server still starts: `uv run python -m recursive_companion_mcp`
- ✅ Tools work in MCP context

**If Tests Fail**:
- Review failure logs
- Fix any remaining import issues
- Check for circular imports
- Verify `__init__.py` exports

---

### Phase 5: Documentation & Cleanup (30 minutes)

#### 5.1 Update CLAUDE.md

Add section:
```markdown
## Package Structure (Post-Migration)

**FastMCP Package**: `src/recursive_companion_mcp/`
- `core/` - MCP server instance
- `tools/` - Individual MCP tools (split from monolithic server)
- `transports/` - HTTP and stdio transports
- `legacy/` - Modules from flat src/ structure (backward compatibility)

**Legacy Modules**: Temporarily in `legacy/` subpackage during migration.
Future refactoring will move these into proper architecture:
- `legacy/bedrock_client.py` → `clients/bedrock.py`
- `legacy/incremental_engine.py` → `engines/incremental.py`
- `legacy/convergence.py` → `engines/convergence.py`

See `TEST_MIGRATION_STRATEGY.md` for migration roadmap.
```

#### 5.2 Create Migration Tracking Document

```markdown
# Migration Status

## Phase 1: Test Suite Migration ✅ COMPLETE
- Created `legacy/` subpackage
- Updated tool imports
- Updated test imports
- All 91 tests collecting without errors

## Phase 2: Gradual Refactoring (FUTURE)
Modules to migrate from `legacy/` to proper structure:

### Priority 1: High-Frequency Modules
- [ ] bedrock_client.py → clients/bedrock.py
- [ ] incremental_engine.py → engines/incremental.py
- [ ] convergence.py → engines/convergence.py

### Priority 2: Medium-Frequency
- [ ] refine_engine.py → engines/refine.py
- [ ] validation.py → validation/security.py
- [ ] circuit_breaker.py → utils/circuit_breaker.py

### Priority 3: Low-Frequency
- [ ] All remaining modules

**Migration Process Per Module**:
1. Create new module in proper location
2. Update imports in that module only
3. Run tests - verify no breakage
4. Update tool files to import from new location
5. Add deprecation warning to legacy module
6. After 2 releases, remove legacy module
```

#### 5.3 Add Comments to Legacy Modules

Add to top of each file in `legacy/`:
```python
"""
LEGACY MODULE - Scheduled for refactoring

This module is in the legacy/ subpackage for backward compatibility.
Future location: [target path]
Migration tracking: See TEST_MIGRATION_STRATEGY.md
"""
```

---

## 6. Risk Assessment & Mitigation

### Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Import errors after migration | LOW | Medium | Test each phase incrementally |
| Circular imports in legacy/ | LOW | Low | Modules already worked in flat structure |
| Tool functionality breaks | VERY LOW | High | No code changes, only moves |
| Test failures unrelated to imports | MEDIUM | Low | Pre-existing; track separately |
| Git history loss | VERY LOW | Medium | Use `git mv` not `mv` |

### Rollback Plan

**If Phase 1 Fails**:
```bash
git reset --hard HEAD
# Restores src/ flat structure
```

**If Phase 3 Fails** (partial test migration):
```bash
# Restore test files from backups
cd tests
for f in *.py.bak; do
    mv "$f" "${f%.bak}"
done
```

**If Server Breaks**:
```bash
git checkout src/recursive_companion_mcp/tools/
# Restores tool files to previous state
```

### Testing Strategy

**Incremental Verification**:
1. After Phase 1: Test imports work
2. After Phase 2: Test server starts
3. After Phase 3 (per file): Test that file collects
4. After Phase 3 (complete): Full test suite
5. After Phase 5: Integration test with Claude Desktop

**What NOT to Fix During Migration**:
- Pre-existing test failures (track separately)
- Code quality issues in legacy modules
- Architecture improvements

**Focus**: Import errors only, everything else is follow-up.

---

## 7. Timeline & Resource Estimate

### Detailed Timeline

| Phase | Duration | Effort | Risk | Dependencies |
|-------|----------|--------|------|--------------|
| **Phase 1**: Legacy subpackage | 30 min | Easy | LOW | None |
| **Phase 2**: Tool file updates | 15 min | Easy | LOW | Phase 1 |
| **Phase 3**: Test file updates | 45 min | Medium | LOW | Phase 1-2 |
| **Phase 4**: Full verification | 15 min | Easy | VERIFY | Phase 1-3 |
| **Phase 5**: Documentation | 30 min | Easy | LOW | Phase 1-4 |
| **TOTAL** | **2h 15min** | Medium | **LOW** | Linear |

### Resource Requirements

**Human Resources**:
- 1 developer (familiar with codebase)
- No specialist knowledge required

**Technical Requirements**:
- Git (for history-preserving moves)
- Python environment with uv
- Text editor with find/replace
- Pytest for verification

**No External Dependencies**: Can execute entirely offline

---

## 8. Future Refactoring Roadmap

### Post-Migration Architecture (Target State)

```
src/recursive_companion_mcp/
├── clients/
│   └── bedrock/
│       ├── __init__.py
│       ├── client.py           # BedrockClient
│       ├── embeddings.py       # EmbeddingService
│       └── cache.py            # Caching logic
├── engines/
│   ├── __init__.py
│   ├── incremental.py          # IncrementalRefineEngine
│   ├── refine.py               # RefineEngine
│   └── convergence/
│       ├── __init__.py
│       ├── detector.py         # ConvergenceDetector
│       └── config.py           # ConvergenceConfig
├── validation/
│   ├── __init__.py
│   ├── security.py             # SecurityValidator
│   └── sanitizer.py            # CredentialSanitizer
├── session/
│   ├── __init__.py
│   ├── manager.py              # SessionTracker
│   └── persistence.py          # SessionPersistenceManager
├── utils/
│   ├── __init__.py
│   ├── domains.py              # DomainDetector
│   ├── config.py               # Configuration
│   ├── errors.py               # Error handling
│   └── circuit_breaker.py      # CircuitBreaker
├── cognitive/
│   ├── __init__.py
│   ├── base.py                 # BaseCognitivePattern
│   ├── cot.py                  # Chain-of-thought
│   └── enhancement.py          # CoTEnhancement
├── core/                       # Existing
├── tools/                      # Existing
├── transports/                 # Existing
└── legacy/                     # DEPRECATED - remove after migration
```

### Migration Phases (Post-Test Migration)

**Phase A: Extract High-Frequency Modules** (3-4 weeks)
1. Create `clients/bedrock/` package
   - Split BedrockClient into client, embeddings, cache
   - Update imports in tools
   - Run tests, verify
2. Create `engines/` package
   - Migrate incremental_engine.py
   - Migrate convergence.py
   - Update imports
3. Create `validation/` package
   - Migrate validation.py
   - Migrate security_utils.py

**Phase B: Extract Medium-Frequency** (2-3 weeks)
4. Create `session/` package
5. Create `utils/` package
6. Update all tool references

**Phase C: Extract Low-Frequency** (1-2 weeks)
7. Create `cognitive/` package
8. Final cleanup

**Phase D: Remove Legacy** (1 week)
9. Add deprecation warnings
10. Wait 2 release cycles
11. Delete `legacy/` subpackage

### Continuous Verification

**Per-Module Checklist**:
- [ ] Create new module structure
- [ ] Move code with tests
- [ ] Update imports in new module
- [ ] Update tool imports
- [ ] Run affected tests
- [ ] Update documentation
- [ ] Add deprecation warning to legacy
- [ ] Commit with clear message

**Success Metrics**:
- Zero import errors
- All tests passing
- No performance regression
- Documentation current

---

## 9. Alternative Approaches Considered

### Why Not Option A (Test-Only Updates)?

**Pros Reconsidered**:
- Would get tests passing fastest (30 min)

**Critical Cons**:
- Doesn't fix the root problem (sys.path manipulation)
- Makes future refactoring harder
- Tests become coupled to hack
- No path to proper architecture

**Decision**: Short-term gain, long-term pain

### Why Not Option B (Complete Migration)?

**Pros Reconsidered**:
- Perfect architecture immediately
- No technical debt

**Critical Cons**:
- 6-8 hour effort with HIGH risk
- Requires changing 21 working modules
- Potential circular import issues unknown
- Could break production server
- No incremental validation possible

**Decision**: Perfect is enemy of good

### Why Option C+ Is Optimal

**Balances**:
- Risk: LOW (only moves, no code changes)
- Effort: 2 hours (achievable in one session)
- Value: Unblocks tests, enables future refactoring
- Flexibility: Can migrate modules gradually

**Enables**:
- Tests pass immediately
- Server keeps working
- Clear upgrade path
- Gradual refactoring at safe pace

---

## 10. Success Criteria

### Immediate (End of Migration)

- ✅ 91 tests collected with 0 import errors
- ✅ All test files use package imports (no sys.path hacks)
- ✅ Server starts without errors
- ✅ Tools functional in MCP context
- ✅ Git history preserved for all modules
- ✅ Documentation updated

### Short-Term (1 week post-migration)

- ✅ CI/CD passing with new structure
- ✅ No regression in functionality
- ✅ Developer team trained on new imports
- ✅ Migration documentation complete

### Long-Term (3-6 months)

- ✅ High-frequency modules migrated to proper structure
- ✅ `legacy/` subpackage deprecated
- ✅ Clean architecture achieved
- ✅ Technical debt reduced

---

## 11. Appendix: Command Reference

### Quick Commands

```bash
# Create legacy subpackage
mkdir -p src/recursive_companion_mcp/legacy

# Move modules (preserves git history)
cd src
git mv bedrock_client.py recursive_companion_mcp/legacy/

# Test imports
python -c "from recursive_companion_mcp.legacy import BedrockClient"

# Run tests
uv run pytest tests/ --collect-only
uv run pytest tests/ -v

# Verify server
uv run python -m recursive_companion_mcp
```

### Git Workflow

```bash
# Create feature branch
git checkout -b migrate/test-suite-fastmcp

# After Phase 1
git add src/recursive_companion_mcp/legacy/
git commit -m "feat: create legacy subpackage for backward compatibility"

# After Phase 2
git add src/recursive_companion_mcp/tools/
git commit -m "refactor: update tool imports to use legacy subpackage"

# After Phase 3
git add tests/
git commit -m "test: update imports to use recursive_companion_mcp.legacy"

# After Phase 5
git add *.md
git commit -m "docs: update migration strategy and package structure docs"

# Merge to main
git checkout main
git merge --no-ff migrate/test-suite-fastmcp
```

---

## 12. Conclusion

**Recommendation**: **Execute Option C+ (Hybrid with Legacy Subpackage)**

**Rationale**:
- Minimal risk (no code changes)
- Maximum value (unblocks testing)
- Clear path forward (gradual refactoring)
- Achievable timeline (2 hours)

**Next Steps**:
1. Review this strategy with team
2. Get approval for 2-hour migration window
3. Execute Phases 1-5 sequentially
4. Verify success criteria
5. Plan future refactoring phases

**Questions?** See `TEST_MIGRATION_STRATEGY.md` or contact architecture team.

---

**Document Version**: 1.0
**Author**: Architect Agent
**Date**: 2025-10-03
**Status**: Ready for Implementation
