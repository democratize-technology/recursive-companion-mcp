"""
Legacy modules from flat src/ structure.

These modules are preserved here for backward compatibility during
the FastMCP migration. Future refactoring will move these into
proper submodules (clients/, engines/, utils/, etc.).

Migration tracking: See TEST_MIGRATION_STRATEGY.md
"""

# Re-export core classes needed by tools
from .bedrock_client import BedrockClient
from .incremental_engine import IncrementalRefineEngine
from .session_manager import SessionTracker

# Re-export classes needed for type checking and advanced usage
try:
    from .convergence import ConvergenceConfig, ConvergenceDetector, EmbeddingService
except ImportError:
    pass

try:
    from .domains import DomainDetector
except ImportError:
    pass

try:
    from .validation import SecurityValidator
except ImportError:
    pass

try:
    from .session_manager import RefinementIteration, RefinementResult
except ImportError:
    pass

try:
    from .refine_engine import RefineEngine
except ImportError:
    pass

try:
    from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerManager
except ImportError:
    pass

try:
    from .security_utils import CredentialSanitizer
except ImportError:
    pass

try:
    from .error_handling import create_ai_error_response
except ImportError:
    pass

try:
    from .session_persistence import SessionPersistenceManager
except ImportError:
    pass

try:
    from .internal_cot import create_processor, is_available
except ImportError:
    pass

try:
    from .config import config
except ImportError:
    pass

try:
    from .cot_enhancement import CoTEnhancement
except ImportError:
    pass

try:
    from .configuration_manager import ConfigurationManager
except ImportError:
    pass

try:
    from .progress_tracker import ProgressTracker
except ImportError:
    pass

try:
    from .refinement_types import RefinementSession, RefinementStatus
except ImportError:
    pass

__all__ = [
    "BedrockClient",
    "IncrementalRefineEngine",
    "SessionTracker",
    "ConvergenceDetector",
    "ConvergenceConfig",
    "EmbeddingService",
    "DomainDetector",
    "SecurityValidator",
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
    "CoTEnhancement",
    "ConfigurationManager",
    "ProgressTracker",
    "RefinementStatus",
    "RefinementSession",
]
