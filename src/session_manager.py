"""
Session management for refinement operations.
Tracks current and recent sessions for better UX.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import config

logger = logging.getLogger(__name__)


@dataclass
class RefinementIteration:
    """Represents a single iteration in the refinement process."""
    iteration_number: int
    draft: str
    critiques: List[str]
    revision: str
    convergence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    

@dataclass
class RefinementResult:
    """Complete result of the refinement process."""
    final_answer: str
    domain: str
    iterations: List[RefinementIteration]
    total_iterations: int
    convergence_achieved: bool
    execution_time: float
    metadata: Dict[str, Any]


class SessionTracker:
    """Tracks active and recent refinement sessions."""
    
    def __init__(self):
        self.current_session_id: Optional[str] = None
        self.session_history: List[Dict[str, Any]] = []
        self.max_history = 5
    
    def set_current_session(self, session_id: str, prompt: str) -> None:
        """
        Set the current active session.
        
        Args:
            session_id: The session ID
            prompt: The prompt for this session
        """
        self.current_session_id = session_id
        
        # Add to history
        preview = prompt[:config.prompt_preview_length] + '...' if len(prompt) > config.prompt_preview_length else prompt
        self.session_history.insert(0, {
            'session_id': session_id,
            'prompt_preview': preview,
            'started_at': datetime.utcnow().isoformat()
        })
        
        # Trim history
        if len(self.session_history) > self.max_history:
            self.session_history = self.session_history[:self.max_history]
    
    def get_current_session(self) -> Optional[str]:
        """Get the current session ID."""
        return self.current_session_id
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get the session history."""
        return self.session_history
    
    def clear_current_session(self) -> None:
        """Clear the current session."""
        self.current_session_id = None