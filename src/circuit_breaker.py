"""
Circuit breaker implementation for handling AWS API failures gracefully.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open before closing
    timeout: float = 60.0  # Seconds before trying half-open
    failure_rate_threshold: float = 0.5  # Failure rate to open (50%)
    min_calls: int = 10  # Minimum calls before evaluating failure rate

    # Specific error types that trigger the breaker
    tracked_exceptions: tuple = (Exception,)  # Track all by default

    # Exceptions that should bypass the circuit breaker
    excluded_exceptions: tuple = (
        KeyboardInterrupt,
        SystemExit,
        asyncio.CancelledError,
    )


@dataclass
class CircuitBreakerStats:
    """Statistics for monitoring circuit breaker behavior."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    circuit_opens: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: list = field(default_factory=list)

    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    def reset_consecutive_counts(self):
        """Reset consecutive counters when state changes."""
        self.consecutive_failures = 0
        self.consecutive_successes = 0


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are blocked
    - HALF_OPEN: Testing recovery, limited requests allowed
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker
            config: Configuration settings
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._state_lock = asyncio.Lock()
        self._half_open_lock = asyncio.Lock()
        self._last_state_change = time.time()

    async def _should_open_circuit(self) -> bool:
        """Determine if circuit should open based on failures."""
        # Check consecutive failure threshold
        if self.stats.consecutive_failures >= self.config.failure_threshold:
            return True

        # Check failure rate if we have enough calls
        if self.stats.total_calls >= self.config.min_calls:
            if self.stats.failure_rate() >= self.config.failure_rate_threshold:
                return True

        return False

    async def _should_close_circuit(self) -> bool:
        """Determine if circuit should close from half-open state."""
        return self.stats.consecutive_successes >= self.config.success_threshold

    async def _can_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.stats.last_failure_time is None:
            return True

        time_since_failure = time.time() - self.stats.last_failure_time
        return time_since_failure >= self.config.timeout

    async def _change_state(self, new_state: CircuitState):
        """Change circuit state and log the transition."""
        async with self._state_lock:
            old_state = self.state
            if old_state != new_state:
                self.state = new_state
                self._last_state_change = time.time()
                self.stats.state_changes.append(
                    {
                        "from": old_state.value,
                        "to": new_state.value,
                        "timestamp": self._last_state_change,
                        "stats": {
                            "total_calls": self.stats.total_calls,
                            "failure_rate": self.stats.failure_rate(),
                            "consecutive_failures": self.stats.consecutive_failures,
                        },
                    }
                )

                if new_state == CircuitState.OPEN:
                    self.stats.circuit_opens += 1

                logger.warning(
                    f"Circuit breaker '{self.name}' state change: "
                    f"{old_state.value} -> {new_state.value} "
                    f"(failures: {self.stats.consecutive_failures}, "
                    f"rate: {self.stats.failure_rate():.2%})"
                )

                # Reset consecutive counts on state change
                if new_state == CircuitState.CLOSED:
                    self.stats.reset_consecutive_counts()

    async def call(
        self, func: Callable[..., T], *args, fallback: Optional[Callable[..., T]] = None, **kwargs
    ) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            fallback: Optional fallback function if circuit is open
            **kwargs: Keyword arguments for func

        Returns:
            Result from func or fallback

        Raises:
            CircuitBreakerOpenError: If circuit is open and no fallback
            Original exception: If circuit is closed and func fails
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if await self._can_attempt_reset():
                await self._change_state(CircuitState.HALF_OPEN)
            else:
                # Circuit is open and timeout hasn't expired
                if fallback:
                    logger.debug(f"Circuit breaker '{self.name}' is OPEN, using fallback")
                    return await self._execute_function(fallback, *args, **kwargs)
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Service unavailable for {self.config.timeout - (time.time() - self.stats.last_failure_time):.1f}s"
                    )

        # Handle half-open state with limited concurrency
        if self.state == CircuitState.HALF_OPEN:
            # Only allow one test call at a time in half-open state
            if self._half_open_lock.locked():
                if fallback:
                    return await self._execute_function(fallback, *args, **kwargs)
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is testing recovery"
                    )

        # Try to execute the function
        try:
            # Use lock for half-open state
            if self.state == CircuitState.HALF_OPEN:
                async with self._half_open_lock:
                    result = await self._execute_function(func, *args, **kwargs)
            else:
                result = await self._execute_function(func, *args, **kwargs)

            # Record success
            await self._record_success()
            return result

        except self.config.excluded_exceptions:
            # Don't track excluded exceptions
            raise

        except self.config.tracked_exceptions as e:
            # Record failure
            await self._record_failure()

            # Use fallback if available
            if fallback and self.state == CircuitState.OPEN:
                logger.debug(f"Using fallback due to error: {e}")
                return await self._execute_function(fallback, *args, **kwargs)

            raise

    async def _execute_function(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function handling both sync and async."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in executor to not block
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)

    async def _record_success(self):
        """Record successful call and update state if needed."""
        self.stats.total_calls += 1
        self.stats.successful_calls += 1
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0
        self.stats.last_success_time = time.time()

        # Check if we should close circuit from half-open
        if self.state == CircuitState.HALF_OPEN:
            if await self._should_close_circuit():
                await self._change_state(CircuitState.CLOSED)

    async def _record_failure(self):
        """Record failed call and update state if needed."""
        self.stats.total_calls += 1
        self.stats.failed_calls += 1
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.last_failure_time = time.time()

        # Check if we should open circuit
        if self.state == CircuitState.CLOSED:
            if await self._should_open_circuit():
                await self._change_state(CircuitState.OPEN)
        elif self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test, reopen circuit
            await self._change_state(CircuitState.OPEN)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics and state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "stats": {
                "total_calls": self.stats.total_calls,
                "successful_calls": self.stats.successful_calls,
                "failed_calls": self.stats.failed_calls,
                "failure_rate": f"{self.stats.failure_rate():.2%}",
                "consecutive_failures": self.stats.consecutive_failures,
                "consecutive_successes": self.stats.consecutive_successes,
                "circuit_opens": self.stats.circuit_opens,
                "time_since_last_failure": (
                    time.time() - self.stats.last_failure_time
                    if self.stats.last_failure_time
                    else None
                ),
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
                "failure_rate_threshold": f"{self.config.failure_rate_threshold:.0%}",
            },
        }

    async def reset(self):
        """Manually reset the circuit breaker to closed state."""
        await self._change_state(CircuitState.CLOSED)
        self.stats = CircuitBreakerStats()
        logger.info(f"Circuit breaker '{self.name}' manually reset")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different services."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    async def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()


# Global circuit breaker manager instance
circuit_manager = CircuitBreakerManager()
