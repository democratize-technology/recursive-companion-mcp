"""
Tests for circuit breaker implementation.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch

from circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerManager,
    CircuitState,
    circuit_manager,
)


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout == 60.0
        assert config.failure_rate_threshold == 0.5
        assert config.min_calls == 10

    def test_custom_config(self):
        """Test custom configuration"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout=30.0,
            failure_rate_threshold=0.3,
            min_calls=5,
        )

        assert config.failure_threshold == 3
        assert config.success_threshold == 1
        assert config.timeout == 30.0
        assert config.failure_rate_threshold == 0.3
        assert config.min_calls == 5


class TestCircuitBreakerStates:
    """Test circuit breaker state transitions"""

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self):
        """Test that circuit starts in closed state"""
        breaker = CircuitBreaker("test")
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_open_on_consecutive_failures(self):
        """Test circuit opens after consecutive failures"""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)

        async def failing_func():
            raise Exception("Test failure")

        # Trigger failures
        for i in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.consecutive_failures == 3
        assert breaker.stats.circuit_opens == 1

    @pytest.mark.asyncio
    async def test_open_on_failure_rate(self):
        """Test circuit opens based on failure rate"""
        config = CircuitBreakerConfig(
            failure_threshold=100,  # High threshold so consecutive won't trigger
            failure_rate_threshold=0.5,
            min_calls=4,
        )
        breaker = CircuitBreaker("test", config)

        async def failing_func():
            raise Exception("Test failure")

        async def success_func():
            return "success"

        # 2 successes, then failures until circuit opens
        await breaker.call(success_func)
        await breaker.call(success_func)

        # First 2 failures bring rate to 50%, opening the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        # Circuit should now be open with 2/4 = 50% failure rate
        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.failure_rate() == 0.5

        # Additional failure attempts should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_circuit_stays_closed_below_threshold(self):
        """Test circuit stays closed when below failure threshold"""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)

        async def intermittent_func():
            if breaker.stats.total_calls % 2 == 0:
                return "success"
            raise Exception("Intermittent failure")

        # Alternating success and failure
        for i in range(6):
            try:
                await breaker.call(intermittent_func)
            except Exception:
                pass

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.consecutive_failures < 3

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self):
        """Test circuit moves to half-open after timeout"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout=0.1,  # 100ms timeout for testing
        )
        breaker = CircuitBreaker("test", config)

        async def failing_func():
            raise Exception("Test failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Next call should transition to half-open
        with pytest.raises(Exception):
            await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN  # Failed in half-open, back to open

    @pytest.mark.asyncio
    async def test_close_from_half_open(self):
        """Test circuit closes from half-open after successes"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout=0.1,
        )
        breaker = CircuitBreaker("test", config)

        call_count = 0

        async def recovering_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Still failing")
            return "recovered"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(recovering_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Successfully recover
        result1 = await breaker.call(recovering_func)
        assert result1 == "recovered"
        assert breaker.state == CircuitState.HALF_OPEN

        result2 = await breaker.call(recovering_func)
        assert result2 == "recovered"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_fallback_when_open(self):
        """Test fallback function is used when circuit is open"""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)

        async def failing_func():
            raise Exception("Primary failed")

        async def fallback_func():
            return "fallback_result"

        # Open the circuit
        with pytest.raises(Exception):
            await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Use fallback
        result = await breaker.call(failing_func, fallback=fallback_func)
        assert result == "fallback_result"

    @pytest.mark.asyncio
    async def test_error_when_open_no_fallback(self):
        """Test CircuitBreakerOpenError when open and no fallback"""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)

        async def failing_func():
            raise Exception("Failed")

        # Open the circuit
        with pytest.raises(Exception):
            await breaker.call(failing_func)

        # Should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await breaker.call(failing_func)

        assert "Circuit breaker 'test' is OPEN" in str(exc_info.value)


class TestCircuitBreakerFunctionality:
    """Test circuit breaker with different function types"""

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Test with async functions"""
        breaker = CircuitBreaker("test")

        async def async_func(x):
            await asyncio.sleep(0.01)
            return x * 2

        result = await breaker.call(async_func, 5)
        assert result == 10
        assert breaker.stats.successful_calls == 1

    @pytest.mark.asyncio
    async def test_sync_function(self):
        """Test with sync functions"""
        breaker = CircuitBreaker("test")

        def sync_func(x):
            return x * 3

        result = await breaker.call(sync_func, 4)
        assert result == 12
        assert breaker.stats.successful_calls == 1

    @pytest.mark.asyncio
    async def test_function_with_kwargs(self):
        """Test functions with keyword arguments"""
        breaker = CircuitBreaker("test")

        async def func_with_kwargs(a, b=10, c=20):
            return a + b + c

        result = await breaker.call(func_with_kwargs, 5, b=15, c=25)
        assert result == 45

    @pytest.mark.asyncio
    async def test_excluded_exceptions(self):
        """Test that excluded exceptions don't trigger circuit"""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            excluded_exceptions=(ValueError, TypeError),
        )
        breaker = CircuitBreaker("test", config)

        async def func_with_value_error():
            raise ValueError("This should not open circuit")

        # ValueError should not open circuit
        with pytest.raises(ValueError):
            await breaker.call(func_with_value_error)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.failed_calls == 0

    @pytest.mark.asyncio
    async def test_tracked_exceptions(self):
        """Test only tracked exceptions trigger circuit"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            tracked_exceptions=(RuntimeError,),
        )
        breaker = CircuitBreaker("test", config)

        async def func_with_runtime_error():
            raise RuntimeError("Tracked")

        async def func_with_value_error():
            raise ValueError("Not tracked")

        # ValueError should not affect circuit
        with pytest.raises(ValueError):
            await breaker.call(func_with_value_error)
        assert breaker.state == CircuitState.CLOSED

        # RuntimeError should affect circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(func_with_runtime_error)

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerStats:
    """Test circuit breaker statistics"""

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test statistics are properly tracked"""
        breaker = CircuitBreaker("test")

        async def sometimes_fails(should_fail):
            if should_fail:
                raise Exception("Failed")
            return "success"

        # Some successes
        for _ in range(3):
            await breaker.call(sometimes_fails, False)

        # Some failures
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(sometimes_fails, True)

        stats = breaker.get_stats()

        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["stats"]["total_calls"] == 5
        assert stats["stats"]["successful_calls"] == 3
        assert stats["stats"]["failed_calls"] == 2
        assert stats["stats"]["failure_rate"] == "40.00%"

    @pytest.mark.asyncio
    async def test_reset_stats(self):
        """Test manual reset clears statistics"""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test", config)

        async def failing_func():
            raise Exception("Fail")

        # Generate some stats
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.total_calls > 0

        # Reset
        await breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.total_calls == 0
        assert breaker.stats.failed_calls == 0


class TestCircuitBreakerManager:
    """Test circuit breaker manager"""

    def test_get_or_create(self):
        """Test getting or creating circuit breakers"""
        manager = CircuitBreakerManager()

        # Create new
        breaker1 = manager.get_or_create("service1")
        assert breaker1.name == "service1"

        # Get existing
        breaker2 = manager.get_or_create("service1")
        assert breaker1 is breaker2

        # Create another
        breaker3 = manager.get_or_create("service2")
        assert breaker3.name == "service2"
        assert breaker3 is not breaker1

    def test_custom_config(self):
        """Test creating with custom config"""
        manager = CircuitBreakerManager()
        config = CircuitBreakerConfig(failure_threshold=10)

        breaker = manager.get_or_create("custom", config)
        assert breaker.config.failure_threshold == 10

    @pytest.mark.asyncio
    async def test_get_all_stats(self):
        """Test getting stats for all breakers"""
        manager = CircuitBreakerManager()

        breaker1 = manager.get_or_create("service1")
        breaker2 = manager.get_or_create("service2")

        # Generate some activity
        async def success():
            return "ok"

        await breaker1.call(success)
        await breaker2.call(success)

        all_stats = manager.get_all_stats()

        assert "service1" in all_stats
        assert "service2" in all_stats
        assert all_stats["service1"]["stats"]["successful_calls"] == 1
        assert all_stats["service2"]["stats"]["successful_calls"] == 1

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """Test resetting all circuit breakers"""
        manager = CircuitBreakerManager()
        config = CircuitBreakerConfig(failure_threshold=1)

        breaker1 = manager.get_or_create("service1", config)
        breaker2 = manager.get_or_create("service2", config)

        async def fail():
            raise Exception("Fail")

        # Open both circuits
        with pytest.raises(Exception):
            await breaker1.call(fail)
        with pytest.raises(Exception):
            await breaker2.call(fail)

        assert breaker1.state == CircuitState.OPEN
        assert breaker2.state == CircuitState.OPEN

        # Reset all
        await manager.reset_all()

        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED


class TestCircuitBreakerConcurrency:
    """Test circuit breaker under concurrent load"""

    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test circuit breaker with concurrent calls"""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("test", config)

        call_counter = 0

        async def concurrent_func():
            nonlocal call_counter
            call_counter += 1
            await asyncio.sleep(0.01)
            if call_counter % 3 == 0:
                raise Exception("Periodic failure")
            return call_counter

        # Make concurrent calls
        tasks = []
        for _ in range(10):
            tasks.append(asyncio.create_task(self._safe_call(breaker, concurrent_func)))

        results = await asyncio.gather(*tasks)

        # Check results
        success_count = sum(1 for r in results if r is not None)
        failure_count = sum(1 for r in results if r is None)

        assert success_count + failure_count == 10
        assert breaker.stats.total_calls == 10

    async def _safe_call(self, breaker, func):
        """Helper to safely call function through breaker"""
        try:
            return await breaker.call(func)
        except Exception:
            return None

    @pytest.mark.asyncio
    async def test_half_open_concurrency_limit(self):
        """Test that half-open state limits concurrent test calls"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout=0.1,
        )
        breaker = CircuitBreaker("test", config)

        async def failing_func():
            raise Exception("Fail")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Try concurrent calls when transitioning to half-open
        async def slow_func():
            await asyncio.sleep(0.1)
            return "ok"

        async def fallback_func():
            return "fallback"

        # Multiple concurrent calls in half-open should use fallback
        tasks = [
            asyncio.create_task(breaker.call(slow_func, fallback=fallback_func)) for _ in range(3)
        ]

        results = await asyncio.gather(*tasks)

        # At least some should have used fallback due to half-open limiting
        fallback_count = sum(1 for r in results if r == "fallback")
        assert fallback_count >= 1
