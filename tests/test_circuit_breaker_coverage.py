"""
Surgical tests for CircuitBreaker to achieve 100% coverage.
Specifically targets missing lines: 143, 223, 250-251
"""

import time

import pytest

from recursive_companion_mcp.legacy.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
)

# sys.path removed - using package imports


class TestCircuitBreakerCoverage:
    """Surgical tests targeting specific missing lines in CircuitBreaker."""

    @pytest.fixture
    def circuit_config(self):
        """Create test circuit breaker config"""
        return CircuitBreakerConfig(
            failure_threshold=2, timeout=1.0, tracked_exceptions=(Exception,)
        )

    @pytest.fixture
    def circuit(self, circuit_config):
        """Create test circuit breaker"""
        return CircuitBreaker("test_circuit", circuit_config)

    @pytest.mark.asyncio
    async def test_can_attempt_reset_no_last_failure_time(self, circuit):
        """Test line 143: _can_attempt_reset when last_failure_time is None"""
        # Ensure last_failure_time is None (fresh circuit)
        assert circuit.stats.last_failure_time is None

        # This should trigger line 143: return True
        can_reset = await circuit._can_attempt_reset()
        assert can_reset is True

    @pytest.mark.asyncio
    async def test_half_open_no_fallback_raise_error(self, circuit):
        """Test line 223: Half-open state with locked half-open lock and no fallback"""
        # Force circuit into half-open state
        circuit.state = CircuitState.HALF_OPEN

        # Manually acquire the half-open lock to simulate concurrent access
        await circuit._half_open_lock.acquire()

        try:

            async def test_func():
                return "success"

            # This should trigger line 223 - half-open state with locked lock, no fallback
            with pytest.raises(CircuitBreakerOpenError) as exc_info:
                await circuit.call(test_func)

            assert "is testing recovery" in str(exc_info.value)
        finally:
            # Clean up
            circuit._half_open_lock.release()

    @pytest.mark.asyncio
    async def test_fallback_with_logging_when_open(self, circuit, caplog):
        """Test lines 250-251: Fallback usage with logging when circuit is open"""
        # Force circuit into open state
        circuit.state = CircuitState.OPEN

        async def failing_func():
            raise Exception("Primary function failed")

        async def fallback_func():
            return "fallback_result"

        # Clear previous logs
        caplog.clear()

        # Call with fallback - should trigger lines 250-251
        import logging

        with caplog.at_level(logging.DEBUG):
            result = await circuit.call(failing_func, fallback=fallback_func)

        # Verify fallback was used
        assert result == "fallback_result"

        # Verify logging occurred (line 250)
        log_messages = [record.message for record in caplog.records]
        assert any("Using fallback due to error:" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_fallback_execution_in_open_state_complete_flow(self, circuit):
        """Test complete fallback flow that triggers both failure recording and fallback usage"""

        # Force circuit into open state by triggering failures
        async def always_fail():
            raise ValueError("Persistent failure")

        async def reliable_fallback():
            return "fallback_success"

        # Trigger enough failures to open circuit
        for _ in range(circuit.config.failure_threshold + 1):
            try:
                await circuit.call(always_fail)
            except Exception:
                pass

        # Verify circuit is now open
        assert circuit.state == CircuitState.OPEN

        # Now call with fallback - this should use fallback without raising
        result = await circuit.call(always_fail, fallback=reliable_fallback)
        assert result == "fallback_success"

    @pytest.mark.asyncio
    async def test_half_open_state_edge_case(self, circuit):
        """Test edge case in half-open state - need success_threshold successes to close"""
        # Manually set circuit to half-open state
        circuit.state = CircuitState.HALF_OPEN

        async def test_func():
            return "success"

        # Need success_threshold (default 2) successes to close circuit
        for _ in range(circuit.config.success_threshold):
            result = await circuit.call(test_func)
            assert result == "success"

        # Circuit should now be closed after enough successful calls
        assert circuit.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_can_attempt_reset_with_recent_failure(self, circuit):
        """Test _can_attempt_reset when last_failure_time is recent"""
        # Set a recent failure time
        circuit.stats.last_failure_time = time.time() - 0.5  # Very recent failure

        # Should not be able to reset yet
        can_reset = await circuit._can_attempt_reset()
        assert can_reset is False

    @pytest.mark.asyncio
    async def test_can_attempt_reset_after_timeout(self, circuit):
        """Test _can_attempt_reset after timeout has passed"""
        # Set an old failure time (beyond timeout)
        circuit.stats.last_failure_time = time.time() - (circuit.config.timeout + 1)

        # Should be able to reset now
        can_reset = await circuit._can_attempt_reset()
        assert can_reset is True
