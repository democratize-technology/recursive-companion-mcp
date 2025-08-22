"""
Comprehensive tests for base_cognitive.py module to achieve 100% coverage.
Tests cover all defensive programming branches, error conditions, and edge cases.
"""

import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, "./src")

from base_cognitive import (
    CognitiveConfig,
    CognitiveEnhancer,
    EnhancedThinkingTool,
    IterationResult,
    iterate_until_convergence,
    with_convergence,
)


class TestCognitiveConfig:
    """Test CognitiveConfig dataclass"""

    def test_default_configuration(self):
        """Test default config values"""
        config = CognitiveConfig(tool_name="test_tool")

        assert config.tool_name == "test_tool"
        assert config.max_iterations == 10
        assert config.convergence_threshold is None
        assert config.enable_convergence is True
        assert config.enable_logging is True
        assert config.log_level == "INFO"

    def test_custom_configuration(self):
        """Test custom config values"""
        config = CognitiveConfig(
            tool_name="custom_tool",
            max_iterations=5,
            convergence_threshold=0.9,
            enable_convergence=False,
            enable_logging=False,
            log_level="DEBUG",
        )

        assert config.tool_name == "custom_tool"
        assert config.max_iterations == 5
        assert config.convergence_threshold == 0.9
        assert config.enable_convergence is False
        assert config.enable_logging is False
        assert config.log_level == "DEBUG"


class TestIterationResult:
    """Test IterationResult dataclass"""

    def test_default_creation(self):
        """Test default IterationResult creation"""
        result = IterationResult(iteration=1, content="test content")

        assert result.iteration == 1
        assert result.content == "test content"
        assert isinstance(result.timestamp, datetime)
        assert result.convergence_score is None
        assert result.metadata == {}

    def test_custom_creation(self):
        """Test IterationResult with all fields"""
        timestamp = datetime.utcnow()
        metadata = {"key": "value"}

        result = IterationResult(
            iteration=2,
            content="custom content",
            timestamp=timestamp,
            convergence_score=0.95,
            metadata=metadata,
        )

        assert result.iteration == 2
        assert result.content == "custom content"
        assert result.timestamp == timestamp
        assert result.convergence_score == 0.95
        assert result.metadata == metadata


class TestCognitiveEnhancer:
    """Test CognitiveEnhancer class"""

    def test_initialization_with_logging_enabled(self):
        """Test initialization with logging enabled"""
        config = CognitiveConfig(tool_name="test_tool", enable_logging=True, log_level="DEBUG")

        with patch("base_cognitive.logger") as mock_logger:
            enhancer = CognitiveEnhancer(config)

            assert enhancer.config == config
            assert enhancer.convergence_detector is not None
            assert enhancer.iteration_history == []
            assert isinstance(enhancer.start_time, datetime)
            mock_logger.setLevel.assert_called_once()
            mock_logger.info.assert_called()

    def test_initialization_with_logging_disabled(self):
        """Test initialization with logging disabled"""
        config = CognitiveConfig(tool_name="test_tool", enable_logging=False)

        with patch("base_cognitive.logger") as mock_logger:
            CognitiveEnhancer(config)

            # Should not call setLevel when logging disabled
            mock_logger.setLevel.assert_not_called()

    def test_initialization_with_convergence_disabled(self):
        """Test initialization with convergence disabled"""
        config = CognitiveConfig(tool_name="test_tool", enable_convergence=False)

        enhancer = CognitiveEnhancer(config)
        assert enhancer.convergence_detector is None

    @pytest.mark.asyncio
    async def test_check_convergence_disabled(self):
        """Test convergence check when disabled"""
        config = CognitiveConfig(tool_name="test_tool", enable_convergence=False)
        enhancer = CognitiveEnhancer(config)

        converged, score = await enhancer.check_convergence("current", "previous")

        assert converged is False
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_check_convergence_no_detector(self):
        """Test convergence check when detector is None"""
        config = CognitiveConfig(tool_name="test_tool", enable_convergence=True)
        enhancer = CognitiveEnhancer(config)
        enhancer.convergence_detector = None  # Simulate detector creation failure

        converged, score = await enhancer.check_convergence("current", "previous")

        assert converged is False
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_check_convergence_enabled(self):
        """Test convergence check when enabled"""
        config = CognitiveConfig(tool_name="test_tool", enable_convergence=True)
        enhancer = CognitiveEnhancer(config)

        # Mock the convergence detector
        mock_detector = AsyncMock()
        mock_detector.is_converged.return_value = (True, 0.95)
        enhancer.convergence_detector = mock_detector

        converged, score = await enhancer.check_convergence("current", "previous")

        assert converged is True
        assert score == 0.95
        mock_detector.is_converged.assert_called_once_with("current", "previous")

    def test_add_iteration_without_metadata(self):
        """Test adding iteration without metadata"""
        config = CognitiveConfig(tool_name="test_tool")
        enhancer = CognitiveEnhancer(config)

        with patch("base_cognitive.logger") as mock_logger:
            result = enhancer.add_iteration("test content")

            assert len(enhancer.iteration_history) == 1
            assert result.iteration == 1
            assert result.content == "test content"
            assert result.metadata == {}  # Test defensive || {} pattern
            mock_logger.debug.assert_called_once()

    def test_add_iteration_with_metadata(self):
        """Test adding iteration with metadata"""
        config = CognitiveConfig(tool_name="test_tool")
        enhancer = CognitiveEnhancer(config)
        metadata = {"key": "value"}

        result = enhancer.add_iteration("test content", metadata)

        assert result.metadata == metadata

    def test_add_iteration_with_none_metadata(self):
        """Test adding iteration with None metadata (defensive programming)"""
        config = CognitiveConfig(tool_name="test_tool")
        enhancer = CognitiveEnhancer(config)

        result = enhancer.add_iteration("test content", None)

        assert result.metadata == {}  # Should use defensive || {} fallback

    @pytest.mark.asyncio
    async def test_process_with_convergence_no_iterations(self):
        """Test processing with no successful iterations"""
        config = CognitiveConfig(tool_name="test_tool")
        enhancer = CognitiveEnhancer(config)

        def failing_processor(input_data, iteration):
            raise Exception("Processing failed")

        with patch("base_cognitive.logger") as mock_logger:
            result = await enhancer.process_with_convergence(
                failing_processor, "initial input", max_iterations=1
            )

            assert result["success"] is False
            assert "No iterations completed" in result["error"]
            assert "total_time" in result
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_with_convergence_single_iteration(self):
        """Test processing with single iteration"""
        config = CognitiveConfig(tool_name="test_tool", enable_convergence=False)
        enhancer = CognitiveEnhancer(config)

        def simple_processor(input_data, iteration):
            return f"processed_{input_data}_{iteration}"

        with patch("base_cognitive.logger") as mock_logger:
            result = await enhancer.process_with_convergence(
                simple_processor, "input", max_iterations=1
            )

            assert result["success"] is True
            assert result["final_result"] == "processed_input_1"
            assert result["total_iterations"] == 1
            assert result["convergence_achieved"] is False
            mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_process_with_convergence_achieves_convergence(self):
        """Test processing that achieves convergence"""
        config = CognitiveConfig(tool_name="test_tool", enable_convergence=True)
        enhancer = CognitiveEnhancer(config)

        # Mock convergence detector
        mock_detector = AsyncMock()
        mock_detector.is_converged.return_value = (True, 0.98)
        mock_detector.config.threshold = 0.95
        enhancer.convergence_detector = mock_detector

        def simple_processor(input_data, iteration):
            return f"iteration_{iteration}"

        with patch("base_cognitive.logger") as mock_logger:
            result = await enhancer.process_with_convergence(
                simple_processor, "input", max_iterations=5
            )

            assert result["success"] is True
            assert result["total_iterations"] == 2  # Should stop after convergence
            assert result["convergence_achieved"] is True
            assert result["final_convergence_score"] == 0.98
            mock_logger.info.assert_any_call("Converged at iteration 2 with score 0.9800")

    @pytest.mark.asyncio
    async def test_process_with_convergence_max_iterations(self):
        """Test processing that hits max iterations without convergence"""
        config = CognitiveConfig(tool_name="test_tool", enable_convergence=True)
        enhancer = CognitiveEnhancer(config)

        # Mock convergence detector that never converges
        mock_detector = AsyncMock()
        mock_detector.is_converged.return_value = (False, 0.5)
        mock_detector.config.threshold = 0.95
        enhancer.convergence_detector = mock_detector

        def simple_processor(input_data, iteration):
            return f"iteration_{iteration}"

        result = await enhancer.process_with_convergence(
            simple_processor, "input", max_iterations=3
        )

        assert result["success"] is True
        assert result["total_iterations"] == 3
        assert result["convergence_achieved"] is False

    @pytest.mark.asyncio
    async def test_process_with_convergence_processing_failure(self):
        """Test processing failure in the middle"""
        config = CognitiveConfig(
            tool_name="test_tool", enable_convergence=False
        )  # Disable convergence to avoid None score issues
        enhancer = CognitiveEnhancer(config)

        def failing_on_second_processor(input_data, iteration):
            if iteration == 2:
                raise ValueError("Second iteration fails")
            return f"iteration_{iteration}"

        with patch("base_cognitive.logger") as mock_logger:
            result = await enhancer.process_with_convergence(
                failing_on_second_processor, "input", max_iterations=3
            )

            assert result["success"] is True  # Should succeed with partial results
            assert result["total_iterations"] == 1  # Only first iteration completed
            assert mock_logger.error.called  # Should log the error

    def test_create_final_result_empty_history(self):
        """Test final result creation with empty iteration history"""
        config = CognitiveConfig(tool_name="test_tool")
        enhancer = CognitiveEnhancer(config)

        result = enhancer._create_final_result()

        assert result["success"] is False
        assert "No iterations completed" in result["error"]
        assert "total_time" in result

    def test_create_final_result_with_convergence_detector(self):
        """Test final result creation with convergence detector"""
        config = CognitiveConfig(tool_name="test_tool")
        enhancer = CognitiveEnhancer(config)

        # Mock convergence detector
        mock_detector = Mock()
        mock_detector.config.threshold = 0.95
        enhancer.convergence_detector = mock_detector

        # Add some iterations
        enhancer.add_iteration("iteration 1")
        enhancer.add_iteration("iteration 2")
        enhancer.iteration_history[-1].convergence_score = 0.98

        result = enhancer._create_final_result()

        assert result["success"] is True
        assert result["convergence_achieved"] is True
        assert result["final_convergence_score"] == 0.98
        assert result["tool_name"] == "test_tool"
        assert result["metadata"]["threshold"] == 0.95

    def test_create_final_result_without_convergence_detector(self):
        """Test final result creation without convergence detector"""
        config = CognitiveConfig(tool_name="test_tool", enable_convergence=False)
        enhancer = CognitiveEnhancer(config)
        enhancer.convergence_detector = None

        enhancer.add_iteration("iteration 1")

        result = enhancer._create_final_result()

        assert result["convergence_achieved"] is False
        assert result["metadata"]["threshold"] is None

    def test_get_iteration_history_empty(self):
        """Test getting iteration history when empty"""
        config = CognitiveConfig(tool_name="test_tool")
        enhancer = CognitiveEnhancer(config)

        history = enhancer.get_iteration_history()
        assert history == []

    def test_get_iteration_history_with_data(self):
        """Test getting iteration history with data"""
        config = CognitiveConfig(tool_name="test_tool")
        enhancer = CognitiveEnhancer(config)

        enhancer.add_iteration("short content", {"key": "value"})
        enhancer.add_iteration("a" * 300, {"key2": "value2"})  # Test truncation
        enhancer.iteration_history[-1].convergence_score = 0.95

        history = enhancer.get_iteration_history()

        assert len(history) == 2
        assert history[0]["content_preview"] == "short content"
        assert history[0]["content_length"] == 13
        assert history[0]["metadata"] == {"key": "value"}

        # Test content truncation
        assert len(history[1]["content_preview"]) == 203  # 200 + "..."
        assert history[1]["content_preview"].endswith("...")
        assert history[1]["convergence_score"] == 0.95

    def test_get_stats_with_convergence_detector(self):
        """Test getting stats with convergence detector"""
        config = CognitiveConfig(tool_name="test_tool")
        enhancer = CognitiveEnhancer(config)

        # Mock convergence detector
        mock_detector = Mock()
        mock_detector.get_stats.return_value = {"detector_stats": "test"}
        enhancer.convergence_detector = mock_detector

        # Add iterations with convergence scores
        enhancer.add_iteration("iteration 1")
        enhancer.iteration_history[-1].convergence_score = 0.8
        enhancer.add_iteration("iteration 2")
        enhancer.iteration_history[-1].convergence_score = 0.9
        enhancer.add_iteration("iteration 3")
        enhancer.iteration_history[-1].convergence_score = 0.95

        stats = enhancer.get_stats()

        assert stats["tool_name"] == "test_tool"
        assert stats["total_iterations"] == 3
        assert stats["convergence_enabled"] is True
        assert stats["convergence_stats"] == {"detector_stats": "test"}
        expected_avg = (0.8 + 0.9 + 0.95) / 3
        assert abs(stats["convergence_summary"]["avg_score"] - expected_avg) < 1e-10
        assert stats["convergence_summary"]["max_score"] == 0.95
        assert stats["convergence_summary"]["min_score"] == 0.8
        assert stats["convergence_summary"]["final_score"] == 0.95

    def test_get_stats_without_convergence_scores(self):
        """Test getting stats without convergence scores"""
        config = CognitiveConfig(tool_name="test_tool")
        enhancer = CognitiveEnhancer(config)

        enhancer.add_iteration("iteration 1")  # No convergence score

        stats = enhancer.get_stats()

        assert "convergence_summary" not in stats
        assert stats["total_iterations"] == 1

    def test_get_stats_without_convergence_detector(self):
        """Test getting stats without convergence detector"""
        config = CognitiveConfig(tool_name="test_tool", enable_convergence=False)
        enhancer = CognitiveEnhancer(config)
        enhancer.convergence_detector = None

        stats = enhancer.get_stats()

        assert "convergence_stats" not in stats
        assert stats["convergence_enabled"] is False


class TestEnhancedThinkingTool:
    """Test EnhancedThinkingTool abstract base class"""

    def test_initialization_with_default_config(self):
        """Test initialization with default config"""

        class TestTool(EnhancedThinkingTool):
            async def process_iteration(self, input_data: str, iteration: int) -> str:
                return f"processed_{input_data}_{iteration}"

        tool = TestTool("test_tool")

        assert tool.tool_name == "test_tool"
        assert tool.config.tool_name == "test_tool"
        assert isinstance(tool.enhancer, CognitiveEnhancer)

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config"""

        class TestTool(EnhancedThinkingTool):
            async def process_iteration(self, input_data: str, iteration: int) -> str:
                return f"processed_{input_data}_{iteration}"

        config = CognitiveConfig(tool_name="custom_tool", max_iterations=5)
        tool = TestTool("test_tool", config)

        assert tool.config.max_iterations == 5

    @pytest.mark.asyncio
    async def test_process_method(self):
        """Test main process method"""

        class TestTool(EnhancedThinkingTool):
            async def process_iteration(self, input_data: str, iteration: int) -> str:
                return f"processed_{input_data}_{iteration}"

        tool = TestTool("test_tool")

        with patch.object(tool.enhancer, "process_with_convergence") as mock_process:
            mock_process.return_value = {"result": "test"}

            result = await tool.process("input")

            assert result == {"result": "test"}
            mock_process.assert_called_once_with(tool.process_iteration, "input")

    def test_get_stats_delegation(self):
        """Test get_stats method delegation"""

        class TestTool(EnhancedThinkingTool):
            async def process_iteration(self, input_data: str, iteration: int) -> str:
                return f"processed_{input_data}_{iteration}"

        tool = TestTool("test_tool")

        with patch.object(tool.enhancer, "get_stats") as mock_stats:
            mock_stats.return_value = {"stats": "test"}

            stats = tool.get_stats()

            assert stats == {"stats": "test"}
            mock_stats.assert_called_once()

    def test_get_history_delegation(self):
        """Test get_history method delegation"""

        class TestTool(EnhancedThinkingTool):
            async def process_iteration(self, input_data: str, iteration: int) -> str:
                return f"processed_{input_data}_{iteration}"

        tool = TestTool("test_tool")

        with patch.object(tool.enhancer, "get_iteration_history") as mock_history:
            mock_history.return_value = [{"history": "test"}]

            history = tool.get_history()

            assert history == [{"history": "test"}]
            mock_history.assert_called_once()

    def test_abstract_method_enforcement(self):
        """Test that abstract method must be implemented"""

        with pytest.raises(TypeError):
            # Should fail because process_iteration is not implemented
            EnhancedThinkingTool("test_tool")


class TestWithConvergenceDecorator:
    """Test @with_convergence decorator"""

    @pytest.mark.asyncio
    async def test_decorator_basic_functionality(self):
        """Test basic decorator functionality"""

        @with_convergence("test_tool", threshold=0.9)
        async def test_function(input_data):
            return f"processed_{input_data}"

        result = await test_function("input")

        assert result["result"] == "processed_input"
        assert "enhanced_stats" in result
        assert result["enhanced_stats"]["tool_name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_decorator_with_none_threshold(self):
        """Test decorator with None threshold (uses default)"""

        @with_convergence("test_tool", threshold=None)
        async def test_function(input_data):
            return f"processed_{input_data}"

        result = await test_function("input")

        assert result["result"] == "processed_input"
        assert "enhanced_stats" in result


class TestIterateUntilConvergence:
    """Test iterate_until_convergence utility function"""

    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        """Test basic iterate_until_convergence functionality"""

        def simple_processor(input_data):
            return f"processed_{input_data}"

        with patch("base_cognitive.CognitiveEnhancer") as mock_enhancer_class:
            mock_enhancer = AsyncMock()  # Use AsyncMock since method is async
            mock_enhancer.process_with_convergence.return_value = {"result": "test"}
            mock_enhancer_class.return_value = mock_enhancer

            result = await iterate_until_convergence(simple_processor, "input", "test_tool", 5, 0.9)

            assert result == {"result": "test"}
            mock_enhancer_class.assert_called_once()
            mock_enhancer.process_with_convergence.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_default_parameters(self):
        """Test with default parameters"""

        def simple_processor(input_data):
            return f"processed_{input_data}"

        with patch("base_cognitive.CognitiveEnhancer") as mock_enhancer_class:
            mock_enhancer = AsyncMock()  # Use AsyncMock since method is async
            mock_enhancer.process_with_convergence.return_value = {"result": "test"}
            mock_enhancer_class.return_value = mock_enhancer

            result = await iterate_until_convergence(simple_processor, "input")

            assert result == {"result": "test"}
            # Verify default parameters were used
            config_call = mock_enhancer_class.call_args[0][0]
            assert config_call.tool_name == "generic"
            assert config_call.max_iterations == 10
            assert config_call.convergence_threshold == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.base_cognitive"])
