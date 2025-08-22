# Convergence Detection Extraction Pattern

This document explains how convergence detection was extracted from recursive-companion-mcp into reusable patterns that can be applied to other thinking tools.

## Overview

The convergence detection pattern helps thinking tools determine when iterative processes have stabilized, avoiding infinite loops and providing natural stopping points. This pattern was successfully extracted from the recursive-companion tool and can be manually replicated in other MCP thinking tools.

## Extracted Components

### 1. Core Convergence Module (`src/convergence.py`)

**Purpose**: Provides drop-in convergence detection for any thinking tool

**Key Classes**:
- `ConvergenceConfig`: Configuration with tool-specific thresholds
- `EmbeddingService`: AWS Bedrock embedding service with caching
- `ConvergenceDetector`: Main convergence detection logic

**Key Features**:
- **Tool-specific thresholds**: Different tools need different convergence thresholds
  - `devil-advocate`: 0.70 (lower - we want diversity)
  - `decision-matrix`: 0.90 (moderate)
  - `rubber-duck`: 0.95 (high - stop loops)
- **Caching**: LRU cache for embeddings to improve performance
- **Fallback detection**: Basic text similarity when embeddings unavailable
- **Statistics tracking**: Comprehensive metrics for monitoring

**Usage Pattern**:
```python
from convergence import create_detector_for_tool

# Create tool-specific detector
detector = create_detector_for_tool("your-tool-name")

# Check convergence between iterations
converged, score = await detector.is_converged(current_text, previous_text)
```

### 2. Base Cognitive Enhancement (`src/base_cognitive.py`)

**Purpose**: Provides common cognitive enhancement patterns for thinking tools

**Key Classes**:
- `CognitiveEnhancer`: Base enhancement functionality
- `EnhancedThinkingTool`: Abstract base class for thinking tools
- `CognitiveConfig`: Configuration for cognitive features

**Key Features**:
- **Iteration tracking**: Automatic tracking of processing iterations
- **Convergence integration**: Built-in convergence detection
- **Performance logging**: Comprehensive statistics and timing
- **Decorator support**: Easy enhancement of existing functions

**Usage Pattern**:
```python
from base_cognitive import EnhancedThinkingTool, CognitiveConfig

class MyThinkingTool(EnhancedThinkingTool):
    def __init__(self):
        config = CognitiveConfig(tool_name="my-tool", max_iterations=10)
        super().__init__("my-tool", config)
    
    async def process_iteration(self, input_data: str, iteration: int) -> str:
        # Your tool's logic here
        return processed_result
```

## Implementation Guide for Other Tools

### Step 1: Extract the Core Files

Copy these files to your tool's `src/` directory:

1. **`src/convergence.py`** - Core convergence detection
2. **`src/base_cognitive.py`** - Base cognitive enhancement classes

### Step 2: Add Dependencies

Add to your `pyproject.toml`:
```toml
dependencies = [
    "numpy>=1.24.0",
    "boto3>=1.26.0",
    # ... other dependencies
]
```

### Step 3: Configure Your Tool

Add your tool to the threshold configuration in `convergence.py`:
```python
# In create_detector_for_tool function
thresholds = {
    "devil-advocate": 0.70,  # Lower - we want diversity
    "decision-matrix": 0.90,  # Moderate
    "your-tool-name": 0.85,  # Add your tool here
    # ... other tools
}
```

### Step 4: Integration Patterns

#### Pattern A: Simple Integration (for existing tools)
```python
from convergence import simple_convergence_check

async def your_existing_function(input_text):
    previous_result = ""
    
    for iteration in range(max_iterations):
        current_result = await process_iteration(input_text, iteration)
        
        if iteration > 0:
            converged = await simple_convergence_check(
                current_result, previous_result, threshold=0.90
            )
            if converged:
                break
        
        previous_result = current_result
    
    return current_result
```

#### Pattern B: Full Enhancement (for new tools)
```python
from base_cognitive import EnhancedThinkingTool, CognitiveConfig

class YourThinkingTool(EnhancedThinkingTool):
    def __init__(self):
        config = CognitiveConfig(
            tool_name="your-tool",
            max_iterations=10,
            convergence_threshold=0.90
        )
        super().__init__("your-tool", config)
    
    async def process_iteration(self, input_data: str, iteration: int) -> str:
        # Your tool's processing logic
        result = await your_processing_logic(input_data)
        return result

# Usage
tool = YourThinkingTool()
result = await tool.process(initial_input)
print(f"Final result: {result['final_result']}")
print(f"Converged: {result['convergence_achieved']}")
```

#### Pattern C: Decorator Pattern (for simple functions)
```python
from base_cognitive import with_convergence

@with_convergence("your-tool", threshold=0.90)
async def your_enhanced_function(input_data):
    # Your existing function
    return processed_result

# Usage automatically includes convergence tracking
result = await your_enhanced_function(input_data)
```

### Step 5: Configure Environment Variables

Add to your `.env` file:
```bash
# AWS Bedrock for embeddings
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1

# Convergence settings
CONVERGENCE_THRESHOLD=0.95
MAX_ITERATIONS=10
```

## Tool-Specific Recommendations

### Threshold Selection Guidelines

- **High Creativity Tools** (0.70-0.80): devil-advocate, brainstorming
  - Need diversity, lower convergence prevents premature stopping
  
- **Balanced Tools** (0.85-0.90): decision-matrix, conversation-tree
  - Balance between exploration and convergence
  
- **Focused Tools** (0.95-0.98): rubber-duck, hindsight
  - Need stability, higher convergence prevents loops

### Performance Optimization

1. **Use Caching**: The embedding cache significantly improves performance
2. **Parallel Processing**: Generate multiple critiques/evaluations in parallel
3. **Fallback Methods**: Implement basic text similarity for when embeddings fail
4. **Tool-Specific Models**: Use faster models (e.g., Claude Haiku) for critiques

## Testing Pattern

Create tests for your convergence integration:

```python
import pytest
from your_tool import YourThinkingTool

@pytest.mark.asyncio
async def test_convergence_detection():
    tool = YourThinkingTool()
    
    # Test with content that should converge
    result = await tool.process("test input")
    
    assert result["success"]
    assert "convergence_achieved" in result
    assert result["total_iterations"] > 0
    
    # Verify statistics are tracked
    stats = tool.get_stats()
    assert "convergence_stats" in stats
```

## Benefits of This Pattern

1. **Consistency**: All thinking tools use the same convergence detection
2. **Performance**: Shared caching and optimizations
3. **Configurability**: Tool-specific thresholds and settings
4. **Monitoring**: Built-in statistics and performance tracking
5. **Fallback**: Graceful degradation when embeddings unavailable
6. **Testability**: Comprehensive test patterns included

## Migration Notes

When applying this pattern to existing tools:

1. **Preserve Existing Logic**: Don't change your tool's core logic
2. **Add Incrementally**: Start with simple convergence checks
3. **Monitor Performance**: Use the built-in statistics to optimize
4. **Test Thoroughly**: Verify convergence behavior with your tool's specific patterns

## Example Tools Using This Pattern

This pattern has been successfully implemented in:
- **recursive-companion-mcp**: Iterative refinement with draft→critique→revise cycles
- **Ready for**: devil-advocate, decision-matrix, rubber-duck, and other thinking tools

The pattern is designed to be tool-agnostic and can be adapted to any iterative thinking process.
