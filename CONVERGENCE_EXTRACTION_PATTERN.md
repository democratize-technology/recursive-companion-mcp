# Convergence Extraction Pattern Documentation

## Overview

This document describes the reusable convergence detection pattern extracted from recursive-companion-mcp. This pattern can be manually replicated in other thinking tools to prevent wasted tokens on repetitive iterations.

## What Was Extracted

### Core Modules

1. **`src/convergence.py`** - Convergence detection engine
   - `ConvergenceDetector` class with AWS Bedrock integration
   - `EmbeddingService` with caching and async support
   - Tool-specific threshold configuration
   - Cosine similarity calculation

2. **`src/base_cognitive.py`** - Cognitive enhancement framework
   - `CognitiveEnhancer` base class for thinking tools
   - `EnhancedThinkingTool` abstract base class
   - Utility functions: `iterate_until_convergence()`, `@with_convergence` decorator
   - Iteration tracking and performance logging

### Key Features

- **Async/await support** for non-blocking operations
- **Embedding caching** to reduce API calls and costs
- **Tool-specific thresholds** (0.85-0.99 range)
- **Graceful fallbacks** when convergence detection fails
- **Performance tracking** and logging
- **Circuit breaker patterns** for reliability

## Manual Replication Guide

### Step 1: Copy Core Files

Copy these files to your thinking tool:

```bash
# In your thinking tool directory
mkdir -p src/
cp /path/to/recursive-companion/src/convergence.py src/
cp /path/to/recursive-companion/src/base_cognitive.py src/
```

### Step 2: Install Dependencies

Add to your `pyproject.toml` or `requirements.txt`:

```toml
numpy = "^1.24.0"
boto3 = "^1.26.0"
```

### Step 3: Basic Integration

#### Simple Convergence Check

```python
# In your main processing function
from convergence import simple_convergence_check

async def your_iterative_function():
    previous_result = None
    
    for iteration in range(max_iterations):
        current_result = await generate_result()
        
        # Check convergence
        if previous_result:
            converged = await simple_convergence_check(
                current_result, 
                previous_result, 
                threshold=0.90
            )
            if converged:
                print(f"Converged at iteration {iteration}")
                break
                
        previous_result = current_result
    
    return current_result
```

#### Advanced Integration with CognitiveEnhancer

```python
from base_cognitive import CognitiveEnhancer, CognitiveConfig, EnhancedThinkingTool

class YourThinkingTool(EnhancedThinkingTool):
    def __init__(self):
        config = CognitiveConfig(
            tool_name="your-tool",
            max_iterations=10,
            convergence_threshold=0.92,
            enable_convergence=True
        )
        super().__init__(config)
    
    async def generate_iteration(self, iteration: int, previous_result: str) -> str:
        # Your tool's logic here
        return await your_generation_logic(previous_result)
    
    async def run_analysis(self, prompt: str) -> str:
        return await self.iterate_until_convergence(
            initial_input=prompt,
            iteration_generator=self.generate_iteration
        )
```

### Step 4: Configuration

#### Environment Variables

```bash
# AWS Configuration (required)
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Tool-specific thresholds (optional)
export CONVERGENCE_THRESHOLD_YOUR_TOOL=0.90
```

#### Tool-Specific Thresholds

Different tools benefit from different convergence thresholds:

- **Devil's Advocate**: 0.85 (looser - challenges can vary significantly)
- **Decision Matrix**: 0.95 (tighter - scores should be precise)
- **Context Switcher**: 0.90 (medium - perspectives can evolve)
- **Rubber Duck**: 0.88 (looser - conversations are exploratory)

### Step 5: Integration Patterns

#### Pattern 1: Loop Integration

```python
async def your_main_loop():
    detector = ConvergenceDetector()
    history = []
    
    for i in range(max_iterations):
        result = await process_iteration()
        history.append(result)
        
        if len(history) >= 2:
            converged, score = await detector.is_converged(
                history[-1], history[-2]
            )
            if converged:
                logger.info(f"Converged at iteration {i} (score: {score:.3f})")
                break
    
    return history[-1]
```

#### Pattern 2: Decorator Integration

```python
from base_cognitive import with_convergence

@with_convergence(threshold=0.90, max_iterations=8)
async def your_iterative_function(input_text: str) -> str:
    # Your function automatically gets convergence detection
    return await your_processing_logic(input_text)
```

#### Pattern 3: Session Integration

```python
class YourSession:
    def __init__(self):
        self.convergence_detector = ConvergenceDetector()
        self.iteration_history = []
    
    async def continue_session(self):
        new_result = await self.process_next_step()
        self.iteration_history.append(new_result)
        
        if len(self.iteration_history) >= 2:
            converged, score = await self.convergence_detector.is_converged(
                self.iteration_history[-1],
                self.iteration_history[-2]
            )
            
            if converged:
                self.status = "CONVERGED"
                return {"status": "complete", "result": new_result}
        
        return {"status": "continue", "result": new_result}
```

## Integration Examples by Tool Type

### Devil's Advocate Tool

```python
# In challenge generation loop
async def generate_challenges(idea: str, context: str):
    detector = ConvergenceDetector()
    challenges = []
    
    for perspective in perspectives:
        challenge = await generate_challenge(idea, context, perspective)
        challenges.append(challenge)
        
        # Check if challenges are becoming repetitive
        if len(challenges) >= 2:
            converged, score = await detector.is_converged(
                challenges[-1], challenges[-2]
            )
            if converged:
                logger.info(f"Challenge convergence detected for {perspective}")
                break
    
    return challenges
```

### Decision Matrix Tool

```python
# In evaluation loop
async def evaluate_criteria(options: List[str], criteria: List[str]):
    detector = ConvergenceDetector()
    
    for criterion in criteria:
        scores = []
        for iteration in range(3):  # Multiple evaluation rounds
            round_scores = await evaluate_round(options, criterion)
            scores.append(round_scores)
            
            if len(scores) >= 2:
                # Check if scores are stabilizing
                current_text = json.dumps(scores[-1])
                previous_text = json.dumps(scores[-2])
                
                converged, score = await detector.is_converged(
                    current_text, previous_text
                )
                if converged:
                    break
        
        final_scores[criterion] = scores[-1]
```

### Context Switcher Tool

```python
# In perspective analysis
async def analyze_from_perspectives(topic: str, perspectives: List[str]):
    detector = ConvergenceDetector()
    analyses = {}
    
    for perspective in perspectives:
        analysis_history = []
        
        for refinement in range(max_refinements):
            analysis = await generate_analysis(topic, perspective, analysis_history)
            analysis_history.append(analysis)
            
            if len(analysis_history) >= 2:
                converged, score = await detector.is_converged(
                    analysis_history[-1], analysis_history[-2]
                )
                if converged:
                    logger.info(f"Analysis converged for {perspective}")
                    break
        
        analyses[perspective] = analysis_history[-1]
    
    return analyses
```

## Troubleshooting

### Common Issues

1. **Convergence never fires**: Lower threshold (try 0.85)
2. **Convergence always fires**: Raise threshold (try 0.95)
3. **AWS credential errors**: Check environment variables
4. **Import errors**: Ensure dependencies are installed
5. **Performance issues**: Enable embedding caching

### Debug Configuration

```python
# Enable debug logging
import logging
logging.getLogger('convergence').setLevel(logging.DEBUG)
logging.getLogger('base_cognitive').setLevel(logging.DEBUG)

# Test convergence detection
from convergence import test_convergence_detection
await test_convergence_detection()
```

### Performance Tuning

```python
# Optimize for your use case
config = ConvergenceConfig(
    threshold=0.90,           # Tool-specific threshold
    cache_size=2000,          # Larger cache for high-volume tools
    max_text_length=12000,    # Adjust for your content size
)

detector = ConvergenceDetector(config)
```

## Benefits Achieved

After implementing this pattern in recursive-companion:

- ✅ **190 tests passing** (was 187 passed, 3 failed)
- ✅ **Zero regressions** - all existing functionality preserved  
- ✅ **Eliminated code duplication** - reduced from 3 implementations to 1
- ✅ **Improved maintainability** - single source of truth for convergence
- ✅ **Better performance** - shared embedding cache across components

## Next Steps

1. **Copy the modules** to your thinking tool
2. **Choose integration pattern** based on your tool's architecture
3. **Set appropriate threshold** for your use case
4. **Test convergence behavior** with real workloads
5. **Monitor and tune** performance based on usage

## Success Metrics

Track these to measure success:

- **Token savings**: Iterations avoided due to convergence
- **Quality maintained**: Output quality doesn't degrade
- **Performance impact**: Overhead of convergence detection
- **Cache hit rate**: Embedding cache effectiveness

Remember: This is experimental. Success means it works without breaking, saves some tokens, and you learn what works for your specific tool.