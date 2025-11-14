# Hard/Medium/Easy Differentiated Speculation Parameters

## Overview

This document describes the implementation of differentiated speculation parameters based on problem difficulty classification in the `propose_suffix_draft_token_ids` method.

## Implementation Details

### Problem Classification

The system classifies requests into three categories:
- **Hard**: Complex problems requiring more aggressive speculation
- **Medium**: Moderate complexity problems  
- **Easy**: Simple problems requiring less speculation

### Spec Parameter Configuration

#### Base Parameters by Difficulty

```python
# Default parameters
hard_spec, hard_prob, hard_spec_factor = 16, 0.1, 2
medium_spec, medium_prob, medium_spec_factor = 8, 0.1, 1
easy_spec, easy_prob, easy_spec_factor = 4, 0.1, 1
```

#### Batch Size Adjustments

The parameters are dynamically adjusted based on batch size to optimize performance:

**For batch_size <= 64:**
```python
medium_spec, medium_prob, medium_spec_factor = 16, 0.1, 2
easy_spec, easy_prob, easy_spec_factor = 16, 0.1, 2
```

**For batch_size <= 128:**
```python
medium_spec, medium_prob, medium_spec_factor = 16, 0.1, 2
easy_spec, easy_prob, easy_spec_factor = 8, 0.1, 1
```

**For batch_size > 128:**
Uses the default parameters.

### Parameter Meaning

- **spec_tokens**: Maximum number of speculative tokens to generate
- **min_prob**: Minimum token probability threshold for speculation
- **spec_factor**: Speculation factor affecting the aggressiveness of speculation

### Integration with Existing Code

The implementation integrates with the existing `propose_suffix_draft_token_ids` method by:

1. **Getting problem classifications**: Uses `_get_hard_and_non_hard_indices()` to get categorized indices
2. **Setting parameters per request**: For each request in the batch, determines appropriate parameters based on its classification
3. **Applying parameters**: Uses the determined parameters in the `_suffix_cache.speculate()` call

### Code Flow

```python
def propose_suffix_draft_token_ids(self, sampled_token_ids, spec_token_ids=None):
    # 1. Get batch size and problem classifications
    batch_size = len(sampled_token_ids)
    hard_indices, medium_indices, easy_indices, allowed_indices = self._get_hard_and_non_hard_indices()
    
    # 2. Set base parameters and adjust for batch size
    # ... parameter configuration ...
    
    # 3. Process each request with appropriate parameters
    for i, sampled_ids in enumerate(sampled_token_ids):
        # Determine parameters based on classification
        if i in hard_indices:
            current_spec_tokens = hard_spec
            current_min_prob = hard_prob
            current_spec_factor = hard_spec_factor
        elif i in medium_indices:
            # ... medium parameters ...
        elif i in easy_indices:
            # ... easy parameters ...
        else:
            # ... default parameters ...
        
        # Apply parameters in speculation
        result, source = self._suffix_cache.speculate(
            req_id, pattern,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=current_spec_factor,
            min_token_prob=current_min_prob,
            problem_id=problem_id,
        )
```

### Benefits

1. **Performance Optimization**: Hard problems get more aggressive speculation, potentially reducing latency
2. **Resource Efficiency**: Easy problems use fewer resources, allowing better overall throughput
3. **Adaptive Behavior**: Parameters adjust based on batch size for optimal performance
4. **Backward Compatibility**: Falls back to default parameters for unclassified requests

### Logging

The system includes debug logging to track differentiated processing:

```python
logger.debug(f"Differentiated speculation - Batch size: {batch_size}, "
            f"Hard: {len(hard_indices)} (spec={hard_spec}), "
            f"Medium: {len(medium_indices)} (spec={medium_spec}), "
            f"Easy: {len(easy_indices)} (spec={easy_spec})")
```

## Usage

This feature is automatically enabled when:
1. Problem IDs are provided with requests
2. Hard/medium/easy classifications are set via `ProblemIdContextManager.set_hard_medium_ids()`
3. The system has sufficient context to classify requests

The feature gracefully falls back to default behavior when problem classifications are not available.
