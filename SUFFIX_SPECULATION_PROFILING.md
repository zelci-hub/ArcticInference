# Suffix Speculation Profiling Documentation

## Overview

This document describes the comprehensive profiling system added to the `propose_suffix_draft_token_ids` method for performance analysis and optimization of suffix speculation with differentiated parameters.

## Profiling Components

### 1. Timing Infrastructure

#### Buffer Configuration
```python
# Environment variables for configuration
ARCTIC_SUFFIX_SPECULATION_BUFFER_SIZE=200  # Buffer size before flush
ARCTIC_SUFFIX_SPECULATION_FLUSH_SEC=3      # Time threshold for flush
```

#### Output Files
- **File**: `suffix_speculation_timing_rank_{rank}_local_{local_rank}.jsonl`
- **Location**: `$ARCTIC_METRICS_DIR/{timestamp}/`
- **Format**: JSON Lines (one JSON object per line)

### 2. Timing Measurements

#### Method-Level Timing
- **Total method execution time**: Complete `propose_suffix_draft_token_ids` duration
- **Indices calculation time**: Time to get hard/medium/easy classifications
- **Processing loop time**: Time for all request processing

#### Request-Level Timing
- **Individual request processing time**: Per-request total time
- **Speculation time**: Time spent in `_suffix_cache.speculate()` call
- **Skip detection**: Tracking of skipped requests with reasons

### 3. Collected Metrics

#### Batch-Level Metrics
```json
{
  "timestamp": 1699123456.789,
  "call_type": "propose_suffix_draft_token_ids",
  "batch_size": 32,
  "method_total_time_ms": 45.67,
  "indices_calculation_time_ms": 2.34,
  "processing_time_ms": 42.11,
  "total_speculation_time_ms": 38.90,
  "avg_speculation_time_ms": 1.22
}
```

#### Difficulty Distribution
```json
{
  "difficulty_distribution": {
    "hard": 8,
    "medium": 12,
    "easy": 10,
    "default": 2,
    "skipped": 0
  }
}
```

#### Spec Parameters Used
```json
{
  "spec_parameters": {
    "hard": {"spec": 16, "prob": 0.1, "factor": 2},
    "medium": {"spec": 8, "prob": 0.1, "factor": 1},
    "easy": {"spec": 4, "prob": 0.1, "factor": 1}
  }
}
```

#### Individual Request Details
```json
{
  "individual_requests": [
    {
      "request_index": 0,
      "req_id": "req_123",
      "problem_id": "prob_456",
      "difficulty_category": "hard",
      "speculation_time_ms": 2.45,
      "total_request_time_ms": 3.12,
      "spec_params": {
        "max_spec_tokens": 16,
        "max_spec_factor": 2,
        "min_token_prob": 0.1
      },
      "result_score": 8.5,
      "result_tokens": 12,
      "source": "problem_prob_456",
      "pattern_length": 64,
      "skipped": false
    }
  ]
}
```

### 4. Skip Tracking

Requests can be skipped for various reasons, all tracked with timing:

#### Skip Reasons
- `"no_sampled_ids"`: No sampled tokens available
- `"max_model_len_exceeded"`: Sequence length exceeds model limits

#### Skip Data Format
```json
{
  "request_index": 5,
  "processing_time_ms": 0.05,
  "skipped": true,
  "reason": "no_sampled_ids"
}
```

### 5. Performance Analysis Features

#### Timing Breakdown
- **Method overhead**: `method_total_time - processing_time`
- **Speculation efficiency**: `speculation_time / total_request_time`
- **Batch processing efficiency**: `total_speculation_time / method_total_time`

#### Difficulty-Based Analysis
- Performance comparison across hard/medium/easy categories
- Parameter effectiveness measurement
- Resource allocation optimization insights

#### Source Analysis
- Cache hit analysis by source (`local`, `global`, `problem_*`)
- Problem-specific cache effectiveness
- Speculation success rates by difficulty

### 6. Buffer Management

#### Automatic Flushing
```python
# Flush conditions
should_flush_by_n = len(buffer) >= flush_every_n
should_flush_by_time = (current_time - last_flush_time) >= flush_every_s
```

#### Manual Flushing
- Process exit handlers ensure data persistence
- Force flush capability for immediate data availability

### 7. Integration with Existing Systems

#### Environment Variables
- Respects existing `ARCTIC_METRICS_DIR` structure
- Uses consistent rank/local_rank naming
- Configurable buffer sizes and flush intervals

#### Error Handling
- Graceful degradation on logging failures
- Non-blocking profiling (doesn't affect speculation performance)
- Comprehensive exception handling

### 8. Usage Examples

#### Analyzing Performance by Difficulty
```bash
# Extract hard problem performance
cat suffix_speculation_timing_*.jsonl | \
  jq '.individual_requests[] | select(.difficulty_category == "hard") | .speculation_time_ms' | \
  awk '{sum+=$1; count++} END {print "Avg hard speculation time:", sum/count, "ms"}'
```

#### Batch Size Impact Analysis
```bash
# Correlation between batch size and processing time
cat suffix_speculation_timing_*.jsonl | \
  jq '{batch_size: .batch_size, avg_time: .avg_speculation_time_ms}' | \
  sort_by(.batch_size)
```

#### Cache Source Effectiveness
```bash
# Count speculation sources
cat suffix_speculation_timing_*.jsonl | \
  jq '.individual_requests[].source' | \
  sort | uniq -c
```

### 9. Performance Considerations

#### Minimal Overhead
- Buffered I/O reduces file system impact
- JSON serialization optimized with custom handler
- Timing measurements use high-precision `perf_counter()`

#### Memory Management
- Automatic buffer clearing after flush
- Bounded buffer sizes prevent memory leaks
- Process exit cleanup ensures data persistence

### 10. Troubleshooting

#### Common Issues
1. **Missing output files**: Check `ARCTIC_METRICS_DIR` permissions
2. **Incomplete data**: Verify buffer flush settings
3. **Performance impact**: Adjust buffer sizes if needed

#### Debug Information
- Process rank and local rank included in all records
- Timestamp precision for correlation with other logs
- Comprehensive error logging without affecting main execution

## Benefits

1. **Performance Optimization**: Identify bottlenecks in speculation pipeline
2. **Parameter Tuning**: Data-driven optimization of spec parameters
3. **Resource Allocation**: Understand computational costs by difficulty
4. **Cache Analysis**: Measure effectiveness of different cache sources
5. **Scalability Insights**: Batch size impact on processing efficiency
6. **Problem-Specific Optimization**: Per-problem performance analysis
