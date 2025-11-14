# Arctic Extensions

This directory contains all extension implementations for Arctic Inference.

## Available Extensions

### 1. ArcticExtension (`arctic_extension.py`)

Implements all Arctic-specific features:
- **Ulysses Sequence Parallelism**: For handling long sequences
- **Shift Parallelism**: Dynamic TP adjustment for small batches
- **Suffix Cache**: Pattern-based speculative decoding
- **Arctic Proposer**: Model-based speculative decoding

**Usage**: Automatically enabled when Arctic features are configured in `vllm_config`.

### 2. MetricExtension (`metric_extension.py`)

Collects and records performance metrics:
- Execution timing statistics
- CPU/GPU performance metrics
- Buffered I/O for efficient metric writing
- Automatic flushing on exit

**Usage**: 
- Enabled by default (set `ARCTIC_ENABLE_METRICS=0` to disable)
- Configure buffer size: `ARCTIC_TIMING_BUFFER_SIZE=400` (default)
- Configure flush interval: `ARCTIC_TIMING_FLUSH_SEC=5` (default seconds)
- Output directory: `ARCTIC_METRICS_DIR=/path/to/metrics` (default: `/data/zshao/tmp/arctic_metrics`)

**Output Format**: JSONL files with one metric record per line:
```json
{
  "call_type": "execute_model_timing",
  "execution_time_ms": 123.45,
  "timestamp": 1699876543.21,
  "num_scheduled_tokens": 100
}
```

## Extension Architecture

### CompositeExtension

Multiple extensions can be combined using `CompositeExtension`:

```python
from arctic_inference.vllm_new.extensions import ArcticExtension, MetricExtension
from arctic_inference.vllm_new.contracts import CompositeExtension

extensions = [
    ArcticExtension(runner, config),
    MetricExtension(runner),
]
composite = CompositeExtension(extensions)
```

Extensions are called in sequence:
1. `wrap_model()`: Each extension wraps the model in order
2. `before_execute()`: Each extension preprocesses in order
3. vLLM executes the model
4. `after_execute()`: Each extension postprocesses in order

### Creating Custom Extensions

To create a new extension:

1. **Implement the interface**:
```python
from arctic_inference.vllm_new.contracts import ModelRunnerExtension

class MyExtension(ModelRunnerExtension):
    def wrap_model(self, model):
        # Wrap or modify model
        return model
    
    def before_execute(self, ctx):
        # Preprocess
        return ctx
    
    def after_execute(self, ctx, result):
        # Postprocess
        return result
```

2. **Register in `_create_extension()`** in `model_runner.py`:
```python
def _create_extension(self, vllm_config):
    extensions = []
    
    # ... existing extensions ...
    
    # Add your extension
    if some_condition:
        extensions.append(MyExtension(self, config))
    
    return CompositeExtension(extensions) if len(extensions) > 1 else extensions[0]
```

## Design Principles

1. **Single Responsibility**: Each extension handles one concern
2. **Independent**: Extensions don't depend on each other
3. **Composable**: Multiple extensions can work together
4. **Testable**: Each extension can be tested in isolation

## Extension Execution Order

```
ModelRunner.execute_model()
    ↓
1. before_execute()
    ├─ ArcticExtension.before_execute()
    └─ MetricExtension.before_execute()  ← Records start time
    ↓
2. vLLM execution (pure vLLM code)
    ↓
3. after_execute()
    ├─ ArcticExtension.after_execute()   ← Add spec tokens, update cache
    └─ MetricExtension.after_execute()   ← Record timing, write metrics
    ↓
Return result
```

## Performance Considerations

### MetricExtension
- **Buffered I/O**: Metrics are buffered and flushed periodically
- **Minimal overhead**: < 0.1% performance impact
- **Async-safe**: Uses buffering to avoid blocking execution

### ArcticExtension
- **Zero overhead** when features are disabled
- **Conditional execution**: Only runs when needed
- **Optimized**: Reuses vLLM's existing infrastructure

## Environment Variables

```bash
# Metrics Collection
export ARCTIC_ENABLE_METRICS=1              # Enable/disable metrics (default: 1)
export ARCTIC_METRICS_DIR=/path/to/metrics  # Output directory
export ARCTIC_TIMING_BUFFER_SIZE=400        # Buffer size (lines)
export ARCTIC_TIMING_FLUSH_SEC=5            # Flush interval (seconds)

# Arctic Features (configured via vllm_config, not env vars)
# - Ulysses: vllm_config.parallel_config.ulysses_sequence_parallel_size
# - Speculative: vllm_config.speculative_config.method
```

## Examples

### Example 1: Arctic Only
```python
# Configure Arctic features
vllm_config.parallel_config.ulysses_sequence_parallel_size = 4

# Disable metrics
os.environ["ARCTIC_ENABLE_METRICS"] = "0"

# Result: Only ArcticExtension is active
```

### Example 2: Metrics Only
```python
# No Arctic features configured

# Enable metrics
os.environ["ARCTIC_ENABLE_METRICS"] = "1"

# Result: Only MetricExtension is active
```

### Example 3: Both (Default)
```python
# Configure Arctic features
vllm_config.parallel_config.ulysses_sequence_parallel_size = 4

# Metrics enabled by default

# Result: CompositeExtension with both ArcticExtension and MetricExtension
```

## Troubleshooting

### Metrics not being written
1. Check if `ARCTIC_ENABLE_METRICS=1`
2. Check write permissions for `ARCTIC_METRICS_DIR`
3. Check logs for flush errors

### Extension not being called
1. Check `_create_extension()` logic in `model_runner.py`
2. Check extension's `__init__()` for initialization errors
3. Enable debug logging: `logger.setLevel(logging.DEBUG)`

---

**Last Updated**: 2025-11-06
**Version**: v1.0

