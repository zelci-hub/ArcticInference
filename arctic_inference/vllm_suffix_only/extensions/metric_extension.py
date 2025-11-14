# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Metric Extension for Arctic Inference

This extension handles performance metrics collection and timing statistics.
All metric-related code is centralized here, separate from core model execution logic.
"""

import os
import json
import time
import atexit
import logging
from datetime import datetime
from typing import Any

import numpy as np

from ..contracts.model_runner_extension import (
    ModelRunnerExtension,
    ExecutionContext,
    ExecutionResult,
)

logger = logging.getLogger(__name__)


class MetricExtension(ModelRunnerExtension):
    """Extension for collecting and recording performance metrics.
    
    This extension handles:
    - Execution timing statistics
    - CPU/GPU metrics
    - Buffered I/O for efficient metric writing
    - Automatic flushing on exit
    
    Metrics are written to JSONL files for easy analysis.
    """
    
    def __init__(self, model_runner, config=None):
        """Initialize metric collection.
        
        Args:
            model_runner: The vLLM ModelRunner instance
            config: Optional configuration (currently unused, for future extension)
        """
        self.runner = model_runner
        self.config = config
        
        # Initialize CPU timing buffer
        self._cpu_timing_buffer: list[str] = []
        self._timing_flush_every_n: int = int(
            os.getenv("ARCTIC_TIMING_BUFFER_SIZE", "400")
        )
        self._timing_flush_every_s: float = float(
            os.getenv("ARCTIC_TIMING_FLUSH_SEC", "5")
        )
        self._timing_last_flush_time: float = time.monotonic()
        
        # Setup output directory and file path
        self._setup_output_paths()
        
        # Ensure buffer flushes on process exit
        atexit.register(lambda: self._flush_timing_buffer(force=True))
        
        logger.info("Metric extension initialized")
    
    def _setup_output_paths(self):
        """Setup output directory and file paths for metrics."""
        # Get or create metrics directory
        root_dir = os.getenv("ARCTIC_METRICS_DIR", "/data/zshao/tmp/arctic_metrics")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(root_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # Propagate to env so other writers use the same timestamped directory
        os.environ["ARCTIC_METRICS_DIR"] = output_dir
        
        # Create file path with rank information
        local_rank = os.getenv("LOCAL_RANK", "0")
        rank = os.getenv("RANK", "0")
        self._timing_file_path_cpu = os.path.join(
            output_dir, f"CPU_execution_timing_rank_{rank}_local_{local_rank}.jsonl"
        )
        
        logger.info(f"Metrics will be written to: {self._timing_file_path_cpu}")
    
    # ========== ModelRunnerExtension Interface Implementation ==========
    
    def wrap_model(self, model: Any) -> Any:
        """No model wrapping needed for metrics."""
        return model
    
    def before_execute(self, ctx: ExecutionContext) -> ExecutionContext:
        """Record execution start time."""
        # Add start time to metadata
        ctx.metadata['metric_start_time'] = time.perf_counter()
        return ctx
    
    def after_execute(
        self,
        ctx: ExecutionContext,
        result: ExecutionResult,
    ) -> ExecutionResult:
        """Record execution timing and write metrics.
        
        This is called after each model execution to collect timing data.
        """
        # Calculate execution time
        start_time = ctx.metadata.get('metric_start_time')
        if start_time is not None:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Collect timing data from result metadata
            timing_data = {
                "call_type": "execute_model_timing",
                "execution_time_ms": execution_time * 1000,
                "timestamp": time.time(),
                "num_scheduled_tokens": ctx.num_scheduled_tokens,
            }
            
            # Add any additional metrics from result metadata
            if result.metadata:
                timing_data.update(result.metadata)
            
            # Write timing stats
            self._write_timing_stats(timing_data)
        
        return result
    
    # ========== Metric Collection Methods ==========
    
    def _write_timing_stats(self, timing_data: dict):
        """Buffer execution timing data and flush periodically to reduce I/O.
        
        Args:
            timing_data: Dictionary containing timing information
        """
        self._cpu_timing_buffer.append(
            json.dumps(timing_data, default=self._json_serializable)
        )
        
        # Flush conditions: buffer size or time threshold
        should_flush_by_n = len(self._cpu_timing_buffer) >= self._timing_flush_every_n
        should_flush_by_time = (
            time.monotonic() - self._timing_last_flush_time
        ) >= self._timing_flush_every_s
        
        if should_flush_by_n or should_flush_by_time:
            self._flush_timing_buffer()
    
    def _flush_timing_buffer(self, force: bool = False):
        """Flush buffered timing lines to disk.
        
        Args:
            force: If True, flush unconditionally (e.g., at exit)
        """
        try:
            if not self._cpu_timing_buffer and not force:
                return
            
            # Nothing to write if empty and not forced
            if not self._cpu_timing_buffer:
                self._timing_last_flush_time = time.monotonic()
                return
            
            # Write all pending lines at once
            with open(self._timing_file_path_cpu, "a") as f:
                f.write("\n".join(self._cpu_timing_buffer) + "\n")
            
            self._cpu_timing_buffer.clear()
            self._timing_last_flush_time = time.monotonic()
            
        except Exception as e:
            logger.error(f"Failed to flush timing stats: {e}")
    
    def _json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-serializable types.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle scalar numpy types
            return obj.item()
        else:
            return str(obj)

