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
Parallel Utilities for Arctic Inference

These utilities were extracted from the original model_runner.py to keep
the code organized. They handle shift parallel mode switching.

Note: These functions manage global state for parallel execution modes.
"""

import contextlib
from typing import Optional

import vllm.distributed.parallel_state as parallel_state

# Global state for shift parallel mode
SP_TP_MODE = None


@contextlib.contextmanager
def set_shift_parallel_mode(mode: Optional[bool]):
    """
    Context manager to temporarily switch between shift parallel and normal mode.
    
    When shift parallel is enabled, the tensor parallel group is swapped between:
    - SP_TP mode: For small batches, uses more tensor parallelism
    - Original TP mode: For large batches, uses sequence parallelism
    
    Args:
        mode: True to enable shift parallel, False to use normal mode,
              None to do nothing
    
    Usage:
        with set_shift_parallel_mode(True):
            # Execute with shift parallel mode
            model_output = shift_model(...)
    
    Note:
        This function manipulates vLLM's internal parallel state.
        It's carefully designed to restore state even if exceptions occur.
    """
    if mode is None:
        yield
        return

    global SP_TP_MODE

    if not is_shift_parallel_mode():
        assert not parallel_state._TP_STATE_PATCHED
        parallel_state._ORIG_TP = parallel_state._TP

    old_mode = SP_TP_MODE
    old_tp_group = parallel_state.get_tp_group()
    SP_TP_MODE = mode

    parallel_state._TP = (parallel_state._SP_TP
                          if mode else parallel_state._ORIG_TP)

    try:
        yield
    finally:
        # restore the original state
        SP_TP_MODE = old_mode
        parallel_state._TP = old_tp_group


def is_shift_parallel_mode() -> bool:
    """
    Check if shift parallel mode is currently enabled.
    
    Returns:
        True if shift parallel mode is active, False otherwise
    """
    global SP_TP_MODE
    return SP_TP_MODE is True


def round_up(value: int, multiple: int) -> int:
    """
    Round up a value to the nearest multiple.
    
    Args:
        value: The value to round up
        multiple: The multiple to round up to
    
    Returns:
        The rounded up value
    
    Example:
        >>> round_up(10, 4)
        12
        >>> round_up(12, 4)
        12
    """
    return ((value + multiple - 1) // multiple) * multiple

