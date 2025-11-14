# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Arctic vLLM Integration (Suffix Cache Only)

这是一个简洁的架构，只包含 Suffix Cache 推测解码。

特点：
- vLLM 代码保持 99% 不变
- Suffix 逻辑完全独立
- 只有一个清晰的 hook 点
- 易于测试和维护

Usage:
    from arctic_inference.vllm_new.model_runner import GPUModelRunnerPatch
    from arctic_inference.vllm_new.extensions import SuffixCacheExtension
"""

from .model_runner import GPUModelRunnerPatch
from .extensions import SuffixCacheExtension, SuffixResult

__all__ = [
    "GPUModelRunnerPatch",
    "SuffixCacheExtension",
    "SuffixResult",
]
