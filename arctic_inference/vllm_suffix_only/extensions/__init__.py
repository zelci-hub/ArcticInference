# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Arctic Extensions for vLLM

简洁的扩展架构：只包含 Suffix Cache 推测解码。

这个架构的特点：
- 完全独立于 vLLM 内部实现
- 不需要 hidden_states
- 不修改 vLLM 执行流程
- 纯后处理逻辑
- 易于测试和维护
"""

from .suffix_cache_extension import SuffixCacheExtension, SuffixResult
from .metric_extension import MetricExtension

__all__ = [
    "SuffixCacheExtension",
    "SuffixResult",
    "MetricExtension",
]
