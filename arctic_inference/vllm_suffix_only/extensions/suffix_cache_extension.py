# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Suffix Cache Extension

纯 Suffix Cache 推测解码逻辑，完全独立于 vLLM 内部实现。

特点：
- 不需要 hidden_states
- 不需要 model forward
- 只需要采样后的 token IDs
- 纯后处理逻辑
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass
class SuffixResult:
    """Suffix cache 推测结果"""
    token_ids: List[int]
    score: float


class SuffixCacheExtension:
    """
    Suffix Cache 扩展（完全独立）。
    
    这个类完全不依赖 vLLM 内部实现：
    - 不访问 hidden_states
    - 不修改 model forward
    - 只处理采样后的数据
    
    使用方式：
        # 创建
        suffix_ext = SuffixCacheExtension(vllm_config)
        
        # 更新 cache
        suffix_ext.update_cache(
            sampled_token_ids=sampled_ids,
            req_ids=req_ids,
            ...
        )
        
        # 推测 tokens
        results = suffix_ext.propose_tokens(
            sampled_token_ids=sampled_ids,
            req_ids=req_ids,
            ...
        )
    """
    
    def __init__(self, vllm_config):
        """
        初始化 Suffix Cache Extension。
        
        Args:
            vllm_config: vLLM 配置对象
        """
        self.config = vllm_config
        self.suffix_cache = None
        self.enabled = False
        
        # 配置参数
        self.method = None
        self.num_speculative_tokens = 0
        self.suffix_cache_max_depth = 64
        self.suffix_max_spec_factor = 1.0
        self.suffix_max_spec_offset = 0
        self.suffix_min_token_prob = 0.1
        
        # 检查并初始化
        if hasattr(vllm_config, 'speculative_config') and vllm_config.speculative_config:
            self._setup_suffix_cache(vllm_config.speculative_config)
    
    def _setup_suffix_cache(self, spec_config):
        """
        设置 suffix cache。
        
        Args:
            spec_config: Speculative decoding 配置
        """
        # 检查是否启用 suffix decoding
        if not spec_config.enable_suffix_decoding:
            return
        
        # 验证方法兼容性
        if spec_config.method not in ("arctic", "suffix", "mlp_speculator"):
            raise ValueError(
                f"Suffix decoding is only supported with 'arctic', 'suffix', "
                f"or 'mlp_speculator' methods. Got: {spec_config.method}"
            )
        
        # 导入并创建 suffix cache
        try:
            from arctic_inference.suffix_decoding import SuffixDecodingCache
        except ImportError:
            logger.warning("Failed to import SuffixDecodingCache, suffix cache disabled")
            return
        
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=spec_config.suffix_cache_max_depth,
            max_cached_requests=spec_config.suffix_cache_max_requests,
        )
        
        # 保存配置参数
        self.method = spec_config.method
        self.num_speculative_tokens = spec_config.num_speculative_tokens
        self.suffix_cache_max_depth = spec_config.suffix_cache_max_depth
        self.suffix_max_spec_factor = spec_config.suffix_max_spec_factor
        self.suffix_max_spec_offset = spec_config.suffix_max_spec_offset
        self.suffix_min_token_prob = spec_config.suffix_min_token_prob
        
        self.enabled = True
        
        logger.info(
            f"SuffixCacheExtension initialized: "
            f"method={self.method}, "
            f"max_depth={self.suffix_cache_max_depth}, "
            f"num_spec_tokens={self.num_speculative_tokens}"
        )
    
    def is_enabled(self) -> bool:
        """
        检查 suffix cache 是否启用。
        
        Returns:
            True 如果启用，False 否则
        """
        return self.enabled and self.suffix_cache is not None
    
    def update_cache(
        self,
        sampled_token_ids: List[List[int]],
        req_ids: List[str],
        req_id_to_index: Dict[str, int],
        token_ids_cpu: torch.Tensor,
        num_prompt_tokens: torch.Tensor,
    ) -> None:
        """
        更新 suffix cache（添加新生成的 tokens）。
        
        这个方法在每次采样后调用，用于维护 suffix cache 的状态。
        
        Args:
            sampled_token_ids: 采样的 token IDs，List[List[int]]
            req_ids: 请求 IDs
            req_id_to_index: 请求 ID 到索引的映射
            token_ids_cpu: CPU 上的 token IDs tensor
            num_prompt_tokens: 每个请求的 prompt token 数量
        """
        if not self.is_enabled():
            return
        
        seen_req_ids = set()
        
        for i, sampled_ids in enumerate(sampled_token_ids):
            req_id = req_ids[i]
            seen_req_ids.add(req_id)
            
            if not sampled_ids:
                continue
            
            index = req_id_to_index[req_id]
            
            # 如果是新请求，启动它
            if req_id not in self.suffix_cache.active_requests:
                # 如果在 cached_requests 中，先驱逐
                if req_id in self.suffix_cache.cached_requests:
                    self.suffix_cache.evict_cached_response(req_id)
                
                # 启动新请求
                num_prompt = int(num_prompt_tokens[index])
                prompt_tokens = token_ids_cpu[index, :num_prompt]
                self.suffix_cache.start_request(req_id, prompt_tokens)
            
            # 添加新生成的 tokens
            self.suffix_cache.add_active_response(req_id, sampled_ids)
        
        # 停止不再活跃的请求
        for req_id in list(self.suffix_cache.active_requests):
            if req_id not in seen_req_ids:
                self.suffix_cache.stop_request(req_id)
    
    def propose_tokens(
        self,
        sampled_token_ids: List[List[int]],
        req_ids: List[str],
        token_ids_cpu: torch.Tensor,
        num_tokens_no_spec: torch.Tensor,
        max_model_len: int,
    ) -> List[SuffixResult]:
        """
        从 suffix cache 推测 tokens。
        
        这个方法使用 suffix cache 中的历史模式来预测接下来的 tokens。
        
        Args:
            sampled_token_ids: 采样的 token IDs
            req_ids: 请求 IDs
            token_ids_cpu: CPU 上的 token IDs tensor
            num_tokens_no_spec: 每个请求已有的 token 数量（不包括推测）
            max_model_len: 最大模型长度
        
        Returns:
            每个请求的 SuffixResult（包含 token_ids 和 score）
        """
        if not self.is_enabled():
            return [SuffixResult([], 0.0) for _ in sampled_token_ids]
        
        results = []
        
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                results.append(SuffixResult([], 0.0))
                continue
            
            req_id = req_ids[i]
            start_idx = int(num_tokens_no_spec[i])
            end_idx = start_idx + len(sampled_ids)
            
            # 提取用于匹配的 pattern
            pattern_size = min(end_idx, self.suffix_cache_max_depth)
            pattern = token_ids_cpu[i, end_idx - pattern_size:end_idx]
            pattern = pattern.tolist()
            
            # 计算可以推测的最大 token 数量
            max_spec_tokens = min(
                self.num_speculative_tokens,
                max_model_len - end_idx - 1,
            )
            
            if max_spec_tokens <= 0:
                results.append(SuffixResult([], 0.0))
                continue
            
            # 从 suffix cache 推测
            result = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=max_spec_tokens,
                max_spec_factor=self.suffix_max_spec_factor,
                max_spec_offset=self.suffix_max_spec_offset,
                min_token_prob=self.suffix_min_token_prob,
            )
            
            results.append(SuffixResult(result.token_ids, result.score))
        
        return results
    
    def get_min_score(self) -> float:
        """
        获取 suffix cache 的最小 score 阈值。
        
        如果 method 是 "suffix"，总是使用 suffix（min_score=0）。
        否则，只有 score >= num_speculative_tokens 才使用 suffix。
        
        Returns:
            最小 score 阈值
        """
        if not self.is_enabled():
            return float('inf')
        
        if self.method == "suffix":
            return 0.0  # 总是使用 suffix
        else:
            return float(self.num_speculative_tokens)
