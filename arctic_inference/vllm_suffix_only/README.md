# Arctic vLLM Integration - Suffix Cache Only

## 📋 概述

这是一个**极简架构**，实现了 vLLM 和 Suffix Cache 的**完全分离**。

### 🎯 设计目标

1. **✅ vLLM 代码保持 99% 不变**
2. **✅ Suffix 逻辑完全独立**
3. **✅ 只有一个清晰的 hook 点**
4. **✅ 易于测试和维护**

---

## 🏗️ 架构设计

### 极简架构

```
┌────────────────────────────────────┐
│ vLLM execute_model (unchanged)     │
│                                    │
│ 1. Prepare inputs                  │
│ 2. Model forward                   │
│ 3. Compute logits                  │
│ 4. Sampling                        │
│    ↓ sampled_token_ids            │
└────────────────────────────────────┘
         │
         │ Hook (1 function call)
         ▼
┌────────────────────────────────────┐
│ SuffixCacheExtension               │
│ (completely independent)           │
│                                    │
│ 1. update_cache()                  │
│ 2. propose_tokens()                │
│ 3. return spec_token_ids           │
└────────────────────────────────────┘
```

### 代码量对比

| Component | Lines | 描述 |
|-----------|-------|------|
| **model_runner.py** | ~150 | 最小化 patch，只有 2 个 patch 点 |
| **suffix_cache_extension.py** | ~250 | 纯 Suffix 逻辑，完全独立 |
| **Total** | ~400 | vs 原版本 983 lines (↓60%) |

---

## 📁 文件结构

```
vllm_new/
├── __init__.py                  # Package exports
├── model_runner.py              # Minimal patch (150 lines)
├── parallel_utils.py            # Helper functions
├── extensions/
│   ├── __init__.py
│   ├── suffix_cache_extension.py  # Suffix cache logic (250 lines)
│   └── metric_extension.py        # Metrics (optional)
└── README.md                    # This file
```

---

## 🚀 使用方式

### 1. 内部使用（在 vLLM 中）

#### 启用 Suffix Cache

在 vLLM 配置中启用 suffix decoding：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="your-model",
    speculative_config={
        "method": "suffix",  # 或 "arctic"
        "enable_suffix_decoding": True,
        "suffix_cache_max_depth": 10,
        "suffix_cache_max_requests": 1000,
        "num_speculative_tokens": 5,
    }
)
```

### 2. 自动使用 Suffix Cache

Suffix cache 会自动工作：

```python
# Generate
outputs = llm.generate(prompts, sampling_params)

# Suffix cache 自动：
# 1. 更新 cache（学习模式）
# 2. 推测 tokens（加速生成）
# 3. 返回结果
```

### 3. 检查 Suffix Cache 状态

```python
# 查看日志
# INFO: ✓ Suffix Cache Extension enabled
# INFO: SuffixCacheExtension initialized: method=suffix, max_depth=10, num_spec_tokens=5
```

---

## 🔍 工作原理

### Suffix Cache 的执行流程

```python
# ========== Step 1: vLLM Sampling ==========
# vLLM 正常执行，生成 sampled_token_ids
sampled_token_ids = vllm.sampler(logits)  # [[1, 2, 3], [4, 5]]

# ========== Step 2: Update Cache ==========
# 学习这些 token 的模式
suffix_extension.update_cache(
    sampled_token_ids=sampled_token_ids,
    req_ids=["req_1", "req_2"],
    ...
)

# ========== Step 3: Propose Speculative Tokens ==========
# 根据历史模式推测接下来的 tokens
suffix_results = suffix_extension.propose_tokens(
    sampled_token_ids=sampled_token_ids,
    ...
)
# suffix_results[0] = SuffixResult([6, 7, 8], score=3.5)
# suffix_results[1] = SuffixResult([], score=0.0)

# ========== Step 4: Filter by Score ==========
# 只使用 score 足够高的推测
min_score = suffix_extension.get_min_score()  # 0.0 or 5.0
spec_token_ids = []
for result in suffix_results:
    if result.score >= min_score:
        spec_token_ids.append(result.token_ids)  # Use it
    else:
        spec_token_ids.append([])  # Skip it

# ========== Step 5: Return ==========
# 返回给 vLLM
return ModelRunnerOutput(
    sampled_token_ids=sampled_token_ids,
    spec_token_ids=spec_token_ids,  # [[6,7,8], []]
)
```

### Suffix Cache 不需要什么

Suffix cache 只需要采样后的数据，**不需要**：

- ❌ `hidden_states`
- ❌ `sample_hidden_states`
- ❌ Model forward 修改
- ❌ vLLM 内部变量

这使得它可以**完全独立**！

---

## 🧪 测试

### 单元测试 Suffix Extension

```python
from arctic_inference.vllm_new.extensions import SuffixCacheExtension
import torch

def test_suffix_cache():
    # 创建 mock config
    class MockConfig:
        class SpecConfig:
            enable_suffix_decoding = True
            method = "suffix"
            suffix_cache_max_depth = 10
            suffix_cache_max_requests = 100
            num_speculative_tokens = 5
            suffix_max_spec_factor = 2.0
            suffix_max_spec_offset = 2
            suffix_min_token_prob = 0.1
        
        speculative_config = SpecConfig()
    
    # 创建 extension
    suffix_ext = SuffixCacheExtension(MockConfig())
    
    assert suffix_ext.is_enabled() == True
    
    # 测试 update_cache
    suffix_ext.update_cache(
        sampled_token_ids=[[1, 2, 3]],
        req_ids=["test_req"],
        req_id_to_index={"test_req": 0},
        token_ids_cpu=torch.tensor([[1, 2, 3, 0, 0]]),
        num_prompt_tokens=torch.tensor([0]),
    )
    
    # 测试 propose_tokens
    results = suffix_ext.propose_tokens(
        sampled_token_ids=[[1, 2, 3]],
        req_ids=["test_req"],
        token_ids_cpu=torch.tensor([[1, 2, 3, 0, 0]]),
        num_tokens_no_spec=torch.tensor([0]),
        max_model_len=1024,
    )
    
    assert len(results) == 1
    assert isinstance(results[0].token_ids, list)
    assert isinstance(results[0].score, float)
    
    print("✓ All tests passed!")
```

### 集成测试

```python
from vllm import LLM, SamplingParams

# 创建 LLM（会自动使用 vllm_new 的 patch）
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    speculative_config={
        "method": "suffix",
        "enable_suffix_decoding": True,
        "num_speculative_tokens": 5,
    }
)

# Generate
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

# 检查结果
for output in outputs:
    print(output.outputs[0].text)
```

---

## 📊 性能对比

### 代码复杂度

| Metric | Original | Refactored | Improvement |
|--------|----------|------------|-------------|
| **Total Lines** | 983 | ~400 | ↓ 60% |
| **Patch Points** | 6+ | 2 | ↓ 67% |
| **Dependencies** | Mixed | Separated | ✅ Clean |
| **Test Coverage** | Difficult | Easy | ✅ Improved |

### 运行性能

- **无性能损失**：只增加一个函数调用（可忽略）
- **Suffix Cache 加速**：推测解码可以加速生成
- **内存开销**：Suffix cache 的内存（可配置）

---

## 🔧 配置选项

### Suffix Cache 配置

```python
speculative_config = {
    # 方法
    "method": "suffix",  # 或 "arctic", "mlp_speculator"
    
    # 启用 suffix decoding
    "enable_suffix_decoding": True,
    
    # Suffix cache 参数
    "suffix_cache_max_depth": 10,      # 匹配深度
    "suffix_cache_max_requests": 1000,  # 最大请求数
    "num_speculative_tokens": 5,        # 推测 token 数量
    "suffix_max_spec_factor": 2.0,      # 最大推测因子
    "suffix_max_spec_offset": 2,        # 最大推测偏移
    "suffix_min_token_prob": 0.1,       # 最小 token 概率
}
```

### Min Score 说明

- **method="suffix"**: `min_score=0.0` (总是使用 suffix)
- **method="arctic"**: `min_score=num_speculative_tokens` (只有 score 足够高才使用)

---

## 🎯 设计原则

### 1. 最小化修改

```python
# vLLM 的 execute_model 只有 1% 的修改
def execute_model(self, scheduler_output, ...):
    # 99%: vLLM 原始逻辑
    vllm_result = self._orig_execute_model(...)
    
    # 1%: Suffix hook
    vllm_result.spec_token_ids = self._apply_suffix_cache(...)
    
    return vllm_result
```

### 2. 完全分离

```python
# SuffixCacheExtension 完全不访问 vLLM 内部
class SuffixCacheExtension:
    def propose_tokens(self, sampled_token_ids, ...):
        # 只使用传入的参数
        # 不访问 self.runner.xxx
        # 不依赖 vLLM 内部状态
        ...
```

### 3. 清晰边界

```python
# 只有一个 hook 点
vllm_result.spec_token_ids = self._apply_suffix_cache(
    vllm_result.sampled_token_ids  # 唯一的输入
)
```

---

## 🔄 升级 vLLM

当 vLLM 更新时：

```python
# 1. 检查 execute_model 的返回值是否变化
if hasattr(vllm_result, 'sampled_token_ids'):
    # OK, 兼容
    pass
else:
    # 需要调整

# 2. 检查 input_batch 的属性是否变化
self.input_batch.req_ids
self.input_batch.token_ids_cpu
# ...

# 3. 如果都没变，无需修改！
```

---

## 🐛 调试

### 检查 Suffix Cache 是否工作

```python
import logging
logging.basicConfig(level=logging.INFO)

# 查看日志
# INFO: ✓ Suffix Cache Extension enabled
# INFO: SuffixCacheExtension initialized: ...
```

### 禁用 Suffix Cache

```python
# 方法 1: 配置中禁用
speculative_config = {
    "enable_suffix_decoding": False,
}

# 方法 2: 不提供 speculative_config
# Suffix cache 会自动禁用
```

---

## 📚 相关文档

- `SUFFIX_ONLY_CLEAN_ARCHITECTURE.md` - 详细的架构设计
- `extensions/suffix_cache_extension.py` - 代码实现
- `model_runner.py` - Patch 实现

---

## ✅ 总结

### 这个架构的优势

1. **✅ 简洁**：总共 ~400 行代码
2. **✅ 清晰**：vLLM 和 Suffix 完全分离
3. **✅ 可测**：Suffix 可以独立测试
4. **✅ 可维**：vLLM 更新影响最小
5. **✅ 高效**：无性能损失

### 适用场景

- ✅ **只需要 Suffix Cache** 推测解码
- ✅ 需要**易于维护**的代码
- ✅ 需要**易于测试**的架构
- ✅ 希望**最小化对 vLLM 的修改**

### 不适用场景

- ❌ 需要 Arctic Proposer（需要 hidden_states）
- ❌ 需要 Ulysses Parallelism
- ❌ 需要 Shift Parallel

---

**Last Updated**: 2025-11-06
