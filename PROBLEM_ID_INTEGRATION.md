# Problem ID Integration in ArcticInference

## 概述

本文档描述了在新版本ArcticInference中成功集成的problem_id支持功能，包括hard_medium_indices相关内容。

## 🎯 实现的功能

### 1. **ProblemIdContextManager**
- **位置**: `arctic_inference/vllm/model_runner.py`
- **功能**: 线程安全的problem_id上下文管理
- **特性**:
  - 支持批处理problem_ids设置和获取
  - req_id到problem_id的映射管理
  - hard/medium/easy问题分类支持
  - 线程本地存储确保并发安全

### 2. **LLM层面的problem_id支持**
- **位置**: `arctic_inference/vllm/llm.py`
- **功能**: 扩展vLLM的LLM.generate()方法
- **特性**:
  - 支持`problem_ids`参数
  - 自动建立req_id到problem_id的映射
  - 向后兼容（不传入problem_id时正常工作）

### 3. **SuffixDecodingCache增强**
- **位置**: `arctic_inference/suffix_decoding/cache.py`
- **功能**: 支持problem_id的缓存和推测
- **特性**:
  - 每个problem_id维护独立的SuffixTree
  - `prebuild_problemtree()` 方法支持问题级别的预构建
  - `add_active_response()` 支持problem_id参数
  - `speculate()` 支持problem_id特定的推测

### 4. **Problem_id提取机制**
- **位置**: `arctic_inference/vllm/model_runner.py`
- **功能**: 从多种格式的prompt中提取problem_id
- **支持格式**:
  - 字典格式: `{"problem_id": "prob_001"}`
  - 嵌套格式: `{"meta": {"problem_id": "prob_001"}}`
  - 字符串格式: `"problem_id:prob_001"`
  - 括号格式: `"[PROBLEM_ID: prob_001]"`
  - 对象属性: `prompt.problem_id`

### 5. **Hard/Medium/Easy分类**
- **功能**: 支持基于problem_id的难度分类
- **用途**: 分配不同的计算资源给不同难度的问题
- **方法**:
  - `set_hard_medium_ids()` 设置难度分类
  - `_get_hard_and_non_hard_indices()` 计算批次中的难度索引

## 🔧 使用方式

### 1. 基本使用

```python
from vllm import LLM, SamplingParams

# 创建LLM实例
llm = LLM(
    model="your-model",
    speculative_config={
        "method": "suffix",
        "enable_suffix_decoding": True,
        "suffix_cache_max_depth": 10,
        "suffix_cache_max_requests": 1000,
        "num_speculative_tokens": 5,
    }
)

# 使用problem_ids进行生成
prompts = ["What is 2+2?", "What is 3+3?"]
problem_ids = ["math_001", "math_002"]

outputs = llm.generate(
    prompts=prompts,
    problem_ids=problem_ids,  # 关键：传入problem_ids
    sampling_params=SamplingParams(temperature=0.8)
)
```

### 2. 难度分类使用

```python
from arctic_inference.vllm.model_runner import ProblemIdContextManager

# 设置问题难度分类
hard_ids = ["complex_math_001", "complex_math_002"]
medium_ids = ["medium_math_001"]
easy_ids = ["simple_math_001"]

ProblemIdContextManager.set_hard_medium_ids(hard_ids, medium_ids, easy_ids)

# 后续的推理会根据难度分配不同的资源
```

### 3. 预构建问题树

```python
from arctic_inference.suffix_decoding.cache import SuffixDecodingCache

cache = SuffixDecodingCache(max_tree_depth=10, max_cached_requests=100)

# 预构建特定问题的缓存树
cache.prebuild_problemtree(
    seq_id=1,
    problem_id="math_001",
    prompt_token_ids=[1, 2, 3, 4],
    token_ids=[5, 6, 7, 8]
)
```

## 🏗️ 架构设计

### 数据流

```
用户调用 LLM.generate(prompts, problem_ids)
    ↓
LLM patches 设置 ProblemIdContextManager 上下文
    ↓
ModelRunner 获取 problem_id 并传递给 SuffixCache
    ↓
SuffixCache 使用 problem_id 进行推测和缓存更新
    ↓
返回结果给用户
```

### 关键组件交互

1. **LLM Layer**: 接收problem_ids参数，设置上下文
2. **Context Manager**: 管理problem_id映射和分类
3. **Model Runner**: 获取problem_id并集成到推理流程
4. **Suffix Cache**: 使用problem_id进行隔离的缓存和推测

## 🧪 测试结果

运行 `test_problem_id_simple.py` 的结果：

```
🎉 All tests passed! problem_id integration is working correctly.

📋 Summary of implemented features:
   ✅ ProblemIdContextManager for thread-safe context management
   ✅ Problem_id extraction from multiple prompt formats
   ✅ Hard/medium/easy problem classification
   ✅ SuffixDecodingCache with problem_id support
   ✅ Complete integration workflow
```

## 🔄 与旧版本的兼容性

### 主要差异

| 功能 | 旧版本 | 新版本 | 状态 |
|------|--------|--------|------|
| **LLM.generate()** | 支持problem_ids参数 | 支持problem_ids参数 | ✅ 兼容 |
| **Suffix Cache** | `update_response()` | `add_active_response()` | ✅ 已适配 |
| **推测方法** | `speculate()` | `speculate()` + source | ✅ 已适配 |
| **问题树管理** | `prebuild_problemtree()` | `prebuild_problemtree()` | ✅ 兼容 |
| **难度分类** | `_get_hard_and_non_hard_indices()` | `_get_hard_and_non_hard_indices()` | ✅ 兼容 |

### 迁移指南

从旧版本迁移到新版本时：

1. **API保持不变**: `LLM.generate(prompts, problem_ids=...)` 语法完全相同
2. **自动应用**: 插件系统会自动应用所有必要的patches
3. **配置兼容**: 所有speculative_config参数保持兼容

## 🚀 性能优势

### 1. **问题级别隔离**
- 每个problem_id有独立的SuffixTree
- 避免不同问题类型之间的模式干扰
- 提高推测准确性

### 2. **智能资源分配**
- 基于问题难度的资源分配
- hard问题获得更多推测tokens
- 提高整体推理效率

### 3. **缓存优化**
- 问题特定的缓存预热
- 相同问题类型的模式复用
- 减少冷启动时间

## 🔧 配置选项

### Speculative Config

```python
speculative_config = {
    "method": "suffix",  # 或 "arctic"
    "enable_suffix_decoding": True,
    "suffix_cache_max_depth": 10,
    "suffix_cache_max_requests": 1000,
    "num_speculative_tokens": 5,
    "suffix_max_spec_factor": 2.0,
    "suffix_max_spec_offset": 2,
    "suffix_min_token_prob": 0.1,
}
```

### 环境变量

- `ARCTIC_INFERENCE_SKIP_VERSION_CHECK`: 跳过vLLM版本检查
- `ARCTIC_METRICS_DIR`: 指标输出目录
- `ARCTIC_TIMING_BUFFER_SIZE`: 时间缓冲区大小

## 📝 注意事项

### 1. **线程安全**
- 所有problem_id操作都是线程安全的
- 使用线程本地存储避免竞争条件

### 2. **内存管理**
- 每个problem_id会创建独立的SuffixTree
- 使用`evict_problem()`及时清理不需要的问题树

### 3. **性能考虑**
- problem_id数量不宜过多（建议<1000）
- 定期清理长时间未使用的问题树

## 🔮 未来扩展

### 计划中的功能

1. **动态难度调整**: 根据推测准确率动态调整问题难度
2. **问题相似度**: 基于问题内容的自动分类
3. **缓存持久化**: 支持问题树的磁盘持久化
4. **分布式缓存**: 跨节点的问题树共享

---

**最后更新**: 2025-11-13
**版本**: ArcticInference v2.0
**状态**: ✅ 完成并测试通过

