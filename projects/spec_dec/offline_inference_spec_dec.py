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

import vllm
from vllm import LLM, SamplingParams
import pandas as pd

import os
os.environ["VLLM_USE_V1"] = "1"

vllm.plugins.load_general_plugins()

# 读取 parquet 文件
data_path = "/data/zshao/rllm/rllm/data/datasets/deepscaler_math/val_gsm8k_fixed.parquet"
df = pd.read_parquet(data_path)
print(f"Loaded {len(df)} problems from {data_path}")

# 从 dataframe 中提取对话数据
conversations = []
for idx, row in df.iterrows():
    # problem 列包含对话消息的数组
    problem_messages = row['problem']
    # 将其转换为标准的对话格式
    conversation = [{"role": msg["role"], "content": msg["content"]} for msg in problem_messages]
    conversations.append(conversation)

print(f"Prepared {len(conversations)} conversations")

llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    quantization="fp8",
    tensor_parallel_size=1,
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 5,
        "enable_suffix_decoding": True,
        "disable_by_batch_size": 128,
        "suffix_cache_max_depth": 32,
    },
    seed=0,
)

print("=" * 80)

sampling_params = SamplingParams(temperature=0.6, max_tokens=16384)

# 批量处理所有对话
outputs = llm.chat(conversations, sampling_params=sampling_params)

# 打印结果
for idx, output in enumerate(outputs):
    print(f"\n{'='*80}")
    print(f"Problem {idx + 1}/{len(outputs)} (ID: {df.iloc[idx]['problem_id']}):")
    print(f"Question: {df.iloc[idx]['question'][:100]}...")
    print(f"Ground Truth: {df.iloc[idx]['ground_truth']}")
    print(f"\nGenerated Answer:")
    print(output.outputs[0].text)
    print(f"{'='*80}")
