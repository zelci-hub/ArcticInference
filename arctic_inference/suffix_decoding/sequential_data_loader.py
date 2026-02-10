#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential Data Loader for Arctic Inference Simulator
基于simulator.py，实现从指定目录按顺序读取连续n个文件作为训练数据，第n+1个文件作为测试数据
"""

import argparse
import itertools
import multiprocessing as mp
import os
import random
import time
import json
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# 导入原有的simulator模块
import sys
import os
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from arctic_inference.suffix_decoding import SuffixDecodingCache

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 从原simulator.py复制的核心函数
def suffix_decode(
    suffix_cache: SuffixDecodingCache,
    request_id: int,
    problem_id: int,
    prompt: List[int],
    ground_truth_response: List[int],
    max_spec_tokens: int,
    max_spec_factor: float,
    min_token_prob: float,
    use_tree_spec: bool,
    use_cached_prompt: bool,
) -> List[Dict]:
    """从原simulator.py复制的suffix_decode函数"""
    if not max_spec_tokens:
        max_spec_tokens = suffix_cache.max_tree_depth

    suffix_cache.start_request(request_id, problem_id=problem_id, prompt_token_ids=prompt if use_cached_prompt else [])

    assert isinstance(prompt, list) and isinstance(ground_truth_response, list)

    results = []
    response = []
    while len(response) < len(ground_truth_response):
        text = prompt + response

        start_time = time.perf_counter()
        result, source = suffix_cache.speculate(
            request_id,
            text,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            min_token_prob=min_token_prob,
            use_tree_spec=use_tree_spec,
            problem_id=problem_id,
        )
        end_time = time.perf_counter()
        spec_time = end_time - start_time

        # Verify speculated tokens
        accepted_tokens = []
        node = -1
        for token_id in ground_truth_response[len(response):]:
            children = [i for i, p in enumerate(result.parents) if p == node]
            for c in children:
                if result.token_ids[c] == token_id:
                    accepted_tokens.append(token_id)
                    node = c
                    break
            else:
                break

        new_tokens = accepted_tokens.copy()
        response.extend(accepted_tokens)
        assert len(response) <= len(ground_truth_response)
        if len(response) < len(ground_truth_response):
            # Add bonus token
            bonus_token = ground_truth_response[len(response)]
            new_tokens.append(bonus_token)
            response.append(bonus_token)

        # Update suffix cache
        start_time = time.perf_counter()
        suffix_cache.add_active_response(request_id, problem_id=problem_id, token_ids=new_tokens)
        end_time = time.perf_counter()
        update_time = end_time - start_time

        results.append({
            "step": len(results),
            "match_len": result.match_len,
            "score": result.score,
            "num_spec_toks": len(result.token_ids),
            "num_accept_toks": len(accepted_tokens),
            "num_out_toks": len(new_tokens),
            "spec_ms": spec_time * 1000,
            "update_ms": update_time * 1000,
        })

    assert response == ground_truth_response
    suffix_cache.stop_request(request_id)
    return results


def detect_available_files(data_dir: str) -> List[int]:
    """
    检测数据目录中所有可用的文件索引
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        可用文件索引的排序列表
    """
    data_dir = Path(data_dir)
    available_indices = []
    
    for file_path in data_dir.glob("*.jsonl"):
        try:
            # 尝试从文件名中提取数字索引
            filename = file_path.stem
            if filename.isdigit():
                available_indices.append(int(filename))
        except ValueError:
            continue
    
    available_indices.sort()
    print(f"Found {len(available_indices)} files in {data_dir}: {available_indices[:10]}{'...' if len(available_indices) > 10 else ''}")
    return available_indices


def load_sequential_files(data_dir: str, start_idx: int, n: int) -> pd.DataFrame:
    """
    从指定目录按顺序加载连续n个文件
    
    Args:
        data_dir: 数据目录路径
        start_idx: 起始文件索引
        n: 要加载的文件数量
    
    Returns:
        包含所有数据的DataFrame
    """
    data_dir = Path(data_dir)
    all_data = []
    
    print(f"Loading {n} files starting from index {start_idx}...")
    
    for i in range(start_idx, start_idx + n):
        file_path = data_dir / f"{i}.jsonl"
        
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist, skipping...")
            continue
            
        print(f"Loading file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    # 添加文件信息和行号
                    data['file_idx'] = i
                    data['line_num'] = line_num
                    all_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num} in {file_path}: {e}")
                    continue
    
    if not all_data:
        raise ValueError(f"No valid data found in files {start_idx} to {start_idx + n - 1}")
    
    print(f"Loaded {len(all_data)} examples from {n} files")
    return pd.DataFrame(all_data)


def prepare_data_for_simulator(df: pd.DataFrame, tokenizer_name: Optional[str] = None) -> pd.DataFrame:
    """
    将加载的数据转换为simulator所需的格式
    
    Args:
        df: 原始数据DataFrame
        tokenizer_name: tokenizer名称，必须提供以进行tokenization
    
    Returns:
        转换后的DataFrame，包含'prompt'和'response'列（已tokenized）
    """
    print("Preparing data for simulator...")
    
    # 检查是否提供了tokenizer
    if not tokenizer_name:
        raise ValueError("tokenizer_name is required for SuffixDecodingCache. Please provide a valid tokenizer name.")
    
    # 提取prompt和response
    prompts = []
    responses = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
        # 使用input作为prompt，output作为response
        prompts.append(row['input_token_ids'])
        responses.append(row['output_token_ids'])
    
    result_df = pd.DataFrame({
        'prompt': prompts,
        'response': responses,
        'file_idx': df['file_idx'],
        'line_num': df['line_num'],
        'problem_id': df.get('problem_id', ''),
        'score': df.get('score', 0.0),
    })
    
    # 进行tokenization（现在是必需的）
    print(f"Tokenizing data with {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    tokenized_prompts = []
    tokenized_responses = []
    
    for _, row in tqdm(result_df.iterrows(), total=len(result_df), desc="Tokenizing"):
        try:
            # Tokenize并确保返回的是整数列表
            prompt_tokens = row['prompt']
            response_tokens = row['response']
            
            # 验证tokenization结果
            if not isinstance(prompt_tokens, list) or not all(isinstance(t, int) for t in prompt_tokens):
                raise ValueError(f"Prompt tokenization failed: expected list of ints, got {type(prompt_tokens)}")
            if not isinstance(response_tokens, list) or not all(isinstance(t, int) for t in response_tokens):
                raise ValueError(f"Response tokenization failed: expected list of ints, got {type(response_tokens)}")
            
            tokenized_prompts.append(prompt_tokens)
            tokenized_responses.append(response_tokens)
            
        except Exception as e:
            print(f"Error tokenizing row {len(tokenized_prompts)}: {e}")
            print(f"Prompt: {row['prompt'][:100]}...")
            print(f"Response: {row['response'][:100]}...")
            raise
    
    result_df['prompt'] = tokenized_prompts
    result_df['response'] = tokenized_responses
    
    print(f"Tokenization completed. Sample sizes - Prompts: {len(tokenized_prompts[0]) if tokenized_prompts else 0}, Responses: {len(tokenized_responses[0]) if tokenized_responses else 0}")
    
    return result_df


def process_sequential_task(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    task_id: int,
    max_depth: int,
    max_spec_tokens: int,
    max_spec_factor: float,
    min_token_prob: float,
    use_tree_spec: bool,
    use_cached_prompt: bool,
    evict_fraction: float,
    evict_strategy: str,
    max_cached_requests: int,
    seed: int,
) -> List[Dict]:
    """
    处理顺序数据的任务，基于原simulator.py的process_task函数
    """
    print(f"Processing task {task_id} with {len(train_data)} training examples and {len(test_data)} test examples")
    
    # 创建suffix cache
    suffix_cache = SuffixDecodingCache(max_tree_depth=max_depth,
                                       max_cached_requests=max_cached_requests)
    
    # 构建训练缓存
    train_request_ids = []
    num_cached_tokens = {}
    
    print("Building cache from training data...")
    for idx, (request_id, example) in enumerate(tqdm(train_data.iterrows(), 
                                                     total=len(train_data),
                                                     desc="Building cache")):
        # 使用负数request_id来标识训练样本
        train_request_id = -1 - idx
        suffix_cache.start_request(train_request_id, problem_id=example["problem_id"], prompt_token_ids=example["prompt"])
        suffix_cache.add_active_response(train_request_id, problem_id=example["problem_id"], token_ids=example["response"])
        suffix_cache.stop_request(train_request_id)
        train_request_ids.append(train_request_id)
        num_cached_tokens[train_request_id] = len(example["response"])
    
    # # 缓存驱逐策略
    # if evict_fraction > 0:
    #     cached_request_ids = list(suffix_cache.cached_requests)
    #     num_evict = round(len(cached_request_ids) * evict_fraction)
    #     if evict_strategy == "oldest":
    #         evict_ids = cached_request_ids[:num_evict]
    #     elif evict_strategy == "newest":
    #         evict_ids = cached_request_ids[-num_evict:]
    #     else:
    #         assert evict_strategy == "random"
    #         rng = random.Random(seed)
    #         evict_ids = rng.sample(cached_request_ids, num_evict)
        
    #     for request_id in tqdm(evict_ids, desc="Evicting cached responses"):
    #         suffix_cache.evict_cached_response(request_id)
    
    # 检查缓存完整性
    # print("Checking cache integrity...", end=" ", flush=True)
    # if ret := suffix_cache._global_tree.check_integrity():
    #     raise RuntimeError(f"Cache integrity check failed: {ret}")
    # else:
    #     print("OK")
    
    # # num_cached_tokens 已经在上面的循环中完全构建好了，不需要重新过滤
    # print("Tokens in cache:", sum(num_cached_tokens.values()))
    # print("Memory estimate:", suffix_cache._global_tree.estimate_memory())
    
    # 在测试数据上运行
    records = []
    print("Running inference on test data...")
    for idx, (request_id, example) in enumerate(tqdm(test_data.iterrows(),
                                                     total=len(test_data),
                                                     desc="Running requests")):
        results = suffix_decode(
            suffix_cache,
            idx,  # 使用索引作为request_id
            problem_id=example["problem_id"],
            prompt=example["prompt"],
            ground_truth_response=example["response"],
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            min_token_prob=min_token_prob,
            use_tree_spec=use_tree_spec,
            use_cached_prompt=use_cached_prompt
        )
        
        for result in results:
            result.update({
                "task_id": task_id,
                "request_id": idx,
                "original_request_id": request_id,
                "file_idx": example.get("file_idx", -1),
                "problem_id": example.get("problem_id", ""),
                "num_train": len(train_data),
                "num_test": len(test_data),
                "seed": seed,
                "max_depth": max_depth,
                "max_spec_tokens": max_spec_tokens,
                "max_spec_factor": max_spec_factor,
                "min_token_prob": min_token_prob,
                "use_tree_spec": use_tree_spec,
                "use_cached_prompt": use_cached_prompt,
                "evict_fraction": evict_fraction,
                "evict_strategy": evict_strategy,
                "max_cached_requests": max_cached_requests,
            })
        records.extend(results)
    
    return records


def results_summary(df: pd.DataFrame, config_cols: List[str]) -> pd.DataFrame:
    """从原simulator.py复制的结果汇总函数"""
    # Compute per-request speedup.
    speedup = df.groupby(["task_id", "request_id"]).agg(
        sum_out_toks=("num_out_toks", "sum"),
        num_steps=("step", "count"),
    )
    speedup["speedup"] = speedup["sum_out_toks"] / speedup["num_steps"]
    speedup = speedup.groupby(["task_id"]).agg(
        req_speedup=("speedup", "mean"),
    )
    # Compute summary statistics.
    config_cols = ["task_id"] + list(config_cols)
    summary = df.groupby(config_cols).agg(
        sum_accept_toks=("num_accept_toks", "sum"),
        sum_spec_toks=("num_spec_toks", "sum"),
        sum_out_toks=("num_out_toks", "sum"),
        avg_accept_toks=("num_accept_toks", "mean"),
        avg_spec_toks=("num_spec_toks", "mean"),
        sum_spec_ms=("spec_ms", "sum"),
        sum_update_ms=("update_ms", "sum"),
    ).reset_index()
    summary["accept_rate"] = (
        summary["sum_accept_toks"] / summary["sum_spec_toks"])
    summary["req_speedup"] = speedup["req_speedup"]
    summary["spec_ms_per_tok"] = (
        summary["sum_spec_ms"] / summary["sum_spec_toks"])
    summary["update_ms_per_tok"] = (
        summary["sum_update_ms"] / summary["sum_out_toks"])
    # Calculate columns to drop from the summary
    drop_cols = [col for col in config_cols[1:] if summary[col].nunique() == 1]
    drop_cols.extend([
        "sum_accept_toks",
        "sum_spec_toks",
        "sum_out_toks",
        "sum_spec_ms",
        "sum_update_ms"])
    return summary.drop(columns=drop_cols).set_index("task_id")


def main():
    parser = argparse.ArgumentParser(description="Sequential Data Loader for Arctic Inference")
    
    # 数据相关参数
    parser.add_argument("--data-dir", type=str, required=True,
                        help="数据目录路径")
    parser.add_argument("--start-idx", type=int, default=None,
                        help="起始文件索引 (如果不指定则遍历所有可能的起始位置)")
    parser.add_argument("-n", "--num-train-files", type=int, required=True,
                        help="用于训练的连续文件数量")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="HuggingFace tokenizer名称（必需）")
    parser.add_argument("--iterate-all", action="store_true",
                        help="遍历所有可能的start_idx值")
    
    # 输出参数
    parser.add_argument("-o", "--output", type=str,
                        help="输出CSV文件路径")
    
    # Simulator参数
    parser.add_argument("--max-depth", type=int, default=32,
                        help="suffix tree的最大深度")
    parser.add_argument("--max-spec-tokens", type=int, default=0,
                        help="最大推测token数量 (0表示使用max_depth)")
    parser.add_argument("--max-spec-factor", type=float, default=1.0,
                        help="最大推测token数量作为前缀长度的倍数")
    parser.add_argument("--min-token-prob", type=float, default=0.1,
                        help="考虑token的最小概率")
    parser.add_argument("--use-tree-spec", action="store_true", default=True,
                        help="是否使用基于树的推测")
    parser.add_argument("--use-cached-prompt", action="store_true", default=True,
                        help="是否使用缓存的prompt")
    parser.add_argument("--max-cached-requests", type=int, default=-1,
                        help="最大缓存请求数量 (-1表示无限制)")
    parser.add_argument("--evict-fraction", type=float, default=0.0,
                        help="在运行请求前驱逐缓存序列的比例")
    parser.add_argument("--evict-strategy", type=str, default="random",
                        choices=["random", "oldest", "newest"],
                        help="缓存驱逐策略")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 检测可用文件
    available_files = detect_available_files(args.data_dir)
    if len(available_files) < args.num_train_files + 1:
        raise ValueError(f"需要至少 {args.num_train_files + 1} 个文件，但只找到 {len(available_files)} 个文件")
    
    # 确定要遍历的start_idx列表
    if args.iterate_all or args.start_idx is None:
        # 遍历所有可能的起始位置
        max_start_idx = available_files[-1] - args.num_train_files
        start_indices = [idx for idx in available_files if idx <= max_start_idx]
        print(f"将遍历 {len(start_indices)} 个起始位置: {start_indices[:10]}{'...' if len(start_indices) > 10 else ''}")
    else:
        # 使用指定的起始位置
        start_indices = [args.start_idx]
        print(f"使用指定的起始位置: {args.start_idx}")
    
    all_records = []
    all_summaries = []
    
    # 对每个起始位置运行实验
    for task_id, start_idx in enumerate(start_indices):
        print(f"\n{'='*60}")
        print(f"运行任务 {task_id + 1}/{len(start_indices)}: start_idx={start_idx}")
        print(f"{'='*60}")
        
        # 检查所需文件是否存在
        required_files = list(range(start_idx, start_idx + args.num_train_files + 1))
        missing_files = [f for f in required_files if f not in available_files]
        if missing_files:
            print(f"警告: 缺少文件 {missing_files}，跳过 start_idx={start_idx}")
            continue
        
        try:
            # 加载训练数据 (连续n个文件)
            print(f"Loading training data from files {start_idx} to {start_idx + args.num_train_files - 1}")
            train_data = load_sequential_files(args.data_dir, start_idx, args.num_train_files)
            
            # 加载测试数据 (第n+1个文件)
            test_file_idx = start_idx + args.num_train_files
            print(f"Loading test data from file {test_file_idx}")
            test_data = load_sequential_files(args.data_dir, test_file_idx, 1)
            
            # 首先从test_data中提取所有的problem_id
            print("Extracting problem_ids from test data...")
            test_problem_ids = set(test_data['problem_id'].unique())
            print(f"Found {len(test_problem_ids)} unique problem_ids in test data")
            print(f"Test problem_ids: {list(test_problem_ids)[:10]}{'...' if len(test_problem_ids) > 10 else ''}")
            
            # 筛选train_data，只保留与test_data中problem_id对应的数据
            print(f"Filtering training data by problem_ids...")
            original_train_size = len(train_data)
            train_data = train_data[train_data['problem_id'].isin(test_problem_ids)]
            print(f"Filtered training data: {original_train_size} -> {len(train_data)} examples")
            
            if len(train_data) == 0:
                print(f"警告: 筛选后训练数据为空，跳过 start_idx={start_idx}")
                continue
            
            # 准备数据
            train_data = prepare_data_for_simulator(train_data, args.tokenizer)
            test_data = prepare_data_for_simulator(test_data, args.tokenizer)
            
            print(f"Training data: {len(train_data)} examples")
            print(f"Test data: {len(test_data)} examples")
            
            # 运行任务
            records = process_sequential_task(
                train_data=train_data,
                test_data=test_data,
                task_id=task_id,
                max_depth=args.max_depth,
                max_spec_tokens=args.max_spec_tokens,
                max_spec_factor=args.max_spec_factor,
                min_token_prob=args.min_token_prob,
                use_tree_spec=args.use_tree_spec,
                use_cached_prompt=args.use_cached_prompt,
                evict_fraction=args.evict_fraction,
                evict_strategy=args.evict_strategy,
                max_cached_requests=args.max_cached_requests,
                seed=args.seed,
            )
            
            # 为每条记录添加start_idx信息
            for record in records:
                record["start_idx"] = start_idx
            
            all_records.extend(records)
            
            # 生成当前任务的汇总
            if records:
                task_df = pd.DataFrame.from_records(records)
                config_cols = [
                    "num_train", "num_test", "seed", "max_depth", "max_spec_tokens",
                    "max_spec_factor", "min_token_prob", "use_tree_spec", "use_cached_prompt",
                    "evict_fraction", "evict_strategy", "max_cached_requests"
                ]
                task_summary = results_summary(task_df, config_cols)
                task_summary["start_idx"] = start_idx
                all_summaries.append(task_summary)
                
                print(f"\n任务 {task_id + 1} 汇总:")
                print(task_summary.to_string())
            
        except Exception as e:
            import traceback
            print(f"处理 start_idx={start_idx} 时出错: {type(e).__name__}: {e}")
            print("完整错误信息:")
            traceback.print_exc()
            continue
    
    if not all_records:
        print("没有成功处理任何任务")
        return
    
    print(f"\n{'='*60}")
    print("所有任务完成，准备最终结果...")
    print(f"{'='*60}")
    
    # 合并所有结果
    df = pd.DataFrame.from_records(all_records)
    
    # 生成总体汇总结果
    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        print("\n所有任务的汇总结果:")
        print(combined_summary.to_string())
        
        # 计算平均性能
        avg_summary = combined_summary.select_dtypes(include=[float, int]).mean()
        print(f"\n平均性能指标:")
        for metric, value in avg_summary.items():
            if metric not in ['task_id', 'start_idx']:
                print(f"{metric}: {value:.4f}")
    
    # 保存详细结果
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n详细结果已保存到: {args.output}")
        
        # 保存汇总结果
        if all_summaries:
            summary_output = args.output.replace('.csv', '_summary.csv')
            combined_summary.to_csv(summary_output, index=False)
            print(f"汇总结果已保存到: {summary_output}")
        
        # 保存平均性能指标
        if all_summaries:
            avg_output = args.output.replace('.csv', '_average.csv')
            avg_summary.to_frame(name='average_value').to_csv(avg_output)
            print(f"平均性能指标已保存到: {avg_output}")


if __name__ == "__main__":
    main()
