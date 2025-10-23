#!/usr/bin/env python3
"""
分析 SuffixCache.speculate() 的性能profiling数据

这个脚本读取 suffix_cache_profiling_*.jsonl 文件，
并显示各个步骤的详细时间统计，帮助找出性能瓶颈。
"""

import json
import glob
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_profiling_data(metrics_dir):
    """从指定目录加载所有profiling数据文件"""
    pattern = os.path.join(metrics_dir, "suffix_cache_profiling_*.jsonl")
    files = glob.glob(pattern)
    
    if not files:
        print(f"错误: 在 {metrics_dir} 中没有找到 suffix_cache_profiling_*.jsonl 文件")
        return []
    
    print(f"找到 {len(files)} 个profiling数据文件:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    print()
    
    all_data = []
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        all_data.append(data)
        except Exception as e:
            print(f"警告: 读取文件 {filepath} 时出错: {e}")
    
    return all_data


def analyze_speculate_timing(data_list):
    """分析speculate方法的timing数据"""
    if not data_list:
        print("没有数据可分析")
        return
    
    print("=" * 100)
    print("SuffixCache.speculate() 性能分析")
    print("=" * 100)
    print(f"总调用次数: {len(data_list)}")
    print()
    
    # 提取各个步骤的时间
    prep_times = [e.get('prep_time_ms', 0) for e in data_list]
    prompt_spec_times = [e.get('prompt_spec_time_ms', 0) for e in data_list]
    problem_lookup_times = [e.get('problem_lookup_time_ms', 0) for e in data_list]
    problem_spec_times = [e.get('problem_spec_time_ms', 0) for e in data_list]
    compare_times = [e.get('compare_time_ms', 0) for e in data_list]
    total_times = [e.get('total_time_ms', 0) for e in data_list]
    
    # 各个步骤的时间统计
    print("=" * 100)
    print("各步骤时间统计 (毫秒)")
    print("=" * 100)
    print(f"{'步骤':<30} {'平均值':>12} {'中位数':>12} {'最小值':>12} {'最大值':>12} {'总和':>15}")
    print("-" * 100)
    
    steps = [
        ("1. 准备和验证", prep_times),
        ("2. Prompt Tree Speculate", prompt_spec_times),
        ("3. Problem Tree查找", problem_lookup_times),
        ("4. Problem Tree Speculate", problem_spec_times),
        ("5. 结果比较", compare_times),
        ("总时间", total_times),
    ]
    
    for name, times in steps:
        if times:
            print(f"{name:<30} {np.mean(times):>12.4f} {np.median(times):>12.4f} "
                  f"{np.min(times):>12.4f} {np.max(times):>12.4f} {np.sum(times):>15.2f}")
    
    # 时间比例统计
    print()
    print("=" * 100)
    print("各步骤时间占比 (%)")
    print("=" * 100)
    print(f"{'步骤':<30} {'平均占比':>12} {'中位数占比':>12} {'最小占比':>12} {'最大占比':>12}")
    print("-" * 100)
    
    prep_ratios = [e.get('prep_time_ratio', 0) * 100 for e in data_list]
    prompt_spec_ratios = [e.get('prompt_spec_time_ratio', 0) * 100 for e in data_list]
    problem_lookup_ratios = [e.get('problem_lookup_time_ratio', 0) * 100 for e in data_list]
    problem_spec_ratios = [e.get('problem_spec_time_ratio', 0) * 100 for e in data_list]
    compare_ratios = [e.get('compare_time_ratio', 0) * 100 for e in data_list]
    
    ratio_steps = [
        ("1. 准备和验证", prep_ratios),
        ("2. Prompt Tree Speculate", prompt_spec_ratios),
        ("3. Problem Tree查找", problem_lookup_ratios),
        ("4. Problem Tree Speculate", problem_spec_ratios),
        ("5. 结果比较", compare_ratios),
    ]
    
    for name, ratios in ratio_steps:
        if ratios:
            print(f"{name:<30} {np.mean(ratios):>11.2f}% {np.median(ratios):>11.2f}% "
                  f"{np.min(ratios):>11.2f}% {np.max(ratios):>11.2f}%")
    
    # 总体时间占比 (基于累计时间)
    total_prep = np.sum(prep_times)
    total_prompt_spec = np.sum(prompt_spec_times)
    total_problem_lookup = np.sum(problem_lookup_times)
    total_problem_spec = np.sum(problem_spec_times)
    total_compare = np.sum(compare_times)
    total_all = total_prep + total_prompt_spec + total_problem_lookup + total_problem_spec + total_compare
    
    print()
    print("=" * 100)
    print("总体时间占比 (基于累计总时间)")
    print("=" * 100)
    if total_all > 0:
        print(f"  1. 准备和验证:           {total_prep/total_all*100:>6.2f}% ({total_prep:>12.2f} ms)")
        print(f"  2. Prompt Tree Speculate: {total_prompt_spec/total_all*100:>6.2f}% ({total_prompt_spec:>12.2f} ms)")
        print(f"  3. Problem Tree查找:      {total_problem_lookup/total_all*100:>6.2f}% ({total_problem_lookup:>12.2f} ms)")
        print(f"  4. Problem Tree Speculate:{total_problem_spec/total_all*100:>6.2f}% ({total_problem_spec:>12.2f} ms) ⚠️")
        print(f"  5. 结果比较:              {total_compare/total_all*100:>6.2f}% ({total_compare:>12.2f} ms)")
        print(f"  {'='*30}")
        print(f"  总计:                     100.00% ({total_all:>12.2f} ms)")
    
    # 结果统计
    print()
    print("=" * 100)
    print("结果统计")
    print("=" * 100)
    
    # 统计使用了哪个树
    selected_sources = [e.get('selected_source', 'none') for e in data_list]
    source_counts = defaultdict(int)
    for source in selected_sources:
        source_counts[source] += 1
    
    print("选择的数据源:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count} ({count/len(data_list)*100:.1f}%)")
    
    # 结果token数量统计
    result_token_counts = [e.get('final_result_tokens', 0) for e in data_list]
    match_lens = [e.get('final_result_match_len', 0) for e in data_list]
    
    print()
    print(f"结果Token数量:")
    print(f"  平均: {np.mean(result_token_counts):.2f}")
    print(f"  中位数: {np.median(result_token_counts):.0f}")
    print(f"  最大: {np.max(result_token_counts):.0f}")
    
    print()
    print(f"匹配长度:")
    print(f"  平均: {np.mean(match_lens):.2f}")
    print(f"  中位数: {np.median(match_lens):.0f}")
    print(f"  最大: {np.max(match_lens):.0f}")
    
    # Pattern长度统计
    pattern_lens = [e.get('pattern_length', 0) for e in data_list]
    pattern_truncated = sum(1 for e in data_list if e.get('pattern_truncated', False))
    
    print()
    print(f"Pattern长度:")
    print(f"  平均: {np.mean(pattern_lens):.2f}")
    print(f"  中位数: {np.median(pattern_lens):.0f}")
    print(f"  被截断: {pattern_truncated} ({pattern_truncated/len(data_list)*100:.1f}%)")
    
    # 性能问题识别
    print()
    print("=" * 100)
    print("⚠️  性能瓶颈识别")
    print("=" * 100)
    
    # 找出最慢的步骤
    step_times = {
        "准备和验证": total_prep,
        "Prompt Tree Speculate": total_prompt_spec,
        "Problem Tree查找": total_problem_lookup,
        "Problem Tree Speculate": total_problem_spec,
        "结果比较": total_compare,
    }
    
    sorted_steps = sorted(step_times.items(), key=lambda x: x[1], reverse=True)
    
    print("时间消耗排名:")
    for i, (step, time_ms) in enumerate(sorted_steps, 1):
        percentage = (time_ms / total_all * 100) if total_all > 0 else 0
        marker = " 🔴 BOTTLENECK" if i == 1 else ""
        print(f"  {i}. {step:<30} {time_ms:>12.2f} ms ({percentage:>5.1f}%){marker}")
    
    # 慢调用分析 (>10ms的调用)
    slow_calls = [e for e in data_list if e.get('total_time_ms', 0) > 10]
    if slow_calls:
        print()
        print(f"慢调用分析 (>10ms): {len(slow_calls)} 次 ({len(slow_calls)/len(data_list)*100:.1f}%)")
        slow_total = sum(e.get('total_time_ms', 0) for e in slow_calls)
        print(f"  慢调用总耗时: {slow_total:.2f} ms ({slow_total/np.sum(total_times)*100:.1f}% of total)")
        
        # 慢调用中各步骤的占比
        slow_problem_spec = sum(e.get('problem_spec_time_ms', 0) for e in slow_calls)
        slow_prompt_spec = sum(e.get('prompt_spec_time_ms', 0) for e in slow_calls)
        print(f"  慢调用中Problem Tree Speculate: {slow_problem_spec:.2f} ms ({slow_problem_spec/slow_total*100:.1f}%)")
        print(f"  慢调用中Prompt Tree Speculate: {slow_prompt_spec:.2f} ms ({slow_prompt_spec/slow_total*100:.1f}%)")


def main():
    if len(sys.argv) > 1:
        metrics_dir = sys.argv[1]
    else:
        # 默认使用环境变量或默认路径
        metrics_dir = os.getenv("ARCTIC_METRICS_DIR", "/tmp/arctic_metrics")
    
    # 如果指定的是一个目录，查找最新的时间戳子目录
    if os.path.isdir(metrics_dir):
        subdirs = [d for d in glob.glob(os.path.join(metrics_dir, "*")) if os.path.isdir(d)]
        if subdirs:
            # 按修改时间排序，使用最新的
            latest_dir = max(subdirs, key=os.path.getmtime)
            print(f"使用最新的metrics目录: {latest_dir}\n")
            metrics_dir = latest_dir
    
    if not os.path.exists(metrics_dir):
        print(f"错误: 目录 {metrics_dir} 不存在")
        print(f"\n使用方法: {sys.argv[0]} [metrics_directory]")
        print(f"  metrics_directory: 包含profiling数据文件的目录 (默认: $ARCTIC_METRICS_DIR 或 /tmp/arctic_metrics)")
        sys.exit(1)
    
    print(f"正在从以下目录加载数据: {metrics_dir}")
    print()
    
    # 加载数据
    data_list = load_profiling_data(metrics_dir)
    
    if not data_list:
        print("没有加载到任何数据")
        sys.exit(1)
    
    # 分析统计
    analyze_speculate_timing(data_list)
    
    print()
    print("=" * 100)
    print("分析完成")
    print("=" * 100)
    print()
    print("💡 优化建议:")
    print("   1. 如果 Problem Tree Speculate 占比很高 (>70%):")
    print("      - 检查 C++ SuffixTree.speculate() 实现")
    print("      - 考虑优化后缀树查找算法")
    print("      - 检查是否有不必要的内存复制")
    print()
    print("   2. 如果 Prompt Tree Speculate 占比很高 (>30%):")
    print("      - 考虑减少prompt tree的使用")
    print("      - 或者优化prompt tree的结构")
    print()
    print("   3. 如果 Problem Tree查找 占比较高 (>5%):")
    print("      - 考虑使用更高效的数据结构 (如C++ unordered_map)")
    print()


if __name__ == "__main__":
    main()



