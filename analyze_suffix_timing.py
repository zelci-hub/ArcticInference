#!/usr/bin/env python3
"""
分析 propose_suffix_draft_token_ids 的时间统计数据

这个脚本读取生成的 suffix_speculation_timing_*.jsonl 文件，
并显示 extract_time, submit_time, wait_time 的总体时间比例统计。
"""

import json
import glob
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_timing_data(metrics_dir):
    """从指定目录加载所有timing数据文件"""
    pattern = os.path.join(metrics_dir, "suffix_speculation_timing_*.jsonl")
    files = glob.glob(pattern)
    
    if not files:
        print(f"错误: 在 {metrics_dir} 中没有找到 suffix_speculation_timing_*.jsonl 文件")
        return []
    
    print(f"找到 {len(files)} 个timing数据文件:")
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


def analyze_timing_stats(data_list):
    """分析时间统计数据"""
    if not data_list:
        print("没有数据可分析")
        return
    
    # 按call_type分组统计
    by_call_type = defaultdict(list)
    for entry in data_list:
        call_type = entry.get('call_type', 'unknown')
        by_call_type[call_type].append(entry)
    
    print("=" * 100)
    print("总体统计摘要")
    print("=" * 100)
    print(f"总记录数: {len(data_list)}")
    print(f"Call类型分布:")
    for call_type, entries in by_call_type.items():
        print(f"  - {call_type}: {len(entries)} 条记录")
    print()
    
    # 对每种call_type进行详细分析
    for call_type, entries in sorted(by_call_type.items()):
        print("=" * 80)
        print(f"详细统计: {call_type}")
        print("=" * 80)
        
        # 提取时间数据
        extract_times = [e.get('extract_time_ms', 0) for e in entries]
        submit_times = [e.get('submit_time_ms', 0) for e in entries]
        wait_times = [e.get('wait_time_ms', 0) for e in entries]
        total_times = [e.get('total_time_ms', 0) for e in entries]
        num_tasks = [e.get('num_tasks', 0) for e in entries]
        
        # 计算统计指标
        print(f"\n时间统计 (毫秒):")
        print(f"{'指标':<20} {'平均值':>12} {'中位数':>12} {'最小值':>12} {'最大值':>12} {'总和':>15}")
        print("-" * 85)
        
        for name, times in [
            ("Extract Time", extract_times),
            ("Submit Time", submit_times),
            ("Wait Time", wait_times),
            ("Total Time", total_times),
        ]:
            if times:
                print(f"{name:<20} {np.mean(times):>12.2f} {np.median(times):>12.2f} "
                      f"{np.min(times):>12.2f} {np.max(times):>12.2f} {np.sum(times):>15.2f}")
        
        # 计算时间比例
        print(f"\n时间比例统计 (%):")
        extract_ratios = [e.get('extract_time_ratio', 0) * 100 for e in entries]
        submit_ratios = [e.get('submit_time_ratio', 0) * 100 for e in entries]
        wait_ratios = [e.get('wait_time_ratio', 0) * 100 for e in entries]
        
        print(f"{'指标':<20} {'平均值':>12} {'中位数':>12} {'最小值':>12} {'最大值':>12}")
        print("-" * 72)
        
        for name, ratios in [
            ("Extract Ratio", extract_ratios),
            ("Submit Ratio", submit_ratios),
            ("Wait Ratio", wait_ratios),
        ]:
            if ratios:
                print(f"{name:<20} {np.mean(ratios):>11.2f}% {np.median(ratios):>11.2f}% "
                      f"{np.min(ratios):>11.2f}% {np.max(ratios):>11.2f}%")
        
        # 任务数量统计
        if num_tasks:
            print(f"\n任务数量统计:")
            print(f"  平均任务数: {np.mean(num_tasks):.2f}")
            print(f"  中位数: {np.median(num_tasks):.0f}")
            print(f"  最小值: {np.min(num_tasks):.0f}")
            print(f"  最大值: {np.max(num_tasks):.0f}")
        
        # 总体时间占比 (基于总时间的累计)
        total_extract = np.sum(extract_times)
        total_submit = np.sum(submit_times)
        total_wait = np.sum(wait_times)
        total_all = total_extract + total_submit + total_wait
        
        if total_all > 0:
            print(f"\n总体时间占比 (基于累计总时间):")
            print(f"  Extract: {total_extract/total_all*100:.2f}% ({total_extract:.2f} ms)")
            print(f"  Submit:  {total_submit/total_all*100:.2f}% ({total_submit:.2f} ms)")
            print(f"  Wait:    {total_wait/total_all*100:.2f}% ({total_wait:.2f} ms) ⚠️")
            print(f"  Total:   100.00% ({total_all:.2f} ms)")
        
        # 🔍 并行效率分析
        parallelism_efficiencies = [e.get('parallelism_efficiency', 0) for e in entries if 'parallelism_efficiency' in e]
        actual_speedups = [e.get('actual_speedup', 1.0) for e in entries if 'actual_speedup' in e]
        num_active_threads = [e.get('num_active_threads', 0) for e in entries if 'num_active_threads' in e]
        
        # New metrics
        first_completions = [e.get('first_completion_time_ms', 0) for e in entries if 'first_completion_time_ms' in e]
        last_completions = [e.get('last_completion_time_ms', 0) for e in entries if 'last_completion_time_ms' in e]
        completion_spreads = [e.get('completion_spread_ms', 0) for e in entries if 'completion_spread_ms' in e]
        avg_task_times = [e.get('avg_task_time_from_threads_ms', 0) for e in entries if 'avg_task_time_from_threads_ms' in e]
        
        if parallelism_efficiencies:
            print(f"\n🔍 并行效率分析:")
            print(f"  平均并行效率: {np.mean(parallelism_efficiencies):.2f}x")
            print(f"  中位数并行效率: {np.median(parallelism_efficiencies):.2f}x")
            print(f"  平均实际加速比: {np.mean(actual_speedups):.2f}x")
            print(f"  平均活跃线程数: {np.mean(num_active_threads):.1f}")
            
            # New: Task completion analysis
            if first_completions and last_completions:
                print(f"\n  任务完成时间分析:")
                print(f"    第一个任务完成: {np.mean(first_completions):.2f}ms (平均)")
                print(f"    最后一个任务完成: {np.mean(last_completions):.2f}ms (平均)")
                print(f"    完成时间跨度: {np.mean(completion_spreads):.2f}ms (平均)")
                
                # Calculate expected times for comparison
                if avg_task_times:
                    avg_single_task = np.mean(avg_task_times)
                    avg_wait = np.mean(wait_times)
                    print(f"    单个任务平均耗时: {avg_single_task:.2f}ms")
                    print(f"    wait_time测量值: {avg_wait:.2f}ms")
                    
                    # Diagnosis
                    ratio = avg_wait / avg_single_task if avg_single_task > 0 else 0
                    print(f"\n  📊 诊断:")
                    print(f"    wait_time / 单任务时间 = {ratio:.2f}x")
                    
                    if ratio > 5:
                        print(f"    🔴 严重问题: wait_time是单任务的{ratio:.1f}倍！")
                        print(f"       最可能原因: GIL未释放，任务串行执行")
                        print(f"       建议: 运行 python check_gil_release.py")
                    elif ratio > 2:
                        print(f"    🟡 部分串行: wait_time是单任务的{ratio:.1f}倍")
                        print(f"       可能原因: 锁竞争或任务分布不均")
                    else:
                        print(f"    ✅ 良好并行: wait_time接近单任务时间")
            
            # 并行效率判断
            avg_efficiency = np.mean(parallelism_efficiencies)
            if avg_efficiency < 1.5:
                print(f"\n  ⚠️  警告: 并行效率很低 ({avg_efficiency:.2f}x < 1.5x)")
                print(f"      可能原因: GIL锁竞争、任务太小、或CPU资源不足")
            elif avg_efficiency < 3.0:
                print(f"  ⚠️  注意: 并行效率偏低 ({avg_efficiency:.2f}x < 3.0x)")
                print(f"      仍有优化空间")
            else:
                print(f"  ✅ 并行效率良好 ({avg_efficiency:.2f}x)")
        
        print()


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
        print(f"  metrics_directory: 包含timing数据文件的目录 (默认: $ARCTIC_METRICS_DIR 或 /tmp/arctic_metrics)")
        sys.exit(1)
    
    print(f"正在从以下目录加载数据: {metrics_dir}")
    print()
    
    # 加载数据
    data_list = load_timing_data(metrics_dir)
    
    if not data_list:
        print("没有加载到任何数据")
        sys.exit(1)
    
    # 分析统计
    analyze_timing_stats(data_list)
    
    print("=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

