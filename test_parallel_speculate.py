#!/usr/bin/env python3
"""
精确测试：找出speculate并行性能瓶颈

这个测试会：
1. 直接测试C++的speculate（绕过Python字典）
2. 测试Python层的开销
3. 测试不同的并行方式
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# 需要先import你的suffix_cache
import sys
sys.path.insert(0, '/data/zshao/rllm/ArcticInference')

from arctic_inference.common.suffix_cache import SuffixCache


def test_1_single_thread_baseline(suffix_cache, problem_id, pattern, num_calls=10):
    """基线测试：单线程调用"""
    print("=" * 80)
    print("Test 1: 单线程基线测试")
    print("=" * 80)
    
    times = []
    for i in range(num_calls):
        start = time.perf_counter()
        result = suffix_cache.speculate(
            req_id=f"test_{i}",
            problem_id=problem_id,
            pattern=pattern,
            max_spec_tokens=5,
            use_cached_prompt=False
        )
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    print(f"调用次数: {num_calls}")
    print(f"平均时间: {avg_time:.3f}ms")
    print(f"总时间: {sum(times):.3f}ms")
    print(f"预期串行总时间: {avg_time * num_calls:.3f}ms")
    print()
    
    return avg_time


def test_2_multi_thread_same_problem(suffix_cache, problem_id, pattern, num_threads=9, num_calls=16):
    """测试：多线程访问同一个problem_id"""
    print("=" * 80)
    print(f"Test 2: 多线程测试 ({num_threads} workers, {num_calls} tasks)")
    print("=" * 80)
    
    def worker(task_id):
        thread_id = threading.current_thread().ident
        start = time.perf_counter()
        
        result = suffix_cache.speculate(
            req_id=f"test_{task_id}",
            problem_id=problem_id,
            pattern=pattern,
            max_spec_tokens=5,
            use_cached_prompt=False
        )
        
        end = time.perf_counter()
        return {
            'task_id': task_id,
            'thread_id': thread_id,
            'time_ms': (end - start) * 1000
        }
    
    overall_start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        submit_start = time.perf_counter()
        futures = [executor.submit(worker, i) for i in range(num_calls)]
        submit_time = (time.perf_counter() - submit_start) * 1000
        
        # 等待完成（使用as_completed）
        wait_start = time.perf_counter()
        results = []
        completion_times = []
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completion_times.append((time.perf_counter() - wait_start) * 1000)
        
        wait_time = (time.perf_counter() - wait_start) * 1000
    
    overall_time = (time.perf_counter() - overall_start) * 1000
    
    # 分析结果
    task_times = [r['time_ms'] for r in results]
    avg_task_time = np.mean(task_times)
    
    # 统计每个线程执行了多少任务
    thread_counts = {}
    for r in results:
        tid = r['thread_id']
        thread_counts[tid] = thread_counts.get(tid, 0) + 1
    
    print(f"提交时间: {submit_time:.3f}ms")
    print(f"等待时间: {wait_time:.3f}ms")
    print(f"总时间: {overall_time:.3f}ms")
    print()
    
    print(f"任务统计:")
    print(f"  单任务平均时间: {avg_task_time:.3f}ms")
    print(f"  最快任务: {np.min(task_times):.3f}ms")
    print(f"  最慢任务: {np.max(task_times):.3f}ms")
    print()
    
    print(f"完成时间分析:")
    print(f"  第一个任务完成: {completion_times[0]:.3f}ms")
    print(f"  最后一个任务完成: {completion_times[-1]:.3f}ms")
    print(f"  完成时间跨度: {completion_times[-1] - completion_times[0]:.3f}ms")
    print()
    
    print(f"线程分布:")
    print(f"  活跃线程数: {len(thread_counts)}")
    for tid, count in sorted(thread_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    Thread {tid}: {count} 个任务")
    print()
    
    # 计算并行效率
    serial_time = avg_task_time * num_calls
    parallel_efficiency = serial_time / wait_time if wait_time > 0 else 0
    
    print(f"并行效率分析:")
    print(f"  串行等价时间: {serial_time:.3f}ms")
    print(f"  实际等待时间: {wait_time:.3f}ms")
    print(f"  并行效率: {parallel_efficiency:.2f}x")
    print(f"  wait_time / 单任务时间: {wait_time / avg_task_time:.2f}x")
    print()
    
    # 诊断
    ratio = wait_time / avg_task_time
    if ratio > num_calls * 0.8:
        print(f"  🔴 严重问题: 几乎完全串行执行!")
        print(f"     wait_time ({wait_time:.1f}ms) ≈ 串行时间 ({serial_time:.1f}ms)")
        print(f"     可能原因: Python层的GIL竞争")
    elif ratio > num_calls / num_threads * 1.5:
        print(f"  🟡 部分串行: 并行效率不佳")
        print(f"     可能原因: 锁竞争或负载不均")
    else:
        print(f"  ✅ 良好并行")
    
    return {
        'wait_time': wait_time,
        'avg_task_time': avg_task_time,
        'parallel_efficiency': parallel_efficiency,
        'ratio': ratio
    }


def test_3_pure_cpp_call(suffix_cache, problem_id, pattern, num_threads=9, num_calls=16):
    """测试：直接调用C++对象（最小化Python开销）"""
    print("=" * 80)
    print(f"Test 3: 最小化Python开销测试")
    print("=" * 80)
    
    # 预先获取problem_tree对象，避免在线程中访问字典
    if problem_id not in suffix_cache._problem_tree:
        print("错误: problem_id不存在")
        return None
    
    problem_tree = suffix_cache._problem_tree[problem_id]
    print(f"已获取problem_tree对象，避免字典访问")
    print()
    
    def worker(task_id):
        start = time.perf_counter()
        
        # 直接调用C++对象，不经过Python字典
        candidate = problem_tree.speculate(
            pattern,
            5,  # max_spec_tokens
            1.0,  # max_spec_factor
            -1,  # max_spec_offset
            0.1,  # min_token_prob
            False  # use_tree_spec
        )
        
        end = time.perf_counter()
        return (end - start) * 1000
    
    overall_start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_calls)]
        
        wait_start = time.perf_counter()
        times = []
        for future in as_completed(futures):
            times.append(future.result())
        wait_time = (time.perf_counter() - wait_start) * 1000
    
    overall_time = (time.perf_counter() - overall_start) * 1000
    
    avg_time = np.mean(times)
    serial_time = avg_time * num_calls
    parallel_efficiency = serial_time / wait_time if wait_time > 0 else 0
    
    print(f"等待时间: {wait_time:.3f}ms")
    print(f"单任务平均时间: {avg_time:.3f}ms")
    print(f"串行等价时间: {serial_time:.3f}ms")
    print(f"并行效率: {parallel_efficiency:.2f}x")
    print()
    
    if parallel_efficiency < 1.5:
        print(f"  🔴 即使绕过Python字典，仍然串行!")
        print(f"     问题在C++层或更底层")
    elif parallel_efficiency < 3.0:
        print(f"  🟡 有改善但不够")
        print(f"     Python开销占一部分")
    else:
        print(f"  ✅ 绕过Python字典后并行良好")
        print(f"     说明问题在Python层的字典访问")
    
    return {
        'wait_time': wait_time,
        'avg_task_time': avg_time,
        'parallel_efficiency': parallel_efficiency
    }


def main():
    print("初始化SuffixCache...")
    
    # 创建一个简单的测试树
    suffix_cache = SuffixCache(max_depth=64, thread_safe=True, max_threads=10)
    
    problem_id = "test_problem"
    prompt_tokens = list(range(100))  # 简单的prompt
    response_tokens = list(range(100, 200))  # 简单的response
    
    # 构建树
    suffix_cache.prebuild_problemtree(0, problem_id, prompt_tokens, response_tokens)
    
    # 测试pattern
    pattern = list(range(150, 165))  # 15个token的pattern
    
    print(f"测试配置:")
    print(f"  Problem ID: {problem_id}")
    print(f"  Pattern length: {len(pattern)}")
    print(f"  Thread-safe mode: {suffix_cache._thread_safe}")
    print()
    
    # 运行测试
    avg_single = test_1_single_thread_baseline(suffix_cache, problem_id, pattern, num_calls=10)
    
    test2_result = test_2_multi_thread_same_problem(suffix_cache, problem_id, pattern, 
                                                     num_threads=9, num_calls=16)
    
    test3_result = test_3_pure_cpp_call(suffix_cache, problem_id, pattern,
                                        num_threads=9, num_calls=16)
    
    # 对比分析
    print("=" * 80)
    print("对比分析")
    print("=" * 80)
    
    if test2_result and test3_result:
        test2_eff = test2_result['parallel_efficiency']
        test3_eff = test3_result['parallel_efficiency']
        
        print(f"Test 2 (通过Python) 并行效率: {test2_eff:.2f}x")
        print(f"Test 3 (直接C++)  并行效率: {test3_eff:.2f}x")
        print(f"效率提升: {(test3_eff / test2_eff - 1) * 100:.1f}%")
        print()
        
        if test3_eff > test2_eff * 1.5:
            print("🎯 结论: Python字典访问是主要瓶颈!")
            print("   解决方案: 考虑在C++层实现problem_id到tree的映射")
        elif test3_eff < 2.0 and test2_eff < 2.0:
            print("🎯 结论: 问题在更底层（C++或系统层）")
            print("   可能原因:")
            print("   - CPU核心数限制（检查Docker配置）")
            print("   - C++内部有隐藏的锁")
            print("   - 内存带宽瓶颈")
        else:
            print("🎯 结论: 混合问题")
            print("   Python层和C++层都有优化空间")


if __name__ == "__main__":
    main()


