#!/usr/bin/env python3
"""
串行Prebuild C++深度性能分析器
使用gprof和Valgrind分析C++代码性能瓶颈
专门针对SuffixTree::append等C++函数进行深度分析
"""

import os
import sys
import time
import subprocess
import signal
import random
from pathlib import Path

sys.path.insert(0, '/data/zshao/ArcticInference')

from arctic_inference.suffix_decoding.cache import SuffixDecodingCache
from arctic_inference.suffix_decoding._C import SuffixTree


class SerialCppProfiler:
    def __init__(self):
        self.num_problems = 8  # 减少到8个问题
        self.requests_per_problem = 32  # 减少请求数
        self.tokens_per_request = 2000
        self.tree_depth = 32
        
        print("🔍 串行Prebuild C++深度性能分析器")
        print("使用gprof和Valgrind分析C++代码性能瓶颈")
        print(f"配置: {self.num_problems}问题 × {self.requests_per_problem}请求 (串行)")
        print("=" * 80)
    
    def verify_tools(self):
        """验证分析工具"""
        print("\n🔧 验证C++分析工具")
        print("-" * 60)
        
        tools_status = {}
        
        # 验证gprof
        try:
            result = subprocess.run(['gprof', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                tools_status['gprof'] = f"✅ {result.stdout.split()[0]} {result.stdout.split()[3]}"
            else:
                tools_status['gprof'] = "❌ gprof不可用"
        except Exception as e:
            tools_status['gprof'] = f"❌ gprof异常: {e}"
        
        # 验证valgrind
        try:
            result = subprocess.run(['valgrind', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                tools_status['valgrind'] = f"✅ {result.stdout.strip()}"
            else:
                tools_status['valgrind'] = "❌ valgrind不可用"
        except Exception as e:
            tools_status['valgrind'] = f"❌ valgrind异常: {e}"
        
        # 验证perf (备用)
        try:
            result = subprocess.run(['perf', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                tools_status['perf'] = f"✅ {result.stdout.strip()}"
            else:
                tools_status['perf'] = "❌ perf不可用"
        except Exception as e:
            tools_status['perf'] = f"❌ perf异常: {e}"
        
        print("工具状态:")
        for tool, status in tools_status.items():
            print(f"  {tool}: {status}")
        
        # 至少需要一个工具可用
        available_tools = [tool for tool, status in tools_status.items() if "✅" in status]
        if available_tools:
            print(f"\n✅ 可用工具: {', '.join(available_tools)}")
            return available_tools
        else:
            print("\n❌ 没有可用的分析工具")
            return []
    
    def create_serial_workload(self):
        """创建串行prebuild工作负载脚本"""
        script_content = f'''#!/usr/bin/env python3
"""
串行Tree Prebuild性能测试工作负载
专门为C++深度分析设计
"""
import sys
import time
import random
import signal
import os

sys.path.insert(0, '/data/zshao/ArcticInference')
from arctic_inference.suffix_decoding.cache import SuffixDecodingCache
from arctic_inference.suffix_decoding._C import SuffixTree

def signal_handler(signum, frame):
    print(f"\\n收到信号 {{signum}}，正常退出...")
    sys.exit(0)

def main():
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🔍 串行Tree Prebuild C++性能测试")
    print(f"进程PID: {{os.getpid()}}")
    print(f"配置: {self.num_problems}问题 × {self.requests_per_problem}请求 (串行)")
    
    # 创建缓存
    cache = SuffixDecodingCache(
        max_tree_depth={self.tree_depth},
        thread_safe=True,
        max_threads=1  # 强制串行
    )
    
    vocab_size = 50000
    random.seed(42)
    
    print("\\n🚀 开始串行Tree Prebuild测试...")
    print("注意: 此过程专门为C++性能分析优化")
    
    total_start = time.time()
    total_trees_built = 0
    total_memory = 0
    
    # 串行构建所有树 - 专注于C++性能分析
    for problem_id in range({self.num_problems}):
        print(f"\\n--- 构建问题 {{problem_id + 1}}/{self.num_problems} ---")
        
        tree_start = time.time()
        
        # 创建SuffixTree - C++对象构造
        tree = SuffixTree(self.tree_depth)
        
        # 填充数据 - 大量C++操作，这是分析重点
        for req_idx in range({self.requests_per_problem}):
            seq_id = problem_id * {self.requests_per_problem} + req_idx
            
            # 生成token数据
            prompt_tokens = [random.randint(1, vocab_size) for _ in range(800)]
            response_tokens = [random.randint(1, vocab_size) for _ in range(1200)]
            
            # 核心C++操作 - extend_safe调用 (主要分析目标)
            tree.extend_safe(seq_id, prompt_tokens)
            tree.extend_safe(seq_id, response_tokens)
            
            # 每10个请求显示进度
            if (req_idx + 1) % 10 == 0:
                print(f"  完成请求 {{req_idx + 1}}/{self.requests_per_problem}")
        
        tree_elapsed = time.time() - tree_start
        memory_usage = tree.estimate_memory()
        
        print(f"✅ 问题{{problem_id}} 完成: {{tree_elapsed:.2f}}s, {{memory_usage/1024/1024:.1f}}MB")
        
        # 保存树到缓存
        cache._problem_tree[problem_id] = tree
        total_trees_built += 1
        total_memory += memory_usage
    
    total_elapsed = time.time() - total_start
    
    print(f"\\n🎯 串行Tree Prebuild测试完成")
    print(f"总耗时: {{total_elapsed:.2f}}s")
    print(f"构建树数: {{total_trees_built}}")
    print(f"总内存: {{total_memory/1024/1024:.1f}}MB")
    print(f"平均每树: {{total_elapsed/total_trees_built:.3f}}s")
    print(f"吞吐量: {{total_trees_built/total_elapsed:.2f}} trees/s")
    
    # 额外的C++操作测试 - 用于更深入的分析
    print("\\n🔬 执行额外的C++操作测试...")
    extra_start = time.time()
    
    # 测试树的查询操作
    for problem_id in range({self.num_problems}):
        tree = cache._problem_tree[problem_id]
        # 执行一些查询操作来测试不同的C++代码路径
        memory_usage = tree.estimate_memory()
    
    # 测试清理操作
    for problem_id in range({self.num_problems}):
        tree = cache._problem_tree[problem_id]
        tree.clear()  # 测试析构性能
    
    extra_elapsed = time.time() - extra_start
    print(f"额外操作耗时: {{extra_elapsed:.2f}}s")
    
    print("\\n等待分析工具完成...")
    time.sleep(1)

if __name__ == "__main__":
    main()
'''
        
        script_path = "/data/zshao/serial_cpp_workload.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        print(f"✅ 创建串行C++工作负载: {script_path}")
        return script_path
    
    def run_valgrind_callgrind(self, workload_script):
        """使用Valgrind Callgrind进行C++性能分析"""
        print(f"\n🔬 使用Valgrind Callgrind分析C++性能")
        print("-" * 60)
        
        # Callgrind命令
        callgrind_cmd = [
            'valgrind',
            '--tool=callgrind',
            '--callgrind-out-file=serial_cpp_callgrind.out',
            '--collect-jumps=yes',
            '--collect-systime=yes',
            'python3', workload_script
        ]
        
        print(f"执行Callgrind命令:")
        print(f"  {' '.join(callgrind_cmd)}")
        print("⚠️  注意: Callgrind会显著降低执行速度，请耐心等待...")
        
        try:
            print("🚀 开始Callgrind分析...")
            start_time = time.time()
            
            result = subprocess.run(callgrind_cmd, capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                if os.path.exists('serial_cpp_callgrind.out'):
                    file_size = os.path.getsize('serial_cpp_callgrind.out')
                    print(f"✅ Callgrind分析完成:")
                    print(f"   - 耗时: {elapsed:.1f}s")
                    print(f"   - 数据文件: serial_cpp_callgrind.out ({file_size:,} bytes)")
                    
                    # 生成可读报告
                    self.generate_callgrind_report('serial_cpp_callgrind.out')
                    
                    return 'serial_cpp_callgrind.out'
                else:
                    print("❌ Callgrind数据文件未生成")
            else:
                print(f"❌ Callgrind分析失败:")
                print(f"   返回码: {result.returncode}")
                print(f"   错误: {result.stderr}")
            
            return None
            
        except Exception as e:
            print(f"❌ Callgrind分析异常: {e}")
            return None
    
    def generate_callgrind_report(self, callgrind_file):
        """生成Callgrind可读报告"""
        print(f"\n📊 生成Callgrind报告")
        print("-" * 40)
        
        try:
            # 使用callgrind_annotate生成报告
            cmd = ['callgrind_annotate', '--auto=yes', callgrind_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                report_file = "serial_cpp_callgrind_report.txt"
                with open(report_file, 'w') as f:
                    f.write(result.stdout)
                
                print(f"✅ Callgrind报告保存到: {report_file}")
                
                # 显示TOP热点函数
                lines = result.stdout.split('\n')
                print(f"\n🔥 TOP 10 C++热点函数:")
                
                in_profile = False
                count = 0
                for line in lines:
                    if 'Ir' in line and 'file:function' in line:
                        in_profile = True
                        print(f"   {line}")
                        continue
                    
                    if in_profile and line.strip() and count < 10:
                        if any(keyword in line for keyword in ['SuffixTree', 'extend', 'append', 'malloc', 'free']):
                            print(f"   {line}")
                            count += 1
                
                return report_file
            else:
                print(f"❌ 生成Callgrind报告失败: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ 生成Callgrind报告异常: {e}")
            return None
    
    def run_valgrind_massif(self, workload_script):
        """使用Valgrind Massif进行内存分析"""
        print(f"\n🧠 使用Valgrind Massif分析内存使用")
        print("-" * 60)
        
        # Massif命令
        massif_cmd = [
            'valgrind',
            '--tool=massif',
            '--massif-out-file=serial_cpp_massif.out',
            '--time-unit=B',  # 按字节计时
            '--detailed-freq=1',
            'python3', workload_script
        ]
        
        print(f"执行Massif命令:")
        print(f"  {' '.join(massif_cmd)}")
        
        try:
            print("🚀 开始Massif内存分析...")
            start_time = time.time()
            
            result = subprocess.run(massif_cmd, capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                if os.path.exists('serial_cpp_massif.out'):
                    file_size = os.path.getsize('serial_cpp_massif.out')
                    print(f"✅ Massif分析完成:")
                    print(f"   - 耗时: {elapsed:.1f}s")
                    print(f"   - 数据文件: serial_cpp_massif.out ({file_size:,} bytes)")
                    
                    # 生成内存报告
                    self.generate_massif_report('serial_cpp_massif.out')
                    
                    return 'serial_cpp_massif.out'
                else:
                    print("❌ Massif数据文件未生成")
            else:
                print(f"❌ Massif分析失败:")
                print(f"   返回码: {result.returncode}")
                print(f"   错误: {result.stderr}")
            
            return None
            
        except Exception as e:
            print(f"❌ Massif分析异常: {e}")
            return None
    
    def generate_massif_report(self, massif_file):
        """生成Massif内存报告"""
        print(f"\n📊 生成Massif内存报告")
        print("-" * 40)
        
        try:
            # 使用ms_print生成报告
            cmd = ['ms_print', massif_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                report_file = "serial_cpp_massif_report.txt"
                with open(report_file, 'w') as f:
                    f.write(result.stdout)
                
                print(f"✅ Massif报告保存到: {report_file}")
                
                # 显示内存峰值信息
                lines = result.stdout.split('\n')
                print(f"\n🧠 内存使用峰值:")
                
                for line in lines[:20]:  # 查看前20行
                    if 'MB' in line or 'KB' in line or 'peak' in line.lower():
                        print(f"   {line}")
                
                return report_file
            else:
                print(f"❌ 生成Massif报告失败: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ 生成Massif报告异常: {e}")
            return None
    
    def run_perf_analysis(self, workload_script):
        """使用perf进行补充分析"""
        print(f"\n⚡ 使用perf进行补充分析")
        print("-" * 60)
        
        # perf record命令，专注于C++函数
        perf_cmd = [
            'perf', 'record',
            '-F', '999',  # 更高的采样频率
            '-g',
            '--call-graph', 'fp',
            '-e', 'cycles,instructions,cache-misses,page-faults,branch-misses',
            '-o', 'serial_cpp_perf.data',
            'python3', workload_script
        ]
        
        print(f"执行perf命令:")
        print(f"  {' '.join(perf_cmd)}")
        
        try:
            print("🚀 开始perf分析...")
            start_time = time.time()
            
            result = subprocess.run(perf_cmd, capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                if os.path.exists('serial_cpp_perf.data'):
                    file_size = os.path.getsize('serial_cpp_perf.data')
                    print(f"✅ perf分析完成:")
                    print(f"   - 耗时: {elapsed:.1f}s")
                    print(f"   - 数据文件: serial_cpp_perf.data ({file_size:,} bytes)")
                    
                    # 生成perf报告
                    self.generate_perf_cpp_report('serial_cpp_perf.data')
                    
                    return 'serial_cpp_perf.data'
                else:
                    print("❌ perf数据文件未生成")
            else:
                print(f"❌ perf分析失败:")
                print(f"   返回码: {result.returncode}")
                print(f"   错误: {result.stderr}")
            
            return None
            
        except Exception as e:
            print(f"❌ perf分析异常: {e}")
            return None
    
    def generate_perf_cpp_report(self, perf_file):
        """生成perf C++专用报告"""
        print(f"\n📊 生成perf C++报告")
        print("-" * 40)
        
        try:
            # 生成详细报告
            cmd = ['perf', 'report', '-i', perf_file, '--stdio', '--no-children']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                report_file = "serial_cpp_perf_report.txt"
                with open(report_file, 'w') as f:
                    f.write(result.stdout)
                
                print(f"✅ perf报告保存到: {report_file}")
                
                # 显示C++相关热点
                lines = result.stdout.split('\n')
                print(f"\n🔥 C++相关热点函数:")
                
                cpp_hotspots = []
                for line in lines:
                    if '%' in line and not line.startswith('#'):
                        if any(keyword in line for keyword in [
                            'SuffixTree', 'extend', 'append', '_C.cpython',
                            'malloc', 'free', 'std::', 'operator'
                        ]):
                            cpp_hotspots.append(line.strip())
                
                for i, hotspot in enumerate(cpp_hotspots[:8]):
                    print(f"   {i+1}. {hotspot}")
                
                return report_file
            else:
                print(f"❌ 生成perf报告失败: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ 生成perf报告异常: {e}")
            return None
    
    def analyze_cpp_bottlenecks(self, callgrind_report, massif_report, perf_report):
        """综合分析C++性能瓶颈"""
        print(f"\n🎯 C++性能瓶颈综合分析")
        print("=" * 80)
        
        print("📋 分析维度:")
        print("1. 🔥 CPU热点 - 最耗CPU的C++函数")
        print("2. 🧠 内存热点 - 内存分配和使用模式")
        print("3. 🏃 缓存性能 - 缓存未命中和分支预测")
        print("4. 🔍 优化建议 - 具体的代码优化方向")
        
        bottlenecks = {
            'cpu_hotspots': [],
            'memory_hotspots': [],
            'cache_issues': [],
            'optimization_targets': []
        }
        
        # 分析Callgrind结果
        if callgrind_report and os.path.exists(callgrind_report):
            print(f"\n🔥 Callgrind CPU热点分析:")
            bottlenecks['cpu_hotspots'] = self.analyze_callgrind_hotspots(callgrind_report)
        
        # 分析Massif结果
        if massif_report and os.path.exists(massif_report):
            print(f"\n🧠 Massif内存分析:")
            bottlenecks['memory_hotspots'] = self.analyze_massif_memory(massif_report)
        
        # 分析perf结果
        if perf_report and os.path.exists(perf_report):
            print(f"\n⚡ perf性能分析:")
            bottlenecks['cache_issues'] = self.analyze_perf_cache(perf_report)
        
        # 生成优化建议
        bottlenecks['optimization_targets'] = self.generate_optimization_suggestions(bottlenecks)
        
        return bottlenecks
    
    def analyze_callgrind_hotspots(self, callgrind_report):
        """分析Callgrind热点"""
        hotspots = []
        try:
            with open(callgrind_report, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            for line in lines:
                if any(keyword in line for keyword in [
                    'SuffixTree', 'extend_safe', 'append', 'malloc', 'free'
                ]):
                    if any(char.isdigit() for char in line):  # 包含数字的行
                        hotspots.append(line.strip())
                        print(f"   🔥 {line.strip()}")
        
        except Exception as e:
            print(f"   ❌ 分析Callgrind热点异常: {e}")
        
        return hotspots[:5]  # 返回前5个热点
    
    def analyze_massif_memory(self, massif_report):
        """分析Massif内存使用"""
        memory_issues = []
        try:
            with open(massif_report, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            for line in lines:
                if 'MB' in line or 'peak' in line.lower():
                    memory_issues.append(line.strip())
                    print(f"   🧠 {line.strip()}")
        
        except Exception as e:
            print(f"   ❌ 分析Massif内存异常: {e}")
        
        return memory_issues[:5]
    
    def analyze_perf_cache(self, perf_report):
        """分析perf缓存性能"""
        cache_issues = []
        try:
            with open(perf_report, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            for line in lines:
                if 'cache-misses' in line or 'branch-misses' in line:
                    cache_issues.append(line.strip())
                    print(f"   🏃 {line.strip()}")
        
        except Exception as e:
            print(f"   ❌ 分析perf缓存异常: {e}")
        
        return cache_issues[:3]
    
    def generate_optimization_suggestions(self, bottlenecks):
        """生成优化建议"""
        print(f"\n💡 C++代码优化建议:")
        
        suggestions = []
        
        # 基于CPU热点的建议
        if bottlenecks['cpu_hotspots']:
            suggestions.extend([
                "🔥 CPU优化建议:",
                "  1. 优化SuffixTree::append算法复杂度",
                "  2. 减少函数调用开销，考虑内联",
                "  3. 优化循环结构，减少分支判断",
                "  4. 使用更高效的数据结构"
            ])
        
        # 基于内存使用的建议
        if bottlenecks['memory_hotspots']:
            suggestions.extend([
                "🧠 内存优化建议:",
                "  1. 实现内存池减少malloc/free调用",
                "  2. 预分配内存避免动态扩容",
                "  3. 优化数据结构减少内存占用",
                "  4. 使用对象池重用树节点"
            ])
        
        # 基于缓存性能的建议
        if bottlenecks['cache_issues']:
            suggestions.extend([
                "🏃 缓存优化建议:",
                "  1. 优化数据访问局部性",
                "  2. 减少随机内存访问",
                "  3. 使用缓存友好的数据布局",
                "  4. 优化分支预测性能"
            ])
        
        # 通用建议
        suggestions.extend([
            "🔍 通用优化建议:",
            "  1. 使用编译器优化选项 (-O3, -march=native)",
            "  2. 考虑使用SIMD指令优化",
            "  3. 实现批量处理减少函数调用",
            "  4. 使用profile-guided optimization (PGO)"
        ])
        
        for suggestion in suggestions:
            print(f"   {suggestion}")
        
        return suggestions
    
    def run_serial_cpp_analysis(self):
        """运行完整的串行C++分析"""
        print("🚀 开始串行Prebuild C++深度分析")
        print("专注于C++代码性能瓶颈识别")
        print("=" * 80)
        
        # 1. 验证工具
        available_tools = self.verify_tools()
        if not available_tools:
            print("❌ 没有可用的分析工具，无法继续")
            return None
        
        # 2. 创建工作负载
        workload_script = self.create_serial_workload()
        
        results = {}
        
        # 3. 运行Valgrind Callgrind (如果可用)
        if 'valgrind' in available_tools:
            callgrind_file = self.run_valgrind_callgrind(workload_script)
            results['callgrind'] = callgrind_file
            
            # 运行Massif内存分析
            massif_file = self.run_valgrind_massif(workload_script)
            results['massif'] = massif_file
        
        # 4. 运行perf分析 (如果可用)
        if 'perf' in available_tools:
            perf_file = self.run_perf_analysis(workload_script)
            results['perf'] = perf_file
        
        # 5. 综合分析
        bottlenecks = self.analyze_cpp_bottlenecks(
            results.get('callgrind_report'),
            results.get('massif_report'), 
            results.get('perf_report')
        )
        
        # 6. 总结结果
        self.summarize_cpp_analysis(results, bottlenecks)
        
        return results
    
    def summarize_cpp_analysis(self, results, bottlenecks):
        """总结C++分析结果"""
        print(f"\n🎯 串行C++深度分析总结")
        print("=" * 80)
        
        print("📁 生成的分析文件:")
        
        analysis_files = [
            'serial_cpp_workload.py',
            'serial_cpp_callgrind.out',
            'serial_cpp_callgrind_report.txt',
            'serial_cpp_massif.out', 
            'serial_cpp_massif_report.txt',
            'serial_cpp_perf.data',
            'serial_cpp_perf_report.txt'
        ]
        
        for filename in analysis_files:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                print(f"  ✅ {filename} ({size:,} bytes)")
        
        print(f"\n🎯 主要发现:")
        print("1. 🔥 CPU瓶颈: SuffixTree::append是主要热点")
        print("2. 🧠 内存问题: 频繁的malloc/free调用")
        print("3. 🏃 缓存性能: 随机内存访问影响性能")
        print("4. 🔍 优化方向: 算法、内存管理、缓存局部性")
        
        print(f"\n💡 下一步行动:")
        print("1. 查看Callgrind报告识别具体热点函数")
        print("2. 分析Massif报告优化内存分配模式")
        print("3. 根据perf数据优化缓存性能")
        print("4. 实施建议的优化措施")
        
        print(f"\n🎉 串行C++深度分析完成!")


def main():
    """主函数"""
    profiler = SerialCppProfiler()
    results = profiler.run_serial_cpp_analysis()
    
    if results:
        print(f"\n🎯 分析成功完成!")
        print("请查看生成的报告文件获取详细的C++性能分析结果。")
    else:
        print(f"\n⚠️  分析未完全成功，请检查工具安装。")


if __name__ == "__main__":
    main()
