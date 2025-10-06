#!/usr/bin/env python3
"""
批量运行suffix cache simulator来分析不同problem_id训练集的性能差异
"""

import os
import subprocess
import json
import pandas as pd
from pathlib import Path
import time

def run_simulator(train_file, test_file, output_dir, problem_id):
    """运行simulator并返回结果"""
    
    # 创建输出目录
    output_path = Path(output_dir) / f"results_{problem_id}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 构建simulator命令
    cmd = [
        "python", "arctic_inference/common/suffix_cache/simulator.py",
        str(test_file),
        "--format", "jsonl",
        "--train-dataset", str(train_file),
        "--tokenizer", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "--output", str(output_path),
        "--prompt-column", "input_token_ids",
        "--response-column", "output_token_ids"
    ]
    
    print(f"运行 {problem_id}: {' '.join(cmd)}")
    
    try:
        # 运行命令
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ {problem_id} 完成")
            
            # 读取结果
            results_file = output_path / "results.csv"
            if results_file.exists():
                df = pd.read_csv(results_file)
                # 计算关键指标
                avg_acc_len = df['num_accept_toks'].mean()
                accept_rate = df['num_accept_toks'].sum() / df['num_spec_toks'].sum() if df['num_spec_toks'].sum() > 0 else 0
                
                return {
                    'problem_id': problem_id,
                    'avg_acc_len': avg_acc_len,
                    'accept_rate': accept_rate,
                    'total_steps': len(df),
                    'status': 'success'
                }
            else:
                return {
                    'problem_id': problem_id,
                    'status': 'no_results_file'
                }
        else:
            print(f"✗ {problem_id} 失败: {result.stderr}")
            return {
                'problem_id': problem_id,
                'status': 'failed',
                'error': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        print(f"✗ {problem_id} 超时")
        return {
            'problem_id': problem_id,
            'status': 'timeout'
        }
    except Exception as e:
        print(f"✗ {problem_id} 异常: {e}")
        return {
            'problem_id': problem_id,
            'status': 'exception',
            'error': str(e)
        }

def main():
    # 设置路径
    base_dir = Path(".")
    train_dir = base_dir / "tests" / "individual_problems"
    test_file = base_dir / "tests" / "problem0019.jsonl"
    output_dir = base_dir / "similarity_analysis_results"
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有训练文件（先测试前10个）
    train_files = sorted(list(train_dir.glob("train_prob_*.jsonl")))[:10]
    
    print(f"找到 {len(train_files)} 个训练文件，开始测试...")
    
    results = []
    
    for i, train_file in enumerate(train_files):
        problem_id = train_file.stem.replace("train_", "")
        print(f"\n进度: {i+1}/{len(train_files)} - 处理 {problem_id}")
        
        result = run_simulator(train_file, test_file, output_dir, problem_id)
        results.append(result)
        
        # 短暂休息避免资源占用过高
        time.sleep(1)
    
    # 保存汇总结果
    results_df = pd.DataFrame(results)
    summary_file = output_dir / "summary_results.csv"
    results_df.to_csv(summary_file, index=False)
    
    print(f"\n=== 汇总结果 ===")
    print(results_df)
    print(f"\n结果已保存到: {summary_file}")
    
    # 分析成功的结果
    successful_results = results_df[results_df['status'] == 'success']
    if len(successful_results) > 0:
        print(f"\n=== 性能分析 ===")
        print(f"成功运行: {len(successful_results)}/{len(results)} 个problem_id")
        print(f"平均 avg_acc_len: {successful_results['avg_acc_len'].mean():.4f}")
        print(f"平均 accept_rate: {successful_results['accept_rate'].mean():.4f}")
        print(f"avg_acc_len 标准差: {successful_results['avg_acc_len'].std():.4f}")
        print(f"accept_rate 标准差: {successful_results['accept_rate'].std():.4f}")
        
        # 显示最好和最差的结果
        best_acc_len = successful_results.loc[successful_results['avg_acc_len'].idxmax()]
        worst_acc_len = successful_results.loc[successful_results['avg_acc_len'].idxmin()]
        
        print(f"\n最高 avg_acc_len: {best_acc_len['problem_id']} = {best_acc_len['avg_acc_len']:.4f}")
        print(f"最低 avg_acc_len: {worst_acc_len['problem_id']} = {worst_acc_len['avg_acc_len']:.4f}")

if __name__ == "__main__":
    main()

