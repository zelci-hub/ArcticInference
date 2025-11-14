#!/usr/bin/env python3
"""
绘制cross-trajectory相似度矩阵的热力图
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_heatmap(
    csv_path: str,
    output_path: str = None,
    figsize: tuple = (12, 10),
    cmap: str = 'viridis',
    vmin: float = None,
    vmax: float = None,
    title: str = None,
    show_colorbar: bool = True,
    show_labels: bool = False,
    dpi: int = 300,
):
    """
    绘制相似度矩阵的热力图
    
    Args:
        csv_path: CSV文件路径
        output_path: 输出图片路径，如果为None则不保存
        figsize: 图片大小
        cmap: 颜色映射，可选: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 
              'coolwarm', 'RdYlBu_r', 'seismic'
        vmin: 颜色映射的最小值
        vmax: 颜色映射的最大值
        title: 图片标题
        show_colorbar: 是否显示colorbar
        show_labels: 是否显示行列标签（对于大矩阵建议关闭）
        dpi: 保存图片的DPI
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path, index_col=0)
    
    # 转换为numpy数组
    matrix = df.values
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Value range: [{matrix.min():.2f}, {matrix.max():.2f}]")
    print(f"Mean: {matrix.mean():.2f}, Std: {matrix.std():.2f}")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    im = ax.imshow(
        matrix,
        cmap=cmap,
        aspect='auto',
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax
    )
    
    # 设置标题
    if title is None:
        title = f'Cross-Trajectory Similarity Matrix\n(Average Accepted Tokens)'
    ax.set_title(title, fontsize=14, pad=20)
    
    # 设置轴标签
    ax.set_xlabel('Evaluation Trajectory Index', fontsize=12)
    ax.set_ylabel('Training Trajectory Index', fontsize=12)
    
    # 是否显示刻度标签
    if show_labels:
        ax.set_xticks(range(len(df.columns)))
        ax.set_yticks(range(len(df.index)))
        ax.set_xticklabels(df.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(df.index, fontsize=6)
    else:
        # 对于大矩阵，只显示部分刻度
        n_ticks = 10
        tick_positions = np.linspace(0, len(df) - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_positions)
        ax.set_yticklabels(tick_positions)
    
    # 添加colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Average Accepted Tokens', rotation=270, labelpad=20, fontsize=12)
    
    # 添加对角线（自我相似度）
    ax.plot([0, len(df) - 1], [0, len(df) - 1], 'r--', linewidth=1, alpha=0.5, label='Diagonal (self-similarity)')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Heatmap saved to: {output_path}")
    
    # 显示图片
    plt.show()
    
    return fig, ax


def plot_statistics(csv_path: str, output_path: str = None):
    """
    绘制矩阵统计信息
    
    Args:
        csv_path: CSV文件路径
        output_path: 输出图片路径
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path, index_col=0)
    matrix = df.values
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 对角线值分布（自我相似度）
    diagonal = np.diag(matrix)
    axes[0, 0].hist(diagonal, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Diagonal Values Distribution\n(Self-Similarity)', fontsize=12)
    axes[0, 0].set_xlabel('Average Accepted Tokens')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(diagonal.mean(), color='r', linestyle='--', label=f'Mean: {diagonal.mean():.2f}')
    axes[0, 0].legend()
    
    # 2. 非对角线值分布（交叉相似度）
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    axes[0, 1].hist(off_diagonal, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Off-Diagonal Values Distribution\n(Cross-Similarity)', fontsize=12)
    axes[0, 1].set_xlabel('Average Accepted Tokens')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(off_diagonal.mean(), color='r', linestyle='--', label=f'Mean: {off_diagonal.mean():.2f}')
    axes[0, 1].legend()
    
    # 3. 每行的平均值（作为训练数据的平均相似度）
    row_means = matrix.mean(axis=1)
    axes[1, 0].plot(row_means, linewidth=1)
    axes[1, 0].set_title('Average Similarity per Training Trajectory', fontsize=12)
    axes[1, 0].set_xlabel('Training Trajectory Index')
    axes[1, 0].set_ylabel('Average Accepted Tokens')
    axes[1, 0].axhline(row_means.mean(), color='r', linestyle='--', label=f'Overall Mean: {row_means.mean():.2f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 每列的平均值（作为评估数据的平均相似度）
    col_means = matrix.mean(axis=0)
    axes[1, 1].plot(col_means, linewidth=1)
    axes[1, 1].set_title('Average Similarity per Evaluation Trajectory', fontsize=12)
    axes[1, 1].set_xlabel('Evaluation Trajectory Index')
    axes[1, 1].set_ylabel('Average Accepted Tokens')
    axes[1, 1].axhline(col_means.mean(), color='r', linestyle='--', label=f'Overall Mean: {col_means.mean():.2f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Statistics plot saved to: {output_path}")
    
    plt.show()
    
    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description='绘制cross-trajectory相似度矩阵的热力图'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='CSV文件路径'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出图片路径（默认：与CSV同名的PNG文件）'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[12, 10],
        help='图片大小 (width height)，默认: 12 10'
    )
    parser.add_argument(
        '--cmap',
        type=str,
        default='viridis',
        choices=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                 'coolwarm', 'RdYlBu_r', 'seismic', 'hot', 'jet'],
        help='颜色映射，默认: viridis'
    )
    parser.add_argument(
        '--vmin',
        type=float,
        default=None,
        help='颜色映射的最小值'
    )
    parser.add_argument(
        '--vmax',
        type=float,
        default=None,
        help='颜色映射的最大值'
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='图片标题'
    )
    parser.add_argument(
        '--no-colorbar',
        action='store_true',
        help='不显示colorbar'
    )
    parser.add_argument(
        '--show-labels',
        action='store_true',
        help='显示所有行列标签（对于大矩阵不建议）'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='保存图片的DPI，默认: 300'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='同时绘制统计信息图'
    )
    
    args = parser.parse_args()
    
    # 确定输出路径
    if args.output is None:
        csv_path = Path(args.csv_path)
        args.output = str(csv_path.with_suffix('.png'))
    
    # 绘制热力图
    print("绘制热力图...")
    plot_heatmap(
        csv_path=args.csv_path,
        output_path=args.output,
        figsize=tuple(args.figsize),
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        title=args.title,
        show_colorbar=not args.no_colorbar,
        show_labels=args.show_labels,
        dpi=args.dpi,
    )
    
    # 绘制统计信息
    if args.stats:
        print("\n绘制统计信息...")
        stats_output = Path(args.output).with_stem(Path(args.output).stem + '_stats')
        plot_statistics(
            csv_path=args.csv_path,
            output_path=str(stats_output)
        )


if __name__ == '__main__':
    main()





