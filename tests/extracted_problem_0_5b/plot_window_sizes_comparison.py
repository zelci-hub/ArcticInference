#!/usr/bin/env python3
"""
Plot current_step vs avg_accept_toks for all window_size directories
comparing problem_2bf0439a_sorted.csv files
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def main():
    base_dir = "/data/zshao/rllm/ArcticInference/tests/extracted_problem_0_5b/window_only"
    
    # Ensure base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return
    
    # Find all window_size directories
    window_dirs = glob.glob(os.path.join(base_dir, "window_size_*"))
    window_dirs.sort(key=lambda x: int(x.split('_')[-1]))  # Sort by window size number
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(window_dirs)))
    
    for i, window_dir in enumerate(window_dirs):
        window_size = os.path.basename(window_dir).split('_')[-1]
        csv_file = os.path.join(window_dir, "problem_2bf0439a_sorted.csv")
        
        if os.path.exists(csv_file):
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Check if required columns exist
                if 'current_step' in df.columns and 'avg_accept_toks' in df.columns:
                    # Plot the data
                    plt.plot(df['current_step'], df['avg_accept_toks'], 
                            marker='o', markersize=4, linewidth=2, 
                            color=colors[i], label=f'Window Size {window_size}',
                            alpha=0.8)
                    
                    print(f"✓ Plotted window_size_{window_size}: {len(df)} data points")
                else:
                    print(f"✗ Missing columns in window_size_{window_size}")
                    
            except Exception as e:
                print(f"✗ Error reading window_size_{window_size}: {e}")
        else:
            print(f"✗ File not found: window_size_{window_size}/problem_2bf0439a_sorted.csv")
    
    # Customize the plot
    plt.xlabel('Current Step', fontsize=12)
    plt.ylabel('Average Accept Tokens', fontsize=12)
    plt.title('Average Accept Tokens vs Current Step\nComparison Across Different Window Sizes', fontsize=14, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    output_file = "/data/zshao/rllm/ArcticInference/tests/extracted_problem_0_5b/window_sizes_comparison_problem_2bf0439a.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_file}")
    
    # Show statistics
    print("\n📈 Statistics Summary:")
    for i, window_dir in enumerate(window_dirs):
        window_size = os.path.basename(window_dir).split('_')[-1]
        csv_file = os.path.join(window_dir, "problem_2bf0439a_sorted.csv")
        
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                if 'avg_accept_toks' in df.columns:
                    avg_accept = df['avg_accept_toks'].mean()
                    max_accept = df['avg_accept_toks'].max()
                    min_accept = df['avg_accept_toks'].min()
                    print(f"Window Size {window_size:2s}: Avg={avg_accept:.3f}, Max={max_accept:.3f}, Min={min_accept:.3f}")
            except:
                pass
    
    # plt.show()  # Commented out for headless environment

if __name__ == "__main__":
    main()
