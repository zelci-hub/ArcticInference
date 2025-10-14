import argparse
import csv
import os
import glob
from typing import List, Tuple

import matplotlib
# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_avg_accept_and_avg_angle(csv_path: str) -> Tuple[List[float], List[float]]:
    """Read avg_accept_toks and avg_angle from a CSV file."""
    avg_accepts: List[float] = []
    avg_angles: List[float] = []
    
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows missing required fields
            if "avg_accept_toks" not in row or "avg_angle" not in row:
                continue
            try:
                avg_accept = float(row["avg_accept_toks"])
                avg_angle = float(row["avg_angle"])
            except (ValueError, TypeError):
                # Skip malformed rows
                continue
            avg_accepts.append(avg_accept)
            avg_angles.append(avg_angle)
    
    return avg_accepts, avg_angles


def read_all_csv_files_in_directory(directory: str) -> Tuple[List[float], List[float], List[str]]:
    """Read all CSV files in the directory and combine the data."""
    all_avg_accepts: List[float] = []
    all_avg_angles: List[float] = []
    file_names: List[str] = []
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    print(f"Found {len(csv_files)} CSV files in {directory}")
    
    for csv_file in csv_files:
        print(f"Reading {csv_file}...")
        avg_accepts, avg_angles = read_avg_accept_and_avg_angle(csv_file)
        
        # Add data from this file
        all_avg_accepts.extend(avg_accepts)
        all_avg_angles.extend(avg_angles)
        
        # Keep track of which file each data point came from
        file_name = os.path.basename(csv_file)
        file_names.extend([file_name] * len(avg_accepts))
        
        print(f"  Added {len(avg_accepts)} data points from {file_name}")
    
    print(f"Total data points: {len(all_avg_accepts)}")
    return all_avg_accepts, all_avg_angles, file_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot avg_accept_toks vs avg_angle for all CSV files in a directory.")
    parser.add_argument("--directory", required=True, help="Directory containing CSV files")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--title", default="avg_accept_toks vs avg_angle", help="Plot title")
    args = parser.parse_args()

    # Read all CSV files in the directory
    avg_accepts, avg_angles, file_names = read_all_csv_files_in_directory(args.directory)
    
    if not avg_accepts:
        print("No data found in CSV files!")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8), dpi=150)
    plt.scatter(avg_angles, avg_accepts, alpha=0.6, s=20, color="#1f77b4")
    
    plt.xlabel("avg_angle")
    plt.ylabel("avg_accept_toks")
    plt.title(args.title)
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    
    # Add some statistics to the plot
    import numpy as np
    if len(avg_accepts) > 1:
        correlation = np.corrcoef(avg_angles, avg_accepts)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}\nData points: {len(avg_accepts)}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()

