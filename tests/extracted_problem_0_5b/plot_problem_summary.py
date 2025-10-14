import os
from typing import List
import argparse
import pandas as pd
import matplotlib.pyplot as plt


#CSV_PATH = "/data/zshao/rllm/ArcticInference/tests/extracted_problem_data/problem_f444_summary.csv"
RESULTS_DIR = "/data/zshao/rllm/ArcticInference/tests/extracted_problem_data/results"


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}. Found columns: {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser(description="Step-based suffix cache simulation")
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the CSV file",
    )
    args = parser.parse_args()
    CSV_PATH = args.csv_path
    OUTPUT_PATH = os.path.join(RESULTS_DIR, f"problem_{os.path.basename(CSV_PATH).split('.')[0]}_summary_scatter.png")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required = ["current_step", "avg_accept_toks", "avg_angle", "avg_score"]
    validate_columns(df, required)

    steps = df["current_step"]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), constrained_layout=True)

    # current_step vs avg_accept_toks
    axes[0].scatter(steps, df["avg_accept_toks"], s=16, alpha=0.75, edgecolor="none")
    axes[0].set_title("current_step vs avg_accept_toks")
    axes[0].set_xlabel("current_step")
    axes[0].set_ylabel("avg_accept_toks")
    axes[0].grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    # current_step vs avg_spec_toks (interpreting duplicate request as spec toks)
    axes[1].scatter(steps, df["avg_angle"], s=16, alpha=0.75, edgecolor="none")
    axes[1].set_title("current_step vs avg_angle")
    axes[1].set_xlabel("current_step")
    axes[1].set_ylabel("avg_angle")
    axes[1].grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    # current_step vs avg_score
    axes[2].scatter(steps, df["avg_score"], s=16, alpha=0.75, edgecolor="none")
    axes[2].set_title("current_step vs avg_score")
    axes[2].set_xlabel("current_step")
    axes[2].set_ylabel("avg_score")
    axes[2].grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    fig.suptitle("problem_f444_summary scatter plots", fontsize=14)

    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved scatter plot figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


