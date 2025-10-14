import os
import glob
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


SUMMARY_DIR = "/data/zshao/rllm/ArcticInference/tests/extracted_problem_data/summary"
RESULTS_DIR = "/data/zshao/rllm/ArcticInference/tests/extracted_problem_data/results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "summary_avg_angle_vs_avg_accept_toks.png")


def collect_points(paths: List[str]) -> Tuple[pd.Series, pd.Series, List[str]]:
    xs: List[pd.Series] = []
    ys: List[pd.Series] = []
    skipped: List[str] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            skipped.append(p)
            continue
        if not {"avg_angle", "avg_accept_toks"}.issubset(df.columns):
            skipped.append(p)
            continue
        # drop na rows for safety
        sub = df[["avg_angle", "avg_accept_toks"]].dropna()
        if len(sub) == 0:
            skipped.append(p)
            continue
        xs.append(sub["avg_angle"]) 
        ys.append(sub["avg_accept_toks"]) 
    if xs:
        x = pd.concat(xs, ignore_index=True)
        y = pd.concat(ys, ignore_index=True)
    else:
        x = pd.Series(dtype=float)
        y = pd.Series(dtype=float)
    return x, y, skipped


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(SUMMARY_DIR, "*_system.csv")))
    x, y, skipped = collect_points(csv_paths)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if len(x) == 0:
        ax.text(0.5, 0.5, "No valid data found", ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        ax.scatter(x, y, s=14, alpha=0.75, edgecolor="none")
        ax.set_xlabel("avg_angle")
        ax.set_ylabel("avg_accept_toks")
        ax.set_title("avg_angle vs avg_accept_toks (all summary CSVs)")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved scatter plot to: {OUTPUT_PATH}")
    if skipped:
        print(f"Skipped {len(skipped)} files missing required columns or unreadable.")


if __name__ == "__main__":
    main()


