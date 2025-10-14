import argparse
import csv
from typing import List, Tuple

import matplotlib

# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_step_and_avg_accept(csv_path: str) -> Tuple[List[float], List[float]]:
    steps: List[float] = []
    avg_accepts: List[float] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows missing required fields
            if "current_step" not in row or "avg_accept_toks" not in row:
                continue
            try:
                step_val = float(row["current_step"])  # current_step looks like ints but keep as float
                avg_val = float(row["avg_accept_toks"])  # avg_accept_toks is float
            except (ValueError, TypeError):
                # Skip malformed rows
                continue
            steps.append(step_val)
            avg_accepts.append(avg_val)

    # Sort by step to ensure monotonic x for plotting
    if steps:
        paired = sorted(zip(steps, avg_accepts), key=lambda p: p[0])
        steps, avg_accepts = [p[0] for p in paired], [p[1] for p in paired]
    return steps, avg_accepts


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot step vs avg_accept_toks for two CSVs on one figure.")
    parser.add_argument("--csv1", required=True, help="Path to first CSV file")
    parser.add_argument("--csv2", required=True, help="Path to second CSV file")
    parser.add_argument("--label1", default="csv1", help="Legend label for first CSV")
    parser.add_argument("--label2", default="csv2", help="Legend label for second CSV")
    parser.add_argument("--out", required=True, help="Output PNG path")
    args = parser.parse_args()

    x1, y1 = read_step_and_avg_accept(args.csv1)
    x2, y2 = read_step_and_avg_accept(args.csv2)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.scatter(x1, y1, label=args.label1, color="#1f77b4", alpha=0.6, s=20)
    plt.scatter(x2, y2, label=args.label2, color="#ff7f0e", alpha=0.6, s=20)

    plt.xlabel("current_step")
    plt.ylabel("avg_accept_toks")
    plt.title("Step vs avg_accept_toks")
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)


if __name__ == "__main__":
    main()


