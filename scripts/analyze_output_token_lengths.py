#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from typing import Dict, Tuple


def compute_stats(input_path: str) -> Dict[str, Tuple[float, int, int]]:
    lengths_by_problem: Dict[str, Dict[str, float]] = defaultdict(lambda: {"sum": 0.0, "count": 0, "max": 0})

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            problem_id = obj.get("problem_id")
            token_ids = obj.get("output_token_ids")
            if problem_id is None or token_ids is None:
                continue

            try:
                length = len(token_ids)
            except TypeError:
                # If token_ids is not a list-like, skip
                continue

            bucket = lengths_by_problem[problem_id]
            bucket["sum"] += float(length)
            bucket["count"] += 1
            if length > bucket["max"]:
                bucket["max"] = length

    results: Dict[str, Tuple[float, int, int]] = {}
    for problem_id, agg in lengths_by_problem.items():
        if agg["count"] == 0:
            continue
        avg = agg["sum"] / agg["count"]
        results[problem_id] = (avg, int(agg["max"]), int(agg["count"]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute avg and max output_token_ids length per problem_id.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", default="", help="Optional path to write TSV output")
    parser.add_argument("--ascending", action="store_true", help="Sort ascending by (avg, max). Default is descending.")
    args = parser.parse_args()

    stats = compute_stats(args.input)
    items = [
        (pid, avg, max_len, count)
        for pid, (avg, max_len, count) in stats.items()
    ]

    # Sort by (average, max). Default descending
    items.sort(key=lambda x: (x[1], x[2]), reverse=not args.ascending)

    header = "problem_id\taverage_length\tmax_length\tcount"
    lines = [header]
    for pid, avg, max_len, count in items:
        lines.append(f"{pid}\t{avg:.6f}\t{max_len}\t{count}")

    output_text = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_f:
            out_f.write(output_text)
    else:
        print(output_text, end="")


if __name__ == "__main__":
    main()


