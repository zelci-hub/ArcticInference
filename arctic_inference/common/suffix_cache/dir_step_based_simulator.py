"""
Directory-based step processor using common SuffixCache.

Given an input directory where each step's data is stored in a separate JSONL
file named like "<step>.jsonl", this script processes steps in ascending
numeric order. For each current step S:
  - Read S.jsonl and collect its set of problem_id values
  - For all previous steps (< S), load only rows whose problem_id is in that set
  - Build a SuffixCache from those filtered prior-step rows (prompt + response)
  - Run suffix_decode for each row in S.jsonl and record per-step statistics

This mirrors the behavior of step_based_simulator.py but adapts data loading to
directory-of-steps input and filters training data by the current step's
problem_ids.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from arctic_inference.common.suffix_cache import SuffixCache


def _read_jsonl(file_path: str) -> List[Dict]:
    data: List[Dict] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _tokenize_df(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    prompts: List[List[int]] = []
    responses: List[List[int]] = []
    for _, row in df.iterrows():
        prompts.append(tokenizer.encode(row["input"], add_special_tokens=False))
        responses.append(tokenizer.encode(row["output"], add_special_tokens=False))
    out = df.copy()
    out["prompt"] = prompts
    out["response"] = responses
    return out


def suffix_decode(
    suffix_cache: SuffixCache,
    request_id: int,
    problem_id: str,
    prompt: List[int],
    ground_truth_response: List[int],
    max_spec_tokens: int,
    max_spec_factor: float,
    min_token_prob: float,
    use_tree_spec: bool,
    use_cached_prompt: bool,
    tokenizer=None,
) -> List[Dict]:
    if not max_spec_tokens:
        max_spec_tokens = suffix_cache.max_depth

    if use_cached_prompt:
        suffix_cache.cache_prompt(request_id, prompt, problem_id)

    results: List[Dict] = []
    response: List[int] = []

    while len(response) < len(ground_truth_response):
        text = prompt + response
        pattern = text[-16:]

        result = suffix_cache.speculate(
            request_id,
            problem_id,
            pattern,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            min_token_prob=min_token_prob,
            use_tree_spec=use_tree_spec,
            use_cached_prompt=True,
        )

        # verify along the speculation tree
        accepted_tokens: List[int] = []
        node = -1
        for token_id in ground_truth_response[len(response):]:
            children = [i for i, p in enumerate(result.parents) if p == node]
            for c in children:
                if result.token_ids[c] == token_id:
                    accepted_tokens.append(token_id)
                    node = c
                    break
            else:
                break

        new_tokens = accepted_tokens.copy()
        response.extend(accepted_tokens)
        if len(response) < len(ground_truth_response):
            # add bonus token
            bonus_token = ground_truth_response[len(response)]
            new_tokens.append(bonus_token)
            response.append(bonus_token)

        results.append({
            "step": len(results),
            "match_len": result.match_len,
            "score": result.score,
            "num_spec_toks": len(result.token_ids),
            "num_accept_toks": len(accepted_tokens),
            "num_out_toks": len(new_tokens),
        })

    if use_cached_prompt:
        suffix_cache.evict_prompt(request_id)

    return results


def _summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    speedup = df.groupby(["current_step", "request_id"]).agg(
        sum_out_toks=("num_out_toks", "sum"),
        num_steps=("step", "count"),
    )
    speedup["speedup"] = speedup["sum_out_toks"] / speedup["num_steps"]
    speedup = speedup.groupby(["current_step"]).agg(
        req_speedup=("speedup", "mean"),
    )
    summary = df.groupby(["current_step"]).agg(
        sum_accept_toks=("num_accept_toks", "sum"),
        sum_spec_toks=("num_spec_toks", "sum"),
        sum_out_toks=("num_out_toks", "sum"),
        avg_accept_toks=("num_accept_toks", "mean"),
        avg_spec_toks=("num_spec_toks", "mean"),
        num_eval=("num_eval", "first"),
        num_train=("num_train", "first"),
    ).reset_index()
    summary["accept_rate"] = summary["sum_accept_toks"] / summary["sum_spec_toks"].replace(0, pd.NA)
    summary["req_speedup"] = speedup["req_speedup"].values
    return summary.set_index("current_step")


def run_dir_step_processor(
    input_dir: str,
    tokenizer_name: str,
    max_depth: int = 64,
    max_spec_tokens: int = 0,
    max_spec_factor: float = 1.0,
    min_token_prob: float = 0.1,
    use_tree_spec: bool = True,
    use_cached_prompt: bool = True,
    output_file: Optional[str] = None,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
    window_size: int = 100,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # discover step files (names like 1.jsonl, 2.jsonl)
    entries = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    steps: List[int] = []
    for name in entries:
        base = name[:-6]
        if base.isdigit():
            steps.append(int(base))
    steps = sorted(steps)
    all_steps = steps
    if start_step is not None:
        steps = [s for s in steps if s >= start_step]
    if end_step is not None:
        steps = [s for s in steps if s <= end_step]

    all_results: List[Dict] = []

    prev_step_to_df: Dict[int, pd.DataFrame] = {}

    for idx, current_step in enumerate(tqdm(steps, desc="Processing steps")):
        current_path = os.path.join(input_dir, f"{current_step}.jsonl")
        current_records = _read_jsonl(current_path)
        if len(current_records) == 0:
            continue

        current_df = pd.DataFrame(current_records)
        # require problem_id, input, output
        for col in ("problem_id", "input", "output"):
            if col not in current_df.columns:
                raise ValueError(f"Missing column '{col}' in {current_path}")

        # tokenize current step
        current_df = _tokenize_df(current_df, tokenizer)

        # set of problem_ids in current step
        problem_ids = set(current_df["problem_id"].tolist())

        # collect prior-step rows filtered by these problem_ids (filter first, then tokenize)
        train_frames: List[pd.DataFrame] = []
        for s in tqdm(all_steps, desc="Filtering prior steps"):
            if s < current_step - window_size:
                continue
            if s >= current_step:
                break
            if s not in prev_step_to_df:
                prev_path = os.path.join(input_dir, f"{s}.jsonl")
                prev_records = _read_jsonl(prev_path)
                prev_df = pd.DataFrame(prev_records) if prev_records else pd.DataFrame()
                if not prev_df.empty:
                    # minimal column checks (store raw to enable later filtering without extra tokenize work)
                    for col in ("problem_id", "input", "output"):
                        if col not in prev_df.columns:
                            raise ValueError(f"Missing column '{col}' in {prev_path}")
                prev_step_to_df[s] = prev_df  # cache RAW (un-tokenized) dataframe
            prev_df = prev_step_to_df[s]
            if not prev_df.empty:
                # filter first by problem_id
                filtered = prev_df[prev_df["problem_id"].isin(problem_ids)]
                if len(filtered) > 0:
                    # tokenize only the filtered subset
                    filtered_tok = _tokenize_df(filtered, tokenizer)
                    train_frames.append(filtered_tok)

        train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame(columns=current_df.columns)

        # build suffix cache from train_df
        suffix_cache = SuffixCache(max_depth)
        if len(train_df) > 0:
            for request_id, example in tqdm(train_df.iterrows(), total=len(train_df), desc="Building cache"):
                problem_id = example.get("problem_id", "0")
                suffix_cache.update_response(-1 - request_id, problem_id, example["prompt"])  # prompt tokens
                suffix_cache.update_response(-1 - request_id, problem_id, example["response"])  # response tokens

        # evaluate current step
        step_results: List[Dict] = []
        for request_id, example in tqdm(current_df.iterrows(), total=len(current_df), desc="Running evaluation"):
            problem_id = example.get("problem_id", "0")
            results = suffix_decode(
                suffix_cache,
                request_id,
                problem_id,
                example["prompt"],
                example["response"],
                max_spec_tokens,
                max_spec_factor=max_spec_factor,
                min_token_prob=min_token_prob,
                use_tree_spec=use_tree_spec,
                use_cached_prompt=use_cached_prompt,
                tokenizer=tokenizer,
            )
            for r in results:
                r.update({
                    "current_step": current_step,
                    "request_id": request_id,
                    "num_eval": len(current_df),
                    "num_train": len(train_df),
                    "max_depth": max_depth,
                    "max_spec_tokens": max_spec_tokens,
                    "max_spec_factor": max_spec_factor,
                    "min_token_prob": min_token_prob,
                    "use_tree_spec": use_tree_spec,
                    "use_cached_prompt": use_cached_prompt,
                    "problem_id": problem_id,
                })
            step_results.extend(results)

        all_results.extend(step_results)

        if step_results:
            step_df = pd.DataFrame(step_results)
            step_summary = _summarize_results(step_df)
            print(f"Step {current_step} summary:\n{step_summary.to_string()}\n")

    results_df = pd.DataFrame(all_results)
    if output_file:
        results_df.to_csv(output_file, index=False)
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Directory-based step processor for SuffixCache")
    parser.add_argument("input_dir", type=str, help="Directory containing step JSONL files named <step>.jsonl")
    parser.add_argument("--tokenizer", type=str, required=True, help="HuggingFace tokenizer name")
    parser.add_argument("--max-depth", type=int, default=64, help="Max suffix tree depth")
    parser.add_argument("--max-spec-tokens", type=int, default=0, help="Max speculation tokens (0=use max_depth)")
    parser.add_argument("--max-spec-factor", type=float, default=1.0, help="Spec tokens multiplier vs prefix length")
    parser.add_argument("--min-token-prob", type=float, default=0.1, help="Minimum token probability")
    parser.add_argument("--use-tree-spec", action="store_true", default=True, help="Use tree-based speculation")
    parser.add_argument("--use-cached-prompt", action="store_true", default=True, help="Cache prompt per request")
    parser.add_argument("-o", "--output", type=str, help="Output CSV path")
    parser.add_argument("--start-step", type=int, help="Start step (inclusive)")
    parser.add_argument("--end-step", type=int, help="End step (inclusive)")
    parser.add_argument("--window-size", type=int, default=100, help="Window size for filtering prior steps")
    args = parser.parse_args()

    df = run_dir_step_processor(
        input_dir=args.input_dir,
        tokenizer_name=args.tokenizer,
        max_depth=args.max_depth,
        max_spec_tokens=args.max_spec_tokens,
        max_spec_factor=args.max_spec_factor,
        min_token_prob=args.min_token_prob,
        use_tree_spec=args.use_tree_spec,
        use_cached_prompt=args.use_cached_prompt,
        output_file=args.output,
        start_step=args.start_step,
        end_step=args.end_step,
        window_size=args.window_size,
    )

    if len(df) > 0:
        print(_summarize_results(df).to_string())
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()


