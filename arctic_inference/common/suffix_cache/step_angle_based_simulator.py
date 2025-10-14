"""
Step-based simulator using common SuffixCache.

This mirrors the behavior of arctic_inference/suffix_decoding/step_based_simulator.py
but runs on the shared common.suffix_cache implementation and its suffix_decode logic.
"""
import argparse
import json
import os
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import time
from arctic_inference.common.suffix_cache import SuffixCache

def suffix_decode(
    suffix_cache: SuffixCache,
    request_id: int,
    problem_id: int,
    prompt: List[int],
    ground_truth_response: List[int],
    max_spec_tokens: int,
    max_spec_factor: float,
    min_token_prob: float,
    use_tree_spec: bool,
    use_cached_prompt: bool,
    debug_file_path: Optional[str] = None,
    tokenizer = None,
) -> List[Dict]:
    if not max_spec_tokens:
        max_spec_tokens = suffix_cache.max_depth

    if use_cached_prompt:
        suffix_cache.cache_prompt(request_id, prompt, problem_id)

    assert isinstance(prompt, list) and isinstance(ground_truth_response, list)

    # Open debug file if specified
    debug_file = None
    if debug_file_path:
        debug_file = open(debug_file_path, 'a', encoding='utf-8')

    results = []
    response = []
    step_counter = 0
    while len(response) < len(ground_truth_response):
        text = prompt + response
        pattern = text[-16:]

        start_time = time.perf_counter()
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
        end_time = time.perf_counter()
        spec_time = end_time - start_time

        # Verify scpeculated tokens
        accepted_tokens = []
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
        assert len(response) <= len(ground_truth_response)
        
        # Handle bonus token
        bonus_token = None
        bonus_text = ""
        if len(response) < len(ground_truth_response):
            # Add bonus token
            bonus_token = ground_truth_response[len(response)]
            new_tokens.append(bonus_token)
            response.append(bonus_token)
            
            if tokenizer:
                bonus_text = tokenizer.decode([bonus_token], skip_special_tokens=True)

        # Debug output for each step (after all tokens are processed)
        if debug_file:
            # Match tokens (from result.match_len)
            match_tokens = text[-result.match_len:] if result.match_len > 0 else []
            match_text = ""
            if tokenizer and match_tokens:
                match_text = tokenizer.decode(match_tokens, skip_special_tokens=True)
            
            # Spec tokens (all speculated tokens)
            spec_tokens = result.token_ids
            spec_text = ""
            if tokenizer and spec_tokens:
                spec_text = tokenizer.decode(spec_tokens, skip_special_tokens=True)
            
            # Accept tokens (accepted tokens)
            accept_text = ""
            if tokenizer and accepted_tokens:
                accept_text = tokenizer.decode(accepted_tokens, skip_special_tokens=True)
            
            # Create JSON object for this step with all information
            debug_data = {
                "request_id": request_id,
                "step": step_counter,
                "match_tokens": match_tokens,
                "match_text": match_text,
                "match_length": result.match_len,
                "spec_tokens": spec_tokens,
                "spec_text": spec_text,
                "num_spec_tokens": len(spec_tokens),
                "accept_tokens": accepted_tokens,
                "accept_text": accept_text,
                "num_accept_tokens": len(accepted_tokens),
                "bonus_token": bonus_token,
                "bonus_text": bonus_text,
                "num_output_tokens": len(new_tokens),
                "score": result.score
            }
            
            debug_file.write(json.dumps(debug_data, ensure_ascii=False) + '\n')
            debug_file.flush()

        # Update suffix cache
        # start_time = time.perf_counter()
        # suffix_cache.update_response(request_id, problem_id, new_tokens)
        # end_time = time.perf_counter()
        # update_time = end_time - start_time

        results.append({
            "step": len(results),
            "match_len": result.match_len,
            "score": result.score,
            "num_spec_toks": len(result.token_ids),
            "num_accept_toks": len(accepted_tokens),
            "num_out_toks": len(new_tokens),
            "spec_ms": spec_time * 1000,
        })
        
        step_counter += 1

    assert response == ground_truth_response

    # Close debug file
    if debug_file:
        debug_file.close()

    if use_cached_prompt:
        suffix_cache.evict_prompt(request_id)

    return results

def step_results_summary(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    # Compute speedup via per-example grouping. Some datasets do not contain request_id;
    # in that case, we treat each row's original_step (or current_step) as the example key.
    has_request = "request_id" in df.columns
    group_keys = ["current_step", "request_id"] if has_request else ["current_step", "original_step"]

    speedup = df.groupby(group_keys).agg(
        sum_out_toks=("num_out_toks", "sum"),
        num_steps=("step", "count"),
    )
    speedup["speedup"] = speedup["sum_out_toks"] / speedup["num_steps"]
    speedup = speedup.groupby(["current_step"]).agg(
        req_speedup=("speedup", "mean"),
    )

    # Average angle per step (if angle is provided in the data).
    # Collapse to one angle per request_id to avoid weighting by internal decode steps.
    angle_summary = None
    if "angle" in df.columns:
        angle_per_example = df.groupby(group_keys).agg(
            angle=("angle", "first"),
        )
        angle_summary = angle_per_example.groupby(["current_step"]).agg(
            avg_angle=("angle", "mean"),
        )

    # Average score per step (if score is provided).
    # First average within each request to avoid overweighting by longer responses.
    score_summary = None
    # prefer example_score if present; otherwise fallback to per-step score values
    score_col = "example_score" if "example_score" in df.columns else ("score" if "score" in df.columns else None)
    if score_col is not None:
        score_per_example = df.groupby(group_keys).agg(
            mean_score=(score_col, "mean"),
        )
        score_summary = score_per_example.groupby(["current_step"]).agg(
            avg_score=("mean_score", "mean"),
        )

    summary = df.groupby(["current_step"]).agg(
        sum_accept_toks=("num_accept_toks", "sum"),
        sum_spec_toks=("num_spec_toks", "sum"),
        sum_out_toks=("num_out_toks", "sum"),
        avg_accept_toks=("num_accept_toks", "mean"),
        avg_spec_toks=("num_spec_toks", "mean"),
        sum_spec_ms=("spec_ms", "sum"),
        num_eval=("num_eval", "first"),
        num_train=("num_train", "first"),
    ).reset_index()

    summary["accept_rate"] = (
        summary["sum_accept_toks"] / summary["sum_spec_toks"]
    )
    summary["req_speedup"] = speedup["req_speedup"].values
    summary["spec_ms_per_tok"] = (
        summary["sum_spec_ms"] / summary["sum_spec_toks"]
    )

    summary = summary.set_index("current_step")

    # Join avg_angle if available
    if angle_summary is not None and len(angle_summary) > 0:
        summary = summary.join(angle_summary)

    # Join avg_score if available
    if score_summary is not None and len(score_summary) > 0:
        summary = summary.join(score_summary)

    return summary


def load_jsonl_data(file_path: str) -> pd.DataFrame:
    data: List[Dict] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)


def group_data_by_step(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    return {step: group for step, group in df.groupby('step')}


def tokenize_text_data(df: pd.DataFrame, tokenizer_name: str) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    prompts: List[List[int]] = []
    responses: List[List[int]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing data"):
        prompts.append(tokenizer.encode(row["input"]))
        responses.append(tokenizer.encode(row["output"]))

    result_df = df.copy()
    result_df["prompt"] = prompts
    result_df["response"] = responses
    return result_df


def run_step_based_simulation(
    data_file: str,
    tokenizer_name: str,
    max_depth: int = 64,
    max_spec_tokens: int = 0,
    max_spec_factor: float = 1.0,
    min_token_prob: float = 0.1,
    use_tree_spec: bool = True,
    use_cached_prompt: bool = True,
    output_file: Optional[str] = None,
    max_steps: Optional[int] = None,
    start_step: Optional[int] = None,
    window_size: int = 10,
    add_low_angle: int = 0,
) -> pd.DataFrame:
    df = load_jsonl_data(data_file)
    df = tokenize_text_data(df, tokenizer_name)

    # Prepare per-step average angle from input data. Prefer row-level 'angle'; fallback to signals['angle'].
    if "angle" not in df.columns:
        if "signals" in df.columns:
            df["angle"] = df["signals"].apply(lambda s: s.get("angle") if isinstance(s, dict) else None)
        else:
            df["angle"] = None
    step_avg_angle = df.groupby("step")["angle"].mean()

    step_groups = group_data_by_step(df)
    all_steps = sorted(step_groups.keys())
    if start_step is not None:
        all_steps = [s for s in all_steps if s >= start_step]

    all_results: List[Dict] = []

    for step_idx, current_step in enumerate(tqdm(all_steps, desc="Processing steps")):
        if current_step == all_steps[0] or (max_steps is not None and current_step >= max_steps):
            continue
        eval_data = step_groups[current_step].reset_index(drop=True)

        train_data_frames = []
        current_avg_angle = step_avg_angle.get(current_step, None)
        if add_low_angle == -1:
            lower_bound_step = all_steps[step_idx -  3 * window_size // 2]
            candidate_steps = [s for s in all_steps if s < current_step and s >= lower_bound_step]
            
            # 收集所有候选数据点及其角度值
            all_candidates = []
            for s in candidate_steps:
                if s in step_groups:
                    step_data = step_groups[s]
                    for _, row in step_data.iterrows():
                        angle_val = row.get('angle', None)
                        if pd.notna(angle_val):
                            all_candidates.append((row, angle_val))
            
            # 按角度值降序排序，选择前 window_size * 8 个
            all_candidates.sort(key=lambda t: t[1], reverse=True)
            selected_count = min(len(all_candidates), window_size * 8)
            selected_data = [row for row, _ in all_candidates[:selected_count]]
            
            # 将选中的数据转换为DataFrame并添加到训练数据中
            if selected_data:
                selected_df = pd.DataFrame(selected_data)
                train_data_frames.append(selected_df)
        elif add_low_angle == 0:
            for prev_step in all_steps[step_idx-1::-1]:
                if prev_step < current_step:
                    if len(train_data_frames) >= window_size:
                        break
                    train_data_frames.append(step_groups[prev_step])
        elif add_low_angle == 1:
            for prev_step in all_steps[step_idx-1::-1]:
                if prev_step < current_step:
                    if len(train_data_frames) >= window_size:
                        break
                    prev_avg_angle = step_avg_angle.get(prev_step, None)
                    if pd.notna(prev_avg_angle) and pd.notna(current_avg_angle) and abs(prev_avg_angle - current_avg_angle) > 0.1:
                        train_data_frames.append(step_groups[prev_step])

        if train_data_frames:
            train_data = pd.concat(train_data_frames, ignore_index=True)
        else:
            train_data = pd.DataFrame(columns=eval_data.columns)

        suffix_cache = SuffixCache(max_depth)
        print(f"len(train_data)",len(train_data))

        if len(train_data) > 0:
            for request_id, example in tqdm(train_data.iterrows(),
                                            total=len(train_data),
                                            desc="Building cache"):
                problem_id = example.get("problem_id", "0")
                suffix_cache.update_response(-1 - request_id, problem_id, example["prompt"]) 
                suffix_cache.update_response(-1 - request_id, problem_id, example["response"]) 

        step_results: List[Dict] = []
        for request_id, example in tqdm(eval_data.iterrows(),
                                        total=len(eval_data),
                                        desc="Running evaluation"):
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
                debug_file_path=None, # 'src/ArcticInference/tests/step_simulation_results_2af0_debug.jsonl',
                tokenizer=None ,#AutoTokenizer.from_pretrained(tokenizer_name),
            )
            for result in results:
                # Extract angle from example; support nested signals.angle as well
                angle_value = example.get("angle")
                if angle_value is None:
                    signals = example.get("signals")
                    if isinstance(signals, dict):
                        angle_value = signals.get("angle")
                # Extract example-level score if present in the dataset
                example_score_value = example.get("score")
                result.update({
                    "current_step": current_step,
                    "request_id": request_id,
                    "num_eval": len(eval_data),
                    "num_train": len(train_data),
                    "max_depth": max_depth,
                    "max_spec_tokens": max_spec_tokens,
                    "max_spec_factor": max_spec_factor,
                    "min_token_prob": min_token_prob,
                    "use_tree_spec": use_tree_spec,
                    "use_cached_prompt": use_cached_prompt,
                    "original_step": example.get("step", current_step),
                    "problem_id": problem_id,
                    "angle": angle_value,
                    "example_score": example_score_value,
                })
            step_results.extend(results)

        all_results.extend(step_results)
       # Print intermediate results
        if step_results:
            step_df = pd.DataFrame(step_results)
            step_summary = step_results_summary(step_df)
            print(f"Step {current_step} summary:")
            print(step_summary.to_string())
            print()
    results_df = pd.DataFrame(all_results)
    if output_file:
        #results_df.to_csv(output_file, index=False)
        # Also write summary next to the results file
        try:
            summary_df = step_results_summary(results_df)
            if output_file.lower().endswith('.csv'):
                summary_file = output_file[:-4] + ".csv"
            else:
                summary_file = output_file + ".csv"
            # Ensure index is a column for CSV output
            summary_df.reset_index().to_csv(summary_file, index=False)
            print(f"Summary written to {summary_file}")
        except Exception as e:
            print(f"Error writing summary: {e}")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Step-based suffix cache simulation")
    parser.add_argument(
        "data_file",
        type=str,
        help="Path to the JSONL data file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Name of the HuggingFace tokenizer",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=64,
        help="Max depth of the suffix tree",
    )
    parser.add_argument(
        "--max-spec-tokens",
        type=int,
        default=0,
        help="Max speculation tokens (if 0, defaults to max_depth)",
    )
    parser.add_argument(
        "--max-spec-factor",
        type=float,
        default=1.0,
        help="Max speculation tokens as a multiplier of the prefix length",
    )
    parser.add_argument(
        "--min-token-prob",
        type=float,
        default=0.1,
        help="Minimum probability of the token to be considered",
    )
    parser.add_argument(
        "--use-tree-spec",
        action="store_true",
        default=True,
        help="Whether to use tree-based speculation",
    )
    parser.add_argument(
        "--use-cached-prompt",
        action="store_true",
        default=True,
        help="Whether to use the cached prompt for the request",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum number of steps to process",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        help="Starting step number (skip earlier steps)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Window size for filtering prior steps"
    )
    parser.add_argument(
        "--add-low-angle",
        type=int,
        default=0,
        help="Add low angle data"
    )

    args = parser.parse_args()
    if args.add_low_angle == 1:
        RESULTS_DIR = f"/app/src/ArcticInference/tests/extracted_problem_0_5b/add_low_angle"
    elif args.add_low_angle == -1:
        RESULTS_DIR = f"/app/src/ArcticInference/tests/extracted_problem_0_5b/add_high_angle"
    else:
        RESULTS_DIR = f"/app/src/ArcticInference/tests/extracted_problem_0_5b/window_only/window_size_{args.window_size}"

    # Ensure the results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)

    results_df = run_step_based_simulation(
        data_file=args.data_file,
        tokenizer_name=args.tokenizer,
        max_depth=args.max_depth,
        max_spec_tokens=args.max_spec_tokens,
        max_spec_factor=args.max_spec_factor,
        min_token_prob=args.min_token_prob,
        use_tree_spec=args.use_tree_spec,
        use_cached_prompt=args.use_cached_prompt,
        output_file=os.path.join(RESULTS_DIR, f"{os.path.basename(args.data_file).split('.')[0]}.csv"),
        max_steps=args.max_steps,
        start_step=args.start_step,
        window_size=args.window_size,
        add_low_angle=args.add_low_angle,
    )


if __name__ == "__main__":
    main()


