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

def create_token_visualization_html(
    tokens: List[int], 
    accept_visualize: List[int], 
    tokenizer, 
    output_path: str,
    request_id: int = 0,
    step: int = 0
) -> None:
    """
    Create an HTML file with colored token visualization.
    
    Args:
        tokens: List of token IDs from example["response"]
        accept_visualize: List of accept/reject/bonus values (1=accept/green, 0=reject/red, 2=bonus/black)
        tokenizer: Tokenizer to decode tokens to text
        output_path: Path to save the HTML file
        request_id: Request ID for labeling
        step: Step number for labeling
    """
    if len(tokens) != len(accept_visualize):
        print(f"Warning: tokens length ({len(tokens)}) != accept_visualize length ({len(accept_visualize)})")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Color mapping
    color_map = {
        1: "#28a745",  # Green for accept
        0: "#dc3545",  # Red for reject  
        2: "#000000"   # Black for bonus
    }
    
    # Start HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Token Visualization - Request {request_id} Step {step}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }}
        .header {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .token {{
            display: inline-block;
            padding: 2px 4px;
            margin: 1px;
            border-radius: 3px;
            font-weight: bold;
            color: white;
            font-size: 14px;
        }}
        .legend {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            padding: 5px 10px;
            border-radius: 3px;
            color: white;
            font-weight: bold;
        }}
        .accept {{ background-color: #28a745; }}
        .reject {{ background-color: #dc3545; }}
        .bonus {{ background-color: #000000; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Token Visualization</h1>
        <p><strong>Request ID:</strong> {request_id}</p>
        <p><strong>Step:</strong> {step}</p>
        <p><strong>Total Tokens:</strong> {len(tokens)}</p>
    </div>
    
    <div class="legend">
        <h3>Legend:</h3>
        <span class="legend-item accept">Accept (1)</span>
        <span class="legend-item reject">Reject (0)</span>
        <span class="legend-item bonus">Bonus (2)</span>
    </div>
    
    <div class="content">
        <h3>Token Sequence:</h3>
        <div class="tokens">
"""
    
    # Add each token with appropriate color
    for i, (token_id, accept_status) in enumerate(zip(tokens, accept_visualize)):
        try:
            # Decode individual token
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            # Escape HTML special characters
            token_text = token_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            # Replace newlines and tabs with visible characters
            token_text = token_text.replace('\n', '\\n').replace('\t', '\\t').replace(' ', '·')
            
            color = color_map.get(accept_status, "#6c757d")  # Default gray for unknown values
            status_class = {1: "accept", 0: "reject", 2: "bonus"}.get(accept_status, "unknown")
            
            html_content += f'<span class="token {status_class}" style="background-color: {color}" title="Token {i}: {token_id}, Status: {accept_status}">{token_text}</span>'
        except Exception as e:
            # Fallback for tokens that can't be decoded
            color = color_map.get(accept_status, "#6c757d")
            status_class = {1: "accept", 0: "reject", 2: "bonus"}.get(accept_status, "unknown")
            html_content += f'<span class="token {status_class}" style="background-color: {color}" title="Token {i}: {token_id}, Status: {accept_status}">#{token_id}</span>'
    
    # Close HTML
    html_content += """
        </div>
    </div>
    
    <div style="margin-top: 30px; padding: 15px; background-color: #e9ecef; border-radius: 5px;">
        <h4>Statistics:</h4>
        <p><strong>Accept tokens:</strong> {accept_count} ({accept_percent:.1f}%)</p>
        <p><strong>Reject tokens:</strong> {reject_count} ({reject_percent:.1f}%)</p>
        <p><strong>Bonus tokens:</strong> {bonus_count} ({bonus_percent:.1f}%)</p>
    </div>
</body>
</html>""".format(
        accept_count=accept_visualize.count(1),
        accept_percent=accept_visualize.count(1) / len(accept_visualize) * 100 if accept_visualize else 0,
        reject_count=accept_visualize.count(0),
        reject_percent=accept_visualize.count(0) / len(accept_visualize) * 100 if accept_visualize else 0,
        bonus_count=accept_visualize.count(2),
        bonus_percent=accept_visualize.count(2) / len(accept_visualize) * 100 if accept_visualize else 0,
    )
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Token visualization saved to: {output_path}")

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
        # Ensure the directory exists
        debug_dir = os.path.dirname(debug_file_path)
        if debug_dir and not os.path.exists(debug_dir):
            os.makedirs(debug_dir, exist_ok=True)
        debug_file = open(debug_file_path, 'a', encoding='utf-8')

    results = []
    response = []
    step_counter = 0
    accept_visualize = []
    while len(response) < len(ground_truth_response):
        text = prompt + response
        pattern = text[-64:]

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
        accept_visualize.extend([1] * len(accepted_tokens))
        assert len(response) <= len(ground_truth_response)
        
        # Handle bonus token
        bonus_token = None
        if len(response) < len(ground_truth_response):
            # Add bonus token
            bonus_token = ground_truth_response[len(response)]
            new_tokens.append(bonus_token)
            response.append(bonus_token)
            if len(accepted_tokens) > 0:
                accept_visualize.append(2)
            else:
                accept_visualize.append(0)
        
        # Debug output for each step (after all tokens are processed)
        if debug_file:
            # Match tokens (from result.match_len)
            accept_text_match_text = ""
            if tokenizer and accepted_tokens:
                match_tokens = text[-result.match_len:] if result.match_len > 0 else []
                accept_text_match_text = tokenizer.decode(match_tokens + accepted_tokens, skip_special_tokens=True)
            
            # Create JSON object for this step with all information
            debug_data = {
                "request_id": request_id,
                "step": step_counter,            
                "match_accept_text": accept_text_match_text,
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

    return results, accept_visualize

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
        if "input_token_ids" in row:
            prompts.append(row["input_token_ids"])
        else:
            prompts.append(tokenizer.encode(row["input"]))
        if "output_token_ids" in row:
            responses.append(row["output_token_ids"])
        else:
            responses.append(tokenizer.encode(row["output"]))

    result_df = df.copy()
    result_df["prompt"] = prompts
    result_df["response"] = responses
    return result_df


def run_step_based_simulation(
    data_file: str,
    eval_data_file: str,
    tokenizer_name: str,
    max_depth: int = 64,
    max_spec_tokens: int = 0,
    max_spec_factor: float = 1.0,
    min_token_prob: float = 0.1,
    use_tree_spec: bool = True,
    use_cached_prompt: bool = True,
    output_file: Optional[str] = None,
    enable_debug: bool = False,
    debug_file_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
    window_size: int = 10,
    enable_visualization: bool = False,
    check_model_change: bool = False,
) -> pd.DataFrame:
    df = load_jsonl_data(data_file)
    df = tokenize_text_data(df, tokenizer_name)
    eval_df = load_jsonl_data(eval_data_file)
    step_groups = group_data_by_step(df)
    eval_groups = group_data_by_step(eval_df)
    all_steps = sorted(step_groups.keys())
    all_results: List[Dict] = []
    start_step = start_step or all_steps[0]
    end_step = end_step or all_steps[-1]

    for step_idx, current_step in enumerate(tqdm(all_steps, desc="Processing steps")):
        if current_step < start_step  or (max_steps is not None and current_step >= max_steps):
            continue
        if current_step > end_step:
            break
        eval_data = eval_groups[current_step].reset_index(drop=True)

        train_data_frames = []
        if check_model_change:
            for prev_step in all_steps[0:step_idx-1]:
                if prev_step < current_step:
                    if len(train_data_frames) >= window_size:
                        break
                    train_data_frames.append(step_groups[prev_step])
        else:
            for prev_step in all_steps[step_idx-1::-1]:
                if prev_step < current_step:
                    if len(train_data_frames) >= window_size:
                        break
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
            tokenizer_for_viz = AutoTokenizer.from_pretrained(tokenizer_name)
            results, accept_visualize = suffix_decode(
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
                debug_file_path=debug_file_path if enable_debug else None,
                tokenizer=tokenizer_for_viz if enable_debug else None,
            )
            
            # Create token visualization HTML file
            if enable_visualization:
                # Create visualization directory based on debug_file_path or output_file
                if debug_file_path:
                    viz_dir = debug_file_path.replace('_debug.jsonl', '_viz')
                elif output_file:
                    viz_dir = os.path.dirname(output_file).replace('suffix_results', 'suffix_viz')
                else:
                    viz_dir = os.path.join(os.path.dirname(data_file), 'visualization')
                
                os.makedirs(viz_dir, exist_ok=True)
                viz_file_path = os.path.join(viz_dir, f"request_{request_id}_step_{current_step}_viz.html")
                create_token_visualization_html(
                    tokens=example["response"],
                    accept_visualize=accept_visualize,
                    tokenizer=tokenizer_for_viz,
                    output_path=viz_file_path,
                    request_id=request_id,
                    step=current_step
                )

            
            for result in results:
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
        "--eval_file",
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
        default=16,
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
        default=False,
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
        "--end-step",
        type=int,
        help="End step number (stop at this step)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Window size for filtering prior steps"
    )
    parser.add_argument(
        "--enable-debug",
        action="store_true",
        default=False,
        help="Whether to enable debug mode",
    )
    parser.add_argument(
        "--enable-visualization",
        action="store_true",
        default=False,
        help="Whether to enable token visualization HTML output",
    )
    parser.add_argument(
        "--check-model-change",
        action="store_true",
        default=False,
        help="Whether to check model change",
    )
    args = parser.parse_args()
    RESULTS_DIR = f"/app/src/aaa_sample_data/7b_instruct_hard/suffix_results/window_size_{args.window_size}/check_model_change_{args.check_model_change}"
    DEBUG_FILE_PATH = f"/app/src/aaa_sample_data/7b_instruct_hard/suffix_debug/window_size_{args.window_size}/check_model_change_{args.check_model_change}"
    # Ensure the results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(DEBUG_FILE_PATH, exist_ok=True)

    results_df = run_step_based_simulation(
        data_file=args.data_file,
        eval_data_file=args.eval_file
        tokenizer_name=args.tokenizer,
        max_depth=args.max_depth,
        max_spec_tokens=args.max_spec_tokens,
        max_spec_factor=args.max_spec_factor,
        min_token_prob=args.min_token_prob,
        use_tree_spec=args.use_tree_spec,
        use_cached_prompt=args.use_cached_prompt,
        output_file=os.path.join(RESULTS_DIR, f"{os.path.basename(args.data_file).split('.')[0]}.csv"),
        enable_debug=args.enable_debug,
        debug_file_path=os.path.join(DEBUG_FILE_PATH, f"{os.path.basename(args.data_file).split('.')[0]}_debug.jsonl"),
        max_steps=args.max_steps,
        start_step=args.start_step,
        end_step=args.end_step,
        window_size=args.window_size,
        enable_visualization=args.enable_visualization,
        check_model_change=args.check_model_change,
    )


if __name__ == "__main__":
    main()


