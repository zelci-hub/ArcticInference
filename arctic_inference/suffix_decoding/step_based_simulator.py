# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from arctic_inference.suffix_decoding.cache import SuffixDecodingCache

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
        accept_visualize: List of source/reject values (1=local/green, 0=reject/red, 2=global/blue)
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
        1: "#28a745",  # Green for local source
        0: "#dc3545",  # Red for reject/bonus  
        2: "#007bff"   # Blue for global source
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
        .local {{ background-color: #28a745; }}
        .reject {{ background-color: #dc3545; }}
        .global {{ background-color: #007bff; }}
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
        <span class="legend-item local">Local Source (1)</span>
        <span class="legend-item reject">Reject/Bonus (0)</span>
        <span class="legend-item global">Global Source (2)</span>
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
            status_class = {1: "local", 0: "reject", 2: "global"}.get(accept_status, "unknown")
            
            html_content += f'<span class="token {status_class}" style="background-color: {color}" title="Token {i}: {token_id}, Status: {accept_status}">{token_text}</span>'
        except Exception as e:
            # Fallback for tokens that can't be decoded
            color = color_map.get(accept_status, "#6c757d")
            status_class = {1: "local", 0: "reject", 2: "global"}.get(accept_status, "unknown")
            html_content += f'<span class="token {status_class}" style="background-color: {color}" title="Token {i}: {token_id}, Status: {accept_status}">#{token_id}</span>'
    
    # Close HTML
    html_content += """
        </div>
    </div>
    
    <div style="margin-top: 30px; padding: 15px; background-color: #e9ecef; border-radius: 5px;">
        <h4>Statistics:</h4>
        <p><strong>Local source tokens:</strong> {local_count} ({local_percent:.1f}%)</p>
        <p><strong>Global source tokens:</strong> {global_count} ({global_percent:.1f}%)</p>
        <p><strong>Reject/Bonus tokens:</strong> {reject_count} ({reject_percent:.1f}%)</p>
    </div>
</body>
</html>""".format(
        local_count=accept_visualize.count(1),
        local_percent=accept_visualize.count(1) / len(accept_visualize) * 100 if accept_visualize else 0,
        global_count=accept_visualize.count(2),
        global_percent=accept_visualize.count(2) / len(accept_visualize) * 100 if accept_visualize else 0,
        reject_count=accept_visualize.count(0),
        reject_percent=accept_visualize.count(0) / len(accept_visualize) * 100 if accept_visualize else 0,
    )
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Token visualization saved to: {output_path}")

def suffix_decode(
    suffix_cache: SuffixDecodingCache,
    request_id: int,
    prompt: List[int],
    ground_truth_response: List[int],
    max_spec_tokens: int,
    max_spec_factor: float,
    min_token_prob: float,
    use_tree_spec: bool,
    use_cached_prompt: bool,
    debug_file_path: Optional[str] = None,
    tokenizer = None,
) -> Tuple[List[Dict], List[int]]:
    if not max_spec_tokens:
        max_spec_tokens = suffix_cache.max_tree_depth

    suffix_cache.start_request(request_id, prompt if use_cached_prompt else [])

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

        start_time = time.perf_counter()
        result, source = suffix_cache.speculate(
            request_id,
            text,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            min_token_prob=min_token_prob,
            use_tree_spec=use_tree_spec,
        )
        end_time = time.perf_counter()
        spec_time = end_time - start_time

        # Verify speculated tokens
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
        
        # Mark accepted tokens based on source: 1 for local, 2 for global
        if source == "local":
            accept_visualize.extend([1] * len(accepted_tokens))
        else:  # source == "global"
            accept_visualize.extend([2] * len(accepted_tokens))
        
        assert len(response) <= len(ground_truth_response)
        
        # Handle bonus token
        bonus_token = None
        if len(response) < len(ground_truth_response):
            # Add bonus token
            bonus_token = ground_truth_response[len(response)]
            new_tokens.append(bonus_token)
            response.append(bonus_token)
            accept_visualize.append(0)

        # Update suffix cache
        start_time = time.perf_counter()
        suffix_cache.add_active_response(request_id, new_tokens)
        end_time = time.perf_counter()
        update_time = end_time - start_time
        
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

        results.append({
            "step": len(results),
            "match_len": result.match_len,
            "score": result.score,
            "num_spec_toks": len(result.token_ids),
            "num_accept_toks": len(accepted_tokens),
            "num_out_toks": len(new_tokens),
            "spec_ms": spec_time * 1000,
            "update_ms": update_time * 1000,
        })
        
        step_counter += 1

    assert response == ground_truth_response

    # Close debug file
    if debug_file:
        debug_file.close()

    suffix_cache.stop_request(request_id)

    return results, accept_visualize

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def step_results_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary of results for step-based simulation."""
    if len(df) == 0:
        return pd.DataFrame()
    
    # Compute per-request speedup.
    speedup = df.groupby(["current_step", "request_id"]).agg(
        sum_out_toks=("num_out_toks", "sum"),
        num_steps=("step", "count"),
    )
    speedup["speedup"] = speedup["sum_out_toks"] / speedup["num_steps"]
    speedup = speedup.groupby(["current_step"]).agg(
        req_speedup=("speedup", "mean"),
    )
    
    # Compute summary statistics.
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
        summary["sum_accept_toks"] / summary["sum_spec_toks"])
    summary["req_speedup"] = speedup["req_speedup"].values
    summary["spec_ms_per_tok"] = (
        summary["sum_spec_ms"] / summary["sum_spec_toks"])
    
    return summary.set_index("current_step")


def load_jsonl_data(file_path: str) -> pd.DataFrame:
    """Load JSONL data and return as DataFrame."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)


def group_data_by_step(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Group data by step field."""
    return {step: group for step, group in df.groupby('step')}


def tokenize_text_data(df: pd.DataFrame, tokenizer_name: str) -> pd.DataFrame:
    """Tokenize input and output text data."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    prompts = []
    responses = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing data"):
        # Tokenize input (prompt)
        prompt_tokens = tokenizer.encode(row["input"], add_special_tokens=False)
        prompts.append(prompt_tokens)
        
        # Tokenize output (response)
        response_tokens = tokenizer.encode(row["output"], add_special_tokens=False)
        responses.append(response_tokens)
    
    # Create new DataFrame with tokenized data
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
    max_cached_requests: int = -1,
    output_file: Optional[str] = None,
    enable_debug: bool = False,
    debug_file_path: Optional[str] = None,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
    window_size: int = 10,
    enable_visualization: bool = False,
    prompt_column: str = "input_token_ids",
    response_column: str = "output_token_ids",
) -> pd.DataFrame:
    """
    Run step-based simulation where:
    - eval_data: all data from the current step
    - train_data: all data from previous steps
    """
    print(f"Loading data from {data_file}...")
    df = load_jsonl_data(data_file)
    eval_df = load_jsonl_data(eval_data_file)
    print(f"Tokenizing data using {tokenizer_name}...")
    if prompt_column != "input_token_ids":
        df = tokenize_text_data(df, tokenizer_name)
        eval_df = tokenize_text_data(eval_df, tokenizer_name)
    else:
        df["prompt"] = df[prompt_column]
        df["response"] = df[response_column]
        eval_df["prompt"] = eval_df[prompt_column]
        eval_df["response"] = eval_df[response_column]
    print("Grouping data by step...")
    step_groups = group_data_by_step(df)
    eval_groups = group_data_by_step(eval_df)
    all_steps = sorted(eval_groups.keys())
    print(f"Found {len(all_steps)} steps to process: {all_steps[:10]}{'...' if len(all_steps) > 10 else ''}")
    
    all_results = []
    if window_size == -1:
        window_size = len(all_steps)
    
    for step_idx, current_step in enumerate(tqdm(all_steps, desc="Processing steps")):
        if start_step is not None and current_step < start_step:
            continue   
        if end_step is not None and current_step > end_step:
            break
        print(f"\nProcessing step {current_step} ({step_idx + 1}/{len(all_steps)})")
        
        # Get eval data (current step)
        eval_data = eval_groups[current_step].reset_index(drop=True)
        
        # Get train data (all previous steps)
        train_data_frames = []
        for prev_step in all_steps[max(0, step_idx-window_size):]:
            if prev_step < current_step:
                train_data_frames.append(step_groups[prev_step])
        
        if train_data_frames:
            train_data = pd.concat(train_data_frames, ignore_index=True)
        else:
            # For the first step, use empty training data
            train_data = pd.DataFrame(columns=eval_data.columns)
        
        print(f"  Eval data: {len(eval_data)} examples")
        print(f"  Train data: {len(train_data)} examples")
        
        # Initialize suffix cache
        suffix_cache = SuffixDecodingCache(
            max_tree_depth=max_depth,
            max_cached_requests=max_cached_requests
        )
        
        # Build cache with training data
        train_request_ids = []
        if len(train_data) > 0:
            for request_id, example in tqdm(train_data.iterrows(), 
                                          total=len(train_data),
                                          desc="Building cache"):
                # Use negative request_id for training examples
                train_request_id = -1 - request_id
                suffix_cache.start_request(train_request_id, example["prompt"])
                suffix_cache.add_active_response(train_request_id, example["response"])
                suffix_cache.stop_request(train_request_id)
                train_request_ids.append(train_request_id)
        
        # Check cache integrity
        print("Checking cache integrity...", end=" ", flush=True)
        if ret := suffix_cache._global_tree.check_integrity():
            raise RuntimeError(f"Cache integrity check failed: {ret}")
        else:
            print("OK")
        
        print(f"Memory estimate: {suffix_cache._global_tree.estimate_memory()}")
        
        # Run evaluation on current step data
        step_results = []
        tokenizer_for_viz = AutoTokenizer.from_pretrained(tokenizer_name) if (enable_debug or enable_visualization) else None
        for request_id, example in tqdm(eval_data.iterrows(),
                                      total=len(eval_data),
                                      desc="Running evaluation"):
            try:
                results, accept_visualize = suffix_decode(
                    suffix_cache,
                    request_id,
                    example["prompt"],
                    example["response"],
                    max_spec_tokens if max_spec_tokens > 0 else max_depth,
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
                        "max_cached_requests": max_cached_requests,
                        "original_step": example["step"],
                        "problem_id": '0',
                        "score": example["score"],
                    })
                step_results.extend(results)
                
            except Exception as e:
                print(f"Error processing request {request_id}: {e}")
                continue
        
        all_results.extend(step_results)
        
        if step_results:
            step_df = pd.DataFrame(step_results)
            step_summary = step_results_summary(step_df)
            print(f"Step {current_step} summary:")
            print(step_summary.to_string())
            print()
    
    # Create final results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # if output_file:
    #     results_df.to_csv(output_file, index=False)
    #     print(f"Results saved to {output_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Step-based suffix decoding simulation")
    parser.add_argument(
        "data_file",
        type=str,
        help="Path to the JSONL data file"
    )
    parser.add_argument(
        "--eval_data_file",
        type=str,
        default=None,
        help="Path to the JSONL data file"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Name of the HuggingFace tokenizer"
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="input",
        help="Column name for the prompts in the dataset",
    )
    parser.add_argument(
        "--response-column",
        type=str,
        default="output",
        help="Column name for the responses in the dataset",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=16,
        help="Max depth of the suffix tree"
    )
    parser.add_argument(
        "--max-spec-tokens",
        type=int,
        default=0,
        help="Max speculation tokens (if 0, defaults to max_depth)"
    )
    parser.add_argument(
        "--max-spec-factor",
        type=float,
        default=1.0,
        help="Max speculation tokens as a multiplier of the prefix length"
    )
    parser.add_argument(
        "--min-token-prob",
        type=float,
        default=0.1,
        help="Minimum probability of the token to be considered"
    )
    parser.add_argument(
        "--use-tree-spec",
        action="store_true",
        default=True,
        help="Whether to use tree-based speculation"
    )
    parser.add_argument(
        "--use-cached-prompt",
        action="store_true",
        default=True,
        help="Whether to use the cached prompt for the request"
    )
    parser.add_argument(
        "--max-cached-requests",
        type=int,
        default=-1,
        help="Max number of cached requests (if -1, unlimited)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--end-step",
        type=int,
        help="Maximum number of steps to process"
    )
    parser.add_argument(
        "--start-step",
        type=int,
        help="Starting step number (skip earlier steps)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="Window size for filtering prior steps"
    )
    parser.add_argument(
        "--enable-debug",
        action="store_true",
        default=False,
        help="Whether to enable debug mode"
    )
    parser.add_argument(
        "--debug-file",
        type=str,
        help="Path to debug output file (JSONL format)"
    )
    parser.add_argument(
        "--enable-visualization",
        action="store_true",
        default=False,
        help="Whether to enable token visualization HTML output"
    )
    args = parser.parse_args()
    if args.eval_data_file is None:
        args.eval_data_file = args.data_file
    
    results_df = run_step_based_simulation(
        data_file=args.data_file,
        eval_data_file=args.eval_data_file,
        tokenizer_name=args.tokenizer,
        max_depth=args.max_depth,
        max_spec_tokens=args.max_spec_tokens,
        max_spec_factor=args.max_spec_factor,
        min_token_prob=args.min_token_prob,
        use_tree_spec=args.use_tree_spec,
        use_cached_prompt=args.use_cached_prompt,
        max_cached_requests=args.max_cached_requests,
        output_file=args.output,
        enable_debug=args.enable_debug,
        debug_file_path=args.debug_file,
        end_step=args.end_step,
        start_step=args.start_step,
        window_size=args.window_size,
        enable_visualization=args.enable_visualization,
        prompt_column=args.prompt_column,
        response_column=args.response_column,
    )
    
    # Print final summary
    if len(results_df) > 0:
        print("\nFinal Summary:")
        summary = step_results_summary(results_df)
        if args.output:
            summary.to_csv(args.output)
            print(f"Summary saved to {args.output}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
