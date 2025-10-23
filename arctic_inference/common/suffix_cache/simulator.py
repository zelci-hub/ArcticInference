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


#python src/ArcticInference/arctic_inference/common/suffix_cache/simulator.py /app/src/ArcticInference/tests/cleaned_0019_4.jsonl --format jsonl --train-dataset /app/src/ArcticInference/tests/cleaned_all.jsonl --tokenizer deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output src/data/suffix_simulator

import argparse
import itertools
import json
import multiprocessing as mp
import os
from pickle import FALSE
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# Import suffix cache
from arctic_inference.common.suffix_cache import SuffixCache

try:
    from .vis_tree import SuffixTreeVisualizer
except ImportError:
    try:
        # Fallback to absolute import if relative import fails
        from vis_tree import SuffixTreeVisualizer
    except ImportError:
        print("Warning: Could not import SuffixTreeVisualizer. Visualization will be disabled.")
        SuffixTreeVisualizer = None



os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        start_time = time.perf_counter()
        suffix_cache.update_response(request_id, problem_id, new_tokens)
        end_time = time.perf_counter()
        update_time = end_time - start_time

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

    if use_cached_prompt:
        suffix_cache.evict_prompt(request_id)

    return results


def sample_data(
    dataset: pd.DataFrame,
    train_dataset: Optional[pd.DataFrame],
    num_eval: Optional[int],
    num_train: Optional[int],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if train_dataset is None:
        if num_eval is None:
            num_eval = len(dataset) - num_train
        if num_train is None:
            num_train = len(dataset) - num_eval
        assert num_train + num_eval <= len(dataset)
        shuffled = dataset.sample(frac=1, random_state=seed)
        train_subset = shuffled.head(num_train)
        eval_subset = shuffled.tail(num_eval)
    else:
        if num_eval is None:
            num_eval = len(dataset)
            eval_subset = dataset
        else:
            shuffled = dataset.sample(frac=1, random_state=seed)
            eval_subset = shuffled.head(num_eval)
        if num_train is None:
            num_train = len(train_dataset)
            train_subset = train_dataset
        else:
            shuffled = train_dataset.sample(frac=1, random_state=seed)
            train_subset = shuffled.head(num_train)
    return eval_subset, train_subset


def process_task(
    dataset: pd.DataFrame,
    train_dataset: Optional[pd.DataFrame],
    task_id: int,
    num_eval: int,
    num_train: int,
    seed: int,
    max_depth: int,
    max_spec_tokens: int,
    max_spec_factor: float,
    min_token_prob: float,
    use_tree_spec: bool,
    use_cached_prompt: bool,
    debug_file_path: Optional[str] = None,
    tokenizer = None,
    enable_visualization: bool = False,
    viz_output_dir: Optional[str] = None,
) -> List[Dict]:
    eval_subset, train_subset = sample_data(
        dataset,
        train_dataset,
        num_eval,
        num_train,
        seed,
    )
    suffix_cache = SuffixCache(max_depth)
    for request_id, example in tqdm(train_subset.iterrows(),
                                    total=len(train_subset),
                                    desc=f"Building cache"):
        # Use negative request_id to indicate training examples and avoid
        # conflicts with eval request_ids numbered 0, .., num_eval - 1.
        # Use the real problem_id from data
        problem_id = "0"
        suffix_cache.update_response(-1 - request_id, problem_id, example["prompt"])
        suffix_cache.update_response(-1 - request_id, problem_id, example["response"])

    # # Visualize the suffix tree after building
    # if enable_visualization and SuffixTreeVisualizer is not None:
    #     print("Generating suffix tree visualizations...")
    #     try:
    #         # Create visualizer
    #         viz = SuffixTreeVisualizer(suffix_cache=suffix_cache)
            
    #         # Set tokenizer for better visualization
    #         if tokenizer is not None:
    #             viz.set_tokenizer(tokenizer)
            
    #         # Create output directory for visualizations
    #         if viz_output_dir is None:
    #             viz_output_dir = "suffix_tree_visualizations"
    #         os.makedirs(viz_output_dir, exist_ok=True)
            
    #         # Generate different visualizations
    #         print(f"Saving visualizations to {viz_output_dir}/")
            
    #         # 1. Print tree structure to text file
    #         tree_structure_file = os.path.join(viz_output_dir, f"tree_structure_task_{task_id}.txt")
    #         with open(tree_structure_file, 'w', encoding='utf-8') as f:
    #             import sys
    #             original_stdout = sys.stdout
    #             sys.stdout = f
    #             try:
    #                 viz.print_tree_structure()
    #             finally:
    #                 sys.stdout = original_stdout
    #         print(f"  - Tree structure saved to: {tree_structure_file}")
            
    #         # 2. Generate graph visualization
    #         graph_file = os.path.join(viz_output_dir, f"suffix_tree_graph_task_{task_id}.png")
    #         viz.visualize_tree_graph(max_nodes=100, output_file=graph_file)
    #         print(f"  - Tree graph saved to: {graph_file}")
            
    #         # 3. Print debugging statistics
    #         stats_file = os.path.join(viz_output_dir, f"tree_statistics_task_{task_id}.txt")
    #         with open(stats_file, 'w', encoding='utf-8') as f:
    #             original_stdout = sys.stdout
    #             sys.stdout = f
    #             try:
    #                 viz.debug_statistics()
    #             finally:
    #                 sys.stdout = original_stdout
    #         print(f"  - Tree statistics saved to: {stats_file}")
            
    #         print("Suffix tree visualization completed successfully!")
            
    #     except Exception as e:
    #         print(f"Warning: Failed to generate visualizations: {e}")
    #         import traceback
    #         traceback.print_exc()

    records = []
    for request_id, example in tqdm(eval_subset.iterrows(),
                                    total=len(eval_subset),
                                    desc=f"Running requests"):
        # Use the real problem_id from data
        #problem_id = example["problem_id"]
        problem_id = "0"
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
            debug_file_path=debug_file_path,
            tokenizer=tokenizer,
        )
        for result in results:
            result.update({
                "task_id": task_id,
                "request_id": request_id,
                "num_eval": len(eval_subset),
                "num_train": len(train_subset),
                "seed": seed,
                "max_depth": max_depth,
                "max_spec_tokens": max_spec_tokens,
                "max_spec_factor": max_spec_factor,
                "min_token_prob": min_token_prob,
                "use_tree_spec": use_tree_spec,
                "use_cached_prompt": use_cached_prompt,
            })
        records.extend(results)

    return records


def results_summary(df: pd.DataFrame, config_cols: List[str]) -> pd.DataFrame:
    # Compute per-request speedup.
    speedup = df.groupby(["task_id", "request_id"]).agg(
        sum_out_toks=("num_out_toks", "sum"),
        num_steps=("step", "count"),
    )
    speedup["speedup"] = speedup["sum_out_toks"] / speedup["num_steps"]
    speedup = speedup.groupby(["task_id"]).agg(
        req_speedup=("speedup", "mean"),
    )
    # Compute summary statistics.
    config_cols = ["task_id"] + list(config_cols)
    summary = df.groupby(config_cols).agg(
        sum_accept_toks=("num_accept_toks", "sum"),
        sum_spec_toks=("num_spec_toks", "sum"),
        sum_out_toks=("num_out_toks", "sum"),
        avg_accept_toks=("num_accept_toks", "mean"),
        avg_spec_toks=("num_spec_toks", "mean"),
        sum_spec_ms=("spec_ms", "sum"),
        sum_update_ms=("update_ms", "sum"),
    ).reset_index()
    summary["accept_rate"] = (
        summary["sum_accept_toks"] / summary["sum_spec_toks"])
    summary["req_speedup"] = speedup["req_speedup"]
    summary["spec_ms_per_tok"] = (
        summary["sum_spec_ms"] / summary["sum_spec_toks"])
    summary["update_ms_per_tok"] = (
        summary["sum_update_ms"] / summary["sum_out_toks"])
    # Calculate columns to drop from the summary
    drop_cols = [col for col in config_cols if summary[col].nunique() == 1]
    drop_cols.extend([
        "sum_accept_toks",
        "sum_spec_toks",
        "sum_out_toks",
        "sum_spec_ms",
        "sum_update_ms"])
    return summary.drop(columns=drop_cols)


def read_data_file(
    path: str,
    prompt_column: str,
    response_column: str,
    format: Optional[str] = None,
) -> pd.DataFrame:
    # Read the dataset file into a pandas DataFrame.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if format is None:
        _, ext = os.path.splitext(path)
        format = ext[1:]
    if format == "csv":
        df = pd.read_csv(path)
    elif format == "json":
        df = pd.read_json(path)
    elif format == "jsonl":
        # Prefer pandas for performance, but fall back to a precise parser with
        # clearer errors when files are empty or contain malformed lines.
        try:
            df = pd.read_json(path, lines=True)
        except ValueError as e:
            # Handle empty files or lines with stray whitespace/comments
            import json
            lines: List[str] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        lines.append(stripped)
            if not lines:
                raise ValueError(f"Dataset file is empty: {path}") from e
            records: List[Dict] = []
            for i, line in enumerate(lines, start=1):
                try:
                    records.append(json.loads(line))
                except Exception as je:
                    raise ValueError(
                        f"Invalid JSONL content at {path}:{i}: {je}") from je
            df = pd.DataFrame.from_records(records)
    else:
        raise ValueError(f"Unsupported dataset format: {format}")
    # Ensure that the prompt and response columns are present in the dataset.
    if prompt_column not in df.columns:
        raise ValueError(f"Column '{prompt_column}' not found in dataset")
    if response_column not in df.columns:
        raise ValueError(f"Column '{response_column}' not found in dataset")
    # Keep problem_id if it exists, otherwise create a default one
    if "problem_id" in df.columns:
        df = df[[prompt_column, response_column, "problem_id"]]
        return df.rename(columns={
            prompt_column: "prompt",
            response_column: "response",
            "problem_id": "problem_id",
        })
    else:
        # If no problem_id in data, create default ones
        df = df[[prompt_column, response_column]]
        df["problem_id"] = [f"default_prob_{i}" for i in range(len(df))]
        return df.rename(columns={
            prompt_column: "prompt", 
            response_column: "response",
        })


def tokenize_data(dataset: pd.DataFrame, tokenizer_name: str) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    prompts = []
    responses = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset),
                       desc="Tokenizing dataset"):
        prompts.append(tokenizer.encode(row["prompt"]))
        responses.append(tokenizer.encode(row["response"]))
    return pd.DataFrame({
        "prompt": prompts,
        "response": responses,
    })


def ensure_tokenized(dataset: pd.DataFrame):
    for _, row in dataset.iterrows():
        if not isinstance(row["prompt"], list):
            break
        if not all(isinstance(x, int) for x in row["prompt"]):
            break
        if not isinstance(row["response"], list):
            break
        if not all(isinstance(x, int) for x in row["response"]):
            break
    else:
        return
    raise ValueError(
        "Dataset must be tokenized or a tokenizer must be provided")


def get_data(args: argparse.Namespace) -> Tuple[pd.DataFrame,
                                                Optional[pd.DataFrame]]:
    dataset = read_data_file(args.dataset, args.prompt_column,
                             args.response_column, args.format)
    max_num_eval = max(args.num_eval) if args.num_eval else 1
    max_num_train = max(args.num_train) if args.num_train else 0
    if args.train_dataset is not None:
        train_dataset = read_data_file(args.train_dataset, args.prompt_column,
                                       args.response_column, args.format)
        if args.num_eval and max_num_eval > len(dataset):
            raise ValueError(
                f"Number of evaluation examples ({max_num_eval}) exceeds the "
                f"size of the dataset ({len(dataset)})"
            )
        if args.num_train and max_num_train > len(train_dataset):
            raise ValueError(
                f"Number of training examples ({max_num_train}) exceeds the "
                f"size of the training dataset ({len(train_dataset)})"
            )
    else:
        train_dataset = None
        if max_num_eval + max_num_train > len(dataset):
            raise ValueError(
                f"Number of evaluation examples ({max_num_eval}) and training "
                f"examples ({max_num_train}) exceed the size of the dataset "
                f"({len(dataset)})"
            )
    return dataset, train_dataset


def main(args: argparse.Namespace):
    dataset, train_dataset = get_data(args)
    
    # Initialize tokenizer if provided
    tokenizer = None
    if args.tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        
    # # # Tokenize datasets (if needed)
    # if args.tokenizer is not None:
    #     dataset = tokenize_data(dataset, args.tokenizer)
    #     if train_dataset is not None:
    #         train_dataset = tokenize_data(train_dataset, args.tokenizer)
    # else:
    #     ensure_tokenized(dataset)
    #     if train_dataset is not None:
    #         ensure_tokenized(train_dataset)
    # Create all possible configurations
    num_eval = args.num_eval or [None]
    num_train = args.num_train or [None]
    configs = OrderedDict(
        num_eval=num_eval,
        num_train=num_train,
        seed=args.seed,
        max_depth=args.max_depth,
        max_spec_tokens=args.max_spec_tokens,
        max_spec_factor=args.max_spec_factor,
        min_token_prob=args.min_token_prob,
        use_tree_spec=args.use_tree_spec,
        use_cached_prompt=args.use_cached_prompt,
    )
    config_values = itertools.product(*configs.values())
    
    # Prepare debug file path if output is specified
    debug_file_path = None
    if args.output is not None:
        output_dir = args.output if os.path.isdir(args.output) or os.path.splitext(args.output)[1] == "" else os.path.dirname(args.output)
        debug_file_path = os.path.join(output_dir, "debug_tokens.jsonl")
        # Clear debug file at start
        if os.path.exists(debug_file_path):
            os.remove(debug_file_path)
    
    # Prepare visualization settings
    enable_visualization = getattr(args, 'enable_visualization', False)
    viz_output_dir = getattr(args, 'viz_output', None)
    if viz_output_dir is None and args.output is not None:
        # Use the main output directory for visualizations
        base_output_dir = args.output if os.path.isdir(args.output) or os.path.splitext(args.output)[1] == "" else os.path.dirname(args.output)
        viz_output_dir = os.path.join(base_output_dir, "visualizations")

    config_values = [
        (dataset, train_dataset, i, *v, debug_file_path, tokenizer, enable_visualization, viz_output_dir) 
        for i, v in enumerate(config_values)]

    records = []
    with mp.Pool(args.parallel) as pool:
        for results in pool.starmap(process_task, config_values):
            records.extend(results)

    print("Preparing results...")

    df = pd.DataFrame.from_records(records)

    summary = results_summary(df, list(configs.keys()))
    print("\nSummary of results:\n")
    print(summary.to_string() + "\n")

    if args.output is not None:
        output_path = args.output
        # If the provided output is a directory or has no extension,
        # treat it as a directory and write a default CSV file into it.
        if os.path.isdir(output_path) or os.path.splitext(output_path)[1] == "":
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, "results.csv")
        else:
            parent_dir = os.path.dirname(output_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"Detailed results saved to: {output_path}")
        
    if debug_file_path and os.path.exists(debug_file_path):
        print(f"Debug token information (JSONL format) saved to: {debug_file_path}")


def bool_arg(v):
    if v.lower() not in ("true", "false"):
        raise ValueError(f"Invalid boolean argument '{v}'")
    return v.lower() == "true"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the dataset file",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl", "csv"],
        help="Format of the dataset file, uses its extension if not provided",
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        help="Path to a separate dataset file for training",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="input_token_ids",
        help="Column name for the prompts in the dataset",
    )
    parser.add_argument(
        "--response-column",
        type=str,
        default="output_token_ids",
        help="Column name for the responses in the dataset",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        nargs="+",
        help=("Number of examples to use for training (required if "
              "separate --train-dataset is not provided)"),
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        nargs="+",
        help=("Number of examples to use for evaluation (required if "
              "separate --train-dataset is not provided)"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[0],
        help="Random seed (for train/eval split)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name of the HuggingFace tokenizer",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The path to the output CSV file",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=16,
        help="Max number of parallel processes",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        nargs="+",
        default=[32],
        help="Max depth of the suffix tree",
    )
    parser.add_argument(
        "--max-spec-tokens",
        type=int,
        nargs="+",
        default=[0],
        help="Max speculation tokens (if 0, defaults to max_depth)",
    )
    parser.add_argument(
        "--max-spec-factor",
        type=float,
        nargs="+",
        default=[2.0],
        help="Max speculation tokens as a multiplier of the prefix length",
    )
    parser.add_argument(
        "--min-token-prob",
        type=float,
        nargs="+",
        default=[0.1],
        help="Minimum probability of the token to be considered",
    )
    parser.add_argument(
        "--use-tree-spec",
        type=bool_arg,
        nargs="*",
        default=[True],
        help="Whether to use tree-based speculation (True/False)",
    )
    parser.add_argument(
        "--use-cached-prompt",
        type=bool_arg,
        nargs="*",
        default=[True],
        help=("Whether to use the cached prompt for the request in addition "
              "to the global cache of previous responses (True/False)"),
    )
    parser.add_argument(
        "--enable-visualization",
        default=[False],
        help="Enable suffix tree visualization after building",
    )
    parser.add_argument(
        "--viz-output",
        type=str,
        help="Directory path for visualization outputs (if not specified, uses output directory + '/visualizations')",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.train_dataset is None:
        if args.num_train is None and args.num_eval is None:
            raise ValueError("Must provide --num-train or --num-eval if "
                             "separate --train-dataset is not provided")
    if len(args.use_tree_spec) == 0:
        args.use_tree_spec = [True]
    if len(args.use_cached_prompt) == 0:
        args.use_cached_prompt = [True]
    main(args)
