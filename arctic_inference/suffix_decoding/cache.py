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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, KeysView, List, Optional, Sequence, Union, Tuple
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

from arctic_inference.suffix_decoding._C import SuffixTree, Candidate


@dataclass
class SuffixDecodingDraft:
    """
    A dataclass representing the result of a speculation using SuffixDecoding.

    Attributes:
        token_ids (List[int]): List of token IDs in the speculation result.
        parents (List[int]): List of parent indices for each token used to
            encode the tree structure. The parent token of token_ids[i] is
            token_ids[parents[i]].
        probs (List[float]): List of estimated probabilities for each token.
        score (float): The overall score of the suffix match computed as the
            sum of the estimated probabilities of each speculated token.
        match_len (int): The length of the pattern match that yielded this
            speculation result.
    """
    token_ids: List[int] = field(default_factory=list)
    parents: List[int] = field(default_factory=list)
    probs: List[float] = field(default_factory=list)
    score: float = 0.0
    match_len: int = 0

    @staticmethod
    def from_candidate(candidate: Candidate) -> SuffixDecodingDraft:
        return SuffixDecodingDraft(
            token_ids=candidate.token_ids,
            parents=candidate.parents,
            probs=candidate.probs,
            score=candidate.score,
            match_len=candidate.match_len,
        )


class SuffixDecodingCache:
    
    def __init__(self,
                 max_tree_depth: int = 64,
                 max_cached_requests: int = -1,
                 thread_safe: bool = True,
                 max_threads: int = 4):
        """
        Initialize the SuffixDecodingCache.

        Args:
            max_tree_depth (int): The maximum depth of the suffix trees.
            max_cached_requests (int, optional): The maximum number of cached
                requests. Eviction is triggered when the limit is reached. `-1`
                means no limit on the number of cached requests.
            thread_safe (bool): Whether to use thread-safe operations
            max_threads (int): Maximum number of threads for parallel operations
        """
        if max_cached_requests > 0x7FFFFFFF:
            raise ValueError("max_cached_requests must be at most 2^31")

        self._max_tree_depth = max_tree_depth
        self._max_cached_requests = max_cached_requests
        self._thread_safe = thread_safe
        self._max_threads = max_threads
        # Local suffix trees cache prompts for each active request separately.
        self._local_trees = {}
        
        # Problem trees cache responses for each problem separately
        self._problem_tree = {}
        
    @property
    def max_tree_depth(self) -> int:
        return self._max_tree_depth

    @property
    def max_cached_requests(self) -> int:
        return self._max_cached_requests

    @property
    def active_requests(self) -> KeysView:
        """
        Returns a view of the currently active request IDs. Active requests are
        those that have been started via `start_request` and not yet stopped
        via `stop_request`. The prompts of active requests are stored so they
        can be used during speculation for the same request.
        """
        return self._local_trees.keys()

    def start_request(self, req_id: Hashable, problem_id: Optional[Hashable], prompt_token_ids: Sequence[int]):
        """
        This method should be called when starting to process a new request. It
        will store the prompt for the request, allowing future speculations for
        the same request to use the prompt context. The prompt will be stored
        until `stop_request` is called. If `max_cached_requests != 0`, then a
        new slot is allocated in the global cache for the response, triggering
        cache eviction (FIFO order) if needed.

        Args:
            req_id (Hashable): The request identifier. Must be a hashable value
                that uniquely identifies the request.
            problem_id (Optional[Hashable]): The problem identifier. Must be a hashable value
                that uniquely identifies the problem.
            prompt_token_ids (Sequence[int]): A sequence of token IDs
                representing the prompt of the request.

        Raises:
            ValueError: If a request with the same `req_id` is already active
                or cached.
        """
        if req_id in self._local_trees:
            raise ValueError(f"Request '{req_id}' is already active")
        self._local_trees[req_id] = SuffixTree(self._max_tree_depth)
        self._local_trees[req_id].extend(0, prompt_token_ids)
        assert problem_id is not None
        if problem_id not in self._problem_tree:
            self._problem_tree[problem_id] = SuffixTree(self._max_tree_depth)
            self._problem_tree[problem_id].extend(0, prompt_token_ids)

    def stop_request(self, req_id: Hashable):
        """
        This method should be called when a request is completed. It will evict
        the prompt for the request, freeing up memory. The request's response
        may still be cached in the global cache until it is evicted.

        Args:
            req_id (Hashable): The request identifier. Must be a hashable value
                that uniquely identifies the request.

        Raises:
            ValueError: If the request with the given `req_id` is not active.
        """
        if req_id not in self._local_trees:
            raise ValueError(f"Request '{req_id}' is not active")
        del self._local_trees[req_id]

    def add_active_response(
        self,
        req_id: Hashable,
        problem_id: Optional[Hashable],
        token_ids: Union[int, Sequence[int]],
    ):
        """
        Update the cached response for a given request by appending token(s) to
        its end. Once the response is updated, the new tokens can be used for
        future speculations for all requests.

        Args:
            req_id (Hashable): The unique identifier for the request.
            token_ids (Union[int, Sequence[int]]): Either a single token ID
                (int) or a sequence of token IDs to be appended to the response
                for the given request.

        Raises:
            ValueError: If the request with the given `req_id` is not active.
        """
        if req_id not in self._local_trees:
            raise ValueError(f"Request '{req_id}' is not active")
        assert problem_id is not None
        
        # Ensure problem tree exists (defensive programming)
        # if problem_id not in self._problem_tree:
        #     self._problem_tree[problem_id] = SuffixTree(self._max_tree_depth)
        
        if isinstance(token_ids, Sequence):
            if self._thread_safe:
                # self._problem_tree[problem_id].extend_safe(0, token_ids)
                self._local_trees[req_id].extend_safe(0, token_ids)
            else:
                # self._problem_tree[problem_id].extend(0, token_ids)
                self._local_trees[req_id].extend(0, token_ids)
        else:
            if self._thread_safe:
                # self._problem_tree[problem_id].append_safe(0, token_ids)
                self._local_trees[req_id].append_safe(0, token_ids)
            else:
                # self._problem_tree[problem_id].append(0, token_ids)
                self._local_trees[req_id].append(0, token_ids)

    def speculate(
        self,
        req_id: Hashable,
        pattern: Sequence[int],
        max_spec_tokens: Optional[int] = None,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
        problem_id: Optional[Hashable] = None,
    ) -> tuple[SuffixDecodingDraft, str]:
        """
        Speculates and returns the most likely continuation of a given token
        pattern using the request's prompt and the global cache of previous
        responses. This method can only be called for active requests (i.e.
        after calling `start_request` and before calling `stop_request`).

        Args:
            req_id (Hashable): The unique identifier for the request.
            pattern (Sequence[int]): The sequence of token IDs to match and
                continue from.
            max_spec_tokens (int): Maximum number of tokens to speculate. If 0,
                uses the cache's max_tree_depth.
            max_spec_factor (float): Factor that limits speculation based on
                matched pattern length.
            min_token_prob (float): Minimum estimated probability threshold for
                candidate tokens.
            use_tree_spec (bool): If True, uses tree-based speculation.
        
        Returns:
            The speculation result containing the most likely continuation
            tokens, their probabilities, and overall score.

        Raises:
            ValueError: If the request with the given `req_id` is not active.
        """
        if req_id not in self._local_trees:
            raise ValueError(f"Request '{req_id}' is not active")

        if max_spec_tokens is None:
            max_spec_tokens = self.max_tree_depth

        if len(pattern) > self._max_tree_depth:
            pattern = pattern[-self._max_tree_depth :]

        candidate = self._local_trees[req_id].speculate(
            pattern,
            max_spec_tokens,
            max_spec_factor,
            max_spec_offset,
            min_token_prob,
            use_tree_spec)
        result = SuffixDecodingDraft.from_candidate(candidate)
        source = "local"
        assert problem_id is not None
        # Speculate using problem-specific tree if problem_id is provided
        if problem_id is not None and problem_id in self._problem_tree:
            problem_tree = self._problem_tree[problem_id]
            problem_candidate = problem_tree.speculate(
                pattern,
                max_spec_tokens,
                max_spec_factor,
                max_spec_offset,
                min_token_prob,
                use_tree_spec,
            )
            if problem_candidate.score > result.score:
                result = SuffixDecodingDraft.from_candidate(problem_candidate)
                source = f"problem_{problem_id}"
        return result, source

    def prebuild_problems_parallel(
        self,
        problem_data: List[dict],
    ):
        """
        Pre-build multiple problem trees in parallel using ThreadPoolExecutor.
        
        Args:
            problem_data: List of dict format: 
                         {'problem_id': pid, 'sequences': [{'seq_id': int, 'prompt_tokens': list, 'response_tokens': list}, ...]}
        
        Returns:
            dict: Results containing success status and statistics
        """
        if not problem_data:
            return {"success": True, "problems_built": 0}
        
        import hashlib
        
        # Validate and normalize input data with assertions
        normalized_data = []
        # 🕐 阶段2: 线程分组和负载均衡 (以problem_id为单位分配到不同线程)
        thread_groups = [[] for _ in range(self._max_threads)]
        
        for i, item in enumerate(problem_data):
            # Assert input format
            assert isinstance(item, dict), f"Expected dict at index {i}, got {type(item)}"
            assert 'problem_id' in item, f"Missing 'problem_id' key at index {i}"
            
            problem_id = item['problem_id']
            sequences = item.get('sequences', [])
            assert isinstance(sequences, list), f"'sequences' must be list at index {i}, got {type(sequences)}"
            
            thread_idx = i % self._max_threads
            
            for seq_idx, seq_data in enumerate(sequences):
                assert isinstance(seq_data, dict), f"Sequence at index {i}.{seq_idx} must be dict, got {type(seq_data)}"
                
                seq_id = seq_data.get('seq_id', seq_idx)
                prompt_tokens = seq_data.get('prompt_tokens', [])
                response_tokens = seq_data.get('response_tokens', [])
                
                # Assert data types
                assert isinstance(seq_id, int), f"seq_id must be int at {i}.{seq_idx}, got {type(seq_id)}"
                assert isinstance(prompt_tokens, list), f"prompt_tokens must be list at {i}.{seq_idx}, got {type(prompt_tokens)}"
                assert isinstance(response_tokens, list), f"response_tokens must be list at {i}.{seq_idx}, got {type(response_tokens)}"
                
                thread_groups[thread_idx].append((seq_id, problem_id, prompt_tokens, response_tokens))
            
        def prebuild_problem_group(group_data):
            """Pre-build a group of problems in one thread"""
            built_count = 0
            
            for seq_id, problem_id, prompt_tokens, response_tokens in group_data:
                try:
                    # Thread-safe tree creation
                    if problem_id not in self._problem_tree:
                        # 🔧 修复: SuffixTree构造函数只接受max_depth参数，不支持thread_safe
                        self._problem_tree[problem_id] = SuffixTree(self._max_tree_depth)
                    
                    tree = self._problem_tree[problem_id]
                    
                    # Use thread-safe methods if enabled (with GIL release)
                    if prompt_tokens or response_tokens:
                        if self._thread_safe:
                            tree.extend_safe(seq_id, prompt_tokens + response_tokens)
                        else:
                            tree.extend(seq_id, prompt_tokens + response_tokens)
                    
                    built_count += 1
                    
                except Exception as e:
                    print(f"Error building problem {problem_id}: {e}")
                    continue
            
            return {
                'built': built_count,
            }
        
        # Execute parallel prebuild with ThreadPoolExecutor
        max_workers = len([g for g in thread_groups if g])
        print(f"debug:max_workers: {max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, group in enumerate(thread_groups):
                if group:  # Only submit non-empty groups
                    future = executor.submit(prebuild_problem_group, group)
                    futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                results.append(future.result())
            
            total_built = sum(r['built'] for r in results)
            
            #print(f"Parallel prebuild: {total_built} problems built in {max_thread_time:.3f}s using {len(results)} threads")
        # elapsed = time.time() - start_time
        # print(f"✅ Parallel prebuild completed in {elapsed:.3f}s - {total_built}/{len(problem_data)} problems built")
        
        return {
            "success": True, 
            "problems_built": total_built,
            "total_problems": len(normalized_data)
        }

    def clear_cache(self, problem_ids: Optional[Sequence[Hashable]] = None):
        """
        Clear cached problem trees. If problem_ids is given, only those trees are
        cleared and removed; otherwise all problem trees and local trees are cleared.
        Uses parallel tree deletion with ThreadPoolExecutor.

        Args:
            problem_ids: Optional sequence of problem_id to clean. Only these
                problem trees are cleared and removed from cache. If None, clears
                all problem trees and local trees (full cache reset).

        Note: Parallel cleanup has limited speedup due to:
        1. C++ clear() holds mutex during entire operation (including memory deallocation)
        2. System memory allocator may have global locks
        3. Heavy memory deallocation operations serialize at C++ level

        For large trees (many sequences), expect ~1.4x speedup with 4 threads,
        not close to 4x like prebuild operations.
        """
        import hashlib

        if problem_ids is not None:
            # Only clean the specified problem_ids that exist in cache; deduplicate so each problem_id is cleared once
            tree_by_pid = {
                pid: self._problem_tree[pid]
                for pid in problem_ids
                if pid in self._problem_tree
            }
            trees_to_clear = list(tree_by_pid.items())
            clear_full_cache = False
        else:
            # Clean all problem trees
            trees_to_clear = list(self._problem_tree.items())
            clear_full_cache = True

        num_to_clear = len(trees_to_clear)
        num_local_trees = len(self._local_trees)

        print(f"Starting parallel cleanup of {num_to_clear} problem trees")
        start_time = time.time()

        if trees_to_clear:
            # Group (problem_id, tree) by thread using hash-based load balancing
            thread_groups = [[] for _ in range(self._max_threads)]
            for problem_id, tree in trees_to_clear:
                tree_hash = hashlib.md5(str(problem_id).encode()).hexdigest()
                thread_idx = int(tree_hash, 16) % self._max_threads
                thread_groups[thread_idx].append((problem_id, tree))

            def cleanup_tree_group(group_data):
                """Clear a group of trees in one thread"""
                thread_start = time.perf_counter()
                cleared_count = 0
                for problem_id, tree in group_data:
                    tree.clear()
                    cleared_count += 1
                return {
                    "cleared": cleared_count,
                    "time": time.perf_counter() - thread_start,
                }

            num_workers = len([g for g in thread_groups if g])
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(cleanup_tree_group, group)
                    for group in thread_groups
                    if group
                ]
                results = [f.result() for f in as_completed(futures)]

            total_cleared = sum(r["cleared"] for r in results)
            max_thread_time = max(r["time"] for r in results) if results else 0
            print(
                f"Parallel tree clearing: {total_cleared} trees in {max_thread_time:.3f}s "
                f"using {len(results)} threads"
            )

            if clear_full_cache:
                self._problem_tree.clear()
                self._local_trees.clear()
            else:
                for problem_id, _ in trees_to_clear:
                    self._problem_tree.pop(problem_id, None)

        gc.collect()
        elapsed = time.time() - start_time
        print(
            f"✅ Parallel cleanup completed in {elapsed:.3f}s - {num_to_clear} problem trees cleared"
            + (f", {num_local_trees} local trees" if clear_full_cache else "")
        )
        return {"success": True, "elapsed": elapsed, "cleared_count": num_to_clear}

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        stats = {
            "max_tree_depth": self._max_tree_depth,
            "max_cached_requests": self._max_cached_requests,
            "thread_safe": self._thread_safe,
            "max_threads": self._max_threads,
            "problem_tree_count": len(self._problem_tree),
            "local_tree_count": len(self._local_trees),
            "active_request_count": len(self._local_trees),
        }
        
        if self._thread_safe:
            # Get thread-safe counts
            total_seqs = 0
            for tree in self._problem_tree.values():
                try:
                    total_seqs += tree.num_seqs_safe()
                except:
                    total_seqs += tree.num_seqs()
            stats["total_sequences_in_problem_trees"] = total_seqs
        else:
            total_seqs = sum(tree.num_seqs() for tree in self._problem_tree.values())
            stats["total_sequences_in_problem_trees"] = total_seqs
        
        return stats
