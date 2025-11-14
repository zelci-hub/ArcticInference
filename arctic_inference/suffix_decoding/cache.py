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
                 thread_safe: bool = False,
                 max_threads: int = None):
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
        
        # Configure thread pool size
        if max_threads is None:
            self._max_threads = 10
            print(f"🎯 SuffixDecodingCache使用硬编码线程数: {self._max_threads}")
        else:
            self._max_threads = max_threads
            print(f"🎯 SuffixDecodingCache使用指定线程数: {max_threads}")

        # Global suffix tree caches previous responses in a single tree.
        self._global_tree = SuffixTree(max_tree_depth)

        # Local suffix trees cache prompts for each active request separately.
        self._local_trees = {}
        
        # Problem trees cache responses for each problem separately
        self._problem_tree = {}

        # Maps between Python request ID and int32_t sequence ID. Tracks all
        # request IDs that are in the global tree.
        self._req_to_seq_id = {}
        self._seq_to_req_id = {}

        # Unused sequence ID to assign to a new request ID.
        self._next_seq_id = 0
        
        # Thread safety lock for shared dictionary access
        self._dict_lock = threading.Lock() if thread_safe else None
        
        print(f"SuffixDecodingCache initialized: thread_safe={thread_safe}, max_threads={self._max_threads}")

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

    @property
    def cached_requests(self) -> KeysView:
        """
        Returns a view of all request IDs that have their responses cached in
        the global suffix tree. The response for the cached request can be used
        during speculation for other requests, until the response is evicted.
        """
        return self._req_to_seq_id.keys()

    def start_request(self, req_id: Hashable, prompt_token_ids: Sequence[int]):
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
        if self._max_cached_requests != 0:
            # Global cache is enabled.
            if req_id in self._req_to_seq_id:
                # Evict existing cached response for the request if present.
                self.evict_cached_response(req_id)
            # Allocate a new seq_id for the request.
            self._generate_seq_id(req_id)

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
        token_ids: Union[int, Sequence[int]],
        problem_id: Optional[Hashable] = None,
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
        if isinstance(token_ids, Sequence):
            self._local_trees[req_id].extend(0, token_ids)
        else:
            self._local_trees[req_id].append(0, token_ids)
        # Also update the response if the request is in the global cache (it
        # may be evicted from the global cache before the request is stopped).
        if req_id in self._req_to_seq_id:
            seq_id = self._req_to_seq_id[req_id]
            if isinstance(token_ids, Sequence):
                self._global_tree.extend(seq_id, token_ids)
            else:
                self._global_tree.append(seq_id, token_ids)

        # Also update problem-specific tree if problem_id is provided
        if problem_id is not None:
            # Thread-safe tree creation
            if problem_id not in self._problem_tree:
                if self._dict_lock:
                    with self._dict_lock:
                        if problem_id not in self._problem_tree:
                            self._problem_tree[problem_id] = SuffixTree(self._max_tree_depth)
                else:
                    self._problem_tree[problem_id] = SuffixTree(self._max_tree_depth)
            
            problem_tree = self._problem_tree[problem_id]
            seq_id = self._req_to_seq_id.get(req_id, 0)  # Use seq_id if available
            
            # Thread-safe extend/append to problem_tree
            if self._dict_lock:
                with self._dict_lock:
                    if isinstance(token_ids, Sequence):
                        problem_tree.extend(seq_id, token_ids)
                    else:
                        problem_tree.append(seq_id, token_ids)
            else:
                if isinstance(token_ids, Sequence):
                    problem_tree.extend(seq_id, token_ids)
                else:
                    problem_tree.append(seq_id, token_ids)

    def evict_cached_response(self, req_id: Hashable):
        """
        Evicts the given request's response from the global cache. `req_id` can
        be safely reused for a new request after eviction.

        Args:
            req_id (Hashable): The unique identifier for the request that
                should be evicted.

        Raises:
            ValueError: If no response exists for the given request identifier.
        """
        if req_id not in self._req_to_seq_id:
            raise ValueError(f"Request '{req_id}' is not cached")
        seq_id = self._req_to_seq_id.pop(req_id)
        self._seq_to_req_id.pop(seq_id)
        self._global_tree.remove(seq_id)

    def evict_problem(self, problem_id: Hashable):
        """
        Evicts a problem tree from the cache.

        Args:
            problem_id (Hashable): The unique identifier for the problem whose
                tree should be evicted.

        Raises:
            ValueError: If no problem tree exists for the given problem identifier.
        """
        if problem_id not in self._problem_tree:
            raise ValueError(f"Problem tree does not exist for problem '{problem_id}'")
        del self._problem_tree[problem_id]

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
                uses the cache's max_depth.
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
            max_spec_tokens = self.max_depth

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

        candidate = self._global_tree.speculate(
            pattern,
            max_spec_tokens,
            max_spec_factor,
            max_spec_offset,
            min_token_prob,
            use_tree_spec)
        if candidate.score > result.score:
            result = SuffixDecodingDraft.from_candidate(candidate)
            source = "global"
        # result = SuffixDecodingDraft.from_candidate(candidate)
        # source = "global"

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

    def _generate_seq_id(self, req_id: Hashable) -> int:
        # Find the next available seq_id not used by an active request.
        while True:
            seq_id = self._next_seq_id
            # Increment to the next non-negative int32_t value.
            self._next_seq_id = (self._next_seq_id + 1) & 0x7FFFFFFF
            if (seq_id not in self._seq_to_req_id or
                    self._seq_to_req_id[seq_id] not in self._local_trees):
                break
        # Check if the seq_id is used by an inactive but cached request.
        if seq_id in self._seq_to_req_id:
            # This seq_id is already used, should be a very rare case that
            # only happens when the seq_id has wrapped around and collided.
            # We evict the old cached request to free up the seq_id.
            del self._req_to_seq_id[self._seq_to_req_id[seq_id]]
            del self._seq_to_req_id[seq_id]
            self._global_tree.remove(seq_id)
        # Allocate the seq_id to the new req_id.
        self._req_to_seq_id[req_id] = seq_id
        self._seq_to_req_id[seq_id] = req_id
        self._maybe_evict_requests(seq_id)
        return seq_id

    def _maybe_evict_requests(self, new_seq_id: int):
        if self._max_cached_requests < 0:
            # Negative value means no global cache size limit.
            return
        assert self._max_cached_requests != 0  # Global cache must be enabled.
        while len(self._req_to_seq_id) > self._max_cached_requests:
            # Evict the first eligible request. Should be FIFO order in Python
            # 3.7+ since dict preserves insertion order. Avoid evicting the
            # request that was just added (new_seq_id).
            for req_id, seq_id in self._req_to_seq_id.items():
                if seq_id != new_seq_id:
                    self.evict_cached_response(req_id)
                    break

    def prebuild_problemtree(
        self,
        seq_id: int,
        problem_id: Hashable,
        prompt_token_ids: Sequence[int],
        token_ids: Union[int, Sequence[int]],
    ):
        """
        Pre-build problem tree with given tokens.
        
        Args:
            seq_id: Sequence ID  
            problem_id: Problem ID to identify the tree
            prompt_token_ids: Prompt token sequence
            token_ids: Response token sequence
        """
        # Thread-safe tree creation
        if problem_id not in self._problem_tree:
            if self._dict_lock:
                with self._dict_lock:
                    if problem_id not in self._problem_tree:
                        self._problem_tree[problem_id] = SuffixTree(self._max_tree_depth)
            else:
                self._problem_tree[problem_id] = SuffixTree(self._max_tree_depth)
        
        tree = self._problem_tree[problem_id]
        
        # Use thread-safe methods if enabled (with GIL release)
        if self._thread_safe:
            tree.extend_safe(seq_id, prompt_token_ids)
            tree.extend_safe(seq_id, token_ids)
        else:
            tree.extend(seq_id, prompt_token_ids)
            tree.extend(seq_id, token_ids)

    def clear_all_cache(self):
        """
        Clear all cached data in the suffix cache to free up memory.
        Uses parallel tree deletion with ThreadPoolExecutor.
        
        Note: Parallel cleanup has limited speedup due to:
        1. C++ clear() holds mutex during entire operation (including memory deallocation)
        2. System memory allocator may have global locks
        3. Heavy memory deallocation operations serialize at C++ level
        
        For large trees (many sequences), expect ~1.4x speedup with 4 threads,
        not close to 4x like prebuild operations.
        """
        num_problem_trees = len(self._problem_tree)
        num_local_trees = len(self._local_trees)
        
        print(f"Starting parallel cleanup of {num_problem_trees} problem trees, {num_local_trees} local trees")
        start_time = time.time()
        
        import hashlib
        all_trees = []
        
        # Collect all problem trees
        for problem_id, tree in self._problem_tree.items():
            all_trees.append(("problem", problem_id, tree))
        
        # Collect all local trees
        for req_id, tree in self._local_trees.items():
            all_trees.append(("local", req_id, tree))
        
        if all_trees:
            # Group trees by thread using hash-based load balancing
            # Grouped approach is better than individual tasks for heavy operations
            thread_groups = [[] for _ in range(self._max_threads)]
            for tree_type, tree_id, tree in all_trees:
                tree_hash = hashlib.md5(str(tree_id).encode()).hexdigest()
                thread_idx = int(tree_hash, 16) % self._max_threads
                thread_groups[thread_idx].append((tree_type, tree_id, tree))
            
            def cleanup_tree_group(group_data):
                """Clear a group of trees in one thread"""
                thread_start = time.perf_counter()
                cleared_count = 0
                
                for tree_type, tree_id, tree in group_data:
                    # Each tree's clear() releases GIL but holds C++ mutex
                    tree.clear()
                    cleared_count += 1
                
                return {
                    'cleared': cleared_count,
                    'time': time.perf_counter() - thread_start
                }
            
            # Execute parallel cleanup with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len([g for g in thread_groups if g])) as executor:
                futures = []
                for i, group in enumerate(thread_groups):
                    if group:  # Only submit non-empty groups
                        future = executor.submit(cleanup_tree_group, group)
                        futures.append(future)
                
                # Collect results
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
                
                total_cleared = sum(r['cleared'] for r in results)
                max_thread_time = max(r['time'] for r in results) if results else 0
                
                print(f"Parallel tree clearing: {total_cleared} trees in {max_thread_time:.3f}s using {len(results)} threads")
        
        # Clear dictionaries (fast operation)
        self._problem_tree.clear()
        self._local_trees.clear()
        self._req_to_seq_id.clear()
        self._seq_to_req_id.clear()
        
        # Force GC
        gc.collect()
        
        elapsed = time.time() - start_time
        print(f"✅ Parallel cleanup completed in {elapsed:.3f}s - {num_problem_trees} problem trees, "
              f"{num_local_trees} local trees")
        return {"success": True, "elapsed": elapsed}

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        stats = {
            "max_tree_depth": self._max_tree_depth,
            "max_cached_requests": self._max_cached_requests,
            "thread_safe": self._thread_safe,
            "max_threads": self._max_threads,
            "problem_tree_count": len(self._problem_tree),
            "local_tree_count": len(self._local_trees),
            "cached_request_count": len(self._req_to_seq_id),
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
