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
from typing import Hashable, List, Optional, Sequence, Union, Tuple
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import multiprocessing as mp
import os
import sys

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

from arctic_inference.common.suffix_cache._C import SuffixTree, Candidate


@dataclass
class SuffixSpecResult:
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
    def from_candidate(candidate: Candidate) -> SuffixSpecResult:
        return SuffixSpecResult(
            token_ids=candidate.token_ids,
            parents=candidate.parents,
            probs=candidate.probs,
            score=candidate.score,
            match_len=candidate.match_len,
        )


class SuffixCache:
    
    def __init__(self, max_depth: int = 64, thread_safe: bool = False, max_threads: int = None):
        """
        Initialize SuffixCache
        
        Args:
            max_depth: Maximum depth of suffix trees
            thread_safe: Whether to use thread-safe C++ methods (with GIL release)
            max_threads: Maximum number of threads for parallel operations (None = auto)
        """
        self._max_depth = max_depth
        self._thread_safe = thread_safe
        # 🎯 SuffixCache线程数配置：硬编码为10线程
        if max_threads is None:
            self._max_threads = 10
            print(f"🎯 SuffixCache使用硬编码线程数: {self._max_threads}")
        else:
            self._max_threads = max_threads
            print(f"🎯 SuffixCache使用指定线程数: {max_threads}")
        
        #self._suffix_tree = SuffixTree(max_depth)
        self._problem_tree = {}
        self._prompt_trees = {}
        self._req_to_seq_id = {}
        
        # Thread safety lock for shared dictionary access
        self._dict_lock = threading.Lock() if thread_safe else None
        
        # 🧹 Background cleanup infrastructure (Queue + Thread, like prebuild)
        import queue
        self._cleanup_queue = queue.Queue()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="suffix-cache-cleanup",
            daemon=True
        )
        self._cleanup_thread.start()
        
        print(f"SuffixCache initialized: thread_safe={thread_safe}, max_threads={self._max_threads}")
        print(f"SuffixCache: Background cleanup thread started")

    @property
    def max_depth(self) -> int:
        return self._max_depth

    def has_cached_prompt(self, req_id: Hashable) -> bool:
        return req_id in self._prompt_trees

    def cached_prompt_ids(self) -> List[Hashable]:
        return list(self._prompt_trees.keys())

    def cache_prompt(self, req_id: Hashable, prompt_token_ids: Sequence[int], problem_id: Optional[Hashable] = None):
        """
        Cache a prompt for a specific request ID. Future speculations for the
        same request may also source draft tokens from this prompt.

        Args:
            req_id (Hashable): The request identifier. Must be a hashable value
                that uniquely identifies the request.
            prompt_token_ids (Sequence[int]): A sequence of token IDs
                representing the prompt to be cached.

        Raises:
            ValueError: If a prompt already exists for the given request ID.

        Note:
            The caller should evict the cached prompt using `evict_prompt` once
            the prompt is no longer needed (i.e. the request is completed).
        """
        # Thread-safe prompt creation and caching
        if self._dict_lock:
            with self._dict_lock:
                if req_id in self._prompt_trees:
                    raise ValueError(f"Prompt already exists for request '{req_id}'")
                
                self._prompt_trees[req_id] = SuffixTree(self._max_depth)
                tree = self._prompt_trees[req_id]
        else:
            if req_id in self._prompt_trees:
                raise ValueError(f"Prompt already exists for request '{req_id}'")
            
            self._prompt_trees[req_id] = SuffixTree(self._max_depth)
            tree = self._prompt_trees[req_id]
        
        # Use thread-safe methods if enabled
        if self._thread_safe:
            tree.extend_safe(0, prompt_token_ids)
        else:
            tree.extend(0, prompt_token_ids)

        # Optionally also cache the prompt into the corresponding problem tree
        if problem_id is not None:
            # Ensure problem tree exists
            if problem_id not in self._problem_tree:
                if self._dict_lock:
                    with self._dict_lock:
                        if problem_id not in self._problem_tree:
                            self._problem_tree[problem_id] = SuffixTree(self._max_depth)
                else:
                    self._problem_tree[problem_id] = SuffixTree(self._max_depth)
            problem_tree = self._problem_tree[problem_id]
            if self._thread_safe:
                problem_tree.extend_safe(0, prompt_token_ids)
            else:
                problem_tree.extend(0, prompt_token_ids)

    def evict_prompt(self, req_id: Hashable):
        """
        Evicts a prompt from the cache for a specific request.

        Args:
            req_id (Hashable): The unique identifier for the request whose
                prompt should be evicted.

        Raises:
            ValueError: If no prompt exists for the given request identifier.
        """
        if req_id not in self._prompt_trees:
            raise ValueError(f"Prompt does not exist for request '{req_id}'")
        del self._prompt_trees[req_id]

    def evict_problem(self, problem_id: Hashable):
        if problem_id not in self._problem_tree:
            raise ValueError(f"Prompt does not exist for request '{problem_id}'")
        del self._problem_tree[problem_id]

    def _cleanup_loop(self):
        """🧹 Background cleanup loop (runs in daemon thread, like prebuild loop)"""
        import queue as q
        while True:
            try:
                task = self._cleanup_queue.get()  # 阻塞等待清理任务
                
                old_problem_tree = task["old_problem_tree"]
                old_prompt_trees = task["old_prompt_trees"]
                old_req_to_seq_id = task["old_req_to_seq_id"]
                num_problem_trees = task["num_problem_trees"]
                num_prompt_trees = task["num_prompt_trees"]
                
                print(f"DEBUG: [Cleanup Thread] Starting cleanup of {num_problem_trees} problem trees, {num_prompt_trees} prompt trees")
                start_time = time.time()
                
                # Clear old caches (slow part, ~100s)
                old_problem_tree.clear()
                old_prompt_trees.clear()
                old_req_to_seq_id.clear()
                
                # Force GC
                gc.collect()
                
                elapsed = time.time() - start_time
                print(f"DEBUG: [Cleanup Thread] ✅ Cleanup completed in {elapsed:.3f}s")
                
                self._cleanup_queue.task_done()
                
            except Exception as e:
                print(f"ERROR: [Cleanup Thread] Cleanup failed: {e}")
                import traceback
                traceback.print_exc()
    
    def clear_all_cache(self, use_background_thread: bool = False):
        """
        Clear all cached data in the suffix cache to free up memory.
        Uses Queue + Thread pattern (same as prebuild).
        
        Args:
            use_background_thread: If True, enqueue cleanup to background thread (non-blocking).
                                  If False, perform synchronous cleanup.
        """
        num_problem_trees = len(self._problem_tree)
        num_prompt_trees = len(self._prompt_trees)
        
        if use_background_thread:
            # 🚀 Enqueue cleanup task (non-blocking, like prebuild)
            print(f"DEBUG: [Main] Enqueueing cleanup task: {num_problem_trees} problem trees, {num_prompt_trees} prompt trees")
            
            # Move references to task dict
            task = {
                "old_problem_tree": self._problem_tree,
                "old_prompt_trees": self._prompt_trees,
                "old_req_to_seq_id": self._req_to_seq_id,
                "num_problem_trees": num_problem_trees,
                "num_prompt_trees": num_prompt_trees,
            }
            
            # Create new empty caches
            self._problem_tree = {}
            self._prompt_trees = {}
            self._req_to_seq_id = {}
            
            # Enqueue task (立即返回！)
            self._cleanup_queue.put_nowait(task)
            
            print(f"DEBUG: [Main] Cleanup task enqueued, returning immediately")
            return {"success": True, "method": "background_thread"}
        
        # Synchronous cleanup
        self._problem_tree.clear()
        self._prompt_trees.clear()
        self._req_to_seq_id.clear()
        
        print(f"DEBUG: Cleared all suffix cache data - {num_problem_trees} problem trees, "
              f"{num_prompt_trees} prompt trees, and req_to_seq_id mapping")
        return {"success": True, "method": "synchronous"}
    
    def _clear_all_cache_background_process(self):
        """
        ⚡ Fork a background process to clean up cache in parallel with main process.
        
        This leverages Linux fork's copy-on-write mechanism:
        1. Fork creates child process that inherits parent's memory (including C++ objects)
        2. Parent immediately creates new empty caches and continues
        3. Child slowly cleans up the old caches (with its own GIL, no blocking!)
        4. Child process exits when done
        
        Returns immediately without blocking main process.
        """
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Get counts before clearing
        num_problem_trees = len(self._problem_tree)
        num_prompt_trees = len(self._prompt_trees)
        
        if num_problem_trees == 0 and num_prompt_trees == 0:
            print(f"DEBUG: [{timestamp}] No cache to clean, skipping background cleanup")
            return None
        
        print(f"DEBUG: [{timestamp}] [Parent PID={os.getpid()}] Starting background process cleanup")
        print(f"DEBUG: [{timestamp}] Cache size: {num_problem_trees} problem trees, {num_prompt_trees} prompt trees")
        
        # 🔑 Key: Move references to local variables
        # After fork, both parent and child will have these references
        time_move_refs = time.time()
        old_problem_tree = self._problem_tree
        old_prompt_trees = self._prompt_trees
        old_req_to_seq_id = self._req_to_seq_id
        print(f"DEBUG: [{timestamp}] Moved references in {(time.time() - time_move_refs)*1000:.1f}ms")
        
        # Parent creates new empty caches BEFORE fork
        # This way, parent can immediately start using new caches
        time_new_caches = time.time()
        self._problem_tree = {}
        self._prompt_trees = {}
        self._req_to_seq_id = {}
        print(f"DEBUG: [{timestamp}] Created new empty caches in {(time.time() - time_new_caches)*1000:.1f}ms")
        
        # 🚀 使用 os.fork() 直接 fork，避免 multiprocessing 的所有开销
        # multiprocessing.Process 有隐式同步机制，会导致父进程阻塞
        time_fork_start = time.time()
        
        print(f"DEBUG: [{timestamp}] [Parent PID={os.getpid()}] About to os.fork()...")
        
        child_pid = os.fork()
        
        if child_pid == 0:
            # ========== 子进程代码 ==========
            # 这部分在子进程中执行
            child_pid_actual = os.getpid()
            print(f"DEBUG: [{timestamp}] [Child PID={child_pid_actual}] Background cleanup started")
            start_time = time.time()
            
            try:
                # Clear old caches (this is the slow part, ~100s)
                print(f"DEBUG: [{timestamp}] [Child PID={child_pid_actual}] Clearing {len(old_problem_tree)} problem trees...")
                old_problem_tree.clear()
                
                print(f"DEBUG: [{timestamp}] [Child PID={child_pid_actual}] Clearing {len(old_prompt_trees)} prompt trees...")
                old_prompt_trees.clear()
                
                old_req_to_seq_id.clear()
                
                # Force garbage collection
                print(f"DEBUG: [{timestamp}] [Child PID={child_pid_actual}] Running gc.collect()...")
                gc.collect()
                
                elapsed = time.time() - start_time
                print(f"DEBUG: [{timestamp}] [Child PID={child_pid_actual}] ✅ Background cleanup completed in {elapsed:.3f}s - "
                      f"{num_problem_trees} problem trees, {num_prompt_trees} prompt trees")
                
            except Exception as e:
                print(f"ERROR: [{timestamp}] [Child PID={child_pid_actual}] Background cleanup failed: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
            finally:
                # Child process exits
                print(f"DEBUG: [{timestamp}] [Child PID={child_pid_actual}] Child process exiting")
                os._exit(0)  # 使用 os._exit() 避免触发 atexit 回调
        
        # ========== 父进程代码 ==========
        # child_pid > 0，这是父进程
        fork_time = time.time() - time_fork_start
        print(f"DEBUG: [{timestamp}] ⚡ os.fork() took {fork_time*1000:.1f}ms")
        print(f"DEBUG: [{timestamp}] [Parent PID={os.getpid()}] ⚡ Forked child process (PID={child_pid})")
        print(f"DEBUG: [{timestamp}] [Parent] Returning immediately, cleanup in child process")
        print(f"DEBUG: [{timestamp}] [Parent] About to return, total elapsed={(time.time() - time_fork_start)*1000:.1f}ms")
        
        # 父进程立即返回，不等待子进程
        return {"success": True, "pid": child_pid}

    def _get_or_assign_seq_id(self, req_id: Hashable) -> int:
        if req_id not in self._req_to_seq_id:
            self._req_to_seq_id[req_id] = len(self._req_to_seq_id)
        return self._req_to_seq_id[req_id]

    def update_response(
        self,
        req_id: Hashable,
        problem_id: Hashable,
        token_ids: Union[int | Sequence[int]],
    ):
        """
        Update the cached response for a given request by adding token(s) to
        its end. It does not rely on the prompt being cached for the request,
        and its lifetime does not depend on the prompt's existence. Once the
        response is updated, the new tokens can be used for future speculations
        for all requests.

        Args:
            req_id (Hashable): The unique identifier for the request.
            token_ids (Union[int, Sequence[int]]): Either a single token ID
                (int) or a sequence of token IDs to be appended to the response
                for the given request.

        Notes:
            - If req_id doesn't exist, a new empty sequence will be initialized.
            - If token_ids is a single integer, it's added as a single token.
            - If token_ids is a sequence, all tokens in the sequence are added.
        """
        # Thread-safe tree creation
        if problem_id not in self._problem_tree:
            if self._dict_lock:
                with self._dict_lock:
                    if problem_id not in self._problem_tree:
                        self._problem_tree[problem_id] = SuffixTree(self._max_depth)
            else:
                self._problem_tree[problem_id] = SuffixTree(self._max_depth)
        
        assert problem_id in self._problem_tree
        tree = self._problem_tree[problem_id]
        
        if isinstance(token_ids, int):
            # Use thread-safe methods if enabled
            if self._thread_safe:
                tree.append_safe(0, token_ids)
                if req_id in self._prompt_trees:
                    self._prompt_trees[req_id].append_safe(0, token_ids)
            else:
                tree.append(0, token_ids)
                if req_id in self._prompt_trees:
                    self._prompt_trees[req_id].append(0, token_ids)
        else:
            # Use thread-safe methods if enabled
            if self._thread_safe:
                tree.extend_safe(0, token_ids)
                if req_id in self._prompt_trees:
                    self._prompt_trees[req_id].extend_safe(0, token_ids)
            else:
                tree.extend(0, token_ids)
                if req_id in self._prompt_trees:
                    self._prompt_trees[req_id].extend(0, token_ids)

    def prebuild_problemtree(
        self,
        seq_id: int,
        problem_id: Hashable,
        prompt_token_ids: Sequence[int],
        token_ids: Union[int | Sequence[int]],
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
                        self._problem_tree[problem_id] = SuffixTree(self._max_depth)
            else:
                self._problem_tree[problem_id] = SuffixTree(self._max_depth)
        
        tree = self._problem_tree[problem_id]
        
        # Use thread-safe methods if enabled (with GIL release)
        if self._thread_safe:
            tree.extend_safe(seq_id, prompt_token_ids)
            tree.extend_safe(seq_id, token_ids)
        else:
            tree.extend(seq_id, prompt_token_ids)
            tree.extend(seq_id, token_ids) 

    def speculate(
        self,
        req_id: Hashable,
        problem_id: Hashable,
        pattern: Sequence[int],
        max_spec_tokens: Optional[int] = None,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = -1,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
        use_cached_prompt: bool = True,
    ) -> SuffixSpecResult:
        """
        Speculates and returns the most likely continuation of a given token
        pattern using the request-specific prompt cache (if available) and the
        global cache of previous responses.

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
            use_cached_prompt (bool): If True, uses the cached prompt for the
                request in addition to the global cache of previous responses.
        
        Returns:
            The speculation result containing the most likely continuation
            tokens, their probabilities, and overall score.

        Raises:
            ValueError: If the prompt doesn't exist for the given req_id when
                use_cached_prompt is True, or if the pattern is invalid.
        """
        if use_cached_prompt and req_id not in self._prompt_trees:
            raise ValueError(f"Prompt does not exist for request '{req_id}'")
        # if problem_id not in self._problem_tree:
        #     raise ValueError(f"Prompt does not exist for problem '{problem_id}'")
        if not pattern:
            raise ValueError("Pattern must not be empty")


        if max_spec_tokens is None:
            max_spec_tokens = self.max_depth

        if len(pattern) > self._max_depth:
            pattern = pattern[-self._max_depth :]
        
        result = SuffixSpecResult()        
        if use_cached_prompt:
            prompt_tree = self._prompt_trees[req_id]
            # Use thread-safe speculate if available (though speculate is typically read-only)
            candidate = prompt_tree.speculate(
                pattern,
                max_spec_tokens,
                max_spec_factor,
                max_spec_offset,
                min_token_prob,
                use_tree_spec)
            result = SuffixSpecResult.from_candidate(candidate)
        else:
            result = SuffixSpecResult()

        # Thread-safe access to problem tree (no need for _safe method as speculate is read-only)
        if problem_id in self._problem_tree:
            problem_tree = self._problem_tree[problem_id]
            candidate = problem_tree.speculate(
                pattern,
                max_spec_tokens,
                max_spec_factor,
                max_spec_offset,
                min_token_prob,
                use_tree_spec)
        else:
            candidate = SuffixSpecResult()

        if candidate.score > result.score:
            result = SuffixSpecResult.from_candidate(candidate)
        return result

    # 🆕 Thread-safe parallel processing methods using C++ object-level locking
    
    def prebuild_problems_parallel(
        self, 
        problems_data: List[Tuple[Hashable, List[int], List[List[int]]]]
    ) -> dict:
        """
        Build multiple problem trees in parallel using C++ object-level locking + GIL release.
        
        This method utilizes:
        - Per-object C++ mutex for each SuffixTree  
        - GIL release during C++ execution
        - True parallel execution for different problem_ids
        - Automatic serialization for same problem_id operations
        
        Args:
            problems_data: List of (problem_id, prompt_tokens, sequences)
        
        Returns:
            Dictionary with performance statistics
        """
        if not self._thread_safe:
            # Fallback to serial processing
            return self._build_problems_serial(problems_data)
        
        if not problems_data:
            return {"total_problems": 0, "total_time": 0.0, "method": "no_data"}
        
        import hashlib
        
        start_time = time.perf_counter()
        
        # Group problems by thread based on hash (for load balancing)
        thread_groups = [[] for _ in range(self._max_threads)]
        for problem_id, prompt_tokens, sequences in problems_data:
            problem_hash = hashlib.md5(str(problem_id).encode()).hexdigest()
            thread_id = int(problem_hash, 16) % self._max_threads
            thread_groups[thread_id].append((problem_id, prompt_tokens, sequences))
        
        def process_thread_group(group_data):
            """Process a group of problems in one thread"""
            thread_start = time.perf_counter()
            processed = 0
            operations = 0
            
            for problem_id, prompt_tokens, sequences in group_data:
                for i, token_ids in enumerate(sequences):
                    seq_id = -i-1
                    # ⚡ Key: This calls our C++ thread-safe methods with GIL release!
                    self.prebuild_problemtree(seq_id, problem_id, prompt_tokens, token_ids)
                    operations += 2  # extend prompt + extend tokens
                processed += 1
                
            return {
                'processed': processed,
                'operations': operations, 
                'time': time.perf_counter() - thread_start
            }
        
        # Execute threads in parallel
        with ThreadPoolExecutor(max_workers=len([g for g in thread_groups if g])) as executor:
            futures = []
            for i, group in enumerate(thread_groups):
                if group:  # Only submit non-empty groups
                    future = executor.submit(process_thread_group, group)
                    futures.append(future)
            
            # Collect results with progress bar
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), 
                              desc="Building with C++ object-level locking"):
                results.append(future.result())
        
        # total_time = time.perf_counter() - start_time
        
        # # Aggregate statistics
        # total_processed = sum(r['processed'] for r in results)
        # total_operations = sum(r['operations'] for r in results)
        # thread_times = [r['time'] for r in results]
        
        # parallel_time = max(thread_times) if thread_times else 0.0
        # sequential_equivalent_time = sum(thread_times)  # Time if executed sequentially
        # theoretical_speedup = sequential_equivalent_time / parallel_time if parallel_time > 0 else 1.0
        # actual_speedup = sequential_equivalent_time / total_time if total_time > 0 else 1.0
        
        # print(f"🚀 C++ Object-Level Locking Results:")
        # print(f"  Total time: {total_time:.4f}秒")
        # print(f"  Parallel time: {parallel_time:.4f}秒")
        # print(f"  Processed problems: {total_processed}")
        # print(f"  Total operations: {total_operations}")
        # print(f"  Theoretical speedup: {theoretical_speedup:.2f}x")
        # print(f"  Actual speedup: {actual_speedup:.2f}x")
        # print(f"  Active threads: {len(results)}")
        # print(f"  ✅ Each SuffixTree had independent C++ mutex + GIL release")
        
        # return {
        #     "method": f"cpp_object_locking_{self._max_threads}",
        #     "total_problems": len(problems_data),
        #     "successful_problems": total_processed,
        #     "total_operations": total_operations,
        #     "total_time": total_time,
        #     "parallel_time": parallel_time,
        #     "theoretical_speedup": theoretical_speedup,
        #     "actual_speedup": actual_speedup,
        #     "active_threads": len(results),
        #     "thread_safe": True
        # }
    
    def _build_problems_serial(self, problems_data: List[Tuple[Hashable, List[int], List[List[int]]]]) -> dict:
        """Fallback serial processing method"""
        start_time = time.perf_counter()
        
        processed = 0
        operations = 0
        
        for problem_id, prompt_tokens, sequences in problems_data:
            for i, token_ids in enumerate(sequences):
                seq_id = -i-1
                self.prebuild_problemtree(seq_id, problem_id, prompt_tokens, token_ids)
                operations += 2
            processed += 1
        
        total_time = time.perf_counter() - start_time
        
        return {
            "method": "serial_fallback",
            "total_problems": len(problems_data),
            "successful_problems": processed,
            "total_operations": operations,
            "total_time": total_time,
            "parallel_time": total_time,
            "theoretical_speedup": 1.0,
            "actual_speedup": 1.0,
            "active_threads": 1,
            "thread_safe": False
        }
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        stats = {
            "max_depth": self._max_depth,
            "thread_safe": self._thread_safe,
            "max_threads": self._max_threads,
            "problem_tree_count": len(self._problem_tree),
            "prompt_tree_count": len(self._prompt_trees)
        }
        
        if self._thread_safe:
            # Get thread-safe counts
            total_seqs = 0
            for tree in self._problem_tree.values():
                try:
                    total_seqs += tree.num_seqs_safe()
                except:
                    total_seqs += tree.num_seqs()
            stats["total_sequences"] = total_seqs
        else:
            total_seqs = sum(tree.num_seqs() for tree in self._problem_tree.values())
            stats["total_sequences"] = total_seqs
        
        return stats
