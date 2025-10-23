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

import contextlib
import copy
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union, Optional, TYPE_CHECKING, Hashable
from itertools import tee
from datetime import datetime
import os
import json
import re
import numpy as np
import torch
from transformers import AutoTokenizer
import vllm.distributed.parallel_state as parallel_state
import vllm.envs as envs
from tqdm import tqdm
from vllm.attention.layer import Attention
from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationLevel
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (get_pp_group, get_tp_group,
                                             is_global_first_rank)
from vllm.forward_context import set_forward_context
from vllm.config import VllmConfig
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.utils import round_up
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import MAX_SPEC_LEN, RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_model_runner import GPUModelRunner, logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

from arctic_inference.common.suffix_cache import SuffixCache
from arctic_inference.patching import ArcticPatch
from arctic_inference.vllm.spec_dec.arctic_proposer import ArcticProposer
from arctic_inference.common.suffix_cache import SuffixSpecResult

SP_TP_MODE = None


@contextlib.contextmanager
def set_shift_parallel_mode(mode: Optional[bool]):
    if mode is None:
        yield
        return

    global SP_TP_MODE

    if not is_shift_parallel_mode():
        assert not parallel_state._TP_STATE_PATCHED
        parallel_state._ORIG_TP = parallel_state._TP

    old_mode = SP_TP_MODE
    old_tp_group = parallel_state.get_tp_group()
    SP_TP_MODE = mode

    parallel_state._TP = (parallel_state._SP_TP if mode
                          else parallel_state._ORIG_TP)

    try:
        yield
    finally:
        # restore the original state
        SP_TP_MODE = old_mode
        parallel_state._TP = old_tp_group


def is_shift_parallel_mode() -> bool:
    """Check if the shift parallel mode is enabled."""
    global SP_TP_MODE
    return SP_TP_MODE is True


# Thread-local storage for problem_ids context
_problem_id_context = threading.local()


class ProblemIdContextManager:
    """Context manager for problem_ids with req_id mapping support."""
    
    @staticmethod
    def set_current_batch_problem_ids(problem_ids: list[Optional[str]]):
        """Set problem_ids for the current batch."""
        if not hasattr(_problem_id_context, 'data'):
            _problem_id_context.data = {}
        _problem_id_context.data['problem_ids'] = problem_ids
    
    @staticmethod
    def get_current_batch_problem_ids() -> list[Optional[str]]:
        """Get problem_ids for the current batch."""
        assert _problem_id_context.data.get('problem_ids') is not None, "problem_ids not found in _problem_id_context.data"
        return _problem_id_context.data.get('problem_ids', [])
    
    @staticmethod
    def set_req_id_to_problem_id_mapping(mapping: dict[str, Optional[str]]):
        """Set req_id to problem_id mapping."""
        if not hasattr(_problem_id_context, 'data'):
            _problem_id_context.data = {}
        _problem_id_context.data['req_id_to_problem_id'] = mapping
    
    @staticmethod
    def get_req_id_to_problem_id_mapping() -> dict[str, Optional[str]]:
        """Get the req_id to problem_id mapping."""
        assert _problem_id_context.data.get('req_id_to_problem_id') is not None, "req_id_to_problem_id not found in _problem_id_context.data"
        return _problem_id_context.data.get('req_id_to_problem_id', {})
    
    # @staticmethod
    # def get_problem_id_for_index(index: int) -> Optional[str]:
    #     """Get problem_id for a specific index."""
    #     if not hasattr(_problem_id_context, 'data'):
    #         return None
        
    #     problem_ids = _problem_id_context.data.get('problem_ids', [])
    #     if 0 <= index < len(problem_ids):
    #         return problem_ids[index]
    #     return None
    
    @staticmethod
    def get_problem_id_for_req_id(req_id: str) -> Optional[str]:
        """Get problem_id for a specific req_id."""
        assert _problem_id_context.data.get('req_id_to_problem_id') is not None, "req_id_to_problem_id not found in _problem_id_context.data"
        mapping = _problem_id_context.data.get('req_id_to_problem_id')
        return mapping.get(req_id)
    
    @staticmethod
    def clear_context():
        """Clear the current context."""
        if hasattr(_problem_id_context, 'data'):
            _problem_id_context.data = {}

    
    @staticmethod
    def set_hard_medium_ids(hard_ids: list[Optional[str]],
                            medium_ids: list[Optional[str]],
                            easy_ids: list[Optional[str]]):
        """Store hard and medium problem_id lists for the current batch."""
        if not hasattr(_problem_id_context, 'data'):
            _problem_id_context.data = {}
        _problem_id_context.data['hard_ids'] = hard_ids
        _problem_id_context.data['medium_ids'] = medium_ids
        _problem_id_context.data['easy_ids'] = easy_ids
    
    @staticmethod
    def get_hard_medium_ids() -> tuple[list[Optional[str]], list[Optional[str]], list[Optional[str]]]:
        """Get (hard_ids, medium_ids) tuple for the current batch."""
        assert _problem_id_context.data.get('hard_ids') is not None, "hard_ids not found in _problem_id_context.data"
        assert _problem_id_context.data.get('medium_ids') is not None, "medium_ids not found in _problem_id_context.data"
        assert _problem_id_context.data.get('easy_ids') is not None, "easy_ids not found in _problem_id_context.data"
        return (
            _problem_id_context.data.get('hard_ids', []),
            _problem_id_context.data.get('medium_ids', []),
            _problem_id_context.data.get('easy_ids', []),
        )

    @staticmethod
    def set_hard_medium_indices(hard_indices: list[int], medium_indices: list[int], easy_indices: list[int], allowed_indices: list[int]):
        """Store precomputed hard/medium request indices for the current batch."""
        if not hasattr(_problem_id_context, 'data'):
            _problem_id_context.data = {}
        _problem_id_context.data['hard_indices'] = list(hard_indices or [])
        _problem_id_context.data['medium_indices'] = list(medium_indices or [])
        _problem_id_context.data['easy_indices'] = list(easy_indices or [])
        _problem_id_context.data['allowed_indices'] = list(allowed_indices or [])
        _problem_id_context.data['has_hm_indices'] = True
    
    @staticmethod
    def get_hard_medium_indices() -> tuple[list[int], list[int], list[int], list[int]]:
        """Get stored (hard_indices, medium_indices) or empty lists if none."""
        assert _problem_id_context.data.get('hard_indices') is not None, "hard_indices not found in _problem_id_context.data"
        assert _problem_id_context.data.get('medium_indices') is not None, "medium_indices not found in _problem_id_context.data"
        assert _problem_id_context.data.get('easy_indices') is not None, "easy_indices not found in _problem_id_context.data"
        assert _problem_id_context.data.get('allowed_indices') is not None, "allowed_indices not found in _problem_id_context.data"
        return (
            _problem_id_context.data.get('hard_indices', []) or [],
            _problem_id_context.data.get('medium_indices', []) or [],
            _problem_id_context.data.get('easy_indices', []) or [],
            _problem_id_context.data.get('allowed_indices', []) or [],
        )
    
    @staticmethod
    def has_hard_medium_indices() -> bool:
        """Whether hard/medium indices have been set for the current batch."""
        if not hasattr(_problem_id_context, 'data'):
            return False
        return bool(_problem_id_context.data.get('has_hm_indices', False))
    
    # @staticmethod
    # def clear_hard_medium_indices():
    #     """Clear stored hard/medium indices only."""
    #     if hasattr(_problem_id_context, 'data'):
    #         _problem_id_context.data.pop('hard_indices', None)
    #         _problem_id_context.data.pop('medium_indices', None)
    #         _problem_id_context.data.pop('has_hm_indices', None)
    
    @staticmethod
    def set_current_step(current_step: Optional[int]):
        """Set current training step."""
        if not hasattr(_problem_id_context, 'data'):
            _problem_id_context.data = {}
        _problem_id_context.data['current_step'] = current_step
    
    @staticmethod
    def get_current_step() -> Optional[int]:
        """Get current training step."""
        return _problem_id_context.data.get('current_step')
    
    
    @staticmethod
    @contextlib.contextmanager
    def batch_context(problem_ids: list[Optional[str]]):
        """Context manager for a batch of problem_ids."""
        try:
            ProblemIdContextManager.set_current_batch_problem_ids(problem_ids)
            yield
        finally:
            ProblemIdContextManager.clear_context()


def extract_problem_id_from_prompt(prompt) -> Optional[str]:
    """Extract problem_id from a prompt object.
    
    This function should be customized based on how problem_id is embedded in prompts.
    Current implementation supports vLLMRollout's prompt format.
    """
    try:
        # Method 1: Direct problem_id field in dict (vLLMRollout format)
        if isinstance(prompt, dict) and 'problem_id' in prompt:
            return prompt['problem_id']
        
        # Method 2: If prompt is a dict with prompt_token_ids and problem_id fields
        if isinstance(prompt, dict):
            # Check for vLLM input format: {"prompt_token_ids": [...], "problem_id": "..."}
            if 'problem_id' in prompt:
                return prompt['problem_id']
            
            # Check for meta field containing problem_id
            if 'meta' in prompt:
                meta = prompt['meta']
                if isinstance(meta, dict) and 'problem_id' in meta:
                    return meta['problem_id']
        
        # Method 3: If prompt string contains problem_id pattern
        if isinstance(prompt, str):
            import re
            # Pattern: problem_id:value
            match = re.search(r'problem_id:(\w+)', prompt)
            if match:
                return match.group(1)
            
            # Pattern: [PROBLEM_ID: value]
            match = re.search(r'\[PROBLEM_ID:\s*(\w+)\]', prompt)
            if match:
                return match.group(1)
        
        # Method 4: Handle TextPrompt or other prompt types
        if hasattr(prompt, 'problem_id'):
            return prompt.problem_id
        
        # Method 5: Handle nested structures
        if hasattr(prompt, 'get'):
            return prompt.get('problem_id')
            
        return None
    except Exception:
        return None



class GPUModelRunnerPatch(ArcticPatch[GPUModelRunner]):

    _orig_initialize_kv_cache = GPUModelRunner.initialize_kv_cache
    _orig_prepare_inputs = GPUModelRunner._prepare_inputs
    _orig_profile_run = GPUModelRunner.profile_run
    _orig_load_model = GPUModelRunner.load_model
    _orig_propose_draft_token_ids = GPUModelRunner.propose_draft_token_ids
    _orig_init = GPUModelRunner.__init__

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        # Ulysses sequence parallelism
        if vllm_config.parallel_config.ulysses_sequence_parallel_size > 1:
            self.use_ulysses = True
            pass_config = vllm_config.compilation_config.pass_config
            if pass_config.enable_sequence_parallelism:
                raise ValueError(
                    "Ulysses sequence parallelism is incompatible with native "
                    "sequence parallelism. Set enable_sequence_parallelism "
                    "to False in the pass config to use Ulysses.")
        else:
            self.use_ulysses = False

        # Speculative decoding
        # TODO: Use "arctic" as an umbrella method that also covers the Arctic
        # Inverence version of "mlp_speculator".
        if (vllm_config.speculative_config is not None and \
                vllm_config.speculative_config.method in (
                    "arctic", "suffix", "mlp_speculator")):
            # Delay the creation of the drafter until
            # after the child class has been initialized.
            arctic_speculative_config = vllm_config.speculative_config
            vllm_config.speculative_config = None
        else:
            arctic_speculative_config = None

        self._orig_init(vllm_config, device)

        # Set up speculative decoding.
        self._suffix_cache = None
        # 🚀 Initialize persistent thread pool for suffix speculation
        self._speculation_threadpool = None
        self._speculation_max_workers = 8
        
        if arctic_speculative_config is not None:
            # Restore the speculative config.
            self.vllm_config.speculative_config = arctic_speculative_config
            self.speculative_config = arctic_speculative_config

            if get_pp_group().is_last_rank:
                if (self.speculative_config.method == "arctic" or
                      self.speculative_config.method == "mlp_speculator"):
                    self.drafter = ArcticProposer(self.vllm_config)
                elif self.speculative_config.method != "suffix":
                    raise ValueError("Unknown speculative decoding method: "
                                     f"{self.speculative_config.method}")

                self.rejection_sampler = RejectionSampler()

        # if (self.speculative_config is not None and
        #         self.speculative_config.enable_suffix_decoding):
        #     if self.speculative_config.method not in (
        #             "arctic", "suffix", "mlp_speculator"):
        #         raise ValueError(
        #             "Suffix decoding is only supported with the 'arctic', "
        #             "'mlp_speculator' or 'suffix' spec decoding methods.")
        #     self._suffix_cache = SuffixCache(self.speculative_config.suffix_cache_max_depth)
            # Optionally bootstrap the suffix cache with provided sequences so
            # that multiple LLM instances can share the same logical suffix tree
            # contents without passing non-picklable objects across processes.
# #    ---- Timing buffered writer (to reduce I/O) ----
        import os as _os
        import time as _time
        import atexit as _atexit

        # Buffer config via env with sensible defaults
        self._gpu_timing_buffer: list[str] = []
        self._cpu_timing_buffer: list[str] = []
        self._timing_flush_every_n: int = int(_os.getenv("ARCTIC_TIMING_BUFFER_SIZE", "400"))
        self._timing_flush_every_s: float = float(_os.getenv("ARCTIC_TIMING_FLUSH_SEC", "5"))
        self._timing_last_flush_time: dict[str, float] = {
            "GPU_execution_time": _time.monotonic(),
            "CPU_execution_time": _time.monotonic()
        }

        # Precompute output path for this process
        root_dir = _os.getenv("ARCTIC_METRICS_DIR", "/tmp/arctic_metrics")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _os.path.join(root_dir, timestamp)
        _os.makedirs(output_dir, exist_ok=True)
        # Propagate to env so other writers use the same timestamped directory
        _os.environ["ARCTIC_METRICS_DIR"] = output_dir
        local_rank = _os.getenv("LOCAL_RANK", "0")
        rank = _os.getenv("RANK", "0")
        
        # Store base paths for dynamic file generation with current_step
        self._output_dir = output_dir
        self._rank = rank
        self._local_rank = local_rank
        
        # Keep old paths for backward compatibility (when current_step is not available)
        self._timing_file_path_gpu = _os.path.join(
            output_dir, f"GPU_execution_timing_rank_{rank}_local_{local_rank}.jsonl"
        )
        self._timing_file_path_cpu = _os.path.join(
            output_dir, f"CPU_execution_timing_rank_{rank}_local_{local_rank}.jsonl"
        )

        # Ensure buffer flushes on process exit for both GPU and CPU timing
        _atexit.register(lambda: self._flush_timing_buffer(force=True, doc_type="GPU_execution_time"))
        _atexit.register(lambda: self._flush_timing_buffer(force=True, doc_type="CPU_execution_time"))
        
        # ---- Suffix-tree stats buffered writer (to reduce I/O) ----
        self._suffix_buffer: list[str] = []
        self._suffix_flush_every_n: int = int(_os.getenv("ARCTIC_SUFFIX_BUFFER_SIZE", "400"))
        self._suffix_flush_every_s: float = float(_os.getenv("ARCTIC_SUFFIX_FLUSH_SEC", "5"))
        self._suffix_last_flush_time: float = _time.monotonic()
        self._suffix_file_path = _os.path.join(
            output_dir, f"suffix_tree_stats_rank_{rank}_local_{local_rank}.jsonl"
        )
        _atexit.register(lambda: self._flush_suffix_buffer(force=True))
        
        # ---- Suffix speculation timing stats buffered writer ----
        self._suffix_timing_buffer: list[str] = []
        self._suffix_timing_flush_every_n: int = int(_os.getenv("ARCTIC_SUFFIX_TIMING_BUFFER_SIZE", "400"))
        self._suffix_timing_flush_every_s: float = float(_os.getenv("ARCTIC_SUFFIX_TIMING_FLUSH_SEC", "5"))
        self._suffix_timing_last_flush_time: float = _time.monotonic()
        self._suffix_timing_file_path = _os.path.join(
            output_dir, f"suffix_speculation_timing_rank_{rank}_local_{local_rank}.jsonl"
        )
        _atexit.register(lambda: self._flush_suffix_timing_buffer(force=True))
        
        # Initialize tokenizer for text conversion
        self._tokenizer = None
        try:
            # Try to get tokenizer name from model config or use a default
            tokenizer_name = getattr(vllm_config.model_config, 'tokenizer', None) or getattr(vllm_config.model_config, 'model', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Initialized tokenizer: {tokenizer_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}. Text output will be disabled.")
            self._tokenizer = None

    def profile_run(self) -> None:
        self._orig_profile_run()
        if self.shift_model is not None:
            # Run the shift model to trigger compilation.
            orig_model, self.model = self.model, self.shift_model
            try:
                with set_shift_parallel_mode(True):
                    self._dummy_run(self.max_num_tokens, is_profile=True)
            finally:
                self.model = orig_model

    def _prepare_inputs(self, *args, **kwargs):
        attn_metadata, attention_cuda_graphs, logits_indices, *rest = (
            self._orig_prepare_inputs(*args, **kwargs))
        # SwiftKV requires knowing the logits indices from inside the model
        # definition in order to early-stop the prefill tokens.
        for meta in attn_metadata.values():
            meta.swiftkv_logits_indices = logits_indices
        return attn_metadata, attention_cuda_graphs, logits_indices, *rest

    def monkeypatch_forward(self: GPUModelRunner):
        sp_size = parallel_state._SP.world_size
        sp_rank = parallel_state._SP.rank_in_group
        device_group = parallel_state._SP.device_group
        model_forward = self.model.forward
        input_key = 'inputs_embeds' if  self.is_multimodal_model else 'input_ids'

        def ulysses_forward(*args, **kwargs):
            # update inputs
            input_tensor = kwargs[input_key]
            positions = kwargs['positions']
            # Ulysses parameters
            N = input_tensor.shape[0]

            N_ulysses = N // sp_size
            N_offset = N_ulysses * sp_rank

            # narrow the input
            kwargs[input_key] = input_tensor[N_offset:N_offset + N_ulysses]
            kwargs['positions'] = positions[N_offset:N_offset + N_ulysses]

            with set_shift_parallel_mode(False):
                output = model_forward(*args, **kwargs)

            if output.size(0) == N_ulysses:
                # all-gather model_output
                model_output = torch.empty((N, self.hidden_size),
                                        dtype=output.dtype,
                                        device=output.device)
                torch.distributed.all_gather_into_tensor(model_output,
                                                        output,
                                                        group=device_group)
            else:
                # SwiftKV models will already have all-gathered the output.
                assert output.size(0) == N
                model_output = output
            return model_output

        self.model.forward = ulysses_forward

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:

        # Get process information for data parallel scenarios
        local_rank = os.getenv("LOCAL_RANK", "0")
        world_size = os.getenv("WORLD_SIZE", "1")
        rank = os.getenv("RANK", "0")
        
        self._update_states(scheduler_output)

        # torch.cuda.synchronize()
        # execution_start_time = time.perf_counter()
        # execution_start_timestamp = datetime.now().isoformat()
        
        # Extract problem_ids for the current batch at the very beginning
        # Build req_id to problem_id mapping for the current batch
        self._current_batch_req_id_to_problem_id = {}
        self._current_batch_problem_ids = []
        
        if hasattr(self.input_batch, 'req_ids') and self.input_batch.req_ids:
            batch_size = len(self.input_batch.req_ids)
            
            # Build mapping from req_id to problem_id
            # Get req_id to problem_id mapping from LLM patches (most reliable)
            context_mapping = ProblemIdContextManager.get_req_id_to_problem_id_mapping()
            #print(f"DEBUG: context_mapping: {context_mapping}")
            
            for i, req_id in enumerate(self.input_batch.req_ids):
                problem_id = None
                
                # Method 1: Use mapping from LLM patches (most reliable)
                if req_id in context_mapping:
                    problem_id = context_mapping[req_id]
                
                # Method 2: Fallback to context manager index lookup
                if problem_id is None:
                    print(f"DEBUG: problem_id is None for req_id {req_id}")
                
                self._current_batch_req_id_to_problem_id[req_id] = problem_id
                self._current_batch_problem_ids.append(problem_id)
                
            # Log the mapping for debugging (optional)
            # print(f"DEBUG: execute_model batch mapping: {self._current_batch_req_id_to_problem_id}")
        else:
            batch_size = 0

        
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output)

        # Prepare the decoder inputs.
        (attn_metadata, attention_cuda_graphs, logits_indices,
         spec_decode_metadata,
         num_scheduled_tokens_np) = (self._prepare_inputs(scheduler_output))
        batch_size = len(self.input_batch.req_ids) if hasattr(self, 'input_batch') and self.input_batch else 0

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        use_shift_model = (
            self.use_ulysses and self.shift_model is not None and
            num_scheduled_tokens <= self.shift_parallel_threshold)
        if self.use_ulysses and not use_shift_model:
            # add padding to the batch size to make it a multiple of SP
            sp_size = self.parallel_config.ulysses_sequence_parallel_size
            num_input_tokens = round_up(num_scheduled_tokens, sp_size)
            if (self.use_cuda_graph and num_input_tokens // sp_size
                    <= self.cudagraph_batch_sizes[-1]):
                num_input_tokens = self.vllm_config.pad_for_cudagraph(
                    num_input_tokens // sp_size) * sp_size
        elif (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            # Pad tokens to multiple of tensor_parallel_size when
            # enabled collective fusion for SP
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if self.compilation_config.pass_config. \
                enable_sequence_parallelism and tp_size > 1:
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        if self.is_multimodal_model and get_pp_group().is_first_rank:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        # Some attention backends only support CUDA Graphs in pure decode.
        # If attention doesn't support CUDA Graphs for this batch, but we
        # compiled with full CUDA graphs, we have to skip them entirely.
        skip_cuda_graphs = self.full_cuda_graph and not attention_cuda_graphs

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                skip_cuda_graphs=skip_cuda_graphs,
        ):
            self.maybe_setup_kv_connector(scheduler_output)

            # # ### Record GPU execution start time for monitoring
            torch.cuda.synchronize()
            execution_start_time = time.perf_counter()
            execution_start_timestamp = datetime.now().isoformat()
            
            model = self.shift_model if use_shift_model else self.model
            with set_shift_parallel_mode(use_shift_model):
                model_output = model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = (
                self.get_finished_kv_transfers(scheduler_output))

        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = None

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        broadcast_pp_output = \
            self.parallel_config.distributed_executor_backend \
            == "external_launcher" and len(get_pp_group().ranks) > 0
        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            if not broadcast_pp_output:
                return hidden_states
            assert isinstance(hidden_states, IntermediateTensors)
            get_pp_group().send_tensor_dict(hidden_states.tensors,
                                            all_gather_group=get_tp_group())
            logits = None
        else:
            if self.input_batch.pooling_params:
                return self._pool(hidden_states, num_scheduled_tokens,
                                  num_scheduled_tokens_np, finished_sending,
                                  finished_recving)

            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
        if broadcast_pp_output:
            model_output_broadcast_data = {
                "logits": logits.contiguous(),
            } if logits is not None else {}
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        draft_token_ids = None
        num_draft_tokens = None
        cu_num_draft_tokens = None
        if spec_decode_metadata is None:
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            
            assert logits is not None
            draft_token_ids = spec_decode_metadata.draft_token_ids
            num_draft_tokens = spec_decode_metadata.num_draft_tokens
            cu_num_draft_tokens = spec_decode_metadata.cu_num_draft_tokens
            # print("DEBUG:num_draft_tokens: ", num_draft_tokens)
            # print("DEBUG:draft_token_ids: ", len(draft_token_ids) if draft_token_ids is not None else 0)

            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                None,  # draft_probs
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids

        # #### Record GPU execution end time after all GPU computations are complete
        torch.cuda.synchronize()
        execution_end_time = time.perf_counter()
        execution_duration = execution_end_time - execution_start_time
        self._log_execution_time(execution_start_timestamp, execution_duration, batch_size, 
                                scheduler_output.total_num_scheduled_tokens, early_return=False, doc_type="GPU_execution_time")

        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                # Rewind the generator state as if the token was not sampled.
                # This relies on cuda-specific torch-internal impl details
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)


        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors                                
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output,
        )


        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )

        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        ### profiling suffix_tree_stats: now may have bug 
        self._log_suffix_tree_stats(num_draft_tokens, draft_token_ids, cu_num_draft_tokens, valid_sampled_token_ids)

        #print(f"DEBUG: sampled_token_ids: {sampled_token_ids}")
            
        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            if end_idx > self.max_model_len:
                end_idx = self.max_model_len
                sampled_ids = sampled_ids[:self.max_model_len - start_idx]
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)


        # ### profiling suffix tree decoding (CPU execution time)
        torch.cuda.synchronize()
        cpu_execution_start_time = time.perf_counter()
        cpu_execution_start_timestamp = datetime.now().isoformat()
        

        if self._suffix_cache is not None:
            self._update_suffix_cache(valid_sampled_token_ids)
        if not self.speculative_config:
            # Speculative decoding is not enabled.
            spec_token_ids = None
        else:
            # problem: propose too much token & sequentially, too much time
            spec_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                valid_sampled_token_ids,
                sampler_output.sampled_token_ids,
                sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                attn_metadata,
            )
                       
            # Determine whether to enable confidence-based filtering of spec tokens.
            confidence_based_only = bool(getattr(self.speculative_config, "confidence_based_only", False))
            distribution_aware = bool(getattr(self.speculative_config, "distribution_aware", False))
            # # # # # # 统计和控制 spec_token 数量
            if spec_token_ids is not None and confidence_based_only:
                # 统计总的 spec_token 数量
                total_spec_tokens = sum(len(tokens) for tokens in spec_token_ids if tokens is not None)
                
                # 如果数量超过128，按比例删除一部分
                if total_spec_tokens > 300:
                    # 计算需要保留的比例
                    keep_ratio = 300 / total_spec_tokens
                    
                    # 对每个子列表按比例保留 tokens
                    filtered_spec_token_ids = []
                    for tokens in spec_token_ids:
                        if tokens is not None and len(tokens) > 0:
                            # 计算当前子列表需要保留的数量
                            keep_count = max(1, int(len(tokens) * keep_ratio))  # 至少保留1个
                            # 保留前 keep_count 个 tokens
                            filtered_tokens = tokens[:keep_count]
                            filtered_spec_token_ids.append(filtered_tokens)
                        else:
                            filtered_spec_token_ids.append(tokens)
                    
                    spec_token_ids = filtered_spec_token_ids
                

            # # # # 统计和控制 spec_token 数量（优先 hard，再分配 medium）
            # if spec_token_ids is not None and distribution_aware:
            #     hard_indices, medium_indices = self._get_hard_and_non_hard_indices()

            #     # 先按 hard 分配，再将剩余分配给 medium，配额总量为 200
            #     spec_token_ids = self._apply_quota_to_spec_tokens(
            #         spec_token_ids=spec_token_ids,
            #         hard_indices=hard_indices,
            #         non_hard_indices=medium_indices,
            #         quota=1000,
            #     )
            torch.cuda.synchronize()
            cpu_execution_end_time = time.perf_counter()
            cpu_execution_duration = cpu_execution_end_time - cpu_execution_start_time
            self._log_execution_time(cpu_execution_start_timestamp, cpu_execution_duration, batch_size, 
                                    scheduler_output.total_num_scheduled_tokens, early_return=False, doc_type="CPU_execution_time")

        # Clear KVConnector state after all KVs are generated.
        if has_kv_transfer_group():
            get_kv_transfer_group().clear_connector_metadata()

        # # self.eplb_step()
        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            num_nans_in_logits=num_nans_in_logits,
        )

    def _get_problem_id_for_index(self, index: int) -> Optional[str]:
        """Retrieve problem_id for the i-th request if available.

        Sources:
        - `self.input_batch.problem_id` propagated from rollout layer
        - Per-request state containers
        - Fallback: regex extract like "prob_0001" from request id
        """
        req_id = None
        try:
            req_id = self.input_batch.req_ids[index]
        except Exception:
            pass

        # First try context manager (highest priority)
        try:
            problem_id = ProblemIdContextManager.get_problem_id_for_index(index)
            if problem_id is not None:
                return problem_id
        except Exception:
            pass

        # Try input_batch vectorized field first
        try:
            problem_ids = getattr(self.input_batch, "problem_id", None)
            if problem_ids is not None:
                pid = problem_ids[index]
                if isinstance(pid, bytes):
                    pid = pid.decode()
                # numpy scalar -> python scalar
                if hasattr(pid, "item"):
                    pid = pid.item()
                if isinstance(pid, (list, np.ndarray)):
                    pid = pid[0] if len(pid) > 0 else None
                if isinstance(pid, str):
                    return pid
        except Exception:
            pass

        # Try request state attributes

        if req_id is not None and req_id in self.requests:
            req_state = self.requests[req_id]
            pid = getattr(req_state, "problem_id", None)
            if isinstance(pid, bytes):
                pid = pid.decode()
            if isinstance(pid, str):
                return pid
            for container_name in ("inputs", "input", "meta", "meta_info", "request_kwargs", "extra", "extras"):
                container = getattr(req_state, container_name, None)
                if isinstance(container, dict) and "problem_id" in container:
                    pid = container.get("problem_id")
                    if isinstance(pid, bytes):
                        pid = pid.decode()
                    if isinstance(pid, str):
                        return pid

        # Fallback: extract a common pattern from req_id
        if req_id is not None:
            m = re.search(r"(prob_[0-9]{4,})", str(req_id))
            if m:
                return m.group(1)
        return None

    def _get_hard_and_non_hard_indices(self, top_percent: float = 0.3) -> tuple[list[int], list[int]]:
        """
        计算当前 batch 中的 hard 和 medium 请求索引。

        基于 ProblemIdContextManager.get_hard_medium_ids() 提供的 problem_id 列表，
        将属于 hard_ids 的请求划为 hard，将属于 medium_ids 的请求划为 non-hard（此处表示 medium）。
        其他未列入者不分配配额。
        """
        # 优先使用已缓存的索引，以避免在 generate_sequences 的多次 execute_model 调用中重复计算
        if ProblemIdContextManager.has_hard_medium_indices():
            cached_hard, cached_medium, cached_easy, cached_allowed_indices = ProblemIdContextManager.get_hard_medium_indices()
            return cached_hard, cached_medium, cached_easy, cached_allowed_indices
        else:
            hard_indices: list[int] = []
            medium_indices: list[int] = []  
            easy_indices: list[int] = []
            hard_ids, medium_ids,easy_ids = ProblemIdContextManager.get_hard_medium_ids()
            hard_set = set(str(pid) for pid in (hard_ids or []))
            medium_set = set(str(pid) for pid in (medium_ids or []))
            easy_set = set(str(pid) for pid in (easy_ids or []))
            for i, req_id in enumerate(self.input_batch.req_ids):
                problem_id = self._current_batch_req_id_to_problem_id.get(req_id)
                pid_str = str(problem_id)
                if pid_str in hard_set:
                    hard_indices.append(i)
                elif pid_str in medium_set:
                    medium_indices.append(i)
                elif pid_str in easy_set:
                    easy_indices.append(i)
            allowed_indices = set(hard_indices) | set(medium_indices) | set(easy_indices)

            ProblemIdContextManager.set_hard_medium_indices(hard_indices, medium_indices, easy_indices, allowed_indices)
        
        return hard_indices, medium_indices, easy_indices, allowed_indices

    def _apply_quota_to_spec_tokens(
        self,
        spec_token_ids: list,
        hard_indices: list[int],
        non_hard_indices: list[int],
        quota: int = 200,
    ) -> list:
        """
        在给定配额下调整 `spec_token_ids`：
        - 优先保留 hard 请求；
        - 若 hard 请求总 tokens 超额，则按比例削减 hard，并删除所有 non-hard；
        - 否则，将剩余配额按比例分配给 non-hard。
        """
        # 统计 hard 请求的 tokens 数量
        hard_spec_tokens = 0
        for i in hard_indices:
            if i < len(spec_token_ids) and spec_token_ids[i] is not None:
                hard_spec_tokens += len(spec_token_ids[i])

        if hard_spec_tokens > quota:
            keep_ratio = quota / max(1, hard_spec_tokens)
            filtered_spec_token_ids = []
            for i in range(len(spec_token_ids)):
                if i in hard_indices:
                    if spec_token_ids[i] is not None and len(spec_token_ids[i]) > 0:
                        keep_count = max(1, int(len(spec_token_ids[i]) * keep_ratio))
                        filtered_tokens = spec_token_ids[i][:keep_count]
                        filtered_spec_token_ids.append(filtered_tokens)
                    else:
                        filtered_spec_token_ids.append(spec_token_ids[i])
                else:
                    filtered_spec_token_ids.append([])
            return filtered_spec_token_ids

        remaining_quota = quota - hard_spec_tokens

        # 统计 non-hard 请求的 tokens 数量
        non_hard_spec_tokens = 0
        for i in non_hard_indices:
            if i < len(spec_token_ids) and spec_token_ids[i] is not None:
                non_hard_spec_tokens += len(spec_token_ids[i])

        if non_hard_spec_tokens > 0 and remaining_quota > 0:
            non_hard_ratio = min(1.0, remaining_quota / max(1, non_hard_spec_tokens))
            filtered_spec_token_ids = []
            for i in range(len(spec_token_ids)):
                if i in hard_indices:
                    filtered_spec_token_ids.append(spec_token_ids[i])
                else:
                    if spec_token_ids[i] is not None and len(spec_token_ids[i]) > 0:
                        keep_count = max(0, int(len(spec_token_ids[i]) * non_hard_ratio))
                        if keep_count > 0:
                            filtered_tokens = spec_token_ids[i][:keep_count]
                            filtered_spec_token_ids.append(filtered_tokens)
                        else:
                            filtered_spec_token_ids.append([])
                    else:
                        filtered_spec_token_ids.append(spec_token_ids[i])
            return filtered_spec_token_ids

        if remaining_quota <= 0:
            filtered_spec_token_ids = []
            for i in range(len(spec_token_ids)):
                if i in hard_indices:
                    filtered_spec_token_ids.append(spec_token_ids[i])
                else:
                    filtered_spec_token_ids.append([])
            return filtered_spec_token_ids

        return spec_token_ids

    def get_current_batch_problem_ids(self) -> list[Optional[str]]:
        """
        Get problem_ids for the current batch that were extracted at the beginning of execute_model.
        
        Returns:
            List of problem_ids for the current batch, with None for requests without problem_ids.
        """
        return getattr(self, '_current_batch_problem_ids', [])
    
    def get_problem_id_by_request_id(self, req_id: str) -> Optional[str]:
        """
        Get problem_id for a specific request ID.
        
        Args:
            req_id: The request ID to look up
            
        Returns:
            The problem_id if found, None otherwise
        """
        # Method 1: Use current batch mapping (most efficient and reliable)
        if hasattr(self, '_current_batch_req_id_to_problem_id'):
            problem_id = self._current_batch_req_id_to_problem_id.get(req_id)
            if problem_id is not None:
                return problem_id
        print(f"Failed to get problem_id for request {req_id}")
        return None

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: list[list[int]],
        original_sampled_token_ids: np.ndarray,
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: Optional[torch.Tensor],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        attn_metadata: dict[str, Any],
    ) -> list[list[int]]:
        #print('\n batchsize: ', len(self.input_batch.req_ids))
        disable_spec_decode = (
            self.speculative_config and
            self.speculative_config.disable_by_batch_size and
            len(self.input_batch.req_ids) > self.speculative_config.disable_by_batch_size
        )
        if disable_spec_decode:
            # No speculative decoding is enabled.
            return [[] for _ in sampled_token_ids]

        suffix_spec_token_ids = None
        new_sampled_token_ids = sampled_token_ids.copy()
        if self._suffix_cache is not None:
            results = self.propose_suffix_draft_token_ids(
                new_sampled_token_ids)
            suffix_spec_token_ids = []
            # The score is an estimate of the acceptance length. Thus, the
            # heuristic is to use the suffix decoded tokens if the score is
            # greater than the # of tokens we would speculate otherwise.
            min_score = (self.speculative_config.num_speculative_tokens
                         if self.speculative_config.method != "suffix" else 0)
            min_score = (0 if self.speculative_config.method == "suffix"
                         else self.speculative_config.num_speculative_tokens)
            for i, result in enumerate(results):
                if result.score >= min_score:
                    # Use suffix decoded tokens, disable other speculation
                    # methods for this request.
                    new_sampled_token_ids[i] = []
                    suffix_spec_token_ids.append(result.token_ids)
                else:
                    suffix_spec_token_ids.append([])

        spec_token_ids = None
        if self.speculative_config.method == "suffix":
            pass
        elif (self.speculative_config.method == "arctic" or 
              self.speculative_config.method == "mlp_speculator"):
            assert isinstance(self.drafter, ArcticProposer)
            previous_hidden_states = self.drafter.prepare_hidden_states(
                sample_hidden_states=sample_hidden_states,
                sampled_token_ids=original_sampled_token_ids,
                spec_decode_metadata=spec_decode_metadata,
            )
            spec_token_ids = self.propose_arctic_draft_token_ids(
                scheduler_output,
                new_sampled_token_ids, 
                previous_hidden_states=previous_hidden_states)
        else:
            spec_token_ids = self._orig_propose_draft_token_ids(
                scheduler_output,
                new_sampled_token_ids,
                sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                attn_metadata,
            )

        if spec_token_ids is None:
            spec_token_ids = suffix_spec_token_ids
        elif suffix_spec_token_ids is not None:
            spec_token_ids = [
                suffix_spec_token_ids[i] or spec_token_ids[i]
                for i in range(len(suffix_spec_token_ids))
            ]
        return spec_token_ids


    def _propose_arctic_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: list[list[int]],
        previous_hidden_states: Optional[torch.Tensor] = None,
    ) -> list[list[int]]:
        """Original serial implementation for fallback."""
        last_tokens : list[int] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            
            if (num_sampled_ids == 0):
                if self.speculative_config.enable_suffix_decoding:
                    return [[]] * len(sampled_token_ids)
                req_id = self.input_batch.req_ids[i]
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                sampled_ids = [req_state.get_token_id(seq_len)]

            # Add sampled_token_ids to token_ids_cpu.
            start_idx = self.input_batch.num_tokens_no_spec[i]
            end_idx = start_idx + num_sampled_ids
            self.input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids[-1]
            last_tokens.append(self.input_batch.token_ids_cpu[i, end_idx - 1])

        drafter_output = self.drafter.propose(
            last_tokens,
            previous_hidden_states=previous_hidden_states,
        )

        draft_token_ids = drafter_output.tolist()

        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                draft_token_ids[i] = []

        return draft_token_ids

    def _update_suffix_cache(self, sampled_token_ids: list[list[int]]) -> None:
        seen_req_ids = set()
        seen_problem_ids = set()
        
        # Check if distribution_aware mode is enabled
        distribution_aware = bool(getattr(self.speculative_config, "distribution_aware", False))
        
        # Get hard and medium indices if distribution_aware is enabled
        allowed_indices = None
        if distribution_aware:
            hard_indices, medium_indices, easy_indices, allowed_indices = self._get_hard_and_non_hard_indices()
        
        for i, sampled_ids in enumerate(sampled_token_ids):
            # Only update suffix cache for hard and medium problems when distribution_aware is enabled
            if distribution_aware and allowed_indices is not None and i not in allowed_indices:
                continue
            if not sampled_ids:
                continue
            req_id = self.input_batch.req_ids[i]
            problem_id = self.get_problem_id_by_request_id(req_id)
            seen_req_ids.add(req_id)
            seen_problem_ids.add(problem_id)
            
            index = self.input_batch.req_id_to_index[req_id]
            if not self._suffix_cache.has_cached_prompt(req_id):
                num_prompt_tokens = self.input_batch.num_prompt_tokens[index]
                prompt_token_ids = (
                    self.input_batch.token_ids_cpu[index, :num_prompt_tokens])
                #print(f"DEBUG: Caching prompt for req_id={req_id}, prompt_tokens={num_prompt_tokens}")
                self._suffix_cache.cache_prompt(req_id, prompt_token_ids)

            #print(f"DEBUG: Updating response for req_id={req_id} with {len(sampled_ids)} tokens")
            self._suffix_cache.update_response(req_id, problem_id,sampled_ids)

    def _process_task_batch_v2(self, task_batch):
        """
        🎯 优化：批处理多个speculation任务，增大并行粒度
        
        每个worker处理一批任务，减少线程调度和同步开销
        
        Args:
            task_batch: 一批任务数据的列表
            
        Returns:
            [(task_data, result), ...] 列表
        """
        from arctic_inference.common.suffix_cache import SuffixSpecResult
        
        batch_results = []
        
        for task_data in task_batch:
            # 解包任务数据
            (i, req_id, problem_id, pattern, spec_ids, config, end_idx, max_model_len, spec_len) = task_data
            
            # 准备pattern
            if len(pattern) > config.suffix_cache_max_depth:
                pattern = pattern[-config.suffix_cache_max_depth:]
            
            pattern = pattern + spec_ids
            if len(pattern) > config.suffix_cache_max_depth:
                pattern = pattern[-config.suffix_cache_max_depth:]
            
            # 计算参数
            max_spec_tokens = min(
                config.num_speculative_tokens if config.num_speculative_tokens is not None else config.suffix_cache_max_depth,
                MAX_SPEC_LEN - len(spec_ids),
                config.suffix_cache_max_depth,
                max_model_len - end_idx - 1,
                spec_len
            )
            
            max_spec_factor = config.suffix_max_spec_factor
            max_spec_offset = config.suffix_max_spec_offset - len(spec_ids) * (max_spec_factor + 1)
            
            # 🎯 使用预取的problem_tree对象
            result = self._suffix_cache.speculate(
            req_id,
            problem_id,
            pattern,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            max_spec_offset=max_spec_offset,
            min_token_prob=config.suffix_min_token_prob)
            
            batch_results.append((task_data, result))
        
        return batch_results
    
    def _process_single_speculation_v2(self, task_data):
        """Process a single speculation request with pre-extracted data (NO shared state access)"""
        import threading
        thread_id = threading.current_thread().ident
        start_time = time.perf_counter()
        
        (i, req_id, problem_id, pattern, spec_ids, config, end_idx, max_model_len) = task_data
        
        # All data is already extracted, no need to access self.input_batch!
        if len(pattern) > config.suffix_cache_max_depth:
            pattern = pattern[-config.suffix_cache_max_depth:]
        
        # Add spec_ids to pattern
        pattern = pattern + spec_ids
        if len(pattern) > config.suffix_cache_max_depth:
            pattern = pattern[-config.suffix_cache_max_depth:]
        
        max_spec_tokens = min(
            config.num_speculative_tokens if config.num_speculative_tokens is not None else config.suffix_cache_max_depth,
            MAX_SPEC_LEN - len(spec_ids),
            config.suffix_cache_max_depth,
            max_model_len - end_idx - 1
        )
        max_spec_tokens = 5
        
        max_spec_factor = config.suffix_max_spec_factor
        max_spec_offset = config.suffix_max_spec_offset - len(spec_ids) * (max_spec_factor + 1)
        
        #spec_start = time.perf_counter()
        result = self._suffix_cache.speculate(
            req_id,
            problem_id,
            pattern,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            max_spec_offset=max_spec_offset,
            min_token_prob=config.suffix_min_token_prob)
        # spec_time = (time.perf_counter() - spec_start) * 1000
        # total_time = (time.perf_counter() - start_time) * 1000
        
        # if total_time > 100:
        #     print(f"[Task] thread={thread_id}, req_id={req_id}, spec={spec_time:.0f}ms, total={total_time:.0f}ms")
        
        return result
    
    def _process_single_speculation(self, args):
        """Process a single speculation request for parallel execution (OLD VERSION with shared state)"""
        # import threading
        # thread_id = threading.current_thread().ident
        # start_time = time.perf_counter()
        
        (i, sampled_ids, spec_ids, config) = args
        
        num_sampled_ids = len(sampled_ids)
        if not num_sampled_ids:
            # Skip speculative decoding.
            return SuffixSpecResult()

        req_id = self.input_batch.req_ids[i]
        problem_id = self.get_problem_id_by_request_id(req_id)  # Method 1: Direct lookup
        if problem_id is None:
            print(f"problem_id is None for req_id={req_id}")

        # Add sampled_token_ids to token_ids_cpu.
        end_idx = self.input_batch.num_tokens_no_spec[i]
        # end_idx = start_idx + len(sampled_ids)

        if end_idx >= self.max_model_len:
            return SuffixSpecResult()

        size = min(end_idx, config.suffix_cache_max_depth)
        pattern = self.input_batch.token_ids_cpu[i, end_idx - size:end_idx]
        pattern = pattern.tolist() + spec_ids
        if len(pattern) > config.suffix_cache_max_depth:
            pattern = pattern[-config.suffix_cache_max_depth:]
        max_spec_tokens = min(config.num_speculative_tokens if config.num_speculative_tokens is not None else config.suffix_cache_max_depth,
                              MAX_SPEC_LEN - len(spec_ids),
                              config.suffix_cache_max_depth,
                              self.max_model_len - end_idx - 1)
        max_spec_tokens = 5
        # max_spec_offset is modified to mimic the behavior of the original
        # max_spec_factor and max_spec_offset as if the speculative tokens
        # were generated by suffix decoding. For example, if:
        #   - max_spec_factor = 2
        #   - max_spec_offset = -1
        #   - we've already speculated 3 tokens
        #   - and the suffix match length is 6
        # Then:
        #   - The match length before the already-speculated tokens is 3
        #   - The original config allow up to 5 speculated tokens total
        #   - Already speculated 3 tokens, so should allow 2 more tokens
        # So the new config should map match length 6 to 2 max spec tokens.
        max_spec_factor = config.suffix_max_spec_factor
        max_spec_offset = (config.suffix_max_spec_offset - len(spec_ids) *
                           (max_spec_factor + 1))
        #spec_start = time.perf_counter()
        result = self._suffix_cache.speculate(
            req_id,
            problem_id,
            pattern,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            max_spec_offset=max_spec_offset,
            min_token_prob=config.suffix_min_token_prob)
        #spec_time = (time.perf_counter() - spec_start) * 1000
        #total_time = (time.perf_counter() - start_time) * 1000
        
        # # Log每个任务的执行情况（只记录慢的）
        # if total_time > 100:
        #     print(f"[Task] thread={thread_id}, req_id={req_id}, spec={spec_time:.0f}ms, total={total_time:.0f}ms")

        # # Debug output - capture match and spec token information
        if hasattr(self, '_debug_spec_file') and self._debug_spec_file:
            self._step_counter += 1
            
            # Extract match tokens from pattern based on match length
            match_tokens = pattern[-result.match_len:] if result.match_len > 0 else []
            match_text = ""
            if self._tokenizer and match_tokens:
                try:
                    match_text = self._tokenizer.decode(match_tokens, skip_special_tokens=True)
                except:
                    match_text = ""
            
            # Spec tokens from result
            spec_tokens = result.token_ids if hasattr(result, 'token_ids') else []
            spec_text = ""
            if self._tokenizer and spec_tokens:
                try:
                    spec_text = self._tokenizer.decode(spec_tokens, skip_special_tokens=True)
                except:
                    spec_text = ""
            
            # Pattern tokens (the search pattern used)
            pattern_text = ""
            if self._tokenizer and pattern:
                try:
                    pattern_text = self._tokenizer.decode(pattern, skip_special_tokens=True)
                except:
                    pattern_text = ""
            
            # Create debug data similar to simulator
            debug_data = {
                "request_id": req_id,
                "step": self._step_counter,
                "batch_index": i,
                "pattern": pattern,
                "pattern_text": pattern_text,
                "pattern_length": len(pattern),
                "match_tokens": match_tokens,
                "match_text": match_text,
                "match_length": result.match_len,
                "spec_tokens": spec_tokens,
                "spec_text": spec_text,
                "num_spec_tokens": len(spec_tokens),
                "sampled_tokens": sampled_ids,
                "existing_spec_tokens": spec_ids,
                "score": getattr(result, 'score', 0.0),
                "max_spec_tokens": max_spec_tokens,
                "max_spec_factor": max_spec_factor,
                "max_spec_offset": max_spec_offset
            }
            
            self._debug_spec_file.write(json.dumps(debug_data, ensure_ascii=False) + '\n')
            self._debug_spec_file.flush()

        return result

    def propose_suffix_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
        spec_token_ids: Optional[list[list[int]]] = None,
    ) -> list[list[int]]:
        config = self.speculative_config
        
        # # Initialize debug file for this rank if not already done
        # if not hasattr(self, '_debug_spec_file'):
        #     rank = os.getenv("RANK", "0")
        #     local_rank = os.getenv("LOCAL_RANK", "0") 
        #     debug_dir = os.getenv("ARCTIC_METRICS_DIR", "/app/src")
        #     os.makedirs(debug_dir, exist_ok=True)
        #     debug_file_path = os.path.join(debug_dir, f"spec_debug_rank_{rank}_local_{local_rank}.jsonl")
        #     self._debug_spec_file = open(debug_file_path, 'w', encoding='utf-8')
        #     self._step_counter = 0
        
        # Determine which indices are allowed (hard and medium only)
        try:
            hard_indices, medium_indices, easy_indices, allowed_indices = self._get_hard_and_non_hard_indices()
        except Exception as e:
            print(f"Error getting hard and non-hard indices: {e}")
            hard_indices, medium_indices, easy_indices, allowed_indices = [], [], [], []

        if len(allowed_indices) > 0:
            # Case (1): Only process hard/medium; others return empty
            results: list[SuffixSpecResult] = [SuffixSpecResult() for _ in range(len(sampled_token_ids))]
            # 🚀 Use persistent thread pool to avoid repeated creation overhead
            self._speculation_max_workers = 9
            if self._speculation_threadpool is None:
                self._speculation_threadpool = ThreadPoolExecutor(max_workers=self._speculation_max_workers)
                print(f"[ThreadPool] Created with {self._speculation_max_workers} workers")
            
            
            # 🚀 Pre-extract all data to avoid shared state access in threads
            # batch_start = time.perf_counter()
            # extract_start = time.perf_counter()
            prepared_tasks = []
            hard_spec, medium_spec, easy_spec = 16, 8, 3
            for i in hard_indices:
                if 0 <= i < len(sampled_token_ids):
                    sampled_ids = sampled_token_ids[i]
                    spec_ids = spec_token_ids[i] if spec_token_ids is not None else []
                    req_id = self.input_batch.req_ids[i]
                    end_idx = self.input_batch.num_tokens_no_spec[i]
                    if end_idx >= self.max_model_len:
                        continue
                    size = min(end_idx, config.suffix_cache_max_depth)
                    pattern = self.input_batch.token_ids_cpu[i, end_idx - size:end_idx].tolist()
                    problem_id = self.get_problem_id_by_request_id(req_id)
                    
                    # Pack everything into a self-contained tuple
                    prepared_tasks.append((
                        i, req_id, problem_id, pattern, spec_ids, 
                        config, end_idx, self.max_model_len, hard_spec
                    ))

            for i in medium_indices:
                if 0 <= i < len(sampled_token_ids):
                    sampled_ids = sampled_token_ids[i]
                    spec_ids = spec_token_ids[i] if spec_token_ids is not None else []
                    req_id = self.input_batch.req_ids[i]
                    end_idx = self.input_batch.num_tokens_no_spec[i]
                    if end_idx >= self.max_model_len:
                        continue
                    size = min(end_idx, config.suffix_cache_max_depth)
                    pattern = self.input_batch.token_ids_cpu[i, end_idx - size:end_idx].tolist()
                    problem_id = self.get_problem_id_by_request_id(req_id)
                    prepared_tasks.append((
                        i, req_id, problem_id, pattern, spec_ids, 
                        config, end_idx, self.max_model_len, medium_spec
                    ))
            for i in easy_indices:
                if 0 <= i < len(sampled_token_ids):
                    sampled_ids = sampled_token_ids[i]
                    spec_ids = spec_token_ids[i] if spec_token_ids is not None else []
                    req_id = self.input_batch.req_ids[i]
                    end_idx = self.input_batch.num_tokens_no_spec[i]
                    if end_idx >= self.max_model_len:
                        continue
                    size = min(end_idx, config.suffix_cache_max_depth)
                    pattern = self.input_batch.token_ids_cpu[i, end_idx - size:end_idx].tolist()
                    problem_id = self.get_problem_id_by_request_id(req_id)
                    prepared_tasks.append((
                        i, req_id, problem_id, pattern, spec_ids, 
                        config, end_idx, self.max_model_len, easy_spec
                    ))

            if not prepared_tasks:
                return results
            
            # 🎯 优化：增大并行粒度 - 将多个小任务合并成批处理任务
            # 将任务按worker数量分组，每个worker处理一批任务
            num_workers = self._speculation_max_workers
            task_groups = [[] for _ in range(num_workers)]
            
            # 循环分配任务到各个worker组（负载均衡）
            # submit_start = time.perf_counter()
            for i, task_data in enumerate(prepared_tasks):
                worker_id = i % num_workers
                task_groups[worker_id].append(task_data)
            # submit_time = time.perf_counter() - submit_start
            # 过滤掉空组
            non_empty_groups = [group for group in task_groups if group]
            
            # 提交批处理任务
            future_to_group = {}
            for group in non_empty_groups:
                future = self._speculation_threadpool.submit(self._process_task_batch_v2, group)
                future_to_group[future] = group
            
            # 收集结果
            completed = 0
            # wait_start = time.perf_counter()
            for future in future_to_group:
                try:
                    batch_results = future.result()  # 返回的是一个列表
                    for task_data, result in batch_results:
                        req_index = task_data[0]
                        results[req_index] = result
                        completed += 1
                except Exception as e:
                    print(f"[Error] processing batch: {e}")
                    # Fallback: 对这个批次的所有任务返回空结果
                    group = future_to_group[future]
                    for task_data in group:
                        req_index = task_data[0]
                        results[req_index] = SuffixSpecResult()
            return results
        else:
            # Case (2): Fallback to previous implementation (process all)            
            # Extract phase: prepare batch args
            batch_args = []
            for i, sampled_ids in enumerate(sampled_token_ids):
                spec_ids = spec_token_ids[i] if spec_token_ids is not None else []
                batch_args.append((i, sampled_ids, spec_ids, config))

            # 🚀 Use persistent thread pool to avoid repeated creation overhead
            if self._speculation_threadpool is None:
                self._speculation_threadpool = ThreadPoolExecutor(max_workers=self._speculation_max_workers)
            
            # Submit phase
            submit_start = time.perf_counter()
            future_to_index = {self._speculation_threadpool.submit(self._process_single_speculation, args): i 
                              for i, args in enumerate(batch_args)}

            results = [None] * len(batch_args)
            for future in future_to_index:
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    print(f"Error processing speculation for index {index}: {e}")
                    results[index] = SuffixSpecResult()  # Fallback to empty result

            return results



    def __del__(self):
        """Clean up debug files and thread pool when model runner is destroyed"""
        if hasattr(self, '_debug_spec_file') and self._debug_spec_file:
            try:
                self._debug_spec_file.close()
                self._debug_spec_file = None
            except:
                pass
        
        # Shutdown the persistent thread pool
        if hasattr(self, '_speculation_threadpool') and self._speculation_threadpool:
            try:
                self._speculation_threadpool.shutdown(wait=False)
            except:
                pass

    def load_model(self) -> None:
        load_shift_model = (
            self.vllm_config.parallel_config.enable_shift_parallel)

        if load_shift_model:
            # Make a deep copy of the config before loading the model.
            shift_config = copy.deepcopy(self.vllm_config)

        self._orig_load_model()

        if self.parallel_config.ulysses_sequence_parallel_size > 1:
            self.monkeypatch_forward()

        if load_shift_model:
            shift_config.parallel_config.tensor_parallel_size *= (
                shift_config.parallel_config.ulysses_sequence_parallel_size)
            shift_config.parallel_config.ulysses_sequence_parallel_size = 1
            with set_shift_parallel_mode(True):
                self.shift_model = get_model(vllm_config=shift_config)
            self.shift_parallel_threshold = (
                shift_config.parallel_config.shift_parallel_threshold)
            if "SwiftKV" in self.model.__class__.__name__:
                # HACK: Replace the decode-runner since it always runs in full
                # TP, but the original model is captured using SP * BATCH_SIZE,
                # which does not cover all its cuda graph sizes. The shift-mode
                # model should have all its cuda graphs captured correctly.
                self.model.model.decode_runner = (
                    self.shift_model.model.decode_runner)
        else:
            self.shift_model = None
            self.shift_parallel_threshold = 0

    def capture_model(self) -> None:
        if not self.use_cuda_graph:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "set -O %s and ensure `use_cudagraph` was not manually set to "
                "False", CompilationLevel.PIECEWISE)
            return

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with parallel_state.graph_capture(device=self.device):
            sp_size = self.parallel_config.ulysses_sequence_parallel_size
            full_cg = self.full_cuda_graph
            # capture original model shapes
            compilation_cases = (shape for shape in reversed(self.cudagraph_batch_sizes)
                if shape * sp_size > self.shift_parallel_threshold
                and shape * sp_size <= self.max_num_tokens)
            # Only rank 0 should print progress bar during capture
            if is_global_first_rank():
                print_cases, compilation_cases = tee(compilation_cases)
                logger.info(f"original model shapes {list(print_cases)}")
                compilation_cases = tqdm(list(compilation_cases),
                                         desc="Capturing CUDA graph shapes of original model")
            for num_tokens in compilation_cases:
                # We skip EPLB here since we don't want to record dummy metrics
                for _ in range(self.vllm_config.compilation_config.
                               cudagraph_num_of_warmups):
                    self._dummy_run(num_tokens * sp_size,
                                    capture_attn_cudagraph=full_cg,
                                    skip_eplb=True)
                self._dummy_run(num_tokens * sp_size,
                                capture_attn_cudagraph=full_cg,
                                skip_eplb=True)

            # Capture shift model shapes
            if self.shift_model is not None:
                orig_model, self.model = self.model, self.shift_model
                # Reset compilation cases
                compilation_cases = (shape for shape in reversed(self.cudagraph_batch_sizes)
                    if shape <= self.shift_parallel_threshold
                    or "SwiftKV" in self.model.__class__.__name__)
                # Note: We want to capture all shapes for the SwiftKV shift model.
                # This is necessary since SwiftKV always uses full TP for the decode runner.
                # For all other models, we only capture necessary shapes for the SP_TP mode,
                # yielding less setup time.
                if is_global_first_rank():
                    print_cases, compilation_cases = tee(compilation_cases)
                    logger.info(f"shift model shapes {list(print_cases)}")
                    compilation_cases = tqdm(list(compilation_cases),
                                             desc="Capturing CUDA graph shapes of shift model")
                with set_shift_parallel_mode(True):
                    for num_tokens in compilation_cases:
                        for _ in range(self.vllm_config.compilation_config.
                                       cudagraph_num_of_warmups):
                            self._dummy_run(num_tokens,
                                            capture_attn_cudagraph=full_cg,
                                            skip_eplb=True)
                        self._dummy_run(num_tokens,
                                        capture_attn_cudagraph=full_cg,
                                        skip_eplb=True)
                self.model = orig_model

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        self._orig_initialize_kv_cache(kv_cache_config)

        if self.shift_model is not None:
            # Bind the KV caches to the shift parallel model.
            forward_context = (
                self.vllm_config.compilation_config.static_forward_context)
            for mod in self.shift_model.modules():
                if isinstance(mod, Attention):
                    mod.kv_cache = forward_context[mod.layer_name].kv_cache
    def _log_execution_time(self, start_timestamp, duration_seconds, batch_size, num_scheduled_tokens,early_return=False, doc_type: str = "GPU_execution_time"):
        """Log execution time metrics for execute_model calls"""
        try:
            timing_data = {
                "timestamp": start_timestamp,
                "call_type": "execute_model_timing",
                "execution_duration_seconds": duration_seconds,
                "execution_duration_ms": duration_seconds * 1000,
                "batch_size": batch_size,
                "num_scheduled_tokens": num_scheduled_tokens,
                "process_info": {
                    "rank": int(os.getenv("RANK", "0")),
                    "local_rank": int(os.getenv("LOCAL_RANK", "0")),
                    "world_size": int(os.getenv("WORLD_SIZE", "1"))
                },
                "early_return": early_return
            }
            
            # Write to file
            self._write_timing_stats(timing_data, doc_type=doc_type)
            
        except Exception as e:
            # Log error but don't crash the model
            logger.error(f"Failed to log execution time: {e}")
    
    def _write_timing_stats(self, timing_data, doc_type: str = "GPU_execution_time"):
        """Buffer execution timing data and flush periodically to reduce I/O."""
        if doc_type == "GPU_execution_time":
            buffer = self._gpu_timing_buffer
            buffer.append(json.dumps(timing_data, default=self._json_serializable))
        elif doc_type == "CPU_execution_time":
            buffer = self._cpu_timing_buffer
            buffer.append(json.dumps(timing_data, default=self._json_serializable))
        else:
            raise ValueError(f"Invalid document type: {doc_type}")
        
        # Flush conditions: buffer size or time threshold
        should_flush_by_n = len(buffer) >= self._timing_flush_every_n
        should_flush_by_time = (time.monotonic() - self._timing_last_flush_time[doc_type]) >= self._timing_flush_every_s
        if should_flush_by_n or should_flush_by_time:
            self._flush_timing_buffer(doc_type=doc_type)

    def _flush_timing_buffer(self, force: bool = False, doc_type: str = "GPU_execution_time"):
        """Flush buffered timing lines to disk.

        When force is True, flush unconditionally (e.g., at exit).
        """
        try:
            if doc_type == "GPU_execution_time":
                buffer = self._gpu_timing_buffer
                file_type = "GPU_execution_timing"
            elif doc_type == "CPU_execution_time":
                buffer = self._cpu_timing_buffer
                file_type = "CPU_execution_timing"
            else:
                raise ValueError(f"Invalid document type: {doc_type}")
            if not buffer and not force:
                return
            # Nothing to write if empty and not forced
            if not buffer:
                self._timing_last_flush_time[doc_type] = time.monotonic()
                return

            # Get file path with current_step included
            file_path = self._get_file_path_with_step(file_type)
            
            # Write all pending lines at once
            with open(file_path, "a") as f:
                f.write("\n".join(buffer) + "\n")
            buffer.clear()
            self._timing_last_flush_time[doc_type] = time.monotonic()
        except Exception as e:
            logger.error(f"Failed to flush timing stats: {e}")

    def _get_file_path_with_step(self, file_type: str) -> str:
        """Generate file path with current_step in the filename.
        
        Args:
            file_type: Type of file - "GPU_execution_timing", "CPU_execution_timing", 
                      "suffix_tree_stats", "suffix_speculation_timing", or "token_data"
        
        Returns:
            File path with current_step in the filename if available, otherwise the default path
        """
        import os as _os
        
        # Get current_step from context manager
        current_step = None
        
        # If current_step is not available, return the default path without step suffix
        if current_step is None:
            if file_type == "GPU_execution_timing":
                return self._timing_file_path_gpu
            elif file_type == "CPU_execution_timing":
                return self._timing_file_path_cpu
            elif file_type == "suffix_tree_stats":
                return self._suffix_file_path
            elif file_type == "suffix_speculation_timing":
                return self._suffix_timing_file_path
            elif file_type == "token_data":
                # Default token_data path when current_step is not available
                return _os.path.join(self._output_dir, f"token_data_rank_{self._rank}_local_{self._local_rank}.jsonl")
            else:
                raise ValueError(f"Unknown file_type: {file_type}")
        
        # Generate filename with current_step
        if file_type == "GPU_execution_timing":
            filename = f"GPU_execution_timing_rank_{self._rank}.jsonl"
        elif file_type == "CPU_execution_timing":
            filename = f"CPU_execution_timing_rank_{self._rank}.jsonl"
        elif file_type == "suffix_tree_stats":
            filename = f"suffix_tree_stats_rank_{self._rank}.jsonl"
        elif file_type == "suffix_speculation_timing":
            filename = f"suffix_speculation_timing_rank_{self._rank}.jsonl"
        elif file_type == "token_data":
            filename = f"token_data_rank_{self._rank}.jsonl"
        else:
            raise ValueError(f"Unknown file_type: {file_type}")
        
        return _os.path.join(self._output_dir, filename)
    
    def _json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-serializable types"""
        import numpy as np
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle scalar numpy types
            return obj.item()
        else:
            return str(obj)  # Fallback to string representation

    def _flush_suffix_buffer(self, force: bool = False):
        """Flush buffered suffix-tree stats lines to disk.

        When force is True, flush unconditionally (e.g., at exit).
        """
        try:
            if not self._suffix_buffer and not force:
                return
            # Nothing to write if empty and not forced
            if not self._suffix_buffer:
                self._suffix_last_flush_time = time.monotonic()
                return

            # Get file path with current_step included
            file_path = self._get_file_path_with_step("suffix_tree_stats")
            
            # Write all pending lines at once
            with open(file_path, "a") as f:
                f.write("\n".join(self._suffix_buffer) + "\n")
            self._suffix_buffer.clear()
            self._suffix_last_flush_time = time.monotonic()
        except Exception as e:
            logger.error(f"Failed to flush suffix-tree stats: {e}")
    
    def _flush_suffix_timing_buffer(self, force: bool = False):
        """Flush buffered suffix speculation timing lines to disk.

        When force is True, flush unconditionally (e.g., at exit).
        """
        try:
            if not self._suffix_timing_buffer and not force:
                return
            # Nothing to write if empty and not forced
            if not self._suffix_timing_buffer:
                self._suffix_timing_last_flush_time = time.monotonic()
                return

            # Get file path with current_step included
            file_path = self._get_file_path_with_step("suffix_speculation_timing")
            
            # Write all pending lines at once
            with open(file_path, "a") as f:
                f.write("\n".join(self._suffix_timing_buffer) + "\n")
            self._suffix_timing_buffer.clear()
            self._suffix_timing_last_flush_time = time.monotonic()
        except Exception as e:
            logger.error(f"Failed to flush suffix timing stats: {e}")
    
    def _write_suffix_timing_stats(self, timing_data):
        """Buffer suffix speculation timing data and flush periodically to reduce I/O."""
        try:
            # Enqueue timing data
            self._suffix_timing_buffer.append(json.dumps(timing_data, default=self._json_serializable))

            # Flush conditions: buffer size or time threshold
            should_flush_by_n = len(self._suffix_timing_buffer) >= self._suffix_timing_flush_every_n
            should_flush_by_time = (time.monotonic() - self._suffix_timing_last_flush_time) >= self._suffix_timing_flush_every_s
            if should_flush_by_n or should_flush_by_time:
                self._flush_suffix_timing_buffer()
        except Exception as e:
            # Log error but don't crash the model
            logger.error(f"Failed to buffer suffix timing stats: {e}")

    def _log_suffix_tree_stats(self, num_draft_tokens, draft_token_ids, cu_num_draft_tokens, valid_sampled_token_ids):
        """Log suffix tree decoding statistics for this execute_model call"""
        if not hasattr(self, 'input_batch') or not self.input_batch:
            return
        #print(f"DEBUG: draft_token_ids={draft_token_ids},\n")
        
        # Count total proposed and accepted tokens for this call
        total_proposed = 0
        total_accepted = 0
        per_request_stats = []

        if num_draft_tokens is None:
            # Still need to populate per_request_stats with req_id and problem_id info
            for req_id in self.input_batch.req_ids:
                # Get problem_id for this request
                problem_id = None
                if hasattr(self, '_current_batch_req_id_to_problem_id'):
                    problem_id = self._current_batch_req_id_to_problem_id.get(req_id)
                
                per_request_stats.append({
                    "req_id": req_id,
                    "problem_id": problem_id,
                    "proposed": 0,
                    "accepted": 0
                })
            
            proposed_count = 0
            accepted_count = 0
            stats_data = {
                "timestamp": datetime.now().isoformat(),
                "call_type": "execute_model_suffix_tree",
                "batch_size": len(self.input_batch.req_ids),
                "total_proposed": total_proposed,
                "total_accepted": total_accepted,
                "accept_rate": total_accepted / total_proposed * 100 if total_proposed > 0 else 0,
                "per_request_stats": per_request_stats,
                "process_info": {
                    "rank": int(os.getenv("RANK", "0")),
                    "local_rank": int(os.getenv("LOCAL_RANK", "0")),
                    "world_size": int(os.getenv("WORLD_SIZE", "1"))
                }
            }
            
            # Write to file
            self._write_suffix_tree_stats(stats_data)
            return
        
        for i, req_id in enumerate(self.input_batch.req_ids):
            proposed_count = num_draft_tokens[i]
            accepted_count = len(valid_sampled_token_ids[i])-1

            if accepted_count < 0:
                proposed_count = 0
                accepted_count = 0

            total_proposed += proposed_count
            total_accepted += accepted_count
         
            # Get problem_id for this request
            problem_id = None
            if hasattr(self, '_current_batch_req_id_to_problem_id'):
                problem_id = self._current_batch_req_id_to_problem_id.get(req_id)
         
            # Record per-request stats
            per_request_stats.append({
                "req_id": req_id,
                "problem_id": problem_id,
                "proposed": proposed_count,
                "accepted": accepted_count
            })
        
        # Add debug info for first few calls
        if hasattr(self, '_debug_call_count'):
            self._debug_call_count += 1
        else:
            self._debug_call_count = 1
            
        # Only log if there were spec tokens proposed
        if True:
            stats_data = {
                "timestamp": datetime.now().isoformat(),
                "call_type": "execute_model_suffix_tree",
                "batch_size": len(self.input_batch.req_ids),
                "total_proposed": total_proposed,
                "total_accepted": total_accepted,
                "accept_rate": total_accepted / total_proposed * 100 if total_proposed > 0 else 0,
                "per_request_stats": per_request_stats,
                "process_info": {
                    "rank": int(os.getenv("RANK", "0")),
                    "local_rank": int(os.getenv("LOCAL_RANK", "0")),
                    "world_size": int(os.getenv("WORLD_SIZE", "1"))
                }
            }
            
            # Write to file
            self._write_suffix_tree_stats(stats_data)
            #print("DEBUG:len(draft_token_ids): ", len(draft_token_ids))
            #print("DEBUG:cu_num_draft_tokens: ", len(cu_num_draft_tokens),cu_num_draft_tokens[:5])
            
            # Write token data to separate file
            #self._write_token_data_to_file(draft_token_ids, cu_num_draft_tokens, valid_sampled_token_ids)
    
    def _write_suffix_tree_stats(self, stats_data):
        """Buffer suffix-tree stats and flush periodically to reduce I/O."""
        try:
            # Enqueue stats data
            self._suffix_buffer.append(json.dumps(stats_data, default=self._json_serializable))

            # Flush conditions: buffer size or time threshold
            should_flush_by_n = len(self._suffix_buffer) >= self._suffix_flush_every_n
            should_flush_by_time = (time.monotonic() - self._suffix_last_flush_time) >= self._suffix_flush_every_s
            if should_flush_by_n or should_flush_by_time:
                self._flush_suffix_buffer()
        except Exception as e:
            # Log error but don't crash the model
            logger.error(f"Failed to buffer suffix tree stats: {e}")
    
    def _write_token_data(self, token_data):
        """Write draft and valid token data to a separate file"""
        try:
            # Get file path with current_step included
            token_file = self._get_file_path_with_step("token_data")
            
            # Write with immediate flush to ensure data is written atomically
            with open(token_file, "a") as f:
                f.write(json.dumps(token_data, default=self._json_serializable) + "\n")
                f.flush()  # Force immediate write to disk
                os.fsync(f.fileno())  # Force OS to write to storage
                
        except Exception as e:
            # Log error but don't crash the model
            logger.error(f"Failed to write token data: {e}")
    
    def _write_token_data_to_file(self, draft_token_ids, cu_num_draft_tokens, valid_sampled_token_ids):
        """Collect and write draft tokens, valid tokens, and pre-draft tokens to file"""
        try:
            from datetime import datetime
            
            if not hasattr(self, 'input_batch') or not self.input_batch:
                return
                
            # Prepare token data for each request
            token_data_entries = []
            
            # Convert draft_token_ids to list if needed
            all_draft_tokens = None
            if draft_token_ids is not None:
                if hasattr(draft_token_ids, 'tolist'):
                    all_draft_tokens = draft_token_ids.tolist()
                elif isinstance(draft_token_ids, list):
                    all_draft_tokens = draft_token_ids
                else:
                    all_draft_tokens = list(draft_token_ids) if draft_token_ids is not None else None
            
            for i, req_id in enumerate(self.input_batch.req_ids):
                # Extract draft tokens for this specific request using cumulative indices
                draft_tokens = None
                if all_draft_tokens is not None and cu_num_draft_tokens is not None and i < len(cu_num_draft_tokens):
                    start_idx = cu_num_draft_tokens[i]
                    end_idx = cu_num_draft_tokens[i+1] if i+1 < len(cu_num_draft_tokens) else len(all_draft_tokens)
                    draft_tokens = all_draft_tokens[start_idx:end_idx]
                
                # Get valid sampled tokens for this request
                valid_tokens = None
                if valid_sampled_token_ids and i < len(valid_sampled_token_ids):
                    valid_tokens = valid_sampled_token_ids[i]
                
                # Get pre-draft tokens (16 tokens before draft)
                pre_draft_tokens = []
                try:
                    if hasattr(self.input_batch, 'token_ids_cpu') and hasattr(self.input_batch, 'num_tokens_no_spec'):
                        # Get current position in the sequence
                        current_pos = self.input_batch.num_tokens_no_spec[i]
                        
                        # Get 16 tokens before current position
                        start_pos = max(0, current_pos - 16)
                        end_pos = current_pos
                        
                        if start_pos < end_pos:
                            pre_draft_slice = self.input_batch.token_ids_cpu[i, start_pos:end_pos]
                            if hasattr(pre_draft_slice, 'tolist'):
                                pre_draft_tokens = pre_draft_slice.tolist()
                            else:
                                pre_draft_tokens = list(pre_draft_slice)
                except Exception as e:
                    logger.warning(f"Failed to get pre-draft tokens for req {req_id}: {e}")
                    pre_draft_tokens = []
                
                # Convert tokens to text if tokenizer is available
                draft_text = self._tokens_to_text(draft_tokens)
                valid_text = self._tokens_to_text(valid_tokens)
                pre_draft_text = self._tokens_to_text(pre_draft_tokens)
                
                # Create entry for this request
                token_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "req_id": req_id,
                    "draft_token_ids": draft_tokens,
                    "draft_text": draft_text,
                    "valid_sampled_token_ids": valid_tokens,
                    "valid_sampled_text": valid_text,
                    "pre_draft_16_tokens": pre_draft_tokens,
                    "pre_draft_16_text": pre_draft_text,
                    "process_info": {
                        "rank": int(os.getenv("RANK", "0")),
                        "local_rank": int(os.getenv("LOCAL_RANK", "0")),
                        "world_size": int(os.getenv("WORLD_SIZE", "1"))
                    }
                }
                
                token_data_entries.append(token_entry)
            
            # Write each entry separately to the token data file
            for entry in token_data_entries:
                self._write_token_data(entry)
                
        except Exception as e:
            # Log error but don't crash the model
            logger.error(f"Failed to process token data: {e}")
    
    def _tokens_to_text(self, token_ids):
        """Convert token ids to text using the tokenizer"""
        if not token_ids or not self._tokenizer:
            return None
        
        try:
            # Handle different input formats
            if isinstance(token_ids, list):
                tokens = token_ids
            elif hasattr(token_ids, 'tolist'):
                tokens = token_ids.tolist()
            else:
                tokens = list(token_ids)
            
            # Filter out any invalid token ids
            valid_tokens = [t for t in tokens if isinstance(t, int) and t >= 0]
            if not valid_tokens:
                return None
                
            # Convert to text
            text = self._tokenizer.decode(valid_tokens, skip_special_tokens=True)
            return text
            
        except Exception as e:
            logger.warning(f"Failed to convert tokens to text: {e}")
            return None