# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Arctic Model Runner (Suffix Cache Only) - Controlled Version

Fully controlled implementation with complete control over the speculative decoding flow.

Design Principles:
1. Stability First: Independent of vLLM's internal speculative decoding implementation
2. Full Control: We manage the entire speculative decoding flow (verification + proposal)
3. Clear Layering: vLLM base functionality vs Arctic extension functionality

Advantages:
+ More robust to vLLM updates
+ Complete control over speculative decoding flow
+ Easy to debug and modify
+ Clear architectural boundaries
"""

import logging
from typing import Optional, List, Union, Any, Dict

import torch
import numpy as np

# vLLM imports
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.distributed.parallel_state import (get_pp_group, get_tp_group,
                                             is_global_first_rank)
from vllm.config import VllmConfig
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.outputs import ModelRunnerOutput, EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.sequence import IntermediateTensors
from vllm.forward_context import set_forward_context
from vllm import envs
from vllm.kv_transfer.kv_connector.factory import has_kv_transfer_group
from vllm.utils import round_up

# Arctic imports  
from arctic_inference.vllm.common import ArcticPatch
from .extensions import SuffixCacheExtension, SuffixResult

logger = logging.getLogger(__name__)


class GPUModelRunnerPatch(ArcticPatch[GPUModelRunner]):
    """
    Arctic Controlled Patch for Suffix Cache.
    
    This version has full control over the speculative decoding flow, including:
    - Rejection sampling (verification)
    - Token parsing (parsing)
    - Discard logic (discarding)
    - Suffix cache proposal (proposal)
    
    Independent of vLLM's internal speculative decoding implementation, 
    more stable and easier to maintain.
    """
    
    # ========== Patch Points ==========
    _orig_init = GPUModelRunner.__init__
    _orig_propose_draft_token_ids = GPUModelRunner.propose_draft_token_ids
    # Note: We don't patch execute_model, we completely rewrite it
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """
        Initialize with suffix cache extension and rejection sampler.
        """
        # 1. Temporarily remove Arctic speculative config
        spec_config = vllm_config.speculative_config
        is_suffix_method = (spec_config and spec_config.method == "suffix")
        
        if is_suffix_method:
            vllm_config.speculative_config = None
        
        # 2. Call vLLM's original __init__
        self._orig_init(vllm_config, device)
        
        # 3. Restore Arctic speculative config and setup
        if is_suffix_method:
            self.vllm_config.speculative_config = spec_config
            self.speculative_config = spec_config
            
            # Create rejection sampler (we manage it ourselves)
            if get_pp_group().is_last_rank:
                self.rejection_sampler = RejectionSampler()
        else:
            self.speculative_config = None
        
        # 4. Create suffix cache extension
        self.suffix_extension = SuffixCacheExtension(self.vllm_config)
        
        status = "enabled" if self.suffix_extension.is_enabled() else "disabled"
        logger.info(f"✓ Suffix Cache Extension {status} (Controlled Mode)")
    
    # ========== Backward Compatibility Properties ==========
    
    @property
    def _suffix_cache(self):
        """
        Backward compatibility: Direct access to suffix cache.
        
        This allows external code (e.g., verl training framework) to access
        the suffix cache the same way as before:
            
            suffix_cache = model_runner._suffix_cache
        
        Returns:
            SuffixDecodingCache object if enabled, None otherwise
        """
        if self.suffix_extension and self.suffix_extension.is_enabled():
            return self.suffix_extension.suffix_cache
        return None
    
    @_suffix_cache.setter
    def _suffix_cache(self, value):
        """
        Backward compatibility: Allow setting suffix cache directly.
        
        This allows external code to inject a custom cache:
            
            model_runner._suffix_cache = my_custom_cache
        
        Args:
            value: SuffixDecodingCache instance or None
        """
        if self.suffix_extension:
            self.suffix_extension.suffix_cache = value
            self.suffix_extension.enabled = (value is not None)
            
            if value is not None:
                logger.info("Suffix cache injected from external code")
    
    # ========== Main Execution Flow ==========
    
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        """
        Fully controlled execute_model implementation.
        
        We control the entire speculative decoding flow, including verification and proposal.
        Independent of vLLM's internal speculative decoding implementation.
        
        Flow:
        1. Prepare inputs (reuse vLLM)
        2. Model forward (reuse vLLM)
        3. Sampling + verification (our logic)
        4. Post-processing (our logic)
        5. Suffix cache proposal (our logic)
        """
        # ========== Phase 1: Preparation and checks ==========
        self._update_states(scheduler_output)
        
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output)
        
        # ========== Phase 2: Prepare inputs (reuse vLLM) ==========
        (attn_metadata, attention_cuda_graphs, logits_indices,
         spec_decode_metadata, num_scheduled_tokens_np) = self._prepare_inputs(
            scheduler_output)
        
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        
        # ========== Phase 3: Model forward (reuse vLLM core logic) ==========
        (hidden_states, sample_hidden_states, logits, aux_hidden_states, 
         finished_sending, finished_recving) = (
            self._do_model_forward(
                scheduler_output,
                num_scheduled_tokens,
                num_scheduled_tokens_np,
                logits_indices,
                attn_metadata,
                attention_cuda_graphs,
                intermediate_tensors,
            )
        )        
        # If mid-pipeline stage, return directly
        if logits is None:
            return hidden_states
        
        # If pooling task (marked by "POOLING" string), use vLLM's pooling logic
        if logits == "POOLING":
            return self._pool(
                hidden_states,
                num_scheduled_tokens,
                num_scheduled_tokens_np,
                finished_sending,
                finished_recving,
            )
        
        # ========== Phase 4: Sampling  ==========
        sampling_metadata = self.input_batch.sampling_metadata
        
        if spec_decode_metadata is None:
            # Normal sampling (no speculative decoding)
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            # Speculative decoding sampling 
            sampler_output = self._sample_with_rejection(
                logits,
                spec_decode_metadata,
                sampling_metadata,
            )
        
        # ========== Phase 5: Post-processing  ==========
        # NaN detection
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)
        
        # Get request indices to discard (partial prefill)
        discard_indices = self._get_discard_indices(scheduler_output)
        
        # GPU -> CPU sync
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = (logprobs_tensors.tolists()
                         if logprobs_tensors is not None else None)
        
        # Prompt logprobs
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output,
        )
        
        # Parse sampled tokens
        sampled_token_ids = sampler_output.sampled_token_ids
        valid_sampled_token_ids = self._parse_sampled_tokens(
            sampled_token_ids,
            spec_decode_metadata,
            discard_indices,
        )
        
        # Update token cache
        self._update_token_cache(valid_sampled_token_ids)
        
        # ========== Phase 6: Draft Token Proposal ==========
        if not self.speculative_config:
            # Speculative decoding is not enabled
            spec_token_ids = None
        else:
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
        
        # ========== Phase 7: Return results ==========
        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            num_nans_in_logits=num_nans_in_logits,
        )
    
    # ========== vLLM Base Methods (inherited and reused) ==========
    
    def _update_states(self, scheduler_output: SchedulerOutput):
        """
        Update internal states based on scheduler output.
        
        This method is inherited from vLLM's GPUModelRunner.
        It updates the input batch and other internal states.
        
        We reuse vLLM's implementation directly without modification.
        """
        return GPUModelRunner._update_states(self, scheduler_output)
    
    def _prepare_inputs(self, scheduler_output: SchedulerOutput):
        """
        Prepare model inputs from scheduler output.
        
        This method is inherited from vLLM's GPUModelRunner.
        It prepares attention metadata, logits indices, and spec_decode_metadata.
        
        Returns:
            tuple: (attn_metadata, attention_cuda_graphs, logits_indices,
                   spec_decode_metadata, num_scheduled_tokens_np)
        
        We reuse vLLM's implementation directly without modification.
        """
        return GPUModelRunner._prepare_inputs(self, scheduler_output)
    
    def _get_nans_in_logits(self, logits: Optional[torch.Tensor]) -> Dict[str, int]:
        """
        Get number of NaN values in logits for debugging.
        
        This method is inherited from vLLM's GPUModelRunner.
        
        We reuse vLLM's implementation directly without modification.
        """
        return GPUModelRunner._get_nans_in_logits(self, logits)
    
    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        scheduler_output: SchedulerOutput,
    ) -> Dict[str, List[Optional[Dict[int, float]]]]:
        """
        Compute prompt logprobs if needed.
        
        This method is inherited from vLLM's GPUModelRunner.
        
        We reuse vLLM's implementation directly without modification.
        """
        return GPUModelRunner._get_prompt_logprobs_dict(
            self, hidden_states, scheduler_output)
    
    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        finished_sending: bool,
        finished_recving: bool,
    ):
        """
        Handle pooling tasks (e.g., for embedding models).
        
        This method is inherited from vLLM's GPUModelRunner.
        
        We reuse vLLM's implementation directly without modification.
        """
        return GPUModelRunner._pool(
            self, hidden_states, num_scheduled_tokens,
            num_scheduled_tokens_np, finished_sending, finished_recving)
    
    def kv_connector_no_forward(self, scheduler_output: SchedulerOutput):
        """
        Handle KV transfer when there's no forward pass needed.
        
        This method is inherited from vLLM's GPUModelRunner.
        
        We reuse vLLM's implementation directly without modification.
        """
        return GPUModelRunner.kv_connector_no_forward(self, scheduler_output)
    
    def _execute_mm_encoder(self, scheduler_output: SchedulerOutput):
        """
        Execute multimodal encoder.
        
        This method is inherited from vLLM's GPUModelRunner.
        
        We reuse vLLM's implementation directly without modification.
        """
        return GPUModelRunner._execute_mm_encoder(self, scheduler_output)
    
    def _gather_mm_embeddings(self, scheduler_output: SchedulerOutput):
        """
        Gather multimodal embeddings.
        
        This method is inherited from vLLM's GPUModelRunner.
        
        We reuse vLLM's implementation directly without modification.
        """
        return GPUModelRunner._gather_mm_embeddings(self, scheduler_output)
    
    def sync_and_slice_intermediate_tensors(
        self,
        num_input_tokens: int,
        intermediate_tensors: Optional[IntermediateTensors],
        broadcast: bool,
    ):
        """
        Synchronize and slice intermediate tensors for pipeline parallelism.
        
        This method is inherited from vLLM's GPUModelRunner.
        
        We reuse vLLM's implementation directly without modification.
        """
        return GPUModelRunner.sync_and_slice_intermediate_tensors(
            self, num_input_tokens, intermediate_tensors, broadcast)
    
    def get_dp_padding(self, num_input_tokens: int):
        """
        Get data parallelism padding.
        
        This method is inherited from vLLM's GPUModelRunner.
        
        Returns:
            tuple: (num_pad, num_tokens_across_dp)
        
        We reuse vLLM's implementation directly without modification.
        """
        return GPUModelRunner.get_dp_padding(self, num_input_tokens)
    
    def maybe_setup_kv_connector(self, scheduler_output: SchedulerOutput):
        """
        Setup KV connector if needed.
        
        This method is inherited from vLLM's GPUModelRunner.
        
        We reuse vLLM's implementation directly without modification.
        """
        return GPUModelRunner.maybe_setup_kv_connector(self, scheduler_output)
    
    def maybe_wait_for_kv_save(self):
        """
        Wait for KV save operation if needed.
        
        This method is inherited from vLLM's GPUModelRunner.
        
        We reuse vLLM's implementation directly without modification.
        """
        return GPUModelRunner.maybe_wait_for_kv_save(self)
    
    def get_finished_kv_transfers(self, scheduler_output: SchedulerOutput):
        """
        Get finished KV transfer status.
        
        This method is inherited from vLLM's GPUModelRunner.
        
        Returns:
            tuple: (finished_sending, finished_recving)
        
        We reuse vLLM's implementation directly without modification.
        """
        return GPUModelRunner.get_finished_kv_transfers(self, scheduler_output)
    
    def apply_grammar_bitmask(self, scheduler_output: SchedulerOutput, logits: torch.Tensor):
        """
        Apply grammar bitmask to logits for constrained decoding.
        
        This method is inherited from vLLM's GPUModelRunner.
        
        We reuse vLLM's implementation directly without modification.
        """
        return GPUModelRunner.apply_grammar_bitmask(self, scheduler_output, logits)
    
    # ========== Core Methods - Phase 1-5 (vLLM General Logic) ==========
    
    def _do_model_forward(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        logits_indices: torch.Tensor,
        attn_metadata: Dict[str, Any],
        attention_cuda_graphs: bool,
        intermediate_tensors: Optional[IntermediateTensors],
    ):
        """
        Execute model forward.
        
        This method is extracted from vLLM's execute_model, containing core forward logic.
        We reuse vLLM's implementation without modification.
        
        Args:
            scheduler_output: Output from scheduler
            num_scheduled_tokens: Number of scheduled tokens
            num_scheduled_tokens_np: Number of scheduled tokens as numpy array (for pooling)
            logits_indices: Indices for computing logits
            attn_metadata: Attention metadata
            attention_cuda_graphs: Whether using CUDA graphs
            intermediate_tensors: Intermediate tensors for pipeline parallelism
        
        Returns:
            tuple: (hidden_states, sample_hidden_states, logits, aux_hidden_states, 
                   finished_sending, finished_recving)
                - hidden_states: Model hidden states
                - sample_hidden_states: Hidden states used for computing logits
                - logits: Logits for sampling (None for mid-pipeline stages)
                - aux_hidden_states: Auxiliary hidden states (for MLP speculator, etc.)
                - finished_sending: KV transfer send status
                - finished_recving: KV transfer receive status
        """
        # Calculate padding (reuse vLLM logic)
        num_input_tokens = self._calculate_num_input_tokens(
            num_scheduled_tokens,
            attention_cuda_graphs,
        )
        
        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        # TODO: we currently do not support batch reorder
        mm_embeds = []
        if self.is_multimodal_model:
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        
        # Prepare inputs (input_ids or inputs_embeds)
        input_ids, inputs_embeds = self._prepare_model_inputs(
            num_scheduled_tokens,
            num_input_tokens,
            mm_embeds,
        )
        
        # Prepare positions
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]
        
        # Handle pipeline parallelism
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)
        
        # CUDA graph check
        skip_cuda_graphs = self.full_cuda_graph and not attention_cuda_graphs
        
        # Calculate DP padding
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad
        
        # Model forward
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            skip_cuda_graphs=skip_cuda_graphs,
        ):
            self.maybe_setup_kv_connector(scheduler_output)
            
            # Execute model forward
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
            
            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = (
                self.get_finished_kv_transfers(scheduler_output))
        
        # Extract hidden states
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = None
        
        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping micro-batches
        # https://github.com/vllm-project/vllm/issues/18019
        broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend
            == "external_launcher" and len(get_pp_group().ranks) > 0
        )
        
        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            if not broadcast_pp_output:
                return hidden_states, None, None, None, finished_sending, finished_recving
            assert isinstance(hidden_states, IntermediateTensors)
            get_pp_group().send_tensor_dict(
                hidden_states.tensors,
                all_gather_group=get_tp_group()
            )
            sample_hidden_states = None
            logits = None
        else:
            # Last rank: Check pooling FIRST, before computing logits
            if self.input_batch.pooling_params:
                # Pooling task: Skip logits computation, use special marker
                return hidden_states, None, "POOLING", None, finished_sending, finished_recving
            else:
                # Normal generation: Compute logits
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states, None)
        
        if broadcast_pp_output:
            # Only broadcast actual tensors, not special markers like "POOLING"
            if logits is not None:
                model_output_broadcast_data = {
                    "logits": logits.contiguous(),
                }
                model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]
            # else: logits is None or "POOLING", no broadcast needed
        
        # Apply grammar bitmask if present (only for actual logits tensors)
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)
        
        return hidden_states, sample_hidden_states, logits, aux_hidden_states, finished_sending, finished_recving
    
    def _calculate_num_input_tokens(
        self,
        num_scheduled_tokens: int,
        attention_cuda_graphs: bool,
    ) -> int:
        """Calculate number of input tokens (including padding)"""
        if (self.use_cuda_graph and
                num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            # Pad tokens to multiple of tensor_parallel_size when
            # enabled collective fusion for SP
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if (self.compilation_config.pass_config.enable_sequence_parallelism
                    and tp_size > 1):
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens
        
        return num_input_tokens
    
    def _prepare_model_inputs(
        self,
        num_scheduled_tokens: int,
        num_input_tokens: int,
        mm_embeds: List,
    ):
        """Prepare model inputs (input_ids or inputs_embeds)"""
        if self.is_multimodal_model and get_pp_group().is_first_rank:
            # Multimodal: use embeddings
            input_ids_slice = self.input_ids[:num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids_slice, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids_slice)
            
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # Text-only: use token IDs
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        
        return input_ids, inputs_embeds
    
    def _sample_with_rejection(
        self,
        logits: torch.Tensor,
        spec_decode_metadata,
        sampling_metadata,
    ):
        """
        Verify draft tokens using rejection sampling.
        
        This is our own managed logic, independent of vLLM's internal implementation.
        
        Logic matches original vllm/model_runner.py L412-441:
        - When indexing with bonus_logits_indices, PyTorch creates a new tensor
        - Sample bonus token from bonus_logits
        - Use rejection sampler to verify draft tokens with target_logits
        - draft_probs is set to None (draft tokens without probability distributions)
        """
        # 1. Sample bonus token
        bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
        sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=sampling_metadata,
        )
        bonus_token_ids = sampler_output.sampled_token_ids
        
        # 2. Use rejection sampler to verify draft tokens
        target_logits = logits[spec_decode_metadata.target_logits_indices]
        output_token_ids = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs (not provided for cache-based draft methods)
            target_logits,
            bonus_token_ids,
            sampling_metadata,
        )
        
        # 3. Update sampled token IDs
        sampler_output.sampled_token_ids = output_token_ids
        return sampler_output
    
    def _get_discard_indices(
        self,
        scheduler_output: SchedulerOutput,
    ) -> List[int]:
        """
        Get request indices whose sampled tokens should be discarded (partial prefill).
        
        This is our own managed logic.
        
        Logic matches original vllm/model_runner.py L447-463:
        - Iterate over requests and check if seq_len < num_tokens
        - For partial prefills, rewind generator state by 4 bytes
        - Record indices to discard
        """
        discard_indices = []
        
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                      scheduler_output.num_scheduled_tokens[req_id])
            
            if seq_len < req_state.num_tokens:
                # Partial prefill: need to discard
                # Rewind generator state as if token was not sampled
                # This relies on cuda-specific torch-internal impl details
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                
                discard_indices.append(i)
        
        return discard_indices
    
    def _parse_sampled_tokens(
        self,
        sampled_token_ids: torch.Tensor,
        spec_decode_metadata,
        discard_indices: List[int],
    ) -> List[List[int]]:
        """
        Parse sampled tokens (including speculative decoding tokens).
        
        This is our own managed logic.
        
        Logic matches original vllm/model_runner.py L477-491:
        - If max_gen_len == 1, no spec decode tokens, just tolist()
        - Otherwise, use rejection_sampler.parse_output()
        - Mask out sampled tokens from discard_indices
        """
        max_gen_len = sampled_token_ids.shape[-1]
        
        if max_gen_len == 1:
            # No spec decode tokens
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens, need to parse
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )
        
        # Mask out sampled tokens that should not be sampled
        for i in discard_indices:
            valid_sampled_token_ids[i].clear()
        
        return valid_sampled_token_ids
    
    def _update_token_cache(
        self,
        valid_sampled_token_ids: List[List[int]],
    ):
        """
        Update token cache (vLLM's internal state).
        
        Logic matches original vllm/model_runner.py L493-523:
        - Cache sampled tokens in model runner so scheduler doesn't need to send them back
        - Update token_ids_cpu, num_tokens_no_spec, num_tokens
        - Update request state's output_token_ids
        """
        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue
            
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            
            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            
            # Check not exceeding max length
            assert end_idx <= self.max_model_len, (
                f"Token count {end_idx} exceeds max_model_len {self.max_model_len}")
            
            # Update token_ids_cpu
            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            
            # Update request state
            req_state.output_token_ids.extend(sampled_ids)
    
    # ==================== Phase 6 - Arctic/Suffix Specific Logic ====================
    # 
    # The methods below implement Arctic-specific speculative decoding logic.
    # These are the ONLY methods in this file that contain suffix cache specific code.
    # 
    # All logic above (Phase 1-5) is general vLLM logic that works with any
    # speculative decoding method.
    # ==================================================================================
    
    def propose_draft_token_ids(
        self,
        scheduler_output: SchedulerOutput,
        sampled_token_ids: List[List[int]],
        original_sampled_token_ids: torch.Tensor,
        sampling_metadata: Any,  # SamplingMetadata
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: Optional[torch.Tensor],
        spec_decode_metadata: Optional[Any],  # SpecDecodeMetadata
        attn_metadata: Dict[str, Any],
    ) -> List[List[int]]:
        """
        Propose draft tokens for speculative decoding.
        
        This method supports different speculative decoding methods.
        For suffix-only mode, we only implement suffix cache proposal.
        
        Args:
            scheduler_output: Output from scheduler
            sampled_token_ids: Sampled token IDs (after validation)
            original_sampled_token_ids: Original sampled token IDs (before validation)
            sampling_metadata: Sampling metadata
            hidden_states: Model hidden states
            sample_hidden_states: Hidden states used for sampling
            aux_hidden_states: Auxiliary hidden states (for MLP speculator)
            spec_decode_metadata: Speculative decoding metadata
            attn_metadata: Attention metadata
        
        Returns:
            List of draft token IDs for each request
        """
        # Check if speculative decoding should be disabled for large batches
        disable_spec_decode = (
            self.speculative_config and
            self.speculative_config.disable_by_batch_size and
            len(self.input_batch.req_ids) > self.speculative_config.disable_by_batch_size
        )
        if disable_spec_decode:
            # No speculative decoding is enabled
            return [[] for _ in sampled_token_ids]
        
        new_sampled_token_ids = sampled_token_ids.copy()
        
        # Initialize variables
        suffix_spec_token_ids = None
        spec_token_ids = None
        
        # For suffix-only mode, we only support suffix cache
        if self.speculative_config.method == "suffix":
            if not self.suffix_extension.is_enabled():
                return [[] for _ in sampled_token_ids]
            
            # Apply suffix cache to propose draft tokens
            suffix_spec_token_ids = self._apply_suffix_cache(new_sampled_token_ids)
        else:
            # Delegate to vLLM's original propose_draft_token_ids for other methods
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
        
        # Merge suffix cache results with other spec decode results
        if spec_token_ids is None:
            spec_token_ids = suffix_spec_token_ids
        elif suffix_spec_token_ids is not None:
            spec_token_ids = [
                suffix_spec_token_ids[i] or spec_token_ids[i]
                for i in range(len(suffix_spec_token_ids))
            ]
        
        return spec_token_ids
    
    def _apply_suffix_cache(
        self,
        sampled_token_ids: List[List[int]],
    ) -> Optional[List[List[int]]]:
        """
        Apply suffix cache to propose tokens.
        
        This is Arctic-specific logic:
        1. Update suffix cache with new sampled tokens
        2. Propose speculative tokens from cache
        3. Filter by score threshold
        """
        # 1. Update suffix cache
        self.suffix_extension.update_cache(
            sampled_token_ids=sampled_token_ids,
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            token_ids_cpu=self.input_batch.token_ids_cpu,
            num_prompt_tokens=self.input_batch.num_prompt_tokens,
        )
        
        # 2. Propose speculative tokens
        suffix_results = self.suffix_extension.propose_tokens(
            sampled_token_ids=sampled_token_ids,
            req_ids=self.input_batch.req_ids,
            token_ids_cpu=self.input_batch.token_ids_cpu,
            num_tokens_no_spec=self.input_batch.num_tokens_no_spec,
            max_model_len=self.max_model_len,
        )
        
        # 3. Filter by score
        min_score = self.suffix_extension.get_min_score()
        spec_token_ids = []
        for result in suffix_results:
            if result.score >= min_score:
                spec_token_ids.append(result.token_ids)
            else:
                spec_token_ids.append([])
        
        return spec_token_ids
