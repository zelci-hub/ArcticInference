# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Optional, Union, Sequence, Any, Callable
from vllm.logger import init_logger

from arctic_inference.vllm.model_runner import ProblemIdContextManager

logger = init_logger(__name__)


# Storage for original methods
class LLMPatch:
    """Storage class for original LLM methods."""
    _orig_generate = None
    _orig_validate_and_add_requests = None
    _orig_add_request = None


def apply_llm_patches():
    """Apply LLM patches for problem_id support."""
    try:
        from vllm.entrypoints.llm import LLM
        
        # Check if already patched
        if hasattr(LLM, '_arctic_problem_id_patched'):
            logger.debug("LLM already patched for problem_id support")
            return
        
        # Store original methods in the patch class
        LLMPatch._orig_generate = LLM.generate
        LLMPatch._orig_validate_and_add_requests = LLM._validate_and_add_requests
        LLMPatch._orig_add_request = LLM._add_request
        
        # Apply patches by directly replacing methods
        import types
        
        # Import the deprecation decorator 
        from vllm.utils import deprecate_kwargs
        
        # Create unbound methods and bind them to LLM instances
        @deprecate_kwargs(
            "prompt_token_ids",
            is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
            additional_message="Please use the 'prompts' parameter instead.",
        )
        def generate_patch(llm_self, prompts = None, sampling_params = None, prompt_token_ids = None, 
                          use_tqdm = True, lora_request = None, prompt_adapter_request = None, 
                          guided_options_request = None, priority = None, 
                          problem_ids: Optional[Union[str, Sequence[str]]] = None, **kwargs):
            """Patched generate method that accepts problem_ids parameter and calls patched methods."""
            # Convert single problem_id to list if needed
            if isinstance(problem_ids, str):
                problem_ids = [problem_ids]
            
            # If problem_ids provided, initialize context
            if problem_ids is not None:
                # Set up problem_ids context
                ProblemIdContextManager.set_current_batch_problem_ids(problem_ids)
                logger.debug(f"Setting up problem_ids context with {len(problem_ids)} IDs")
            else:
                print(f"in generate_patch: DEBUG: problem_ids is None")
            
            # Call the patched validate_and_add_requests method
            llm_self._validate_and_add_requests(
                prompts, 
                sampling_params, 
                use_tqdm=use_tqdm, 
                lora_request=lora_request, 
                prompt_adapter_request=prompt_adapter_request, 
                tokenization_kwargs=kwargs.get('tokenization_kwargs'),
                guided_options=guided_options_request, 
                priority=priority
            )
            
            # Generate outputs using the original logic
            outputs = []
            while llm_self.llm_engine.has_unfinished_requests():
                step_outputs = llm_self.llm_engine.step()
                for output in step_outputs:
                    if output.finished:
                        outputs.append(output)
            
            # Sort outputs by request ID to match input order
            outputs = sorted(outputs, key=lambda x: int(x.request_id))
            
            # Clear context after generation
            ProblemIdContextManager.clear_context()
            
            return outputs

        def validate_and_add_requests_patch(llm_self, prompts, params, *,
                                           use_tqdm = True, lora_request = None, prompt_adapter_request = None, 
                                           tokenization_kwargs = None, guided_options = None, priority = None, **kwargs):
            # Handle guided_options deprecation warning (from original code)
            if guided_options is not None:
                warnings.warn(
                    "guided_options is deprecated. Please use guided_options_request instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
            
            # Validate inputs (same as original code)
            if prompts is None:
                raise ValueError("prompts must be provided.")
            if isinstance(prompts, str):
                prompts = [prompts]
            if isinstance(params, dict):
                params = [params for _ in prompts]
            
            num_requests = len(prompts)
            
            # Get problem_ids from context
            problem_ids = None
            try:
                problem_ids = ProblemIdContextManager.get_current_batch_problem_ids()
            except:
                pass  # No problem_ids context set
            
            # Validate problem_ids length if provided
            if problem_ids is not None and len(problem_ids) != num_requests:
                raise ValueError(f"The lengths of prompts and problem_ids "
                               f"must be the same. Got {num_requests} prompts and {len(problem_ids)} problem_ids.")

            # Handle sampling params processing (same as original code)
            for sp in params if isinstance(params, Sequence) else (params, ):
                # Import here to match original behavior
                from vllm.sampling_params import SamplingParams, RequestOutputKind
                if isinstance(sp, SamplingParams):
                    # Add guided params (same as original - this was missing!)
                    llm_self._add_guided_params(sp, guided_options)
                    
                    # We only care about the final output
                    sp.output_kind = RequestOutputKind.FINAL_ONLY

            # Add requests to the engine (modified to include problem_ids)
            it = prompts
            if use_tqdm:
                try:
                    from tqdm import tqdm
                    tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
                    it = tqdm_func(it, desc="Adding requests")
                except ImportError:
                    pass

            for i, prompt in enumerate(it):
                # Safe way to get problem_id that works with both lists and numpy arrays
                current_problem_id = None
                if problem_ids is not None and i < len(problem_ids):
                    current_problem_id = problem_ids[i]
                llm_self._add_request(
                    prompt,
                    params[i] if isinstance(params, Sequence) else params,
                    tokenization_kwargs=tokenization_kwargs,
                    lora_request=lora_request[i] if isinstance(
                        lora_request, Sequence) else lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                    priority=priority[i] if priority else 0,
                    problem_id=current_problem_id,  # Pass problem_id to _add_request
                )
        
        def add_request_patch(llm_self, prompt, params, tokenization_kwargs = None, lora_request = None, 
                            prompt_adapter_request = None, priority = 0, problem_id: Optional[str] = None, **kwargs):
            # Generate req_id exactly like the original method does
            req_id = str(next(llm_self.request_counter))
            
            # Call llm_engine.add_request directly (same as original implementation)
            llm_self.llm_engine.add_request(
                req_id,
                prompt,
                params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority,
            )
            
            # Record the req_id to problem_id mapping if problem_id provided
            if problem_id is not None:
                try:
                    current_mapping = ProblemIdContextManager.get_req_id_to_problem_id_mapping()
                    current_mapping[req_id] = problem_id
                    ProblemIdContextManager.set_req_id_to_problem_id_mapping(current_mapping)
                    logger.debug(f"Mapped req_id {req_id} -> problem_id {problem_id}")
                    #print(f"Mapped req_id {req_id} -> problem_id {problem_id}")
                except Exception as e:
                    logger.warning(f"Failed to record req_id mapping for {req_id}: {e}")
                    print(f"Failed to record req_id mapping for {req_id}: {e}")
            else:
                print(f"in add_request_patch: DEBUG: problem_id is None for req_id {req_id}")
            
            # Return None (same as original method)
            return None
        
        LLM.generate = generate_patch
        LLM._validate_and_add_requests = validate_and_add_requests_patch
        LLM._add_request = add_request_patch
        
        # Mark as patched
        LLM._arctic_problem_id_patched = True
        
        logger.info("LLM patches applied successfully for problem_id support")
        
    except ImportError as e:
        logger.warning(f"Failed to apply LLM patches - vLLM not available: {e}")
    except Exception as e:
        logger.error(f"Error applying LLM patches: {e}")
        raise


def unapply_llm_patches():
    """Remove LLM patches."""
    try:
        from vllm.entrypoints.llm import LLM
        
        if not hasattr(LLM, '_arctic_problem_id_patched'):
            logger.debug("LLM patches not applied, nothing to remove")
            return
        
        # Restore original methods
        if LLMPatch._orig_generate is not None:
            LLM.generate = LLMPatch._orig_generate
        if LLMPatch._orig_validate_and_add_requests is not None:
            LLM._validate_and_add_requests = LLMPatch._orig_validate_and_add_requests
        if LLMPatch._orig_add_request is not None:
            LLM._add_request = LLMPatch._orig_add_request
        
        # Remove patch marker
        delattr(LLM, '_arctic_problem_id_patched')
        
        logger.info("LLM patches removed successfully")
        
    except ImportError as e:
        logger.warning(f"Failed to remove LLM patches - vLLM not available: {e}")
    except Exception as e:
        logger.error(f"Error removing LLM patches: {e}")
        raise