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

import os
import vllm
from vllm.logger import init_logger
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.worker.worker_base import WorkerBase

from arctic_inference.patching import ArcticPatch
from arctic_inference.utils import get_compatible_vllm_version
from arctic_inference.vllm.args import EngineArgsPatch, AsyncEngineArgsPatch
from arctic_inference.vllm.config import (ParallelConfigPatch,
                                          SpeculativeConfigPatch,
                                          VllmConfigPatch,
                                          MLPSpeculatorConfigPatch)
from arctic_inference.vllm.stats import (SpecDecodingStatsPatch, 
                                         SpecDecodingLoggingPatch)
from arctic_inference.vllm.ulysses import apply_shift_parallel_patches


logger = init_logger(__name__)


def apply_minimal_attention_patch():
    """Apply a minimal patch to handle layer_idx and dual_chunk_attention_config parameters
    when Ulysses is disabled, to maintain compatibility with Qwen2 models."""
    from vllm.attention.layer import Attention
    
    # Store the original __init__ method
    _orig_init = Attention.__init__
    
    def patched_init(self, *args, **kwargs):
        # Remove parameters that the original Attention.__init__ doesn't accept
        kwargs.pop("layer_idx", None)
        kwargs.pop("dual_chunk_attention_config", None)
        return _orig_init(self, *args, **kwargs)
    
    # Apply the patch
    Attention.__init__ = patched_init
    logger.info("Applied minimal attention patch to handle layer_idx and dual_chunk_attention_config parameters")


class EngineCoreProcPatch(ArcticPatch[EngineCoreProc]):

    _orig_run_engine_core = EngineCoreProc.run_engine_core

    @staticmethod
    def run_engine_core(*args, **kwargs):
        # When starting the API server, it will spawn a new process to run the
        # EngineCore. We need to load the plugins in the new process before it
        # initializes the Executor.
        vllm.plugins.load_general_plugins()
        return EngineCoreProcPatch._orig_run_engine_core(*args, **kwargs)


class WorkerBasePatch(ArcticPatch[WorkerBase]):

    _orig_init = WorkerBase.__init__

    def __init__(self, *args, **kwargs):
        # Some patches like the GPUModelRunner will import CUDA libraries when
        # they are initialized, which will cause process forking to fail. For
        # these patches, we need to delay the initialization until after the
        # process has been forked (i.e., in the WorkerBase initializer).
        from arctic_inference.vllm.model_runner import GPUModelRunnerPatch

        GPUModelRunnerPatch.apply_patch()

        return self._orig_init(*args, **kwargs)


def arctic_inference_plugin():
    if (vllm.__version__ != get_compatible_vllm_version() and not
            vllm.__version__.startswith("0.1.dev")):  # Make it work with dev
        logger.warning(
            f"ArcticInference requires vllm=={get_compatible_vllm_version()} "
            f"but found vllm=={vllm.__version__}. Ignoring plugin!")
        return
    
    if not vllm.platforms.current_platform.is_cuda():
        logger.warning(
            f"ArcticInference requires the cuda platform. Ignoring plugin!")
        return
    
    if os.getenv("VLLM_USE_V1") == "0":
        logger.warning("ArcticInference only supports vLLM V1, but detected V0 engine. "
                       "Ignoring plugin!\n"
                       "Hint: To strictly enforce the V1 vLLM engine, please set "
                       "VLLM_USE_V1=1.")
        return

    from transformers import AutoConfig
    from arctic_inference.common.swiftkv import LlamaSwiftKVConfig

    # Register SwiftKV model configurations to transformers.
    AutoConfig.register("llama_swiftkv", LlamaSwiftKVConfig)

    from vllm import ModelRegistry
    #from arctic_inference.vllm.swiftkv import LlamaSwiftKVForCausalLM

    # Register SwiftKV model definitions to vLLM.
    ModelRegistry.register_model(
        "LlamaSwiftKVForCausalLM",
        "arctic_inference.vllm.swiftkv:LlamaSwiftKVForCausalLM")

    # Register ArcticSpeculator models to vLLM.
    from arctic_inference.vllm.spec_dec.arctic_speculator import (
        ArcticMLPSpeculator, ArcticLSTMSpeculator)
    ModelRegistry.register_model("ArcticMLPSpeculatorPreTrainedModel",
                                 ArcticMLPSpeculator)
    ModelRegistry.register_model("ArcticLSTMSpeculatorPreTrainedModel",
                                 ArcticLSTMSpeculator)
    # This name is currently used in corvo
    ModelRegistry.register_model("MLPVariantSpeculatorPreTrainedModel",
                                 ArcticLSTMSpeculator)

    # Patches that make later patches work properly.
    EngineCoreProcPatch.apply_patch()
    WorkerBasePatch.apply_patch()
    
    # Apply LLM patches for problem_id support (early application)
    from arctic_inference.vllm.llm import apply_llm_patches
    apply_llm_patches()

    # Patches to vLLM arguments and configuration objects.
    EngineArgsPatch.apply_patch()
    AsyncEngineArgsPatch.apply_patch()
    ParallelConfigPatch.apply_patch()
    SpeculativeConfigPatch.apply_patch()
    SpecDecodingStatsPatch.apply_patch()
    SpecDecodingLoggingPatch.apply_patch()
    VllmConfigPatch.apply_patch()
    MLPSpeculatorConfigPatch.apply_patch()

    # Main optimization patches.
    # Check if Ulysses/shift parallel patches should be disabled
    disable_ulysses = os.getenv("ARCTIC_DISABLE_ULYSSES", "0") == "1"
    if disable_ulysses:
        logger.info("Ulysses sequence parallelism patches disabled by ARCTIC_DISABLE_ULYSSES=1")
        # Apply a minimal patch to handle layer_idx and dual_chunk_attention_config parameters
        # even when Ulysses is disabled, to maintain compatibility with Qwen2 models
        apply_minimal_attention_patch()
    else:
        apply_shift_parallel_patches()
        apply_minimal_attention_patch()
