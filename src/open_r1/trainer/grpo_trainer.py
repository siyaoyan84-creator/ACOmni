# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
import logging
import traceback
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Optional, Union, Sized

import torch
import torch.utils.data
import transformers
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
import datasets
from datasets import Dataset, IterableDataset
from packaging import version

from .grpo_trainer_microbatch_helper import slice_model_inputs_for_batch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
try:
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
except ImportError:
    try:
        from transformers.trainer_utils import is_deepspeed_zero3_enabled
    except ImportError:
        def is_deepspeed_zero3_enabled():
            return False
from transformers.utils import is_peft_available, is_datasets_available
try:
    from transformers.utils import is_rich_available
except ImportError:
    def is_rich_available():
        try:
            import rich
            return True
        except ImportError:
            return False
from transformers.trainer_utils import seed_worker
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, print_prompt_completions_sample
from trl import GRPOTrainer
from trl.import_utils import is_deepspeed_available

# Import DeepSpeed for zero.Init context manager
try:
    import deepspeed
    from deepspeed import zero
    DEEPSPEED_ZERO_INIT_AVAILABLE = True
except ImportError:
    DEEPSPEED_ZERO_INIT_AVAILABLE = False
    deepspeed = None
    zero = None
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings
import torch.distributed as torch_dist


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from open_r1.vlm_modules.vlm_module import VLMBaseModule
from open_r1.bad_sample_handler import BadSampleTracker, SafeCollatorWrapper

# Initialize logger
logger = logging.getLogger(__name__)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def _is_valid_processor_path(path: str) -> bool:
    """
    Check if a directory contains necessary files to be used as a processor path.

    Args:
        path: Directory path to check

    Returns:
        True if the directory contains preprocessor_config.json, False otherwise
    """
    if not os.path.exists(path):
        return False

    # Check for preprocessor_config.json (required)
    preprocessor_config = os.path.join(path, "preprocessor_config.json")
    if not os.path.exists(preprocessor_config):
        return False

    return True


def find_mask_between_patterns_1d(input_tensor: torch.Tensor, 
                                  start_pattern_list: list, 
                                  end_pattern_list: list) -> torch.Tensor:
    """
    (Helper function - same as before)
    Finds the mask for a single 1D tensor.
    """
    assert input_tensor.ndim == 1, "Input tensor must be 1-dimensional"
    
    device = input_tensor.device
    dtype = input_tensor.dtype # Use input tensor's dtype

    # Ensure patterns are tensors on the correct device and dtype
    start_pattern = torch.tensor(start_pattern_list, dtype=dtype, device=device)
    end_pattern = torch.tensor(end_pattern_list, dtype=dtype, device=device)

    n = input_tensor.shape[0]
    len_start = len(start_pattern)
    len_end = len(end_pattern)

    start_idx = -1
    end_idx = -1

    # --- Find start_pattern index ---
    if n >= len_start:
        start_windows = input_tensor.unfold(0, len_start, 1)
        start_matches = (start_windows == start_pattern).all(dim=1)
        start_indices = start_matches.nonzero(as_tuple=True)[0]
        if start_indices.numel() > 0:
            start_idx = start_indices[0].item() # Assume first match
        else:
            # Indicate pattern not found for this row
            return torch.zeros_like(input_tensor, dtype=torch.long, device=device) 
            # raise ValueError("Start pattern not found in the tensor.") # Original behavior
    else:
        return torch.zeros_like(input_tensor, dtype=torch.long, device=device) # Too short

    # --- Find end_pattern index ---
    if n >= len_end:
        # Search *after* the start pattern to ensure correct order if multiple end patterns exist
        # Although problem states "only one region", this adds robustness
        search_area_end = input_tensor[start_idx + len_start:] 
        if search_area_end.numel() >= len_end:
            end_windows = search_area_end.unfold(0, len_end, 1)
            end_matches = (end_windows == end_pattern).all(dim=1)
            end_indices = end_matches.nonzero(as_tuple=True)[0]
            if end_indices.numel() > 0:
                 # Index relative to the start of search_area_end, need to add offset
                relative_end_idx = end_indices[0].item()
                end_idx = start_idx + len_start + relative_end_idx 
            else:
                 # End pattern not found *after* start pattern
                return torch.zeros_like(input_tensor, dtype=torch.long, device=device)
        else:
            # Not enough elements after start pattern to contain end pattern
             return torch.zeros_like(input_tensor, dtype=torch.long, device=device)
    else:
       return torch.zeros_like(input_tensor, dtype=torch.long, device=device) # Too short

    # --- Calculate mask region ---
    mask_start = start_idx + len_start
    mask_end = end_idx # end_idx is the *start* of the end pattern

    # mask_start = start_idx 
    # mask_end = end_idx + len_end 

    # --- Create and fill mask ---
    mask = torch.zeros_like(input_tensor, dtype=torch.long, device=device)

    # if mask_start < mask_end:
    #     mask[mask_start:-1] = 1
    if mask_start < mask_end:
        mask[:mask_end] = 1

    # if mask_start < mask_end:
    #     mask[mask_start:mask_end] = 1
    # else: patterns adjacent or end before start, mask remains zero, no warning needed here

    return mask



def generate_2d_mask(input_tensor_2d: torch.Tensor, 
                       start_pattern_list: list, 
                       end_pattern_list: list) -> torch.Tensor:
    """
    Generates a 2D mask by applying the 1D pattern finding logic to each row.

    Args:
        input_tensor_2d: The input 2D PyTorch Tensor (Batch x SequenceLength).
        start_pattern_list: The start pattern list.
        end_pattern_list: The end pattern list.

    Returns:
        A 2D mask tensor of the same shape as input_tensor_2d (dtype=torch.long),
        where each row's mask is generated based on the patterns found in that row.
        Rows where patterns are not found (or order is wrong) will have a mask of all zeros.
    """
    assert input_tensor_2d.ndim == 2, "Input tensor must be 2-dimensional"
    
    num_rows = input_tensor_2d.shape[0]
    if num_rows == 0:
        return torch.empty_like(input_tensor_2d, dtype=torch.long) # Handle empty input

    row_masks = []
    for i in range(num_rows):
        current_row = input_tensor_2d[i]
        # Call the 1D function for the current row
        # Modify 1D function to return zeros instead of raising error if pattern not found
        mask_1d = find_mask_between_patterns_1d(current_row, start_pattern_list, end_pattern_list)
        row_masks.append(mask_1d)

    # Stack the generated 1D masks along the batch dimension (dim=0)
    mask_2d = torch.stack(row_masks, dim=0)
    
    return mask_2d

class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        # Ensure batch_size is at least 1 to avoid division by zero in range()
        self.batch_size = max(1, int(batch_size))
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # Safety check: ensure batch_size is valid before iteration
        if self.batch_size <= 0:
            raise ValueError(
                f"RepeatRandomSampler: batch_size must be > 0, got {self.batch_size}. "
                f"Context: mini_repeat_count={self.mini_repeat_count}, "
                f"repeat_count={self.repeat_count}, num_samples={self.num_samples}"
            )

        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class VLMGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = True,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        processor_name_or_path: Optional[str] = None,
        bad_sample_tracker: Optional[BadSampleTracker] = None,
        **kwargs,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Log DeepSpeed configuration status
        logger.info("=" * 60)
        logger.info("VLMGRPOTrainer Initialization")
        logger.info("=" * 60)
        logger.info(f"Model: {model}")
        logger.info(f"DeepSpeed enabled: {is_deepspeed_available()}")
        logger.info(f"DeepSpeed ZeRO3 enabled: {is_deepspeed_zero3_enabled()}")
        logger.info(f"DeepSpeed zero.Init available: {DEEPSPEED_ZERO_INIT_AVAILABLE}")
        if args.deepspeed:
            logger.info(f"DeepSpeed config: {args.deepspeed}")
        logger.info(f"PEFT config: {peft_config is not None}")
        logger.info(f"Freeze vision modules: {freeze_vision_modules}")
        logger.info(f"Processor path: {processor_name_or_path or 'auto-detect'}")
        if is_deepspeed_zero3_enabled() and DEEPSPEED_ZERO_INIT_AVAILABLE:
            logger.info("Ref model init mode: LOW-PEAK (zero.Init context)")
        elif is_deepspeed_zero3_enabled():
            logger.warning("Ref model init mode: HIGH-PEAK (zero.Init not available)")
        else:
            logger.info("Ref model init mode: STANDARD (non-ZeRO3)")
        logger.info("=" * 60)

        self.vlm_module = vlm_module
        self.bad_sample_tracker = bad_sample_tracker or BadSampleTracker()

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}

        # [GRPO_ATTN_IMPL] Read attention implementation from environment variable
        grpo_attn_implementation = os.environ.get("GRPO_ATTN_IMPLEMENTATION", None)

        if grpo_attn_implementation is not None:
            print(f"[GRPO_ATTN_IMPL] env GRPO_ATTN_IMPLEMENTATION = {grpo_attn_implementation}", flush=True)
            logger.info(f"[GRPO_ATTN_IMPL] env GRPO_ATTN_IMPLEMENTATION = {grpo_attn_implementation}")
            model_init_kwargs["attn_implementation"] = grpo_attn_implementation
            print(f"[GRPO_ATTN_IMPL] final attn_implementation = {grpo_attn_implementation}", flush=True)
            logger.info(f"[GRPO_ATTN_IMPL] final attn_implementation = {grpo_attn_implementation}")
            print(f"[GRPO_ATTN_IMPL] applied to policy/base model loading = True", flush=True)
            logger.info(f"[GRPO_ATTN_IMPL] applied to policy/base model loading = True")
        else:
            # FIXME
            # Remember to modify it in the invernvl
            model_init_kwargs["attn_implementation"] = attn_implementation
            print(f"[GRPO_ATTN_IMPL] env GRPO_ATTN_IMPLEMENTATION = None", flush=True)
            logger.info(f"[GRPO_ATTN_IMPL] env GRPO_ATTN_IMPLEMENTATION = None")
            print(f"[GRPO_ATTN_IMPL] final attn_implementation = {attn_implementation} (from args)", flush=True)
            logger.info(f"[GRPO_ATTN_IMPL] final attn_implementation = {attn_implementation} (from args)")
            print(f"[GRPO_ATTN_IMPL] applied to policy/base model loading = True", flush=True)
            logger.info(f"[GRPO_ATTN_IMPL] applied to policy/base model loading = True")

        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # model_init_kwargs["enable_audio_output"] = False
        # model_init_kwargs["use_cache"] = (
        #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        # )
        #     # Disable caching if gradient checkpointing is enabled (not supported)
        # model_init_kwargs["use_cache"] = (
        #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        # )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)

        # [GRPO_ADAPTER_LOAD] Check if model_id is an adapter-only STAGE1 checkpoint
        is_adapter_only = (
            os.path.exists(os.path.join(model_id, "adapter_config.json"))
            and os.path.exists(os.path.join(model_id, "adapter_model.safetensors"))
        )

        if is_adapter_only:
            logger.info("[GRPO_ADAPTER_LOAD] Detected adapter-only STAGE1 checkpoint")
            logger.info(f"[GRPO_ADAPTER_LOAD] adapter_path = {model_id}")

            # Determine base model path with priority order
            base_model_path = None

            # Priority 1: GRPO_BASE_MODEL_PATH environment variable
            if os.environ.get("GRPO_BASE_MODEL_PATH"):
                base_model_path = os.environ.get("GRPO_BASE_MODEL_PATH")
                logger.info(f"[GRPO_ADAPTER_LOAD] base_model_path from GRPO_BASE_MODEL_PATH = {base_model_path}")

            # Priority 2: BASE_MODEL environment variable
            elif os.environ.get("BASE_MODEL"):
                base_model_path = os.environ.get("BASE_MODEL")
                logger.info(f"[GRPO_ADAPTER_LOAD] base_model_path from BASE_MODEL = {base_model_path}")

            # Priority 3: adapter_config.json base_model_name_or_path
            else:
                adapter_config_path = os.path.join(model_id, "adapter_config.json")
                try:
                    import json
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                    base_model_path = adapter_config.get("base_model_name_or_path")
                    if base_model_path:
                        logger.info(f"[GRPO_ADAPTER_LOAD] base_model_path from adapter_config.json = {base_model_path}")
                except Exception as e:
                    logger.warning(f"[GRPO_ADAPTER_LOAD] Failed to read adapter_config.json: {e}")

            # Priority 4: Hardcoded default
            if not base_model_path:
                base_model_path = os.environ.get("BASE_MODEL") or os.environ.get("MODEL_PATH")
            if not base_model_path:
                raise RuntimeError("BASE_MODEL or MODEL_PATH must be set for evaluation")
                logger.info(f"[GRPO_ADAPTER_LOAD] base_model_path from hardcoded default = {base_model_path}")

            logger.info(f"[GRPO_ADAPTER_LOAD] final base_model_path = {base_model_path}")

            # Load base model first
            logger.info("[GRPO_ADAPTER_LOAD] loading base model first")
            # Clean model_init_kwargs to remove PEFT-specific parameters
            clean_model_init_kwargs = {k: v for k, v in model_init_kwargs.items() if k not in ['peft_config']}
            base_model = model_cls.from_pretrained(base_model_path, **clean_model_init_kwargs)

            # Load adapter with PeftModel.from_pretrained
            logger.info("[GRPO_ADAPTER_LOAD] loading adapter with PeftModel.from_pretrained")
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, model_id, is_trainable=True)

            # Diagnostics: count loaded LoRA parameters
            loaded_total_lora = 0
            loaded_audio_tower_lora = 0
            loaded_visual_or_vision_lora = 0
            loaded_model_layers_lora = 0

            for n, p in model.named_parameters():
                if 'lora_' in n:
                    loaded_total_lora += 1
                    if 'audio_tower' in n:
                        loaded_audio_tower_lora += 1
                    elif 'visual' in n or 'vision' in n:
                        loaded_visual_or_vision_lora += 1
                    elif 'model.layers' in n:
                        loaded_model_layers_lora += 1

            logger.info(f"[GRPO_ADAPTER_LOAD] loaded_total_lora_params = {loaded_total_lora}")
            logger.info(f"[GRPO_ADAPTER_LOAD] loaded_audio_tower_lora = {loaded_audio_tower_lora}")
            logger.info(f"[GRPO_ADAPTER_LOAD] loaded_visual_or_vision_lora = {loaded_visual_or_vision_lora}")
            logger.info(f"[GRPO_ADAPTER_LOAD] loaded_model_layers_lora = {loaded_model_layers_lora}")

            # Validation: ensure all three LoRA types are loaded
            if loaded_audio_tower_lora == 0:
                raise RuntimeError(f"[GRPO_ADAPTER_LOAD] Failed to load audio_tower LoRA from {model_id}")
            if loaded_visual_or_vision_lora == 0:
                raise RuntimeError(f"[GRPO_ADAPTER_LOAD] Failed to load visual/vision LoRA from {model_id}")
            if loaded_model_layers_lora == 0:
                raise RuntimeError(f"[GRPO_ADAPTER_LOAD] Failed to load model.layers LoRA from {model_id}")

            logger.info("[GRPO_ADAPTER_LOAD] Adapter-only checkpoint loaded successfully")
        else:
            # Standard full model loading
            logger.info("[GRPO_ADAPTER_LOAD] Loading as full model (not adapter-only)")
            model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        # model = model.thinker # for qwen-omni

        # [PEFT_DIAG] After model loading
        self._print_peft_diag(model, "policy][after_load", adapter_path=model_id)

        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()

        # Check if model is already a PEFT model
        is_already_peft = hasattr(model, 'peft_config') or is_peft_model(model)
        logger.info(f"[PEFT_CHECK] Model is already PEFT: {is_already_peft}")

        if peft_config is not None:
            if is_already_peft:
                # Model is already wrapped with PEFT, don't wrap again
                logger.info("[PEFT_CHECK] Model already has PEFT config, skipping get_peft_model to avoid double wrapping")

                # Ensure LoRA parameters are trainable (they may be frozen from SFT checkpoint)
                logger.info("[PEFT_CHECK] Ensuring LoRA parameters are trainable...")
                lora_params_found = 0
                for n, p in model.named_parameters():
                    if 'lora_' in n or 'q_proj' in n or 'v_proj' in n:
                        # Skip vision modules
                        if not any(keyword in n for keyword in self.vision_modules_keywords):
                            if not p.requires_grad:
                                p.requires_grad = True
                                lora_params_found += 1
                logger.info(f"[PEFT_CHECK] Set {lora_params_found} LoRA/projection parameters to requires_grad=True")
            else:
                # Model is not PEFT yet, apply PEFT wrapping
                def find_all_linear_names(model, multimodal_keywords):
                    cls = torch.nn.Linear
                    lora_module_names = set()
                    for name, module in model.named_modules():
                        # LoRA is not applied to the vision modules
                        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                            continue
                        if isinstance(module, cls):
                            lora_module_names.add(name)
                    for m in lora_module_names:  # needed for 16-bit
                        if "embed_tokens" in m:
                            lora_module_names.remove(m)
                    return list(lora_module_names)
                target_modules = find_all_linear_names(model, self.vision_modules_keywords)
                peft_config.target_modules = target_modules
                logger.info(f"[PEFT_CHECK] Applying PEFT with target_modules: {target_modules[:3]}... (showing first 3)")
                model = get_peft_model(model, peft_config)
                logger.info("[PEFT_CHECK] PEFT wrapping completed successfully")

        # [PEFT_DIAG] After adapter loading
        self._print_peft_diag(model, "policy][after_adapter", adapter_path=model_id)

        # [PEFT_INFERENCE_MODE_FIX] Fix inference_mode=True in LoraConfig for audio_tower_only training
        grpo_lora_train_scope = os.environ.get("GRPO_LORA_TRAIN_SCOPE", "model_layers_only")
        if grpo_lora_train_scope == "audio_tower_only" and hasattr(model, 'peft_config'):
            logger.info("[PEFT_INFERENCE_MODE_FIX] Checking and fixing inference_mode for audio_tower_only training")
            peft_config_dict = model.peft_config
            for adapter_name, config in peft_config_dict.items():
                inference_mode_before = getattr(config, 'inference_mode', None)
                logger.info(f"[GRPO_AUDIO_GRAD_DIAG][PEFT_INFERENCE_MODE_FIX] adapter='{adapter_name}' before={inference_mode_before}")

                if inference_mode_before is True:
                    # Set inference_mode to False to enable gradient computation
                    config.inference_mode = False
                    inference_mode_after = config.inference_mode
                    logger.info(f"[GRPO_AUDIO_GRAD_DIAG][PEFT_INFERENCE_MODE_FIX] adapter='{adapter_name}' after={inference_mode_after}")
                else:
                    logger.info(f"[GRPO_AUDIO_GRAD_DIAG][PEFT_INFERENCE_MODE_FIX] adapter='{adapter_name}' already False or None, no change needed")

            # Verify audio_tower LoRA parameters are still requires_grad=True after inference_mode fix
            audio_tower_lora_count = 0
            for n, p in model.named_parameters():
                if 'audio_tower' in n and 'lora_' in n and p.requires_grad:
                    audio_tower_lora_count += 1
            logger.info(f"[GRPO_AUDIO_GRAD_DIAG][PEFT_INFERENCE_MODE_FIX] audio_tower LoRA params with requires_grad=True after fix: {audio_tower_lora_count}")
        else:
            logger.info("[PEFT_INFERENCE_MODE_FIX] Skipping inference_mode fix (not audio_tower_only mode or no peft_config)")

        # [LORA_RESTRICT] REMOVED - replaced by unified _apply_lora_train_scope
        # Old logic only supported model_layers_only, now we use unified scope control
        # that supports model_layers_only, audio_tower_only, and joint modes

        # ===== PEFT ADAPTER LOADING DIAGNOSTICS ===== (legacy, kept for compatibility)
        logger.info("[PEFT_DIAG] ===== PEFT Adapter Loading Diagnostics =====")
        logger.info(f"[PEFT_DIAG] Model type: {type(model)}")
        logger.info(f"[PEFT_DIAG] Has peft_config: {hasattr(model, 'peft_config')}")

        if hasattr(model, 'peft_config'):
            peft_config_dict = model.peft_config
            logger.info(f"[PEFT_DIAG] PEFT config keys: {list(peft_config_dict.keys())}")
            for adapter_name, config in peft_config_dict.items():
                logger.info(f"[PEFT_DIAG]   Adapter '{adapter_name}': {config}")

        # Count LoRA parameters
        lora_a_count = 0
        lora_b_count = 0
        lora_trainable_count = 0
        lora_frozen_count = 0
        lora_param_names = []

        for n, p in model.named_parameters():
            if 'lora_A' in n:
                lora_a_count += 1
            if 'lora_B' in n:
                lora_b_count += 1
            if 'lora_' in n:
                if p.requires_grad:
                    lora_trainable_count += 1
                else:
                    lora_frozen_count += 1
                lora_param_names.append((n, p.shape, p.requires_grad))

        logger.info(f"[PEFT_DIAG] LoRA parameters: lora_A={lora_a_count}, lora_B={lora_b_count}")
        logger.info(f"[PEFT_DIAG] LoRA trainable: {lora_trainable_count}, frozen: {lora_frozen_count}")

        # First 20 LoRA parameter names with shapes and requires_grad status
        logger.info(f"[PEFT_DIAG] First 20 LoRA parameters:")
        for i, (name, shape, requires_grad) in enumerate(lora_param_names[:20]):
            logger.info(f"[PEFT_DIAG]   {i+1}. {name} | shape={shape} | requires_grad={requires_grad}")

        # First 20 trainable parameters
        trainable_param_names = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                trainable_param_names.append((n, p.shape))

        logger.info(f"[PEFT_DIAG] First 20 trainable parameters:")
        for i, (name, shape) in enumerate(trainable_param_names[:20]):
            logger.info(f"[PEFT_DIAG]   {i+1}. {name} | shape={shape}")

        # LoRA parameter counts by module
        module_lora_counts = {}
        for n, p in model.named_parameters():
            if 'lora_' in n:
                # Extract module name (e.g., "model.layers", "audio_tower", "visual", "lm_head")
                if 'model.layers' in n:
                    module = 'model.layers'
                elif 'audio_tower' in n:
                    module = 'audio_tower'
                elif 'visual' in n:
                    module = 'visual'
                elif 'lm_head' in n:
                    module = 'lm_head'
                else:
                    module = 'other'

                if module not in module_lora_counts:
                    module_lora_counts[module] = {'total': 0, 'trainable': 0, 'frozen': 0}
                module_lora_counts[module]['total'] += 1
                if p.requires_grad:
                    module_lora_counts[module]['trainable'] += 1
                else:
                    module_lora_counts[module]['frozen'] += 1

        logger.info(f"[PEFT_DIAG] LoRA parameter counts by module:")
        for module, counts in module_lora_counts.items():
            logger.info(f"[PEFT_DIAG]   {module}: total={counts['total']}, trainable={counts['trainable']}, frozen={counts['frozen']}")

        # Adapter loading path
        if hasattr(model, 'active_adapter'):
            active_adapter = getattr(model, 'active_adapter', None)
            if callable(active_adapter):
                active_adapter = active_adapter()
            logger.info(f"[PEFT_DIAG] Active adapter: {active_adapter}")

        # Warnings and errors
        if 'model.layers' not in module_lora_counts or module_lora_counts['model.layers']['total'] == 0:
            logger.warning("[PEFT_DIAG] WARNING: No LoRA parameters found under model.layers!")

        if len(trainable_param_names) == 0:
            logger.error("[PEFT_DIAG] ERROR: No trainable parameters found in model!")

        logger.info("[PEFT_DIAG] ===== End PEFT Diagnostics =====")

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        # [GRPO_LORA_SCOPE_UNIFIED] Unified LoRA training scope control
        # CRITICAL: Apply scope BEFORE any validation checks
        grpo_lora_train_scope = os.environ.get("GRPO_LORA_TRAIN_SCOPE", "model_layers_only")

        if peft_config is not None and is_already_peft:
            logger.info(f"[GRPO_LORA_SCOPE_UNIFIED] ===== Applying Unified LoRA Training Scope =====")
            logger.info(f"[GRPO_LORA_SCOPE_UNIFIED] GRPO_LORA_TRAIN_SCOPE = {grpo_lora_train_scope}")

            # Call unified scope application method
            self._apply_lora_train_scope(model, reason="after_adapter_load_before_trainable_check")

            logger.info(f"[GRPO_LORA_SCOPE_UNIFIED] ===== Unified LoRA Training Scope Applied =====")

            # Print diagnostics after unified scope application
            self._print_peft_diag(model, "policy][after_unified_scope", adapter_path=model_id)
        # [GRPO_LORA_SCOPE_EARLY] REMOVED - replaced by unified _apply_lora_train_scope above

        # [POLICY_LORA_RESTORE] REMOVED - replaced by unified _apply_lora_train_scope above
        # [GRPO_LORA_SCOPE] REMOVED - replaced by unified _apply_lora_train_scope above

        # [PEFT_DIAG] After freezing vision modules and policy LoRA restore
        self._print_peft_diag(model, "policy][after_freeze", adapter_path=model_id)

        # Diagnostic logging for trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"[TRAINABLE_PARAMS] Total: {total_params:,} | Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

        # Log sample trainable parameters containing LoRA or projection layers
        lora_trainable_samples = []
        for n, p in model.named_parameters():
            if p.requires_grad and ('lora_' in n or 'q_proj' in n or 'v_proj' in n):
                lora_trainable_samples.append(n)
                if len(lora_trainable_samples) >= 5:
                    break
        if lora_trainable_samples:
            logger.info(f"[TRAINABLE_PARAMS] Sample LoRA/projection trainable params: {lora_trainable_samples}")
        else:
            logger.warning("[TRAINABLE_PARAMS] No LoRA/projection parameters found in trainable params!")

        # [TRAINABLE_FIX] Final verification after policy LoRA restore
        if peft_config is not None and is_already_peft:
            logger.info("[TRAINABLE_FIX] ===== Final Trainable Parameter Verification =====")

            # Count trainable parameters by type
            trainable_model_layers_lora = 0
            trainable_audio_tower_lora = 0
            trainable_base_layer = 0
            trainable_vision_module = 0
            trainable_total = 0
            trainable_first_params = []

            for n, p in model.named_parameters():
                if p.requires_grad:
                    trainable_total += 1
                    if len(trainable_first_params) < 20:
                        trainable_first_params.append(n)
                    if 'model.layers' in n and 'lora_' in n:
                        trainable_model_layers_lora += 1
                    elif 'audio_tower' in n and 'lora_' in n:
                        trainable_audio_tower_lora += 1
                    elif 'base_layer' in n:
                        trainable_base_layer += 1
                    elif any(keyword in n for keyword in self.vision_modules_keywords):
                        trainable_vision_module += 1

            logger.info(f"[TRAINABLE_FIX] policy model.layers LoRA trainable count = {trainable_model_layers_lora}")
            logger.info(f"[TRAINABLE_FIX] policy audio_tower LoRA trainable count = {trainable_audio_tower_lora}")
            logger.info(f"[TRAINABLE_FIX] policy base_layer trainable count = {trainable_base_layer}")
            logger.info(f"[TRAINABLE_FIX] policy vision_module trainable count = {trainable_vision_module}")
            logger.info(f"[TRAINABLE_FIX] policy trainable_total_count = {trainable_total}")
            logger.info(f"[TRAINABLE_FIX] first 20 trainable params:")
            for i, name in enumerate(trainable_first_params):
                logger.info(f"[TRAINABLE_FIX]   {i+1}. {name}")

            # [TRAINABLE_FIX_SCOPE] Critical check: validate trainable params based on GRPO_LORA_TRAIN_SCOPE
            logger.info(f"[TRAINABLE_FIX_SCOPE] Validating trainable parameters for GRPO_LORA_TRAIN_SCOPE={grpo_lora_train_scope}")

            if grpo_lora_train_scope == "model_layers_only":
                # Stage 2/3: model.layers LoRA must be trainable, audio_tower LoRA must be frozen
                if trainable_model_layers_lora == 0:
                    logger.error("[TRAINABLE_FIX_SCOPE][FATAL] model_layers_only requested but model.layers LoRA has 0 trainable parameters!")
                    logger.error(f"[TRAINABLE_FIX_SCOPE][FATAL] First 20 LoRA params and their requires_grad status:")
                    lora_param_count = 0
                    for n, p in model.named_parameters():
                        if 'lora_' in n:
                            lora_param_count += 1
                            if lora_param_count <= 20:
                                logger.error(f"[TRAINABLE_FIX_SCOPE][FATAL]   {lora_param_count}. {n} | requires_grad={p.requires_grad}")
                    raise RuntimeError(
                        "[TRAINABLE_FIX_SCOPE][FATAL] model_layers_only validation failed!\n"
                        "model.layers LoRA trainable parameters: 0\n"
                        "This indicates the LoRA adapter was not properly restored."
                    )
                if trainable_audio_tower_lora > 0:
                    logger.warning(f"[TRAINABLE_FIX_SCOPE] WARNING: model_layers_only mode but audio_tower LoRA is trainable ({trainable_audio_tower_lora} params), should be frozen")
                logger.info("[TRAINABLE_FIX_SCOPE] model_layers_only validation passed")

            elif grpo_lora_train_scope == "audio_tower_only":
                # Stage 4a: audio_tower LoRA must be trainable, model.layers LoRA must be frozen
                if trainable_audio_tower_lora == 0:
                    logger.error("[TRAINABLE_FIX_SCOPE][FATAL] audio_tower_only requested but audio_tower LoRA has 0 trainable parameters!")
                    logger.error(f"[TRAINABLE_FIX_SCOPE][FATAL] First 20 LoRA params and their requires_grad status:")
                    lora_param_count = 0
                    for n, p in model.named_parameters():
                        if 'lora_' in n:
                            lora_param_count += 1
                            if lora_param_count <= 20:
                                logger.error(f"[TRAINABLE_FIX_SCOPE][FATAL]   {lora_param_count}. {n} | requires_grad={p.requires_grad}")
                    raise RuntimeError(
                        "[TRAINABLE_FIX_SCOPE][FATAL] audio_tower_only validation failed!\n"
                        "audio_tower LoRA trainable parameters: 0\n"
                        "This indicates the audio_tower LoRA adapter was not properly configured."
                    )
                if trainable_model_layers_lora > 0:
                    logger.warning(f"[TRAINABLE_FIX_SCOPE] WARNING: audio_tower_only mode but model.layers LoRA is trainable ({trainable_model_layers_lora} params), should be frozen")
                logger.info("[TRAINABLE_FIX_SCOPE] audio_tower_only validation passed")

            elif grpo_lora_train_scope == "joint":
                # Stage 6: both model.layers and audio_tower LoRA must be trainable
                if trainable_model_layers_lora == 0:
                    logger.error("[TRAINABLE_FIX_SCOPE][FATAL] joint mode but model.layers LoRA has 0 trainable parameters")
                    logger.error(f"[TRAINABLE_FIX_SCOPE][FATAL] First 20 LoRA params and their requires_grad status:")
                    lora_param_count = 0
                    for n, p in model.named_parameters():
                        if 'lora_' in n:
                            lora_param_count += 1
                            if lora_param_count <= 20:
                                logger.error(f"[TRAINABLE_FIX_SCOPE][FATAL]   {lora_param_count}. {n} | requires_grad={p.requires_grad}")
                    raise RuntimeError("[TRAINABLE_FIX_SCOPE][FATAL] joint mode requires model.layers LoRA trainable")
                if trainable_audio_tower_lora == 0:
                    logger.error("[TRAINABLE_FIX_SCOPE][FATAL] joint mode but audio_tower LoRA has 0 trainable parameters")
                    logger.error(f"[TRAINABLE_FIX_SCOPE][FATAL] First 20 LoRA params and their requires_grad status:")
                    lora_param_count = 0
                    for n, p in model.named_parameters():
                        if 'lora_' in n:
                            lora_param_count += 1
                            if lora_param_count <= 20:
                                logger.error(f"[TRAINABLE_FIX_SCOPE][FATAL]   {lora_param_count}. {n} | requires_grad={p.requires_grad}")
                    raise RuntimeError("[TRAINABLE_FIX_SCOPE][FATAL] joint mode requires audio_tower LoRA trainable")
                if trainable_base_layer > 0:
                    logger.error(f"[TRAINABLE_FIX_SCOPE][FATAL] joint mode but base_layer has {trainable_base_layer} trainable parameters")
                    raise RuntimeError("[TRAINABLE_FIX_SCOPE][FATAL] joint mode requires base_layer frozen")
                if trainable_vision_module > 0:
                    logger.error(f"[TRAINABLE_FIX_SCOPE][FATAL] joint mode but vision has {trainable_vision_module} trainable parameters")
                    raise RuntimeError("[TRAINABLE_FIX_SCOPE][FATAL] joint mode requires vision frozen")
                logger.info("[TRAINABLE_FIX_SCOPE] joint mode validation passed")

            logger.info("[TRAINABLE_FIX] ===== End Trainable Parameter Verification =====")

        # [GRPO_LORA_SCOPE_DIAG] Enhanced LoRA scope diagnostics
        if peft_config is not None and is_already_peft:
            logger.info("[GRPO_LORA_SCOPE_DIAG] ===== Enhanced LoRA Scope Diagnostics =====")
            logger.info(f"[GRPO_LORA_SCOPE_DIAG] GRPO_LORA_TRAIN_SCOPE = {grpo_lora_train_scope}")

            # Count all parameter types
            total_params = 0
            trainable_params = 0
            trainable_lora_params = 0
            trainable_model_layers_lora = 0
            trainable_audio_tower_lora = 0
            trainable_base_layer = 0
            trainable_vision_module = 0
            trainable_first_30 = []

            for n, p in model.named_parameters():
                total_params += 1
                if p.requires_grad:
                    trainable_params += 1
                    if len(trainable_first_30) < 30:
                        trainable_first_30.append((n, p.shape))

                    if 'lora_' in n:
                        trainable_lora_params += 1
                        if 'model.layers' in n:
                            trainable_model_layers_lora += 1
                        if 'audio_tower' in n:
                            trainable_audio_tower_lora += 1
                    elif any(keyword in n for keyword in self.vision_modules_keywords):
                        trainable_vision_module += 1
                    else:
                        trainable_base_layer += 1

            logger.info(f"[GRPO_LORA_SCOPE_DIAG] Total parameter count = {total_params}")
            logger.info(f"[GRPO_LORA_SCOPE_DIAG] Trainable parameter count = {trainable_params}")
            logger.info(f"[GRPO_LORA_SCOPE_DIAG] Trainable LoRA count = {trainable_lora_params}")
            logger.info(f"[GRPO_LORA_SCOPE_DIAG] Trainable model.layers LoRA count = {trainable_model_layers_lora}")
            logger.info(f"[GRPO_LORA_SCOPE_DIAG] Trainable audio_tower LoRA count = {trainable_audio_tower_lora}")
            logger.info(f"[GRPO_LORA_SCOPE_DIAG] Trainable base layer count = {trainable_base_layer}")
            logger.info(f"[GRPO_LORA_SCOPE_DIAG] Trainable vision module count = {trainable_vision_module}")

            logger.info(f"[GRPO_LORA_SCOPE_DIAG] First 30 trainable parameters:")
            for i, (name, shape) in enumerate(trainable_first_30):
                logger.info(f"[GRPO_LORA_SCOPE_DIAG]   {i+1}. {name} | shape={shape}")

            # Count total and frozen for each LoRA type
            total_model_layers_lora = sum(1 for n, p in model.named_parameters() if 'model.layers' in n and 'lora_' in n)
            frozen_model_layers_lora = sum(1 for n, p in model.named_parameters() if 'model.layers' in n and 'lora_' in n and not p.requires_grad)
            total_audio_tower_lora = sum(1 for n, p in model.named_parameters() if 'audio_tower' in n and 'lora_' in n)
            frozen_audio_tower_lora = sum(1 for n, p in model.named_parameters() if 'audio_tower' in n and 'lora_' in n and not p.requires_grad)

            logger.info(f"[GRPO_LORA_SCOPE_DIAG] model.layers LoRA: total={total_model_layers_lora}, trainable={trainable_model_layers_lora}, frozen={frozen_model_layers_lora}")
            logger.info(f"[GRPO_LORA_SCOPE_DIAG] audio_tower LoRA: total={total_audio_tower_lora}, trainable={trainable_audio_tower_lora}, frozen={frozen_audio_tower_lora}")

            # Validation for audio_tower_only mode
            if grpo_lora_train_scope == "audio_tower_only":
                if trainable_audio_tower_lora == 0:
                    logger.error("[GRPO_LORA_SCOPE_DIAG][FATAL] audio_tower_only mode but audio_tower LoRA trainable count is 0!")
                    raise RuntimeError(
                        "[GRPO_LORA_SCOPE_DIAG][FATAL] audio_tower_only validation failed!\n"
                        f"audio_tower LoRA trainable count: {trainable_audio_tower_lora}\n"
                        "Expected: > 0"
                    )
                if trainable_model_layers_lora > 0:
                    logger.warning(f"[GRPO_LORA_SCOPE_DIAG] WARNING: audio_tower_only mode but model.layers LoRA trainable count is {trainable_model_layers_lora} (should be 0)")
                if trainable_base_layer > 0:
                    logger.warning(f"[GRPO_LORA_SCOPE_DIAG] WARNING: audio_tower_only mode but base layer trainable count is {trainable_base_layer} (should be 0)")
                if trainable_vision_module > 0:
                    logger.warning(f"[GRPO_LORA_SCOPE_DIAG] WARNING: audio_tower_only mode but vision module trainable count is {trainable_vision_module} (should be 0)")
                logger.info("[GRPO_LORA_SCOPE_DIAG] audio_tower_only validation passed")

            logger.info("[GRPO_LORA_SCOPE_DIAG] ===== End Enhanced LoRA Scope Diagnostics =====")

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # [GRPO_DDP_FIX] Log DDP and gradient checkpointing configuration
        import torch.distributed as dist
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()

        logger.info("[GRPO_DDP_FIX] ===== DDP Configuration =====")
        logger.info(f"[GRPO_DDP_FIX] gradient_checkpointing = {args.gradient_checkpointing}")
        if args.gradient_checkpointing:
            use_reentrant = args.gradient_checkpointing_kwargs.get("use_reentrant", True) if hasattr(args, 'gradient_checkpointing_kwargs') and args.gradient_checkpointing_kwargs else True
            logger.info(f"[GRPO_DDP_FIX] use_reentrant = {use_reentrant}")
        logger.info(f"[GRPO_DDP_FIX] ddp_find_unused_parameters = {args.ddp_find_unused_parameters}")
        logger.info(f"[GRPO_DDP_FIX] world_size = {world_size}")
        logger.info(f"[GRPO_DDP_FIX] nproc_per_node = {os.environ.get('LOCAL_WORLD_SIZE', 'not set')}")
        logger.info("[GRPO_DDP_FIX] ===== End DDP Configuration =====")

        # Reference model initialization
        # IMPORTANT: For DeepSpeed ZeRO3, we defer ref_model loading to avoid OOM
        # The actual loading happens in __init__ after accelerator.prepare
        if is_deepspeed_zero3_enabled():
            # Store initialization info for deferred loading
            logger.info("DeepSpeed ZeRO3 detected: deferring ref_model initialization to avoid OOM")
            self._ref_model_init_kwargs = {
                "model_cls": model_cls,
                "model_id": model_id,
                "model_init_kwargs": model_init_kwargs,
            }
            self.ref_model = None  # Will be initialized after prepare
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            logger.info("Creating reference model from policy model (no PEFT)")
            self.ref_model = create_reference_model(model)
            # self.ref_model = self.ref_model.thinker # for qwen-omni
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            logger.info("PEFT enabled: ref_model not needed (adapter can be disabled)")
            self.ref_model = None
            self._ref_model_init_kwargs = None

        # Processing class with automatic fallback
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()

            import json

            # Step 0: Detect LoRA adapter and extract base model path
            adapter_path = None
            base_model_path = None
            adapter_config_path = os.path.join(model_id, "adapter_config.json")

            if os.path.exists(adapter_config_path):
                logger.info(f"[PROCESSOR_LOAD] Detected LoRA adapter at: {model_id}")
                adapter_path = model_id
                try:
                    with open(adapter_config_path) as f:
                        adapter_config = json.load(f)
                    base_model_path = adapter_config.get("base_model_name_or_path")
                    if base_model_path:
                        logger.info(f"[PROCESSOR_LOAD] Extracted base_model_name_or_path from adapter_config.json: {base_model_path}")
                    else:
                        logger.warning(f"[PROCESSOR_LOAD] adapter_config.json found but no base_model_name_or_path field")
                except Exception as e:
                    logger.warning(f"[PROCESSOR_LOAD] Failed to read adapter_config.json: {e}")
            else:
                logger.info(f"[PROCESSOR_LOAD] No adapter_config.json found, treating {model_id} as base model")

            # Step 1: Determine initial processor path
            # Priority 1: explicitly provided processor_name_or_path
            # Priority 2: base_model_path (if LoRA adapter detected)
            # Priority 3: model_id (checkpoint directory)
            if processor_name_or_path is not None:
                initial_processor_path = processor_name_or_path
                logger.info(f"[PROCESSOR_LOAD] Using explicitly provided processor path: {initial_processor_path}")
            elif base_model_path is not None:
                initial_processor_path = base_model_path
                logger.info(f"[PROCESSOR_LOAD] Using base_model_path from adapter_config.json: {initial_processor_path}")
            else:
                initial_processor_path = model_id
                logger.info(f"[PROCESSOR_LOAD] No processor_name_or_path provided, trying model checkpoint: {initial_processor_path}")

            # Step 2: Check if initial path is valid for processor loading
            processor_path_to_use = None
            if _is_valid_processor_path(initial_processor_path):
                processor_path_to_use = initial_processor_path
                logger.info(f"[PROCESSOR_LOAD] Processor path validation passed: {processor_path_to_use}")
            else:
                logger.warning(f"[PROCESSOR_LOAD] Processor path validation failed: {initial_processor_path}")
                logger.warning(f"[PROCESSOR_LOAD] Missing preprocessor_config.json in: {initial_processor_path}")

                # Step 3: If initial path is invalid and we're using model_id, try fallback
                if processor_name_or_path is None:
                    logger.info("[PROCESSOR_LOAD] Attempting automatic fallback to base model...")

                    # Try to extract base model path from config.json
                    config_path = os.path.join(model_id, "config.json")
                    fallback_base_model_path = None

                    if os.path.exists(config_path):
                        try:
                            with open(config_path) as f:
                                config = json.load(f)
                            # Check for common base model path fields
                            fallback_base_model_path = config.get("_name_or_path")
                            if fallback_base_model_path:
                                logger.info(f"[PROCESSOR_LOAD] Found base model path in config.json: {fallback_base_model_path}")
                                # Validate base model path
                                if _is_valid_processor_path(fallback_base_model_path):
                                    processor_path_to_use = fallback_base_model_path
                                    logger.info(f"[PROCESSOR_LOAD] Base model path validation passed: {processor_path_to_use}")
                                else:
                                    logger.warning(f"[PROCESSOR_LOAD] Base model path validation failed: {fallback_base_model_path}")
                                    fallback_base_model_path = None
                            else:
                                logger.warning("[PROCESSOR_LOAD] No _name_or_path field found in config.json")
                        except Exception as config_e:
                            logger.warning(f"[PROCESSOR_LOAD] Failed to read or parse config.json: {config_e}")

                    if processor_path_to_use is None:
                        # No valid processor path found
                        error_msg = (
                            f"Cannot load processor from checkpoint: {model_id}\n"
                            f"Reason: Missing preprocessor_config.json in checkpoint directory.\n"
                        )
                        if fallback_base_model_path:
                            error_msg += f"Attempted fallback to base model ({fallback_base_model_path}) also failed.\n"
                        else:
                            error_msg += "No valid base model path found in config.json (_name_or_path field).\n"
                        error_msg += (
                            f"\nSolution: Provide processor_name_or_path parameter pointing to:\n"
                            f"  - The base model directory (e.g., /path/to/base-model)\n"
                            f"  - Any directory containing preprocessor_config.json\n"
                            f"\nExample: --processor_name_or_path /path/to/base-model"
                        )
                        raise RuntimeError(error_msg)
                else:
                    # User explicitly provided processor_name_or_path but it's invalid
                    raise RuntimeError(
                        f"Explicitly provided processor path is invalid: {processor_name_or_path}\n"
                        f"Reason: Missing preprocessor_config.json\n"
                        f"Please provide a valid processor path containing preprocessor_config.json"
                    )

            # Step 4: Load processor from validated path
            try:
                processing_class = processing_cls.from_pretrained(
                    processor_path_to_use,
                    trust_remote_code=model_init_kwargs.get("trust_remote_code", None)
                )
                logger.info(f"[PROCESSOR_LOAD] Successfully loaded processor from: {processor_path_to_use}")
                if processor_path_to_use != model_id:
                    logger.info(f"[PROCESSOR_LOAD] Note: Model weights loaded from: {model_id}")
                    logger.info(f"[PROCESSOR_LOAD]       Processor loaded from: {processor_path_to_use}")

                # Diagnostic logging - ALWAYS print on rank 0
                if torch_dist.is_available() and torch_dist.is_initialized() and torch_dist.get_rank() == 0:
                    print("[PROCESSOR_LOAD] ===== Processor Loading Diagnostics =====", flush=True)
                    print(f"[PROCESSOR_LOAD] adapter_path: {adapter_path}", flush=True)
                    print(f"[PROCESSOR_LOAD] base_model_path: {base_model_path}", flush=True)
                    print(f"[PROCESSOR_LOAD] processor_load_path: {processor_path_to_use}", flush=True)
                    print(f"[PROCESSOR_LOAD] tokenizer_load_path: {processor_path_to_use}", flush=True)
                    print(f"[PROCESSOR_LOAD] generation_config_load_path: {processor_path_to_use}", flush=True)

                    # Get tokenizer info
                    if hasattr(processing_class, 'tokenizer'):
                        tokenizer = processing_class.tokenizer
                        print(f"[PROCESSOR_LOAD] tokenizer class: {type(tokenizer).__name__}", flush=True)
                        print(f"[PROCESSOR_LOAD] tokenizer.eos_token_id: {tokenizer.eos_token_id}", flush=True)
                        print(f"[PROCESSOR_LOAD] tokenizer.pad_token_id: {tokenizer.pad_token_id}", flush=True)
                    elif isinstance(processing_class, PreTrainedTokenizerBase):
                        print(f"[PROCESSOR_LOAD] tokenizer class: {type(processing_class).__name__}", flush=True)
                        print(f"[PROCESSOR_LOAD] tokenizer.eos_token_id: {processing_class.eos_token_id}", flush=True)
                        print(f"[PROCESSOR_LOAD] tokenizer.pad_token_id: {processing_class.pad_token_id}", flush=True)
                    else:
                        print(f"[PROCESSOR_LOAD] processing_class type: {type(processing_class).__name__}", flush=True)

                    # Tokenizer source validation: warn if loaded from adapter path
                    if adapter_path is not None and processor_path_to_use == adapter_path:
                        print(f"[PROCESSOR_LOAD][WARNING] tokenizer is loaded from adapter_path: {adapter_path}", flush=True)
                        print(f"[PROCESSOR_LOAD][WARNING] This may cause incorrect token decoding. Consider using base_model_path instead.", flush=True)

                    print("[PROCESSOR_LOAD] ===== End Processor Loading Diagnostics =====", flush=True)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load processor from validated path: {processor_path_to_use}\n"
                    f"Error: {e}"
                ) from e

            # Apply custom processing keywords
            for processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    setattr(processing_class, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id
        else:
            # processing_class was provided as parameter - print diagnostics
            if torch_dist.is_available() and torch_dist.is_initialized() and torch_dist.get_rank() == 0:
                logger.info("[PROCESSOR_LOAD] ===== Processor Loading Diagnostics (provided as parameter) =====")
                logger.info(f"[PROCESSOR_LOAD] adapter_path: N/A (provided as parameter)")
                logger.info(f"[PROCESSOR_LOAD] base_model_path: N/A (provided as parameter)")
                logger.info(f"[PROCESSOR_LOAD] processor_load_path: N/A (provided as parameter)")
                logger.info(f"[PROCESSOR_LOAD] tokenizer_load_path: N/A (provided as parameter)")
                logger.info(f"[PROCESSOR_LOAD] generation_config_load_path: N/A (provided as parameter)")

                # Get tokenizer info
                if hasattr(processing_class, 'tokenizer'):
                    tokenizer = processing_class.tokenizer
                    logger.info(f"[PROCESSOR_LOAD] tokenizer class: {type(tokenizer).__name__}")
                    logger.info(f"[PROCESSOR_LOAD] tokenizer.eos_token_id: {tokenizer.eos_token_id}")
                    logger.info(f"[PROCESSOR_LOAD] tokenizer.pad_token_id: {tokenizer.pad_token_id}")
                elif isinstance(processing_class, PreTrainedTokenizerBase):
                    logger.info(f"[PROCESSOR_LOAD] tokenizer class: {type(processing_class).__name__}")
                    logger.info(f"[PROCESSOR_LOAD] tokenizer.eos_token_id: {processing_class.eos_token_id}")
                    logger.info(f"[PROCESSOR_LOAD] tokenizer.pad_token_id: {processing_class.pad_token_id}")
                else:
                    logger.info(f"[PROCESSOR_LOAD] processing_class type: {type(processing_class).__name__}")
                logger.info("[PROCESSOR_LOAD] ===== End Processor Loading Diagnostics =====")
        # print(processing_class.tokenizer)
        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)


        # Parameter statistics and safety check
        total_params = 0
        total_trainable_params = 0
        trainable_param_names = []

        for name, p in model.named_parameters():
            total_params += p.numel()
            if p.requires_grad:
                total_trainable_params += p.numel()
                trainable_param_names.append(name)

        # Log parameter statistics
        logger.info("=" * 80)
        logger.info(f"[PARAMETER STATISTICS]")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {total_trainable_params:,}")
        logger.info(f"  Frozen parameters: {total_params - total_trainable_params:,}")
        if trainable_param_names:
            logger.info(f"  First trainable parameters:")
            for name in trainable_param_names[:5]:
                logger.info(f"    - {name}")
            if len(trainable_param_names) > 5:
                logger.info(f"    ... and {len(trainable_param_names) - 5} more")

            # Categorize trainable parameters by module
            logger.info(f"  Trainable parameter modules:")
            module_counts = {}
            for name in trainable_param_names:
                # Extract top-level module name
                parts = name.split('.')
                if len(parts) > 0:
                    top_module = parts[0]
                    module_counts[top_module] = module_counts.get(top_module, 0) + 1
            for module, count in sorted(module_counts.items()):
                logger.info(f"    - {module}: {count} parameters")
        logger.info("=" * 80)

        # Safety check: ensure at least some parameters are trainable
        if total_trainable_params == 0:
            raise RuntimeError(
                f"[CRITICAL] No trainable parameters found in model!\n"
                f"  Total parameters: {total_params:,}\n"
                f"  Trainable parameters: {total_trainable_params:,}\n"
                f"  This likely means:\n"
                f"    1. freeze_vision_modules=True froze all parameters (check vision_modules_keywords)\n"
                f"    2. PEFT/LoRA was not properly attached to the model\n"
                f"    3. All model parameters have requires_grad=False\n"
                f"  Please verify:\n"
                f"    - freeze_vision_modules setting and vision_modules_keywords\n"
                f"    - PEFT configuration (peft_config should not be None for LoRA)\n"
                f"    - Model loading and initialization\n"
                f"  Cannot proceed with training without trainable parameters."
            )

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Store affective reward configuration
        self.use_affective_rewards = getattr(args, 'use_affective_rewards', False)
        self.affective_reward_gate_mode = getattr(args, 'affective_reward_gate_mode', 'auto')
        self.affective_context_weight = getattr(args, 'affective_context_weight', 0.2)
        self.emotion_consistency_weight = getattr(args, 'emotion_consistency_weight', 0.1)

        # If affective rewards are enabled, adjust weights for the last 2 reward functions
        if self.use_affective_rewards and len(reward_funcs) >= 2:
            # Last 2 functions are affective_context and emotion_consistency
            self.reward_weights[-2] = self.affective_context_weight
            self.reward_weights[-1] = self.emotion_consistency_weight
            logger.info(f"[AFFECTIVE_REWARDS] Weights configured: context={self.affective_context_weight}, consistency={self.emotion_consistency_weight}")

        # [TASK_AWARE_ROUTING] Initialize task-aware routing system
        self.task_aware_routing_enabled = os.environ.get("GRPO_TASK_AWARE_ROUTING", "0") == "1"
        self.task_aware_prompt_enabled = os.environ.get("GRPO_TASK_AWARE_PROMPT", "0") == "1"

        if self.task_aware_routing_enabled:
            logger.info("[TASK_AWARE_ROUTING] Task-aware routing enabled via GRPO_TASK_AWARE_ROUTING=1")
            self.task_type_stats = {
                "emotion_video": {"answer_rate": [], "format_reward_mean": [], "count": 0, "samples": []},
                "general_video_qa": {"answer_rate": [], "format_reward_mean": [], "count": 0, "samples": []},
                "social_reasoning": {"answer_rate": [], "format_reward_mean": [], "count": 0, "samples": []},
                "image_math": {"answer_rate": [], "format_reward_mean": [], "count": 0, "samples": []},
                "unknown": {"answer_rate": [], "format_reward_mean": [], "count": 0, "samples": []},
            }
        else:
            self.task_type_stats = None

        if self.task_aware_prompt_enabled:
            logger.info("[TASK_PROMPT] Task-aware prompt routing enabled via GRPO_TASK_AWARE_PROMPT=1")


        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Wrap collator with bad sample handler if enabled
        if self.bad_sample_tracker.enable:
            data_collator = SafeCollatorWrapper(data_collator, self.bad_sample_tracker)

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None  # TODO
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.markov_reward = args.markov_reward


     
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        # [STABLE_GENERATION] Check if stable generation is enabled
        stable_generation_enabled = os.environ.get("GRPO_STABLE_GENERATION", "0") == "1"
        if stable_generation_enabled:
            logger.info("[STABLE_GENERATION] Stable generation enabled via GRPO_STABLE_GENERATION=1")
            stable_temperature = 0.7
            stable_top_p = 0.9
            stable_top_k = 50
        else:
            stable_temperature = self.temperature
            stable_top_p = self.top_p
            stable_top_k = self.top_k

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            min_new_tokens=min(32, max(1, self.max_completion_length // 4)),
            do_sample=True,
            pad_token_id=processing_class.tokenizer.pad_token_id,
            bos_token_id=processing_class.tokenizer.bos_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
            temperature=stable_temperature,
            top_p=stable_top_p,
            top_k=stable_top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            cache_implementation=args.cache_implementation,
            num_return_sequences=1,  # Set to 1 since prompts are already expanded by num_generations
        )
        if hasattr(self.vlm_module, "get_eos_token_id"): # For InternVL
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
            print(222, self.vlm_module.get_eos_token_id(processing_class))
        self.beta = args.beta
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon


        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions

        # [STAGE3_AUDIO_DIAG] Initialize diagnostic counter
        self._stage3_diag_step_count = 0
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # [GRPO_AUDIO_GRAD_DIAG][HOOK] Register backward hooks for audio_tower LoRA gradient diagnostics
        if grpo_lora_train_scope == "audio_tower_only":
            logger.info("[GRPO_AUDIO_GRAD_DIAG][HOOK] Registering backward hooks for audio_tower LoRA params")
            self._audio_tower_hook_handles = []
            hook_count = 0

            for name, param in self.model.named_parameters():
                if param.requires_grad and 'audio_tower' in name and 'lora_' in name and hook_count < 5:
                    def make_hook(param_name):
                        def hook(grad):
                            if grad is not None:
                                grad_norm = grad.norm().item()
                                grad_abs_mean = grad.abs().mean().item()
                                grad_max = grad.abs().max().item()
                                logger.info(f"[GRPO_AUDIO_GRAD_DIAG][HOOK] {param_name}: grad_norm={grad_norm:.6f}, grad_abs_mean={grad_abs_mean:.6f}, grad_max={grad_max:.6f}")
                            else:
                                logger.warning(f"[GRPO_AUDIO_GRAD_DIAG][HOOK] {param_name}: grad is None")
                        return hook

                    handle = param.register_hook(make_hook(name))
                    self._audio_tower_hook_handles.append(handle)
                    hook_count += 1
                    logger.info(f"[GRPO_AUDIO_GRAD_DIAG][HOOK] Registered hook for: {name}")

            logger.info(f"[GRPO_AUDIO_GRAD_DIAG][HOOK] Registered {hook_count} backward hooks")

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        # if self.num_generations not in possible_values:
        #     raise ValueError(
        #         f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
        #         f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
        #         f"batch size, the valid values for the number of generations are: {possible_values}."
        #     )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Prepare reference model with DeepSpeed-aware initialization
        if hasattr(self, '_ref_model_init_kwargs') and self._ref_model_init_kwargs is not None:
            # DeepSpeed ZeRO3: initialize ref_model with low-peak-memory approach
            logger.info("=" * 60)
            logger.info("Initializing ref_model with DeepSpeed ZeRO3 low-peak-memory mode")
            logger.info("=" * 60)

            model_cls = self._ref_model_init_kwargs["model_cls"]
            model_id = self._ref_model_init_kwargs["model_id"]
            model_init_kwargs = self._ref_model_init_kwargs["model_init_kwargs"]

            # [GRPO_REF_ADAPTER_LOAD] Check if model_id is an adapter-only checkpoint
            is_ref_adapter_only = (
                os.path.exists(os.path.join(model_id, "adapter_config.json"))
                and os.path.exists(os.path.join(model_id, "adapter_model.safetensors"))
            )

            if is_ref_adapter_only:
                logger.info("[GRPO_REF_ADAPTER_LOAD] Detected adapter-only checkpoint for ref_model")
                logger.info(f"[GRPO_REF_ADAPTER_LOAD] adapter_path = {model_id}")
                logger.info("[GRPO_REF_ADAPTER_LOAD] using frozen deepcopy of policy model for adapter-only ref_model")

                # Use deepcopy of policy model to avoid ZeRO-3 adapter loading issues
                import copy
                self.ref_model = copy.deepcopy(model)
                logger.info("[GRPO_REF_ADAPTER_LOAD] deepcopy of policy model created")

                # Diagnostics: count loaded LoRA parameters
                ref_loaded_total_lora = 0
                ref_loaded_audio_tower_lora = 0
                ref_loaded_visual_or_vision_lora = 0
                ref_loaded_model_layers_lora = 0

                for n, p in self.ref_model.named_parameters():
                    if 'lora_' in n:
                        ref_loaded_total_lora += 1
                        if 'audio_tower' in n:
                            ref_loaded_audio_tower_lora += 1
                        elif 'visual' in n or 'vision' in n:
                            ref_loaded_visual_or_vision_lora += 1
                        elif 'model.layers' in n:
                            ref_loaded_model_layers_lora += 1

                logger.info(f"[GRPO_REF_ADAPTER_LOAD] ref_loaded_total_lora_params = {ref_loaded_total_lora}")
                logger.info(f"[GRPO_REF_ADAPTER_LOAD] ref_loaded_audio_tower_lora = {ref_loaded_audio_tower_lora}")
                logger.info(f"[GRPO_REF_ADAPTER_LOAD] ref_loaded_visual_or_vision_lora = {ref_loaded_visual_or_vision_lora}")
                logger.info(f"[GRPO_REF_ADAPTER_LOAD] ref_loaded_model_layers_lora = {ref_loaded_model_layers_lora}")

                # Validation: ensure all three LoRA types are loaded
                if ref_loaded_audio_tower_lora == 0:
                    raise RuntimeError(f"[GRPO_REF_ADAPTER_LOAD] Failed to load audio_tower LoRA into ref_model from deepcopy")
                if ref_loaded_visual_or_vision_lora == 0:
                    raise RuntimeError(f"[GRPO_REF_ADAPTER_LOAD] Failed to load visual/vision LoRA into ref_model from deepcopy")
                if ref_loaded_model_layers_lora == 0:
                    raise RuntimeError(f"[GRPO_REF_ADAPTER_LOAD] Failed to load model.layers LoRA into ref_model from deepcopy")

                # Freeze ref_model completely
                logger.info("[GRPO_REF_ADAPTER_LOAD] Freezing all ref_model parameters")
                ref_trainable_param_count = 0
                for param in self.ref_model.parameters():
                    if param.requires_grad:
                        ref_trainable_param_count += 1
                    param.requires_grad_(False)
                self.ref_model.eval()

                logger.info(f"[GRPO_REF_ADAPTER_LOAD] ref_trainable_param_count = {ref_trainable_param_count}")

                # Validation: ensure ref_model is fully frozen
                final_trainable_count = sum(1 for p in self.ref_model.parameters() if p.requires_grad)
                if final_trainable_count != 0:
                    raise RuntimeError(f"[GRPO_REF_ADAPTER_LOAD] FATAL: ref_model still has {final_trainable_count} trainable parameters after freezing!")

                logger.info("[GRPO_REF_ADAPTER_LOAD] frozen SFT adapter ref_model loaded successfully")

                # Diagnostics: check ref_model embedding weight shape
                try:
                    logger.info(f"[GRPO_REF_EMBED_DIAG] ref_model type = {type(self.ref_model)}")

                    # Try to get embedding module (compatible with PeftModel)
                    emb_module = None
                    if hasattr(self.ref_model, 'get_input_embeddings'):
                        emb_module = self.ref_model.get_input_embeddings()
                    elif hasattr(self.ref_model, 'base_model') and hasattr(self.ref_model.base_model, 'model'):
                        if hasattr(self.ref_model.base_model.model, 'get_input_embeddings'):
                            emb_module = self.ref_model.base_model.model.get_input_embeddings()

                    if emb_module is not None:
                        logger.info(f"[GRPO_REF_EMBED_DIAG] embedding module type = {type(emb_module)}")
                        if hasattr(emb_module, 'weight'):
                            emb_weight = emb_module.weight
                            logger.info(f"[GRPO_REF_EMBED_DIAG] embedding weight shape = {emb_weight.shape}")
                            logger.info(f"[GRPO_REF_EMBED_DIAG] embedding weight device = {emb_weight.device}")
                            logger.info(f"[GRPO_REF_EMBED_DIAG] embedding weight dtype = {emb_weight.dtype}")
                            logger.info(f"[GRPO_REF_EMBED_DIAG] embedding weight requires_grad = {emb_weight.requires_grad}")

                            # Validation: embedding weight must be 2D
                            if emb_weight.dim() != 2:
                                raise RuntimeError(
                                    f"[GRPO_REF_EMBED_DIAG] ref_model embedding weight is not 2D\n"
                                    f"  shape = {emb_weight.shape}\n"
                                    f"  This may indicate ZeRO-3 partition placeholder issue"
                                )
                        else:
                            logger.warning("[GRPO_REF_EMBED_DIAG] embedding module has no 'weight' attribute")
                    else:
                        logger.warning("[GRPO_REF_EMBED_DIAG] could not get embedding module")
                except Exception as e:
                    logger.warning(f"[GRPO_REF_EMBED_DIAG] failed to check embedding: {e}")

                # Skip the rest of ref_model initialization since we already have it
                return

            # Use DeepSpeed zero.Init context manager for true low-peak initialization
            # This ensures the model is sharded during loading, not after
            if DEEPSPEED_ZERO_INIT_AVAILABLE and hasattr(self.accelerator.state, 'deepspeed_plugin'):
                logger.info("Using DeepSpeed zero.Init context for ref_model initialization")
                logger.info("This avoids loading the full model on a single GPU")

                # Get DeepSpeed config from accelerator
                ds_plugin = self.accelerator.state.deepspeed_plugin
                ds_config = ds_plugin.deepspeed_config if ds_plugin else None

                if ds_config:
                    try:
                        # Initialize model within zero.Init context
                        # This makes the model sharded from the start
                        with zero.Init(config_dict_or_path=ds_config,
                                      enabled=True,
                                      mem_efficient_linear=False,
                                      mpu=None):
                            logger.info(f"[REF_ZERO_INIT] Loading base model only (without adapter)")
                            # CRITICAL: Load base model WITHOUT adapter to avoid size mismatch
                            # Clean up PEFT-related parameters that the model class may not accept
                            base_model_init_kwargs = model_init_kwargs.copy()

                            # Remove PEFT-specific parameters that should not be passed to from_pretrained
                            peft_params_to_remove = ["load_adapter", "adapter_name", "is_trainable", "peft_config"]
                            for param_key in peft_params_to_remove:
                                base_model_init_kwargs.pop(param_key, None)

                            logger.info(f"Loading ref_model from {model_id} with zero.Init context")
                            self.ref_model = model_cls.from_pretrained(model_id, **base_model_init_kwargs)
                            logger.info("Ref model base loaded within zero.Init context (sharded during load)")

                            # [ADAPTER_FILTER] Legacy path: manually load and filter adapter state_dict
                            logger.info("[REF_ADAPTER_FILTER] Filtering ref_model adapter state_dict")
                            filtered_state_dict = self._filter_ref_model_adapter_state_dict_safe(
                                model_id, self.ref_model
                            )
                            if filtered_state_dict is not None and len(filtered_state_dict) > 0:
                                try:
                                    from peft import set_peft_model_state_dict
                                    logger.info("[REF_ADAPTER_FILTER] Loading filtered adapter state_dict into ref_model")
                                    set_peft_model_state_dict(self.ref_model, filtered_state_dict)
                                    logger.info("[REF_ADAPTER_FILTER] Filtered adapter loaded successfully")
                                except Exception as e:
                                    logger.warning(f"[REF_ADAPTER_FILTER] Failed to load filtered adapter: {e}")
                            elif filtered_state_dict is not None:
                                logger.warning("[REF_ADAPTER_FILTER] No valid adapter keys after filtering")
                            else:
                                logger.info("[REF_ADAPTER_FILTER] No adapter file found or filtering returned None")

                        # [PEFT_DIAG] After ref_model loading in zero.Init
                        self._print_peft_diag(self.ref_model, "ref][after_load_zero_init", adapter_path=model_id)

                        # Model is already sharded, but still need to register with DeepSpeed engine
                        logger.info("Preparing ref_model with DeepSpeed engine")
                        self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
                        logger.info("Ref model prepared with DeepSpeed ZeRO3 successfully")
                        logger.info("Peak memory during ref_model init: LOW (sharded during load)")

                        # Freeze ref_model completely
                        logger.info("[REF_MODEL_FREEZE] Freezing all ref_model parameters")
                        ref_trainable_param_count = 0
                        for param in self.ref_model.parameters():
                            if param.requires_grad:
                                ref_trainable_param_count += 1
                            param.requires_grad_(False)
                        self.ref_model.eval()

                        logger.info(f"[GRPO_REF_ADAPTER_LOAD] ref_trainable_param_count = {ref_trainable_param_count}")

                        # Validation: ensure ref_model is fully frozen
                        if ref_trainable_param_count > 0:
                            logger.warning(f"[REF_MODEL_FREEZE] WARNING: {ref_trainable_param_count} parameters were trainable before freezing")

                        # Verify all parameters are now frozen
                        final_trainable_count = sum(1 for p in self.ref_model.parameters() if p.requires_grad)
                        if final_trainable_count != 0:
                            raise RuntimeError(f"[REF_MODEL_FREEZE] FATAL: ref_model still has {final_trainable_count} trainable parameters after freezing!")

                        logger.info("[REF_MODEL_FREEZE] ref_model is now fully frozen and in eval mode")

                        if ref_adapter_path is not None:
                            logger.info("[GRPO_REF_ADAPTER_LOAD] frozen SFT adapter ref_model loaded successfully")

                    except Exception as e:
                        logger.warning(f"Failed to use zero.Init context: {e}")
                        # Check if fallback is explicitly allowed
                        allow_fallback = os.environ.get("GRPO_ALLOW_REF_FALLBACK", "0") == "1"
                        if allow_fallback:
                            logger.warning("Falling back to standard initialization (GRPO_ALLOW_REF_FALLBACK=1)")
                            self._fallback_ref_model_init(model_cls, model_id, model_init_kwargs)
                        else:
                            logger.error("[REF_ZERO_INIT] fallback disabled by default")
                            logger.error("To enable fallback, set GRPO_ALLOW_REF_FALLBACK=1")
                            raise RuntimeError(
                                f"Failed to initialize ref_model with zero.Init and fallback is disabled:\n"
                                f"  Error: {e}\n"
                                f"  To enable fallback, set environment variable: GRPO_ALLOW_REF_FALLBACK=1"
                            )
                else:
                    logger.warning("DeepSpeed config not found in accelerator, using fallback")
                    allow_fallback = os.environ.get("GRPO_ALLOW_REF_FALLBACK", "0") == "1"
                    if allow_fallback:
                        self._fallback_ref_model_init(model_cls, model_id, model_init_kwargs)
                    else:
                        raise RuntimeError(
                            "DeepSpeed config not found and fallback is disabled. "
                            "Set GRPO_ALLOW_REF_FALLBACK=1 to enable fallback."
                        )
            else:
                logger.warning("DeepSpeed zero.Init not available, using fallback initialization")
                logger.warning("Peak memory may be higher (full model loaded before sharding)")
                allow_fallback = os.environ.get("GRPO_ALLOW_REF_FALLBACK", "0") == "1"
                if allow_fallback:
                    self._fallback_ref_model_init(model_cls, model_id, model_init_kwargs)
                else:
                    raise RuntimeError(
                        "DeepSpeed zero.Init not available and fallback is disabled. "
                        "Set GRPO_ALLOW_REF_FALLBACK=1 to enable fallback."
                    )

            logger.info("=" * 60)

        elif self.ref_model is not None:
            # Non-DeepSpeed path or DeepSpeed without ZeRO3
            if self.is_deepspeed_enabled:
                logger.info("Preparing ref_model with DeepSpeed (non-ZeRO3)")
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                logger.info("Preparing ref_model with Accelerate")
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

            # Freeze ref_model completely
            logger.info("[REF_MODEL_FREEZE] Freezing all ref_model parameters")
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()
            logger.info("[REF_MODEL_FREEZE] ref_model is now fully frozen and in eval mode")
        else:
            logger.info("No ref_model to prepare (PEFT mode or disabled)")

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def create_optimizer(self):
        """
        Override create_optimizer to explicitly handle audio_tower_only mode.
        CRITICAL: This is called from Trainer.__init__, BEFORE DeepSpeed wrapping.
        """
        print("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_ENTER] VLMGRPOTrainer.create_optimizer called", flush=True)
        print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_ENTER] self_class={self.__class__}", flush=True)
        print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_ENTER] model_type={type(self.model)}", flush=True)

        # [GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH] Diagnostic: check audio_tower paths
        print("\n[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH] === Audio Tower Path Diagnostics ===", flush=True)
        print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH] type(model)={type(self.model)}", flush=True)

        unwrapped_model = self.model
        if hasattr(unwrapped_model, 'module'):
            unwrapped_model = unwrapped_model.module
        print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH] type(unwrapped_model)={type(unwrapped_model)}", flush=True)

        print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH] hasattr(model, 'audio_tower')={hasattr(self.model, 'audio_tower')}", flush=True)
        print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH] hasattr(unwrapped_model, 'audio_tower')={hasattr(unwrapped_model, 'audio_tower')}", flush=True)
        print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH] hasattr(model, 'module')={hasattr(self.model, 'module')}", flush=True)
        if hasattr(self.model, 'module'):
            print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH] hasattr(model.module, 'audio_tower')={hasattr(self.model.module, 'audio_tower')}", flush=True)
        print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH] hasattr(model, 'base_model')={hasattr(self.model, 'base_model')}", flush=True)
        if hasattr(self.model, 'base_model'):
            print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH] hasattr(model.base_model, 'audio_tower')={hasattr(self.model.base_model, 'audio_tower')}", flush=True)

        # Count audio_tower modules in named_modules
        audio_tower_modules = []
        for module_name, module in self.model.named_modules():
            if 'audio_tower' in module_name:
                audio_tower_modules.append((module_name, type(module).__name__))

        print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH] audio_tower modules found: {len(audio_tower_modules)}", flush=True)
        for i, (name, mtype) in enumerate(audio_tower_modules[:50]):
            print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH]   {i+1}. {name} ({mtype})", flush=True)

        # Count audio_tower LoRA parameters
        audio_tower_lora_params = []
        for param_name, param in self.model.named_parameters():
            if 'audio_tower' in param_name and 'lora_' in param_name:
                audio_tower_lora_params.append(param_name)

        print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH] audio_tower LoRA parameters found: {len(audio_tower_lora_params)}", flush=True)
        for i, name in enumerate(audio_tower_lora_params[:20]):
            print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH]   {i+1}. {name}", flush=True)

        print("[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_PATH] === End Audio Tower Path Diagnostics ===\n", flush=True)

        grpo_lora_train_scope = os.environ.get("GRPO_LORA_TRAIN_SCOPE", "model_layers_only")
        print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_ENTER] GRPO_LORA_TRAIN_SCOPE={grpo_lora_train_scope}", flush=True)
        print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_ENTER] self.optimizer_is_None={self.optimizer is None}", flush=True)

        # [GRPO_AUDIO_GRAD_DIAG][PRE_OPTIMIZER_PARAM_SOURCE] Diagnostic BEFORE optimizer creation
        logger.info("[GRPO_AUDIO_GRAD_DIAG][PRE_OPTIMIZER_PARAM_SOURCE] ===== Raw Model Parameters BEFORE Optimizer Creation =====")
        logger.info(f"[GRPO_AUDIO_GRAD_DIAG][PRE_OPTIMIZER_PARAM_SOURCE] GRPO_LORA_TRAIN_SCOPE = {grpo_lora_train_scope}")

        audio_tower_lora_trainable = []
        model_layers_lora_trainable = []
        base_layer_trainable = []
        vision_trainable = []
        all_trainable = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                all_trainable.append((name, param.shape))
                if 'audio_tower' in name and 'lora_' in name:
                    audio_tower_lora_trainable.append(name)
                elif 'model.layers' in name and 'lora_' in name:
                    model_layers_lora_trainable.append(name)
                elif 'base_layer' in name:
                    base_layer_trainable.append(name)
                elif 'visual' in name or 'vision' in name:
                    vision_trainable.append(name)

        logger.info(f"[GRPO_AUDIO_GRAD_DIAG][PRE_OPTIMIZER_PARAM_SOURCE] audio_tower LoRA trainable count: {len(audio_tower_lora_trainable)}")
        logger.info(f"[GRPO_AUDIO_GRAD_DIAG][PRE_OPTIMIZER_PARAM_SOURCE] model.layers LoRA trainable count: {len(model_layers_lora_trainable)}")
        logger.info(f"[GRPO_AUDIO_GRAD_DIAG][PRE_OPTIMIZER_PARAM_SOURCE] base_layer trainable count: {len(base_layer_trainable)}")
        logger.info(f"[GRPO_AUDIO_GRAD_DIAG][PRE_OPTIMIZER_PARAM_SOURCE] vision trainable count: {len(vision_trainable)}")
        logger.info(f"[GRPO_AUDIO_GRAD_DIAG][PRE_OPTIMIZER_PARAM_SOURCE] total trainable params: {len(all_trainable)}")

        logger.info(f"[GRPO_AUDIO_GRAD_DIAG][PRE_OPTIMIZER_PARAM_SOURCE] First 30 trainable params:")
        for i, (name, shape) in enumerate(all_trainable[:30]):
            logger.info(f"[GRPO_AUDIO_GRAD_DIAG][PRE_OPTIMIZER_PARAM_SOURCE]   {i+1}. {name} {shape}")

        # Fatal check: audio_tower_only mode must have audio_tower LoRA trainable
        if grpo_lora_train_scope == "audio_tower_only" and len(audio_tower_lora_trainable) == 0:
            logger.error("[GRPO_AUDIO_GRAD_DIAG][PRE_OPTIMIZER_PARAM_SOURCE][FATAL] audio_tower_only mode but no audio_tower LoRA trainable params!")
            raise RuntimeError(
                "[GRPO_AUDIO_GRAD_DIAG][PRE_OPTIMIZER_PARAM_SOURCE][FATAL] audio_tower_only mode but no audio_tower LoRA trainable params!\n"
                f"audio_tower LoRA trainable count: {len(audio_tower_lora_trainable)}\n"
                f"model.layers LoRA trainable count: {len(model_layers_lora_trainable)}\n"
                f"base_layer trainable count: {len(base_layer_trainable)}\n"
                f"vision trainable count: {len(vision_trainable)}\n"
                "This indicates the LoRA adapter structure is not as expected."
            )

        logger.info("[GRPO_AUDIO_GRAD_DIAG][PRE_OPTIMIZER_PARAM_SOURCE] ===== End Raw Model Parameters Diagnostic =====")

        # Now handle optimizer creation based on scope
        if grpo_lora_train_scope == "audio_tower_only":
            logger.info("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER] audio_tower_only mode: creating optimizer with ONLY audio_tower LoRA params")

            # Define no_decay keywords for weight decay exclusion
            no_decay_keywords = ["bias", "LayerNorm.weight", "norm.weight"]

            # Collect ONLY audio_tower LoRA parameters and separate into decay/no_decay groups
            decay_params = []
            no_decay_params = []

            for name, param in self.model.named_parameters():
                if 'audio_tower' in name and 'lora_' in name and param.requires_grad:
                    # Check if this param should have no weight decay
                    if any(keyword in name for keyword in no_decay_keywords):
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)

            total_audio_tower_params = len(decay_params) + len(no_decay_params)
            if total_audio_tower_params == 0:
                logger.error("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER][FATAL] No audio_tower LoRA params collected for optimizer!")
                raise RuntimeError(
                    "[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER][FATAL] No audio_tower LoRA params collected for optimizer!\n"
                    "This should not happen if PRE_OPTIMIZER_PARAM_SOURCE check passed."
                )

            logger.info(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER] Collected {total_audio_tower_params} audio_tower LoRA params: {len(decay_params)} decay, {len(no_decay_params)} no_decay")

            # [CREATE_OPTIMIZER_SOURCE] Print raw selected parameters before optimizer creation
            print("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_SOURCE] ===== Raw Selected Parameters BEFORE Optimizer Creation =====", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_SOURCE] GRPO_LORA_TRAIN_SCOPE={grpo_lora_train_scope}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_SOURCE] selected_audio_tower_lora_count={total_audio_tower_params}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_SOURCE] selected_decay_count={len(decay_params)}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_SOURCE] selected_no_decay_count={len(no_decay_params)}", flush=True)

            # Collect all selected param names for diagnostics
            selected_param_names = []
            for name, param in self.model.named_parameters():
                if 'audio_tower' in name and 'lora_' in name and param.requires_grad:
                    selected_param_names.append(name)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_SOURCE] first_30_selected_names={selected_param_names[:30]}", flush=True)

            # Explicitly construct optimizer param groups with decay/no_decay separation
            optimizer_grouped_parameters = []

            if len(decay_params) > 0:
                optimizer_grouped_parameters.append({
                    "params": decay_params,
                    "weight_decay": self.args.weight_decay,
                })
                logger.info(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER] Param group 0 (decay): {len(decay_params)} params, weight_decay={self.args.weight_decay}")

            if len(no_decay_params) > 0:
                optimizer_grouped_parameters.append({
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                })
                logger.info(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER] Param group {len(optimizer_grouped_parameters)-1} (no_decay): {len(no_decay_params)} params, weight_decay=0.0")

            # Create optimizer with explicit param groups
            import torch.optim as optim
            optimizer_cls = optim.AdamW
            optimizer_kwargs = {
                "lr": self.args.learning_rate,
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            logger.info(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER] Created AdamW optimizer with {len(optimizer_grouped_parameters)} param groups, total {total_audio_tower_params} audio_tower LoRA params")

            # [CREATE_OPTIMIZER_RESULT] Print optimizer param group mapping result using id(param)->name
            print("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT] ===== Optimizer Param Groups (POST-CREATION) =====", flush=True)

            # Build id(param) -> name mapping
            param_id_to_name = {}
            for name, param in self.model.named_parameters():
                param_id_to_name[id(param)] = name

            # Count optimizer params by type
            optimizer_param_group_count = len(self.optimizer.param_groups)
            optimizer_named_param_count = 0
            optimizer_audio_tower_lora_name_count = 0
            optimizer_model_layers_lora_name_count = 0
            optimizer_base_layer_name_count = 0
            optimizer_vision_name_count = 0
            optimizer_param_names = []

            for group_idx, param_group in enumerate(self.optimizer.param_groups):
                for param in param_group['params']:
                    optimizer_named_param_count += 1
                    param_name = param_id_to_name.get(id(param), "UNKNOWN")
                    optimizer_param_names.append(param_name)

                    if 'audio_tower' in param_name and 'lora_' in param_name:
                        optimizer_audio_tower_lora_name_count += 1
                    elif 'model.layers' in param_name and 'lora_' in param_name:
                        optimizer_model_layers_lora_name_count += 1
                    elif 'base_layer' in param_name:
                        optimizer_base_layer_name_count += 1
                    elif 'visual' in param_name or 'vision' in param_name:
                        optimizer_vision_name_count += 1

            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT] optimizer_param_group_count={optimizer_param_group_count}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT] optimizer_named_param_count={optimizer_named_param_count}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT] optimizer_audio_tower_lora_name_count={optimizer_audio_tower_lora_name_count}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT] optimizer_model_layers_lora_name_count={optimizer_model_layers_lora_name_count}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT] optimizer_base_layer_name_count={optimizer_base_layer_name_count}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT] optimizer_vision_name_count={optimizer_vision_name_count}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT] first_30_optimizer_param_names={optimizer_param_names[:30]}", flush=True)

            # Hard check for audio_tower_only mode
            if optimizer_audio_tower_lora_name_count == 0:
                print("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT][FATAL] optimizer_audio_tower_lora_name_count=0!", flush=True)
                raise RuntimeError("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT][FATAL] audio_tower_only mode but optimizer has no audio_tower LoRA params!")
            if optimizer_model_layers_lora_name_count > 0:
                print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT][FATAL] optimizer_model_layers_lora_name_count={optimizer_model_layers_lora_name_count}!", flush=True)
                raise RuntimeError("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT][FATAL] model.layers LoRA should not be in optimizer!")
            if optimizer_base_layer_name_count > 0:
                print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT][FATAL] optimizer_base_layer_name_count={optimizer_base_layer_name_count}!", flush=True)
                raise RuntimeError("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT][FATAL] base_layer should not be in optimizer!")
            if optimizer_vision_name_count > 0:
                print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT][FATAL] optimizer_vision_name_count={optimizer_vision_name_count}!", flush=True)
                raise RuntimeError("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT][FATAL] vision should not be in optimizer!")

            print("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT] ===== End Optimizer Param Groups Diagnostic =====", flush=True)

            # [GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP] Diagnostic AFTER optimizer creation, BEFORE DeepSpeed
            logger.info("[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP] ===== Optimizer Param Groups (POST-CREATION, PRE-DEEPSPEED) =====")

            # Build id(param) -> name mapping
            param_id_to_name = {}
            for name, param in self.model.named_parameters():
                param_id_to_name[id(param)] = name

            # Analyze optimizer param groups
            total_optimizer_params = 0
            audio_tower_lora_in_optimizer = 0
            model_layers_lora_in_optimizer = 0
            base_layer_in_optimizer = 0
            vision_in_optimizer = 0
            optimizer_param_names = []

            for group_idx, param_group in enumerate(self.optimizer.param_groups):
                logger.info(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP] Param group {group_idx}:")
                logger.info(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP]   lr: {param_group.get('lr', 'N/A')}")
                logger.info(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP]   param count: {len(param_group['params'])}")

                for param in param_group['params']:
                    total_optimizer_params += 1
                    param_name = param_id_to_name.get(id(param), f"<unknown_id_{id(param)}>")
                    optimizer_param_names.append(param_name)

                    if 'audio_tower' in param_name and 'lora_' in param_name:
                        audio_tower_lora_in_optimizer += 1
                    elif 'model.layers' in param_name and 'lora_' in param_name:
                        model_layers_lora_in_optimizer += 1
                    elif 'base_layer' in param_name:
                        base_layer_in_optimizer += 1
                    elif 'visual' in param_name or 'vision' in param_name:
                        vision_in_optimizer += 1

            logger.info(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP] Total optimizer params: {total_optimizer_params}")
            logger.info(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP] audio_tower LoRA in optimizer: {audio_tower_lora_in_optimizer}")
            logger.info(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP] model.layers LoRA in optimizer: {model_layers_lora_in_optimizer}")
            logger.info(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP] base_layer in optimizer: {base_layer_in_optimizer}")
            logger.info(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP] vision module in optimizer: {vision_in_optimizer}")

            logger.info(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP] First 30 optimizer param names:")
            for i, name in enumerate(optimizer_param_names[:30]):
                logger.info(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP]   {i+1}. {name}")

            # Fatal check: optimizer must have audio_tower LoRA params
            if audio_tower_lora_in_optimizer == 0:
                logger.error("[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP][FATAL] optimizer has no audio_tower LoRA params!")
                raise RuntimeError(
                    "[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP][FATAL] audio_tower_only mode but optimizer has no audio_tower LoRA params!\n"
                    f"audio_tower LoRA in optimizer: {audio_tower_lora_in_optimizer}\n"
                    f"model.layers LoRA in optimizer: {model_layers_lora_in_optimizer}\n"
                    f"base_layer in optimizer: {base_layer_in_optimizer}\n"
                    f"vision in optimizer: {vision_in_optimizer}\n"
                    f"total optimizer params: {total_optimizer_params}\n"
                    f"First 30 optimizer param names: {optimizer_param_names[:30]}"
                )

            if model_layers_lora_in_optimizer > 0:
                logger.error(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP][FATAL] {model_layers_lora_in_optimizer} model.layers LoRA params in optimizer!")
                raise RuntimeError(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP][FATAL] model.layers LoRA should not be in optimizer!")

            if base_layer_in_optimizer > 0:
                logger.error(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP][FATAL] {base_layer_in_optimizer} base_layer params in optimizer!")
                raise RuntimeError(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP][FATAL] base_layer should not be in optimizer!")

            if vision_in_optimizer > 0:
                logger.error(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP][FATAL] {vision_in_optimizer} vision params in optimizer!")
                raise RuntimeError(f"[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP][FATAL] vision should not be in optimizer!")

            logger.info("[GRPO_AUDIO_GRAD_DIAG][POST_CREATE_OPTIMIZER_PARAM_GROUP] ===== End Optimizer Param Groups Diagnostic =====")

            # [GRPO_AUDIO_GRAD_DIAG][LORA_HOOK_REGISTER] Register backward hooks for audio_tower LoRA params
            verbose_hook = os.environ.get("GRPO_AUDIO_GRAD_DIAG_VERBOSE", "0") == "1"
            if verbose_hook:
                print("\n[GRPO_AUDIO_GRAD_DIAG][LORA_HOOK_REGISTER] === Registering LoRA Backward Hooks ===", flush=True)

            lora_hook_count = 0
            for param_name, param in self.model.named_parameters():
                if 'audio_tower' in param_name and 'lora_' in param_name and param.requires_grad:
                    if lora_hook_count < 5:  # Register hooks for first 5 params
                        def make_hook(name):
                            def hook(grad):
                                verbose_inner = os.environ.get("GRPO_AUDIO_GRAD_DIAG_VERBOSE", "0") == "1"
                                if verbose_inner:
                                    if grad is not None:
                                        grad_norm = grad.norm().item()
                                        grad_abs_mean = grad.abs().mean().item()
                                        grad_max = grad.abs().max().item()
                                        print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_BACKWARD_HOOK] {name}: grad_norm={grad_norm:.6f}, grad_abs_mean={grad_abs_mean:.6f}, grad_max={grad_max:.6f}", flush=True)
                                    else:
                                        print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_BACKWARD_HOOK] {name}: grad=None", flush=True)
                            return hook

                        param.register_hook(make_hook(param_name))
                        if verbose_hook:
                            print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_HOOK_REGISTER] param_name={param_name}", flush=True)
                        lora_hook_count += 1

            if lora_hook_count == 0:
                print("[GRPO_AUDIO_GRAD_DIAG][LORA_HOOK_REGISTER][FATAL] No audio_tower LoRA params found for hook registration!", flush=True)
                raise RuntimeError("[GRPO_AUDIO_GRAD_DIAG][LORA_HOOK_REGISTER][FATAL] No audio_tower LoRA params found for hook registration!")

            if verbose_hook:
                print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_HOOK_REGISTER] Registered {lora_hook_count} backward hooks", flush=True)
                print("[GRPO_AUDIO_GRAD_DIAG][LORA_HOOK_REGISTER] === End LoRA Backward Hooks Registration ===\n", flush=True)

            # [GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_VERIFIED] Set verification flag for post-DeepSpeed diagnostics
            self._grpo_audio_optimizer_verified = True
            self._grpo_audio_optimizer_audio_lora_count = optimizer_audio_tower_lora_name_count
            self._grpo_audio_optimizer_model_lora_count = optimizer_model_layers_lora_name_count
            self._grpo_audio_optimizer_base_layer_count = optimizer_base_layer_name_count
            self._grpo_audio_optimizer_vision_count = optimizer_vision_name_count
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_VERIFIED] Set verification flag: audio_lora={optimizer_audio_tower_lora_name_count}, model_lora={optimizer_model_layers_lora_name_count}, base={optimizer_base_layer_name_count}, vision={optimizer_vision_name_count}", flush=True)

            return self.optimizer

        elif grpo_lora_train_scope == "joint":
            logger.info("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER] joint mode: creating optimizer with audio_tower LoRA + model.layers LoRA params")

            # Define no_decay keywords for weight decay exclusion
            no_decay_keywords = ["bias", "LayerNorm.weight", "norm.weight"]

            # Collect BOTH audio_tower LoRA and model.layers LoRA parameters
            decay_params = []
            no_decay_params = []

            for name, param in self.model.named_parameters():
                # Include both audio_tower LoRA and model.layers LoRA
                if param.requires_grad and 'lora_' in name:
                    if 'audio_tower' in name or 'model.layers' in name:
                        # Check if this param should have no weight decay
                        if any(keyword in name for keyword in no_decay_keywords):
                            no_decay_params.append(param)
                        else:
                            decay_params.append(param)

            total_joint_params = len(decay_params) + len(no_decay_params)
            if total_joint_params == 0:
                logger.error("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER][FATAL] No joint LoRA params collected for optimizer!")
                raise RuntimeError(
                    "[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER][FATAL] No joint LoRA params collected for optimizer!\n"
                    "This should not happen if PRE_OPTIMIZER_PARAM_SOURCE check passed."
                )

            logger.info(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER] Collected {total_joint_params} joint LoRA params: {len(decay_params)} decay, {len(no_decay_params)} no_decay")

            # [CREATE_OPTIMIZER_SOURCE] Print raw selected parameters before optimizer creation
            print("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_SOURCE] ===== Raw Selected Parameters BEFORE Optimizer Creation =====", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_SOURCE] GRPO_LORA_TRAIN_SCOPE={grpo_lora_train_scope}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_SOURCE] selected_joint_lora_count={total_joint_params}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_SOURCE] selected_decay_count={len(decay_params)}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_SOURCE] selected_no_decay_count={len(no_decay_params)}", flush=True)

            # Collect all selected param names for diagnostics
            selected_param_names = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'lora_' in name:
                    if 'audio_tower' in name or 'model.layers' in name:
                        selected_param_names.append(name)
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_SOURCE] first_30_selected_names={selected_param_names[:30]}", flush=True)

            # Explicitly construct optimizer param groups with decay/no_decay separation
            optimizer_grouped_parameters = []

            if len(decay_params) > 0:
                optimizer_grouped_parameters.append({
                    "params": decay_params,
                    "weight_decay": self.args.weight_decay,
                })
                logger.info(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER] Param group 0 (decay): {len(decay_params)} params, weight_decay={self.args.weight_decay}")

            if len(no_decay_params) > 0:
                optimizer_grouped_parameters.append({
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                })
                logger.info(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER] Param group {len(optimizer_grouped_parameters)-1} (no_decay): {len(no_decay_params)} params, weight_decay=0.0")

            # Create optimizer with explicit param groups
            import torch.optim as optim
            optimizer_cls = optim.AdamW
            optimizer_kwargs = {
                "lr": self.args.learning_rate,
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            logger.info(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER] Created AdamW optimizer with {len(optimizer_grouped_parameters)} param groups, total {total_joint_params} joint LoRA params")

            # [CREATE_OPTIMIZER_RESULT] Print optimizer param group mapping result using id(param)->name
            print("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT] ===== Optimizer Param Groups (POST-CREATION) =====", flush=True)

            # Build id(param) -> name mapping
            param_id_to_name = {}
            for name, param in self.model.named_parameters():
                param_id_to_name[id(param)] = name

            # Count optimizer params by type
            optimizer_param_group_count = len(self.optimizer.param_groups)
            optimizer_named_param_count = 0
            optimizer_audio_tower_lora_name_count = 0
            optimizer_model_layers_lora_name_count = 0
            optimizer_base_layer_name_count = 0
            optimizer_vision_name_count = 0
            optimizer_param_names = []

            for group_idx, param_group in enumerate(self.optimizer.param_groups):
                for param in param_group['params']:
                    optimizer_named_param_count += 1
                    param_name = param_id_to_name.get(id(param), "UNKNOWN")
                    optimizer_param_names.append(param_name)

                    if 'audio_tower' in param_name and 'lora_' in param_name:
                        optimizer_audio_tower_lora_name_count += 1
                    elif 'model.layers' in param_name and 'lora_' in param_name:
                        optimizer_model_layers_lora_name_count += 1
                    elif 'base_layer' in param_name:
                        optimizer_base_layer_name_count += 1
                    elif 'visual' in param_name or 'vision' in param_name:
                        optimizer_vision_name_count += 1

            print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] GRPO_LORA_TRAIN_SCOPE={grpo_lora_train_scope}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] optimizer_param_group_count={optimizer_param_group_count}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] optimizer_named_param_count={optimizer_named_param_count}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] optimizer_audio_tower_lora_name_count={optimizer_audio_tower_lora_name_count}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] optimizer_model_layers_lora_name_count={optimizer_model_layers_lora_name_count}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] optimizer_base_layer_name_count={optimizer_base_layer_name_count}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] optimizer_vision_name_count={optimizer_vision_name_count}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] first_30_optimizer_param_names={optimizer_param_names[:30]}", flush=True)

            # Hard check for joint mode
            if optimizer_audio_tower_lora_name_count == 0:
                print("[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER][FATAL] optimizer_audio_tower_lora_name_count=0!", flush=True)
                raise RuntimeError("[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER][FATAL] joint mode but optimizer has no audio_tower LoRA params!")
            if optimizer_model_layers_lora_name_count == 0:
                print("[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER][FATAL] optimizer_model_layers_lora_name_count=0!", flush=True)
                raise RuntimeError("[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER][FATAL] joint mode but optimizer has no model.layers LoRA params!")
            if optimizer_base_layer_name_count > 0:
                print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER][FATAL] optimizer_base_layer_name_count={optimizer_base_layer_name_count}!", flush=True)
                raise RuntimeError("[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER][FATAL] base_layer should not be in optimizer!")
            if optimizer_vision_name_count > 0:
                print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER][FATAL] optimizer_vision_name_count={optimizer_vision_name_count}!", flush=True)
                raise RuntimeError("[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER][FATAL] vision should not be in optimizer!")

            print("[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_RESULT] ===== End Optimizer Param Groups Diagnostic =====", flush=True)

            # [GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_VERIFIED] Set verification flag for post-DeepSpeed diagnostics
            self._grpo_audio_optimizer_verified = True
            self._grpo_audio_optimizer_audio_lora_count = optimizer_audio_tower_lora_name_count
            self._grpo_audio_optimizer_model_lora_count = optimizer_model_layers_lora_name_count
            self._grpo_audio_optimizer_base_layer_count = optimizer_base_layer_name_count
            self._grpo_audio_optimizer_vision_count = optimizer_vision_name_count
            print(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER_VERIFIED] Set verification flag: audio_lora={optimizer_audio_tower_lora_name_count}, model_lora={optimizer_model_layers_lora_name_count}, base={optimizer_base_layer_name_count}, vision={optimizer_vision_name_count}", flush=True)

            return self.optimizer

        else:
            # For non-audio_tower_only and non-joint modes, use parent class implementation
            logger.info(f"[GRPO_AUDIO_GRAD_DIAG][CREATE_OPTIMIZER] Using default Trainer.create_optimizer for {grpo_lora_train_scope} mode")
            return super().create_optimizer()

    def _filter_ref_model_adapter_state_dict(self, model_id):
        """
        Filter adapter state_dict to remove audio_tower LoRA weights before loading into ref_model.
        This prevents size mismatch errors when loading SFT adapter into ref_model under DeepSpeed ZeRO-3.

        Args:
            model_id: Path to checkpoint containing adapter_model.safetensors

        Returns:
            Filtered state_dict with audio_tower keys removed, or None if no adapter file exists
        """
        adapter_file = os.path.join(model_id, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            logger.info(f"[ADAPTER_FILTER] No adapter file found at {adapter_file}")
            return None

        try:
            from safetensors.torch import safe_open, save_file
            import tempfile

            logger.info(f"[ADAPTER_FILTER] Loading adapter from {adapter_file}")
            with safe_open(adapter_file, framework="pt", device="cpu") as f:
                full_state_dict = {k: f.get_tensor(k) for k in f.keys()}

            logger.info(f"[ADAPTER_FILTER] Full adapter state_dict keys: {len(full_state_dict)}")

            # Count keys by module before filtering
            audio_tower_keys_before = [k for k in full_state_dict.keys() if 'audio_tower' in k]
            model_layers_keys = [k for k in full_state_dict.keys() if 'model.layers' in k]

            logger.info(f"[ADAPTER_FILTER] Before filtering:")
            logger.info(f"[ADAPTER_FILTER]   audio_tower keys: {len(audio_tower_keys_before)}")
            logger.info(f"[ADAPTER_FILTER]   model.layers keys: {len(model_layers_keys)}")

            # Filter: remove all keys containing 'audio_tower'
            filtered_state_dict = {k: v for k, v in full_state_dict.items() if 'audio_tower' not in k}

            audio_tower_keys_after = [k for k in filtered_state_dict.keys() if 'audio_tower' in k]
            logger.info(f"[ADAPTER_FILTER] After filtering:")
            logger.info(f"[ADAPTER_FILTER]   audio_tower keys: {len(audio_tower_keys_after)}")
            logger.info(f"[ADAPTER_FILTER]   total keys: {len(filtered_state_dict)}")
            logger.info(f"[ADAPTER_FILTER]   removed keys: {len(full_state_dict) - len(filtered_state_dict)}")

            if len(audio_tower_keys_before) > 0:
                logger.info(f"[ADAPTER_FILTER] Removed audio_tower keys (first 5):")
                for key in audio_tower_keys_before[:5]:
                    logger.info(f"[ADAPTER_FILTER]   - {key}")

            return filtered_state_dict

        except Exception as e:
            logger.error(f"[ADAPTER_FILTER] Error filtering adapter state_dict: {e}")
            return None

    def _filter_ref_model_adapter_state_dict_safe(self, model_id, ref_model):
        """
        Safely filter adapter state_dict by matching against ref_model's actual parameter shapes.
        This is the primary filtering method for ref_model under zero.Init.

        Args:
            model_id: Path to checkpoint containing adapter_model.safetensors
            ref_model: The ref_model to match shapes against

        Returns:
            Filtered state_dict with only keys that exist in ref_model with matching shapes
        """
        adapter_file = os.path.join(model_id, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            logger.info(f"[REF_ADAPTER_FILTER] No adapter file found at {adapter_file}")
            return None

        try:
            from safetensors.torch import safe_open

            logger.info(f"[REF_ADAPTER_FILTER] Loading adapter from {adapter_file}")
            with safe_open(adapter_file, framework="pt", device="cpu") as f:
                full_state_dict = {k: f.get_tensor(k) for k in f.keys()}

            logger.info(f"[REF_ADAPTER_FILTER] before keys: {len(full_state_dict)}")

            # Build ref_model parameter shape map
            ref_model_shapes = {}
            for name, param in ref_model.named_parameters():
                ref_model_shapes[name] = param.shape

            # Filter by multiple criteria
            filtered_state_dict = {}
            dropped_audio_tower = 0
            dropped_missing_key = 0
            dropped_shape_mismatch = 0

            for key, value in full_state_dict.items():
                # Criterion 1: Remove audio_tower keys
                if 'audio_tower' in key:
                    dropped_audio_tower += 1
                    continue

                # Criterion 2: Check if key exists in ref_model
                if key not in ref_model_shapes:
                    dropped_missing_key += 1
                    continue

                # Criterion 3: Check shape match
                if ref_model_shapes[key] != value.shape:
                    logger.warning(f"[REF_ADAPTER_FILTER] Shape mismatch for {key}: "
                                 f"ref_model {ref_model_shapes[key]} vs adapter {value.shape}")
                    dropped_shape_mismatch += 1
                    continue

                # All checks passed, keep this key
                filtered_state_dict[key] = value

            logger.info(f"[REF_ADAPTER_FILTER] after keys: {len(filtered_state_dict)}")
            logger.info(f"[REF_ADAPTER_FILTER] dropped by audio_tower: {dropped_audio_tower}")
            logger.info(f"[REF_ADAPTER_FILTER] dropped by missing key: {dropped_missing_key}")
            logger.info(f"[REF_ADAPTER_FILTER] dropped by shape mismatch: {dropped_shape_mismatch}")

            return filtered_state_dict if len(filtered_state_dict) > 0 else None

        except Exception as e:
            logger.error(f"[REF_ADAPTER_FILTER] Error filtering adapter state_dict: {e}")
            return None

    def _apply_lora_train_scope(self, model, reason=""):
        """
        Apply LoRA training scope based on GRPO_LORA_TRAIN_SCOPE environment variable.

        This method enforces strict requires_grad control:
        - audio_tower_only: Only audio_tower LoRA parameters are trainable
        - model_layers_only: Only model.layers LoRA parameters are trainable

        Args:
            model: The model to apply scope to
            reason: Reason for applying scope (for logging)
        """
        grpo_lora_train_scope = os.environ.get("GRPO_LORA_TRAIN_SCOPE", "model_layers_only")
        verbose = os.environ.get("GRPO_AUDIO_GRAD_DIAG_VERBOSE", "0") == "1"

        if verbose:
            print(f"[GRPO_LORA_SCOPE_REAPPLY] === Applying LoRA Train Scope ===", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] reason={reason}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] GRPO_LORA_TRAIN_SCOPE={grpo_lora_train_scope}", flush=True)

        if grpo_lora_train_scope == "audio_tower_only":
            # audio_tower_only mode: only audio_tower LoRA trainable
            trainable_audio_tower_lora_count = 0
            trainable_model_layers_lora_count = 0
            trainable_base_layer_count = 0
            trainable_vision_count = 0
            frozen_audio_tower_lora_count = 0
            frozen_model_layers_lora_count = 0

            for name, param in model.named_parameters():
                # Rule 1: audio_tower LoRA parameters are trainable
                if 'audio_tower' in name and 'lora_' in name:
                    if not param.requires_grad:
                        param.requires_grad = True
                        frozen_audio_tower_lora_count += 1
                    trainable_audio_tower_lora_count += 1
                # Rule 2: model.layers LoRA parameters are frozen
                elif 'model.layers' in name and 'lora_' in name:
                    if param.requires_grad:
                        param.requires_grad = False
                        frozen_model_layers_lora_count += 1
                    trainable_model_layers_lora_count += 0  # Should be 0
                # Rule 3: All base_layer parameters are frozen
                elif 'base_layer' in name:
                    if param.requires_grad:
                        param.requires_grad = False
                        trainable_base_layer_count += 0
                # Rule 4: All vision/visual parameters are frozen
                elif 'visual' in name or 'vision' in name:
                    if param.requires_grad:
                        param.requires_grad = False
                        trainable_vision_count += 0
                # Rule 5: All other LoRA parameters are frozen
                elif 'lora_' in name:
                    if param.requires_grad:
                        param.requires_grad = False

            # Recount after applying scope
            trainable_audio_tower_lora_count = 0
            trainable_model_layers_lora_count = 0
            trainable_base_layer_count = 0
            trainable_vision_count = 0

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'audio_tower' in name and 'lora_' in name:
                        trainable_audio_tower_lora_count += 1
                    elif 'model.layers' in name and 'lora_' in name:
                        trainable_model_layers_lora_count += 1
                    elif 'base_layer' in name:
                        trainable_base_layer_count += 1
                    elif 'visual' in name or 'vision' in name:
                        trainable_vision_count += 1

            # Always print summary counts (not verbose-gated)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_audio_tower_lora_count={trainable_audio_tower_lora_count}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_model_layers_lora_count={trainable_model_layers_lora_count}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_base_layer_count={trainable_base_layer_count}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_vision_count={trainable_vision_count}", flush=True)

            if verbose:
                print(f"[GRPO_LORA_SCOPE_REAPPLY] frozen_audio_tower_lora_count={frozen_audio_tower_lora_count}", flush=True)
                print(f"[GRPO_LORA_SCOPE_REAPPLY] frozen_model_layers_lora_count={frozen_model_layers_lora_count}", flush=True)

            # Fatal check: audio_tower_only mode must have 0 trainable model.layers LoRA
            if trainable_model_layers_lora_count > 0:
                error_msg = (
                    f"[GRPO_LORA_SCOPE_REAPPLY] FATAL: audio_tower_only mode but found {trainable_model_layers_lora_count} "
                    f"trainable model.layers LoRA params after applying scope! reason={reason}"
                )
                print(error_msg, flush=True)
                raise RuntimeError(error_msg)

            if verbose:
                print(f"[GRPO_LORA_SCOPE_REAPPLY] === audio_tower_only scope applied successfully ===", flush=True)

        elif grpo_lora_train_scope == "model_layers_only":
            # model_layers_only mode: only model.layers LoRA trainable
            trainable_audio_tower_lora_count = 0
            trainable_model_layers_lora_count = 0
            trainable_base_layer_count = 0
            trainable_vision_count = 0

            for name, param in model.named_parameters():
                # Rule 1: model.layers LoRA parameters are trainable
                if 'model.layers' in name and 'lora_' in name:
                    if not param.requires_grad:
                        param.requires_grad = True
                    trainable_model_layers_lora_count += 1
                # Rule 2: audio_tower LoRA parameters are frozen
                elif 'audio_tower' in name and 'lora_' in name:
                    if param.requires_grad:
                        param.requires_grad = False
                # Rule 3: All base_layer parameters are frozen
                elif 'base_layer' in name:
                    if param.requires_grad:
                        param.requires_grad = False
                # Rule 4: All vision/visual parameters are frozen
                elif 'visual' in name or 'vision' in name:
                    if param.requires_grad:
                        param.requires_grad = False
                # Rule 5: All other LoRA parameters are frozen
                elif 'lora_' in name:
                    if param.requires_grad:
                        param.requires_grad = False

            # Recount after applying scope
            trainable_audio_tower_lora_count = 0
            trainable_model_layers_lora_count = 0
            trainable_base_layer_count = 0
            trainable_vision_count = 0

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'audio_tower' in name and 'lora_' in name:
                        trainable_audio_tower_lora_count += 1
                    elif 'model.layers' in name and 'lora_' in name:
                        trainable_model_layers_lora_count += 1
                    elif 'base_layer' in name:
                        trainable_base_layer_count += 1
                    elif 'visual' in name or 'vision' in name:
                        trainable_vision_count += 1

            # Always print summary counts (not verbose-gated)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_audio_tower_lora_count={trainable_audio_tower_lora_count}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_model_layers_lora_count={trainable_model_layers_lora_count}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_base_layer_count={trainable_base_layer_count}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_vision_count={trainable_vision_count}", flush=True)

            if verbose:
                print(f"[GRPO_LORA_SCOPE_REAPPLY] === model_layers_only scope applied successfully ===", flush=True)

        elif grpo_lora_train_scope == "joint":
            # joint mode: both audio_tower and model.layers LoRA trainable
            trainable_audio_tower_lora_count = 0
            trainable_model_layers_lora_count = 0
            trainable_base_layer_count = 0
            trainable_vision_count = 0

            for name, param in model.named_parameters():
                # Rule 1: audio_tower LoRA parameters are trainable
                if 'audio_tower' in name and 'lora_' in name:
                    if not param.requires_grad:
                        param.requires_grad = True
                    trainable_audio_tower_lora_count += 1
                # Rule 2: model.layers LoRA parameters are trainable
                elif 'model.layers' in name and 'lora_' in name:
                    if not param.requires_grad:
                        param.requires_grad = True
                    trainable_model_layers_lora_count += 1
                # Rule 3: All base_layer parameters are frozen
                elif 'base_layer' in name:
                    if param.requires_grad:
                        param.requires_grad = False
                # Rule 4: All vision/visual parameters are frozen
                elif 'visual' in name or 'vision' in name:
                    if param.requires_grad:
                        param.requires_grad = False
                # Rule 5: All other LoRA parameters are frozen
                elif 'lora_' in name:
                    if param.requires_grad:
                        param.requires_grad = False

            # Recount after applying scope
            trainable_audio_tower_lora_count = 0
            trainable_model_layers_lora_count = 0
            trainable_base_layer_count = 0
            trainable_vision_count = 0

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'audio_tower' in name and 'lora_' in name:
                        trainable_audio_tower_lora_count += 1
                    elif 'model.layers' in name and 'lora_' in name:
                        trainable_model_layers_lora_count += 1
                    elif 'base_layer' in name:
                        trainable_base_layer_count += 1
                    elif 'visual' in name or 'vision' in name:
                        trainable_vision_count += 1

            # Always print summary counts
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_audio_tower_lora_count={trainable_audio_tower_lora_count}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_model_layers_lora_count={trainable_model_layers_lora_count}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_base_layer_count={trainable_base_layer_count}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_vision_count={trainable_vision_count}", flush=True)

            if verbose:
                print(f"[GRPO_LORA_SCOPE_REAPPLY] === joint scope applied successfully ===", flush=True)

        elif grpo_lora_train_scope == "multimodal_joint":
            # multimodal_joint mode: audio_tower, visual/vision, and model.layers LoRA trainable
            trainable_audio_tower_lora_count = 0
            trainable_visual_lora_count = 0
            trainable_model_layers_lora_count = 0
            trainable_base_layer_count = 0
            trainable_lm_head_count = 0
            trainable_embed_tokens_count = 0

            for name, param in model.named_parameters():
                # Rule 1: audio_tower LoRA parameters are trainable
                if 'audio_tower' in name and 'lora_' in name:
                    if not param.requires_grad:
                        param.requires_grad = True
                    trainable_audio_tower_lora_count += 1
                # Rule 2: visual/vision LoRA parameters are trainable
                elif ('visual' in name or 'vision' in name) and 'lora_' in name:
                    if not param.requires_grad:
                        param.requires_grad = True
                    trainable_visual_lora_count += 1
                # Rule 3: model.layers LoRA parameters are trainable
                elif 'model.layers' in name and 'lora_' in name:
                    if not param.requires_grad:
                        param.requires_grad = True
                    trainable_model_layers_lora_count += 1
                # Rule 4: All base_layer parameters are frozen
                elif 'base_layer' in name:
                    if param.requires_grad:
                        param.requires_grad = False
                # Rule 5: lm_head is frozen
                elif 'lm_head' in name:
                    if param.requires_grad:
                        param.requires_grad = False
                # Rule 6: embed_tokens is frozen
                elif 'embed_tokens' in name:
                    if param.requires_grad:
                        param.requires_grad = False
                # Rule 7: All non-LoRA visual/vision parameters are frozen
                elif ('visual' in name or 'vision' in name) and 'lora_' not in name:
                    if param.requires_grad:
                        param.requires_grad = False
                # Rule 8: All non-LoRA audio_tower parameters are frozen
                elif 'audio_tower' in name and 'lora_' not in name:
                    if param.requires_grad:
                        param.requires_grad = False
                # Rule 9: All other LoRA parameters are frozen
                elif 'lora_' in name:
                    if param.requires_grad:
                        param.requires_grad = False

            # Recount after applying scope
            trainable_audio_tower_lora_count = 0
            trainable_visual_lora_count = 0
            trainable_model_layers_lora_count = 0
            trainable_base_layer_count = 0
            trainable_lm_head_count = 0
            trainable_embed_tokens_count = 0

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'audio_tower' in name and 'lora_' in name:
                        trainable_audio_tower_lora_count += 1
                    elif ('visual' in name or 'vision' in name) and 'lora_' in name:
                        trainable_visual_lora_count += 1
                    elif 'model.layers' in name and 'lora_' in name:
                        trainable_model_layers_lora_count += 1
                    elif 'lm_head' in name:
                        trainable_lm_head_count += 1
                    elif 'embed_tokens' in name:
                        trainable_embed_tokens_count += 1
                    elif 'lora_' not in name:
                        trainable_base_layer_count += 1

            # Always print summary counts
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_audio_tower_lora_count={trainable_audio_tower_lora_count}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_visual_lora_count={trainable_visual_lora_count}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_model_layers_lora_count={trainable_model_layers_lora_count}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_base_layer_count={trainable_base_layer_count}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_lm_head_count={trainable_lm_head_count}", flush=True)
            print(f"[GRPO_LORA_SCOPE_REAPPLY] trainable_embed_tokens_count={trainable_embed_tokens_count}", flush=True)

            # Validation: ensure all three LoRA types are trainable
            if trainable_audio_tower_lora_count == 0:
                raise RuntimeError(f"[GRPO_LORA_SCOPE_REAPPLY] FATAL: multimodal_joint mode but audio_tower LoRA count is 0! reason={reason}")
            if trainable_visual_lora_count == 0:
                raise RuntimeError(f"[GRPO_LORA_SCOPE_REAPPLY] FATAL: multimodal_joint mode but visual LoRA count is 0! reason={reason}")
            if trainable_model_layers_lora_count == 0:
                raise RuntimeError(f"[GRPO_LORA_SCOPE_REAPPLY] FATAL: multimodal_joint mode but model.layers LoRA count is 0! reason={reason}")
            if trainable_base_layer_count > 0:
                raise RuntimeError(f"[GRPO_LORA_SCOPE_REAPPLY] FATAL: multimodal_joint mode but base_layer count is {trainable_base_layer_count} (expected 0)! reason={reason}")
            if trainable_lm_head_count > 0:
                raise RuntimeError(f"[GRPO_LORA_SCOPE_REAPPLY] FATAL: multimodal_joint mode but lm_head count is {trainable_lm_head_count} (expected 0)! reason={reason}")
            if trainable_embed_tokens_count > 0:
                raise RuntimeError(f"[GRPO_LORA_SCOPE_REAPPLY] FATAL: multimodal_joint mode but embed_tokens count is {trainable_embed_tokens_count} (expected 0)! reason={reason}")

            if verbose:
                print(f"[GRPO_LORA_SCOPE_REAPPLY] === multimodal_joint scope applied successfully ===", flush=True)

        else:
            print(f"[GRPO_LORA_SCOPE_REAPPLY] WARNING: Unknown GRPO_LORA_TRAIN_SCOPE={grpo_lora_train_scope}, skipping scope application", flush=True)

    def _restrict_policy_model_lora_trainable(self, model):
        """
        Restrict policy model LoRA parameters to only allow training on model.layers.
        - Only allow lora_A and lora_B in model.layers to be trainable
        - Freeze all audio_tower LoRA parameters
        - Freeze all base_layer.weight and base_layer.bias parameters
        - Freeze all vision/visual parameters

        Args:
            model: The policy model with PEFT adapter
        """
        import re

        logger.info("=" * 80)
        logger.info("[LORA_RESTRICT] Restricting policy model LoRA trainable parameters")
        logger.info("[LORA_RESTRICT] Rules:")
        logger.info("[LORA_RESTRICT]   1. Only model.layers lora_A and lora_B are trainable")
        logger.info("[LORA_RESTRICT]   2. audio_tower LoRA parameters are frozen")
        logger.info("[LORA_RESTRICT]   3. All base_layer.weight and base_layer.bias are frozen")
        logger.info("[LORA_RESTRICT]   4. All vision/visual parameters are frozen")
        logger.info("=" * 80)

        # Regex to extract layer index from model.layers.{idx}
        layer_pattern = re.compile(r"model\.layers\.(\d+)\.")

        # Statistics before restriction
        total_params = 0
        total_lora_params = 0
        trainable_lora_a_count = 0
        trainable_lora_b_count = 0
        trainable_base_layer_count = 0
        trainable_audio_lora_count = 0
        trainable_vision_count = 0

        # First pass: count before restriction
        for name, param in model.named_parameters():
            total_params += 1
            if 'lora_' in name:
                total_lora_params += 1
                if param.requires_grad:
                    if 'lora_A' in name:
                        trainable_lora_a_count += 1
                    elif 'lora_B' in name:
                        trainable_lora_b_count += 1
                    if 'audio_tower' in name:
                        trainable_audio_lora_count += 1
            if 'base_layer' in name and param.requires_grad:
                trainable_base_layer_count += 1
            if ('visual' in name or 'vision' in name) and param.requires_grad:
                trainable_vision_count += 1

        logger.info(f"[LORA_RESTRICT] Before restriction:")
        logger.info(f"[LORA_RESTRICT]   Total parameters: {total_params}")
        logger.info(f"[LORA_RESTRICT]   Total LoRA parameters: {total_lora_params}")
        logger.info(f"[LORA_RESTRICT]   Trainable lora_A: {trainable_lora_a_count}")
        logger.info(f"[LORA_RESTRICT]   Trainable lora_B: {trainable_lora_b_count}")
        logger.info(f"[LORA_RESTRICT]   Trainable base_layer: {trainable_base_layer_count}")
        logger.info(f"[LORA_RESTRICT]   Trainable audio_tower LoRA: {trainable_audio_lora_count}")
        logger.info(f"[LORA_RESTRICT]   Trainable vision/visual: {trainable_vision_count}")

        # Second pass: apply restrictions
        for name, param in model.named_parameters():
            # Rule 1: Freeze all audio_tower parameters
            if 'audio_tower' in name:
                param.requires_grad = False
                continue

            # Rule 2: Freeze all vision/visual parameters
            if 'visual' in name or 'vision' in name:
                param.requires_grad = False
                continue

            # Rule 3: Freeze all base_layer parameters
            if 'base_layer' in name:
                param.requires_grad = False
                continue

            # Rule 4: Only allow lora_A and lora_B in model.layers to be trainable
            if 'model.layers' in name and 'lora_' in name:
                if 'lora_A' in name or 'lora_B' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            # Freeze all other LoRA parameters
            elif 'lora_' in name:
                param.requires_grad = False

        # Third pass: count after restriction
        trainable_lora_a_after = 0
        trainable_lora_b_after = 0
        trainable_base_layer_after = 0
        trainable_audio_lora_after = 0
        trainable_vision_after = 0
        trainable_model_layers_lora = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'model.layers' in name and 'lora_' in name:
                    trainable_model_layers_lora += 1
                    if 'lora_A' in name:
                        trainable_lora_a_after += 1
                    elif 'lora_B' in name:
                        trainable_lora_b_after += 1
                if 'base_layer' in name:
                    trainable_base_layer_after += 1
                if 'audio_tower' in name and 'lora_' in name:
                    trainable_audio_lora_after += 1
                if 'visual' in name or 'vision' in name:
                    trainable_vision_after += 1

        logger.info(f"[LORA_RESTRICT] After restriction:")
        logger.info(f"[LORA_RESTRICT]   Trainable model.layers LoRA: {trainable_model_layers_lora}")
        logger.info(f"[LORA_RESTRICT]   Trainable lora_A: {trainable_lora_a_after}")
        logger.info(f"[LORA_RESTRICT]   Trainable lora_B: {trainable_lora_b_after}")
        logger.info(f"[LORA_RESTRICT]   Trainable base_layer: {trainable_base_layer_after}")
        logger.info(f"[LORA_RESTRICT]   Trainable audio_tower LoRA: {trainable_audio_lora_after}")
        logger.info(f"[LORA_RESTRICT]   Trainable vision/visual: {trainable_vision_after}")

        # Collect trainable parameters for logging
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)

        logger.info(f"[LORA_RESTRICT] Total trainable parameters: {len(trainable_params)}")
        logger.info(f"[LORA_RESTRICT] First 30 trainable parameters:")
        for i, name in enumerate(trainable_params[:30]):
            logger.info(f"[LORA_RESTRICT]   {i+1}. {name}")

        # Validation checks
        has_error = False
        if trainable_base_layer_after > 0:
            logger.error(f"[LORA_RESTRICT] ERROR: base_layer still has {trainable_base_layer_after} trainable params!")
            has_error = True
        if trainable_audio_lora_after > 0:
            logger.error(f"[LORA_RESTRICT] ERROR: audio_tower still has {trainable_audio_lora_after} trainable LoRA params!")
            has_error = True
        if trainable_vision_after > 0:
            logger.error(f"[LORA_RESTRICT] ERROR: vision/visual still has {trainable_vision_after} trainable params!")
            has_error = True
        if trainable_model_layers_lora == 0:
            logger.error(f"[LORA_RESTRICT] CRITICAL ERROR: model.layers has 0 trainable LoRA params!")
            has_error = True
            raise RuntimeError(f"[LORA_RESTRICT] CRITICAL: model.layers LoRA parameters are not trainable! This is a critical error that must be fixed.")

        # Print success message if all checks pass
        if not has_error:
            logger.info(f"[LORA_RESTRICT] SUCCESS: All restrictions applied correctly")
            logger.info(f"[LORA_RESTRICT]   ✓ base_layer frozen: {trainable_base_layer_after == 0}")
            logger.info(f"[LORA_RESTRICT]   ✓ audio_tower frozen: {trainable_audio_lora_after == 0}")
            logger.info(f"[LORA_RESTRICT]   ✓ vision/visual frozen: {trainable_vision_after == 0}")
            logger.info(f"[LORA_RESTRICT]   ✓ model.layers LoRA trainable: {trainable_model_layers_lora > 0}")

        logger.info("=" * 80)

        # Print trainable parameter check diagnostics (always on rank 0)
        if torch_dist.is_available() and torch_dist.is_initialized() and torch_dist.get_rank() == 0:
            logger.info("[TRAINABLE_CHECK] ===== Trainable Parameter Check =====")
            logger.info(f"[TRAINABLE_CHECK] trainable_lora_A_count={trainable_lora_a_after}")
            logger.info(f"[TRAINABLE_CHECK] trainable_lora_B_count={trainable_lora_b_after}")
            logger.info(f"[TRAINABLE_CHECK] trainable_base_layer_count={trainable_base_layer_after}")
            logger.info(f"[TRAINABLE_CHECK] trainable_audio_lora_count={trainable_audio_lora_after}")
            logger.info(f"[TRAINABLE_CHECK] trainable_vision_count={trainable_vision_after}")
            logger.info(f"[TRAINABLE_CHECK] trainable_model_layers_lora_count={trainable_model_layers_lora}")
            logger.info("[TRAINABLE_CHECK] ===== End Trainable Parameter Check =====")

    def _fallback_ref_model_init(self, model_cls, model_id, model_init_kwargs):
        """
        Fallback method for ref_model initialization when zero.Init is not available.

        This method loads the full model on a single GPU first, then shards it.
        Peak memory usage is higher than zero.Init approach.

        Args:
            model_cls: Model class to instantiate
            model_id: Model checkpoint path
            model_init_kwargs: Keyword arguments for model initialization
        """
        logger.warning("=" * 60)
        logger.warning("FALLBACK: Loading ref_model with standard approach")
        logger.warning("Peak memory will be HIGHER (full model loaded before sharding)")
        logger.warning("=" * 60)

        # Load full model on single GPU (high peak memory)
        logger.info(f"Loading ref_model from {model_id} (full model on single GPU)")
        self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        logger.info("Ref model loaded (full model temporarily on single GPU)")

        # [ADAPTER_FILTER] Filter audio_tower LoRA weights from adapter before they cause size mismatch
        logger.info("[ADAPTER_FILTER] Filtering ref_model adapter state_dict to remove audio_tower LoRA weights")
        filtered_state_dict = self._filter_ref_model_adapter_state_dict(model_id)
        if filtered_state_dict is not None:
            try:
                from peft import set_peft_model_state_dict
                logger.info("[ADAPTER_FILTER] Loading filtered adapter state_dict into ref_model")
                set_peft_model_state_dict(self.ref_model, filtered_state_dict)
                logger.info("[ADAPTER_FILTER] Filtered adapter loaded successfully")
            except Exception as e:
                logger.warning(f"[ADAPTER_FILTER] Failed to load filtered adapter: {e}")

        # [PEFT_DIAG] After ref_model loading in fallback
        self._print_peft_diag(self.ref_model, "ref][after_load_fallback", adapter_path=model_id)

        # Now shard it with DeepSpeed
        logger.info("Preparing ref_model with DeepSpeed ZeRO3 (sharding now)")
        self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
        logger.info("Ref model prepared with DeepSpeed ZeRO3")
        logger.warning("Peak memory during init: HIGH (full model loaded before sharding)")
        logger.warning("=" * 60)

        # Freeze ref_model completely
        logger.info("[REF_MODEL_FREEZE] Freezing all ref_model parameters")
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        logger.info("[REF_MODEL_FREEZE] ref_model is now fully frozen and in eval mode")

    def _print_peft_diag(self, model, tag, adapter_path=None):
        """Print comprehensive PEFT adapter loading diagnostics."""
        print(f"\n{'='*80}")
        print(f"[PEFT_DIAG][{tag}] ===== PEFT Adapter Loading Diagnostics =====")
        print(f"[PEFT_DIAG][{tag}] Model type: {type(model)}")
        print(f"[PEFT_DIAG][{tag}] Has peft_config: {hasattr(model, 'peft_config')}")

        if hasattr(model, 'peft_config'):
            peft_config_dict = model.peft_config
            print(f"[PEFT_DIAG][{tag}] PEFT config keys: {list(peft_config_dict.keys())}")
            for adapter_name, config in peft_config_dict.items():
                print(f"[PEFT_DIAG][{tag}]   Adapter '{adapter_name}': {config}")

        # Count LoRA parameters by type
        lora_a_count = 0
        lora_b_count = 0
        lora_trainable_count = 0
        lora_frozen_count = 0
        lora_param_names = []

        for n, p in model.named_parameters():
            if 'lora_A' in n:
                lora_a_count += 1
            if 'lora_B' in n:
                lora_b_count += 1
            if 'lora_' in n:
                if p.requires_grad:
                    lora_trainable_count += 1
                else:
                    lora_frozen_count += 1
                lora_param_names.append((n, p.shape, p.requires_grad))

        print(f"[PEFT_DIAG][{tag}] Total LoRA params: {lora_a_count + lora_b_count}")
        print(f"[PEFT_DIAG][{tag}] LoRA trainable: {lora_trainable_count}, frozen: {lora_frozen_count}")

        # LoRA parameter counts by module
        module_lora_counts = {}
        for n, p in model.named_parameters():
            if 'lora_' in n:
                if 'model.layers' in n:
                    module = 'model.layers'
                elif 'audio_tower' in n:
                    module = 'audio_tower'
                elif 'visual' in n:
                    module = 'visual'
                elif 'lm_head' in n:
                    module = 'lm_head'
                else:
                    module = 'other'

                if module not in module_lora_counts:
                    module_lora_counts[module] = {'total': 0, 'trainable': 0, 'frozen': 0}
                module_lora_counts[module]['total'] += 1
                if p.requires_grad:
                    module_lora_counts[module]['trainable'] += 1
                else:
                    module_lora_counts[module]['frozen'] += 1

        print(f"[PEFT_DIAG][{tag}] LoRA counts by module:")
        for module in ['model.layers', 'audio_tower', 'visual', 'lm_head', 'other']:
            if module in module_lora_counts:
                counts = module_lora_counts[module]
                print(f"[PEFT_DIAG][{tag}]   {module}: total={counts['total']}, trainable={counts['trainable']}, frozen={counts['frozen']}")

        # First 30 LoRA parameters
        print(f"[PEFT_DIAG][{tag}] First 30 LoRA parameters:")
        for i, (name, shape, requires_grad) in enumerate(lora_param_names[:30]):
            print(f"[PEFT_DIAG][{tag}]   {i+1}. {name} | shape={shape} | requires_grad={requires_grad}")

        # First 30 trainable parameters
        trainable_param_names = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                trainable_param_names.append((n, p.shape))

        print(f"[PEFT_DIAG][{tag}] First 30 trainable parameters:")
        for i, (name, shape) in enumerate(trainable_param_names[:30]):
            print(f"[PEFT_DIAG][{tag}]   {i+1}. {name} | shape={shape}")

        # Check adapter file if path provided
        if adapter_path:
            adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
            print(f"[PEFT_DIAG][{tag}] Checking adapter file: {adapter_file}")

            if os.path.exists(adapter_file):
                print(f"[PEFT_DIAG][{tag}] Adapter file EXISTS")
                try:
                    from safetensors.torch import safe_open
                    with safe_open(adapter_file, framework="pt", device="cpu") as f:
                        adapter_keys = list(f.keys())
                        print(f"[PEFT_DIAG][{tag}] Adapter total keys: {len(adapter_keys)}")

                        # Count keys by module
                        model_layers_keys = [k for k in adapter_keys if 'model.layers' in k]
                        audio_tower_keys = [k for k in adapter_keys if 'audio_tower' in k]
                        lora_a_keys = [k for k in adapter_keys if 'lora_A' in k]
                        lora_b_keys = [k for k in adapter_keys if 'lora_B' in k]

                        print(f"[PEFT_DIAG][{tag}] Adapter keys with model.layers: {len(model_layers_keys)}")
                        print(f"[PEFT_DIAG][{tag}] Adapter keys with audio_tower: {len(audio_tower_keys)}")
                        print(f"[PEFT_DIAG][{tag}] Adapter keys with lora_A: {len(lora_a_keys)}")
                        print(f"[PEFT_DIAG][{tag}] Adapter keys with lora_B: {len(lora_b_keys)}")

                        print(f"[PEFT_DIAG][{tag}] First 30 adapter keys:")
                        for i, key in enumerate(adapter_keys[:30]):
                            print(f"[PEFT_DIAG][{tag}]   {i+1}. {key}")

                        # Check for mismatch
                        if len(model_layers_keys) > 0 and module_lora_counts.get('model.layers', {}).get('total', 0) == 0:
                            print(f"[PEFT_DIAG][{tag}][ERROR] adapter has language LoRA keys but loaded model has none")
                        elif len(model_layers_keys) == 0:
                            print(f"[PEFT_DIAG][{tag}][ERROR] adapter checkpoint has no language-model LoRA keys")
                except Exception as e:
                    print(f"[PEFT_DIAG][{tag}] Error reading adapter file: {e}")
            else:
                print(f"[PEFT_DIAG][{tag}] Adapter file NOT FOUND")

        print(f"[PEFT_DIAG][{tag}] ===== End PEFT Diagnostics =====")
        print(f"{'='*80}\n")

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # [GRPO_DDP_FIX] Get gradient_checkpointing_kwargs with non-reentrant default
        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {"use_reentrant": False}

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            logger.info("[GRPO_DDP_FIX] Enabling gradient checkpointing for PEFT model with kwargs: {gradient_checkpointing_kwargs}")
            model.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        # Enable gradient checkpointing for non-PEFT models
        else:
            try:
                logger.info("[GRPO_DDP_FIX] Enabling gradient checkpointing for non-PEFT model with kwargs: {gradient_checkpointing_kwargs}")
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            except:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False

        use_reentrant = gradient_checkpointing_kwargs.get("use_reentrant", False)

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    @contextmanager
    def _temporarily_disable_adapters(self, model):
        """
        Context manager to temporarily disable adapters for reference logprob computation.

        Compatibility wrapper that handles both disable_adapter() and disable_adapters() APIs.

        Args:
            model: The model (may be wrapped by accelerator)

        Yields:
            None

        Usage:
            with self._temporarily_disable_adapters(self.model):
                ref_logprobs = self._get_per_token_logps(...)
        """
        unwrapped = self.accelerator.unwrap_model(model)

        # Check for disable_adapter (singular) context manager
        if hasattr(unwrapped, 'disable_adapter') and callable(getattr(unwrapped, 'disable_adapter')):
            print("[ADAPTER_CONTEXT] using disable_adapter context manager", flush=True)
            with unwrapped.disable_adapter():
                yield
            print("[ADAPTER_CONTEXT] adapters restored", flush=True)
            return

        # Check for disable_adapters (plural) method
        if hasattr(unwrapped, 'disable_adapters') and callable(getattr(unwrapped, 'disable_adapters')):
            print("[ADAPTER_CONTEXT] using disable_adapters()/enable_adapters() compatibility wrapper", flush=True)
            try:
                unwrapped.disable_adapters()
                yield
            finally:
                # Try enable_adapters() with no args first, then with True
                if hasattr(unwrapped, 'enable_adapters') and callable(getattr(unwrapped, 'enable_adapters')):
                    try:
                        unwrapped.enable_adapters()
                    except TypeError:
                        # If enable_adapters() requires an argument, try True
                        unwrapped.enable_adapters(True)
                print("[ADAPTER_CONTEXT] adapters restored", flush=True)

                # [ADAPTER_CONTEXT][POST_RESTORE_STATE] Verify adapter state after restore
                grpo_lora_train_scope = os.environ.get("GRPO_LORA_TRAIN_SCOPE", "model_layers_only")
                if grpo_lora_train_scope == "audio_tower_only":
                    print("[ADAPTER_CONTEXT][POST_RESTORE_STATE] === Verifying Adapter State After Restore ===", flush=True)
                    # Find audio_tower module
                    audio_tower_module = None
                    for module_name, module in unwrapped.named_modules():
                        if module_name == 'audio_tower' or (module_name.endswith('.audio_tower') and not any(c in module_name[:-len('.audio_tower')] for c in ['lora_', 'base_'])):
                            audio_tower_module = module
                            break

                    if audio_tower_module is not None:
                        lora_layer_count = 0
                        for module_name, module in audio_tower_module.named_modules():
                            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and lora_layer_count < 5:
                                active_adapters = getattr(module, 'active_adapters', None)
                                disable_adapters = getattr(module, 'disable_adapters', getattr(module, '_disable_adapters', None))
                                merged = getattr(module, 'merged', None)
                                print(f"[ADAPTER_CONTEXT][POST_RESTORE_STATE] module={module_name}", flush=True)
                                print(f"[ADAPTER_CONTEXT][POST_RESTORE_STATE]   active_adapters={active_adapters}", flush=True)
                                print(f"[ADAPTER_CONTEXT][POST_RESTORE_STATE]   disable_adapters={disable_adapters}", flush=True)
                                print(f"[ADAPTER_CONTEXT][POST_RESTORE_STATE]   merged={merged}", flush=True)
                                lora_layer_count += 1
                    print("[ADAPTER_CONTEXT][POST_RESTORE_STATE] === End Adapter State Verification ===", flush=True)

                # Reapply LoRA train scope after adapter restore
                self._apply_lora_train_scope(unwrapped, reason="after_adapter_restore")
            return

        # No adapter disable method found, use nullcontext
        print("[ADAPTER_CONTEXT][WARNING] no disable_adapter or disable_adapters found, using nullcontext", flush=True)
        with nullcontext():
            yield
        print("[ADAPTER_CONTEXT] nullcontext exited (no adapters to restore)", flush=True)


    def _compute_loss_with_backward_microbatch(self, model, inputs, input_ids, attention_mask, completion_ids, completion_mask, multimodal_inputs, rank):
        """
        Compute GRPO loss with backward micro-batch enabled.

        This method splits policy logprob computation into micro-batches, computes loss contribution
        for each micro-batch, and immediately calls backward to release computation graph.

        Returns:
            detached_loss: scalar loss tensor (detached) for logging only
        """
        backward_micro_batch_size_env = os.environ.get("GRPO_LOGPROB_BACKWARD_MICRO_BATCH_SIZE", None)
        backward_micro_batch_size = int(backward_micro_batch_size_env)
        original_batch_size = input_ids.shape[0]

        if rank == 0:
            print(f"\n[GRPO_BACKWARD_MICROBATCH] === Backward micro-batch enabled ===", flush=True)
            print(f"[GRPO_BACKWARD_MICROBATCH] enabled = True", flush=True)
            print(f"[GRPO_BACKWARD_MICROBATCH] micro_batch_size = {backward_micro_batch_size}", flush=True)
            print(f"[GRPO_BACKWARD_MICROBATCH] original_batch_size = {original_batch_size}", flush=True)

        # Get advantages from inputs
        advantages = inputs["advantages"]
        patial_advantages = inputs["patial_advantages"]

        for patial_advantage in patial_advantages:
            advantages = advantages + (patial_advantage["reward"].unsqueeze(-1) * patial_advantage["mask"]).sum(dim=1)

        mode = "eval" if self.control.should_evaluate else "train"

        # Compute old_per_token_logps (no grad) for full batch
        if self.num_iterations > 1:
            old_per_token_logps = self._get_per_token_logps_with_microbatch(
                self.model, input_ids, attention_mask, require_grad=False,
                completion_ids=completion_ids, completion_mask=completion_mask,
                **multimodal_inputs
            )
        else:
            # Will compute per chunk and detach
            old_per_token_logps = None

        # Compute ref_per_token_logps (no grad) for full batch if beta > 0
        ref_per_token_logps = None
        beta = 0.0
        if self.beta > 0:
            if self.state.global_step > (self.state.max_steps / 2):
                beta = self.beta * 0.25
            else:
                beta = self.beta * (1 - 0.75 * self.state.global_step / (self.state.max_steps / 2))

            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps_with_microbatch(
                    self.ref_model, input_ids, attention_mask, require_grad=False,
                    completion_ids=completion_ids, completion_mask=completion_mask,
                    **multimodal_inputs
                )
            else:
                with self._temporarily_disable_adapters(self.model):
                    ref_per_token_logps = self._get_per_token_logps_with_microbatch(
                        self.model, input_ids, attention_mask, require_grad=False,
                        completion_ids=completion_ids, completion_mask=completion_mask,
                        **multimodal_inputs
                    )

        # Calculate global denominator (must use full batch completion_mask)
        global_completion_mask_sum = completion_mask.sum().clamp_min(1.0)

        if rank == 0:
            print(f"[GRPO_BACKWARD_MICROBATCH] global_denominator = {global_completion_mask_sum.item()}", flush=True)

        # Calculate number of chunks
        num_chunks = (original_batch_size + backward_micro_batch_size - 1) // backward_micro_batch_size
        if rank == 0:
            print(f"[GRPO_BACKWARD_MICROBATCH] num_chunks = {num_chunks}", flush=True)

        # Prepare model_inputs dict for slicing
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'completion_ids': completion_ids,
            'completion_mask': completion_mask,
        }
        model_inputs.update(multimodal_inputs)

        # Accumulate detached loss for logging
        accumulated_loss = 0.0

        # Process each micro-batch
        for chunk_idx in range(num_chunks):
            start = chunk_idx * backward_micro_batch_size
            end = min(start + backward_micro_batch_size, original_batch_size)

            if rank == 0:
                print(f"\n[GRPO_BACKWARD_MICROBATCH] === Processing chunk {chunk_idx + 1}/{num_chunks} ===", flush=True)
                print(f"[GRPO_BACKWARD_MICROBATCH] chunk {chunk_idx} start/end = {start}/{end}", flush=True)

            # Slice inputs for this micro-batch
            chunk_inputs = slice_model_inputs_for_batch(model_inputs, start, end, rank=rank)

            chunk_input_ids = chunk_inputs['input_ids']
            chunk_attention_mask = chunk_inputs['attention_mask']
            chunk_completion_ids = chunk_inputs['completion_ids']
            chunk_completion_mask = chunk_inputs['completion_mask']

            # Extract custom multimodal inputs
            chunk_multimodal_inputs = {}
            for key in multimodal_inputs.keys():
                if key in chunk_inputs:
                    chunk_multimodal_inputs[key] = chunk_inputs[key]

            # Slice advantages, old_per_token_logps, ref_per_token_logps
            chunk_advantages = advantages[start:end]
            chunk_old_per_token_logps = old_per_token_logps[start:end] if old_per_token_logps is not None else None
            chunk_ref_per_token_logps = ref_per_token_logps[start:end] if ref_per_token_logps is not None else None

            # Compute policy per_token_logps for this chunk WITH gradients
            chunk_policy_per_token_logps = self._get_per_token_logps(
                model=model,
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask,
                require_grad=True,
                completion_ids=chunk_completion_ids,
                completion_mask=chunk_completion_mask,
                **chunk_multimodal_inputs
            )

            if rank == 0:
                print(f"[GRPO_BACKWARD_MICROBATCH] chunk policy_per_token_logps.shape = {chunk_policy_per_token_logps.shape}", flush=True)
                print(f"[GRPO_BACKWARD_MICROBATCH] chunk policy_per_token_logps.requires_grad = {chunk_policy_per_token_logps.requires_grad}", flush=True)

            # If num_iterations == 1, use detached policy logps as old logps
            if chunk_old_per_token_logps is None:
                chunk_old_per_token_logps = chunk_policy_per_token_logps.detach()

            # Ensure all tensors on same device
            loss_device = chunk_policy_per_token_logps.device
            loss_dtype = chunk_policy_per_token_logps.dtype

            if chunk_completion_mask.device != loss_device:
                chunk_completion_mask = chunk_completion_mask.to(loss_device)
            if chunk_completion_mask.dtype != loss_dtype:
                chunk_completion_mask = chunk_completion_mask.to(loss_dtype)

            if chunk_advantages.device != loss_device:
                chunk_advantages = chunk_advantages.to(loss_device)
            if chunk_advantages.dtype != loss_dtype:
                chunk_advantages = chunk_advantages.to(loss_dtype)

            if chunk_old_per_token_logps.device != loss_device:
                chunk_old_per_token_logps = chunk_old_per_token_logps.to(loss_device)

            # Compute per-token loss for this chunk (same GRPO formula)
            coef_1 = torch.exp(chunk_policy_per_token_logps - chunk_old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss1 = coef_1 * chunk_advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * chunk_advantages.unsqueeze(1)
            chunk_per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

            # Add KL penalty if beta > 0
            if self.beta > 0 and chunk_ref_per_token_logps is not None:
                if chunk_ref_per_token_logps.device != loss_device:
                    chunk_ref_per_token_logps = chunk_ref_per_token_logps.to(loss_device)

                per_token_kl = torch.exp(chunk_ref_per_token_logps - chunk_policy_per_token_logps) - (chunk_ref_per_token_logps - chunk_policy_per_token_logps) - 1
                chunk_per_token_loss = chunk_per_token_loss + beta * per_token_kl

                # Log KL divergence for this chunk
                mean_kl = (per_token_kl * chunk_completion_mask).sum() / chunk_completion_mask.sum()
                self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().detach().cpu().item())

            # Compute chunk loss using GLOBAL denominator
            chunk_loss = (chunk_per_token_loss * chunk_completion_mask).sum() / global_completion_mask_sum

            if rank == 0:
                print(f"[GRPO_BACKWARD_MICROBATCH] chunk_loss.requires_grad = {chunk_loss.requires_grad}", flush=True)
                print(f"[GRPO_BACKWARD_MICROBATCH] chunk_loss.item() = {chunk_loss.item()}", flush=True)

            # Accumulate detached loss for logging
            accumulated_loss += chunk_loss.detach().item()

            # Log clip ratio for this chunk
            is_clipped = (per_token_loss1 < per_token_loss2).float()
            clip_ratio = (is_clipped * chunk_completion_mask).sum() / chunk_completion_mask.sum()
            self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().detach().cpu().item())

            # Call backward for this chunk
            # Must account for gradient_accumulation_steps
            if self.args.gradient_accumulation_steps > 1:
                chunk_loss = chunk_loss / self.args.gradient_accumulation_steps

            self.accelerator.backward(chunk_loss)

            if rank == 0:
                print(f"[GRPO_BACKWARD_MICROBATCH] backward done for chunk {chunk_idx}", flush=True)

            # Release computation graph for this chunk
            del chunk_policy_per_token_logps, chunk_per_token_loss, chunk_loss, coef_1, coef_2, per_token_loss1, per_token_loss2, is_clipped
            if self.beta > 0 and chunk_ref_per_token_logps is not None:
                del per_token_kl

        # Return detached loss for logging
        detached_loss = torch.tensor(accumulated_loss, device=input_ids.device, dtype=torch.float32)

        if rank == 0:
            print(f"[GRPO_BACKWARD_MICROBATCH] final detached_loss = {detached_loss.item()}", flush=True)
            print(f"[GRPO_BACKWARD_MICROBATCH] === End backward micro-batch ===\n", flush=True)

        return detached_loss

    # [GRPO_LOGPROB_FORWARD_MICRO_BATCH] Wrapper for micro-batch forward
    def _get_per_token_logps_with_microbatch(self, model, input_ids, attention_mask, require_grad=False, completion_ids=None, completion_mask=None, completion_len=None, **custom_multimodal_inputs):
        """
        Wrapper for _get_per_token_logps that supports micro-batch forward.

        When GRPO_LOGPROB_FORWARD_MICRO_BATCH_SIZE is set, splits the batch into micro-batches
        and processes them sequentially to reduce memory usage.
        """
        # Get rank for logging
        rank = 0
        try:
            import torch.distributed as torch_dist
            if torch_dist.is_available() and torch_dist.is_initialized():
                rank = torch_dist.get_rank()
        except:
            rank = 0

        # Check if micro-batch is enabled
        micro_batch_size_env = os.environ.get("GRPO_LOGPROB_FORWARD_MICRO_BATCH_SIZE", None)
        if micro_batch_size_env is None or micro_batch_size_env == "0":
            # Micro-batch disabled, call original method directly
            return self._get_per_token_logps(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                require_grad=require_grad,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                completion_len=completion_len,
                **custom_multimodal_inputs
            )

        # Micro-batch enabled
        micro_batch_size = int(micro_batch_size_env)
        original_batch_size = input_ids.shape[0]

        if rank == 0:
            print(f"\n[GRPO_LOGPROB_MICROBATCH] === Micro-batch forward enabled ===", flush=True)
            print(f"[GRPO_LOGPROB_MICROBATCH] enabled = True", flush=True)
            print(f"[GRPO_LOGPROB_MICROBATCH] micro_batch_size = {micro_batch_size}", flush=True)
            print(f"[GRPO_LOGPROB_MICROBATCH] original_batch_size = {original_batch_size}", flush=True)

        # If batch size <= micro_batch_size, no need to split
        if original_batch_size <= micro_batch_size:
            if rank == 0:
                print(f"[GRPO_LOGPROB_MICROBATCH] batch_size <= micro_batch_size, no split needed", flush=True)
            return self._get_per_token_logps(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                require_grad=require_grad,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                completion_len=completion_len,
                **custom_multimodal_inputs
            )

        # Calculate number of chunks
        num_chunks = (original_batch_size + micro_batch_size - 1) // micro_batch_size
        if rank == 0:
            print(f"[GRPO_LOGPROB_MICROBATCH] num_chunks = {num_chunks}", flush=True)

        # Prepare model_inputs dict for slicing
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        if completion_ids is not None:
            model_inputs['completion_ids'] = completion_ids
        if completion_mask is not None:
            model_inputs['completion_mask'] = completion_mask
        model_inputs.update(custom_multimodal_inputs)

        # Process each micro-batch
        per_token_logps_chunks = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * micro_batch_size
            end = min(start + micro_batch_size, original_batch_size)

            if rank == 0:
                print(f"\n[GRPO_LOGPROB_MICROBATCH] === Processing chunk {chunk_idx + 1}/{num_chunks} ===", flush=True)
                print(f"[GRPO_LOGPROB_MICROBATCH] chunk {chunk_idx} start/end = {start}/{end}", flush=True)

            # Slice inputs for this micro-batch
            chunk_inputs = slice_model_inputs_for_batch(model_inputs, start, end, rank=rank)

            # Extract sliced tensors
            chunk_input_ids = chunk_inputs['input_ids']
            chunk_attention_mask = chunk_inputs['attention_mask']
            chunk_completion_ids = chunk_inputs.get('completion_ids', None)
            chunk_completion_mask = chunk_inputs.get('completion_mask', None)

            # Extract custom multimodal inputs
            chunk_custom_multimodal_inputs = {}
            for key in custom_multimodal_inputs.keys():
                if key in chunk_inputs:
                    chunk_custom_multimodal_inputs[key] = chunk_inputs[key]

            if rank == 0:
                print(f"[GRPO_LOGPROB_MICROBATCH] chunk input_ids.shape = {chunk_input_ids.shape}", flush=True)
                print(f"[GRPO_LOGPROB_MICROBATCH] chunk attention_mask.shape = {chunk_attention_mask.shape}", flush=True)
                if 'input_features' in chunk_custom_multimodal_inputs and chunk_custom_multimodal_inputs['input_features'] is not None:
                    print(f"[GRPO_LOGPROB_MICROBATCH] chunk input_features.shape = {chunk_custom_multimodal_inputs['input_features'].shape}", flush=True)
                if 'feature_attention_mask' in chunk_custom_multimodal_inputs and chunk_custom_multimodal_inputs['feature_attention_mask'] is not None:
                    print(f"[GRPO_LOGPROB_MICROBATCH] chunk feature_attention_mask.shape = {chunk_custom_multimodal_inputs['feature_attention_mask'].shape}", flush=True)
                if 'video_grid_thw' in chunk_custom_multimodal_inputs and chunk_custom_multimodal_inputs['video_grid_thw'] is not None:
                    print(f"[GRPO_LOGPROB_MICROBATCH] chunk video_grid_thw.shape = {chunk_custom_multimodal_inputs['video_grid_thw'].shape}", flush=True)
                if 'pixel_values_videos' in chunk_custom_multimodal_inputs and chunk_custom_multimodal_inputs['pixel_values_videos'] is not None:
                    print(f"[GRPO_LOGPROB_MICROBATCH] chunk pixel_values_videos.shape = {chunk_custom_multimodal_inputs['pixel_values_videos'].shape}", flush=True)

            # Call original _get_per_token_logps for this micro-batch
            chunk_per_token_logps = self._get_per_token_logps(
                model=model,
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask,
                require_grad=require_grad,
                completion_ids=chunk_completion_ids,
                completion_mask=chunk_completion_mask,
                completion_len=completion_len,
                **chunk_custom_multimodal_inputs
            )

            if rank == 0:
                print(f"[GRPO_LOGPROB_MICROBATCH] chunk per_token_logps.shape = {chunk_per_token_logps.shape}", flush=True)
                print(f"[GRPO_LOGPROB_MICROBATCH] chunk per_token_logps.requires_grad = {chunk_per_token_logps.requires_grad}", flush=True)

            per_token_logps_chunks.append(chunk_per_token_logps)

        # Concatenate all chunks along batch dimension
        final_per_token_logps = torch.cat(per_token_logps_chunks, dim=0)

        if rank == 0:
            print(f"\n[GRPO_LOGPROB_MICROBATCH] === Concatenating chunks ===", flush=True)
            print(f"[GRPO_LOGPROB_MICROBATCH] final per_token_logps.shape = {final_per_token_logps.shape}", flush=True)
            print(f"[GRPO_LOGPROB_MICROBATCH] final per_token_logps.requires_grad = {final_per_token_logps.requires_grad}", flush=True)
            print(f"[GRPO_LOGPROB_MICROBATCH] === End micro-batch forward ===\n", flush=True)

        # Validate output shape
        expected_batch_size = original_batch_size
        if final_per_token_logps.shape[0] != expected_batch_size:
            raise RuntimeError(
                f"[GRPO_LOGPROB_MICROBATCH][FATAL] batch size mismatch after concatenation\n"
                f"  final_per_token_logps.shape[0] = {final_per_token_logps.shape[0]}\n"
                f"  expected_batch_size = {expected_batch_size}"
            )

        return final_per_token_logps

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, require_grad=False, completion_ids=None, completion_mask=None, completion_len=None, **custom_multimodal_inputs):
        # [GRPO_LORA_TRAIN_SCOPE] Define at function entry to avoid UnboundLocalError
        grpo_lora_train_scope = os.environ.get("GRPO_LORA_TRAIN_SCOPE", "model_layers_only")

        # [GRPO_COMPLETION_LEN_FIX] Calculate real completion length from provided parameters
        if completion_len is not None:
            real_completion_len = int(completion_len)
        elif completion_ids is not None:
            real_completion_len = completion_ids.shape[1]
        elif completion_mask is not None:
            real_completion_len = completion_mask.shape[1]
        else:
            raise RuntimeError(
                "[GRPO_COMPLETION_LEN_FIX] missing completion_ids/completion_mask/completion_len\n"
                "  At least one of these parameters must be provided to determine completion length"
            )

        # [GRPO_COMPLETION_LEN_FIX] Validate completion length
        max_completion_length = getattr(self.args, 'max_completion_length', None) or getattr(self, 'max_completion_length', None)

        print(f"[GRPO_COMPLETION_LEN_FIX] input_ids.shape = {input_ids.shape}", flush=True)
        if completion_ids is not None:
            print(f"[GRPO_COMPLETION_LEN_FIX] completion_ids.shape = {completion_ids.shape}", flush=True)
        if completion_mask is not None:
            print(f"[GRPO_COMPLETION_LEN_FIX] completion_mask.shape = {completion_mask.shape}", flush=True)
        print(f"[GRPO_COMPLETION_LEN_FIX] real_completion_len = {real_completion_len}", flush=True)
        print(f"[GRPO_COMPLETION_LEN_FIX] max_completion_length = {max_completion_length}", flush=True)

        # Safety check: completion_len should not be suspiciously large
        if real_completion_len > 2048:
            raise RuntimeError(
                f"[GRPO_COMPLETION_LEN_FIX] completion_len too large, likely using full prompt_completion length\n"
                f"  real_completion_len = {real_completion_len}\n"
                f"  This should be the completion-only length, not prompt+completion"
            )

        # Validate against max_completion_length if available
        if max_completion_length is not None and real_completion_len > max_completion_length:
            print(
                f"[GRPO_COMPLETION_LEN_FIX] WARNING: real_completion_len ({real_completion_len}) > "
                f"max_completion_length ({max_completion_length})", flush=True
            )

        # [GRPO_REF_EMBED_DIAG][GET_LOGPS] Diagnostics for ref_model embedding
        if not require_grad:
            try:
                logger.info(f"[GRPO_REF_EMBED_DIAG][GET_LOGPS] model type = {type(model)}")
                logger.info(f"[GRPO_REF_EMBED_DIAG][GET_LOGPS] input_ids shape = {input_ids.shape}")
                logger.info(f"[GRPO_REF_EMBED_DIAG][GET_LOGPS] input_ids min/max = {input_ids.min().item()}/{input_ids.max().item()}")

                # Try to get embedding module
                emb_module = None
                if hasattr(model, 'get_input_embeddings'):
                    emb_module = model.get_input_embeddings()
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                    if hasattr(model.base_model.model, 'get_input_embeddings'):
                        emb_module = model.base_model.model.get_input_embeddings()

                if emb_module is not None and hasattr(emb_module, 'weight'):
                    emb_weight = emb_module.weight
                    logger.info(f"[GRPO_REF_EMBED_DIAG][GET_LOGPS] embedding weight shape = {emb_weight.shape}")
                    logger.info(f"[GRPO_REF_EMBED_DIAG][GET_LOGPS] embedding weight device = {emb_weight.device}")
                    logger.info(f"[GRPO_REF_EMBED_DIAG][GET_LOGPS] embedding weight dtype = {emb_weight.dtype}")

                    # Validation: embedding weight must be 2D
                    if emb_weight.dim() != 2:
                        raise RuntimeError(
                            f"[GRPO_REF_EMBED_DIAG][GET_LOGPS] ref_model embedding weight is not 2D\n"
                            f"  shape = {emb_weight.shape}\n"
                            f"  This may indicate ZeRO-3 partition placeholder issue"
                        )
                else:
                    logger.warning("[GRPO_REF_EMBED_DIAG][GET_LOGPS] could not get embedding weight")
            except Exception as e:
                logger.warning(f"[GRPO_REF_EMBED_DIAG][GET_LOGPS] failed to check embedding: {e}")

        # Consistency check before model forward
        print(f"[GRPO_AUDIO_GRAD_DIAG][GET_LOGPS_ENTER] require_grad={require_grad}", flush=True)
        print(f"[GRPO_AUDIO_GRAD_DIAG][GET_LOGPS_ENTER] input_ids.shape={getattr(input_ids, 'shape', None)}", flush=True)
        print(f"[GRPO_AUDIO_GRAD_DIAG][GET_LOGPS_ENTER] attention_mask.shape={getattr(attention_mask, 'shape', None)}", flush=True)
        print(f"[GRPO_AUDIO_GRAD_DIAG][GET_LOGPS_ENTER] torch.is_grad_enabled={torch.is_grad_enabled()}", flush=True)

        batch_size = input_ids.size(0)
        if attention_mask.size(0) != batch_size:
            raise RuntimeError(
                f"Batch dimension mismatch before model forward:\n"
                f"  input_ids.shape: {input_ids.shape}\n"
                f"  attention_mask.shape: {attention_mask.shape}\n"
                f"  Batch dimensions must match."
            )

        # [STAGE3_AUDIO_DIAG][MISMATCH] Forward input diagnostics
        rank = 0
        try:
            import torch.distributed as torch_dist
            if torch_dist.is_available() and torch_dist.is_initialized():
                rank = torch_dist.get_rank()
        except:
            rank = 0

        stage3_audio_diag = os.environ.get("STAGE3_AUDIO_DIAG", "0") == "1" or os.environ.get("GRPO_STAGE3_AUDIO_DIAG", "0") == "1"
        if stage3_audio_diag and rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"\n[STAGE3_AUDIO_DIAG][MISMATCH] === _get_per_token_logps forward input check ===", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][MISMATCH] input_ids.shape={input_ids.shape}, dtype={input_ids.dtype}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][MISMATCH] attention_mask.shape={attention_mask.shape}, dtype={attention_mask.dtype}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][MISMATCH] custom_multimodal_inputs keys: {list(custom_multimodal_inputs.keys())}", flush=True)
            for key, value in custom_multimodal_inputs.items():
                if value is not None and isinstance(value, torch.Tensor):
                    print(f"[STAGE3_AUDIO_DIAG][MISMATCH] {key}: shape={value.shape}, dtype={value.dtype}", flush=True)
                    if value.size(0) != batch_size and key not in {'pixel_values_videos', 'pixel_values', 'pixel_values_images', 'video_grid_thw', 'image_grid_thw'}:
                        print(f"[STAGE3_AUDIO_DIAG][MISMATCH] [MISMATCH] {key} batch size mismatch: {value.size(0)} != {batch_size}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][MISMATCH] === End forward input check ===\n", flush=True)

        # Check multimodal inputs batch consistency
        # Note: pixel_values_videos, pixel_values, pixel_values_images are packed tensors
        # whose first dimension is NOT the text batch size, so we skip batch checks for them
        packed_feature_keys = {
            'pixel_values_videos', 'pixel_values', 'pixel_values_images',
            'video_grid_thw', 'image_grid_thw'
        }

        for key, value in custom_multimodal_inputs.items():
            if value is None:
                continue

            # Skip batch check for packed feature tensors and metadata
            if key in packed_feature_keys:
                continue

            # For other tensors, check if they are batch-major
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                # Only check batch dimension for tensors that are clearly batch-major
                # (e.g., audio_values, input_features, etc.)
                if value.size(0) == batch_size or value.size(0) == 1:
                    # Valid: either matches batch_size or is a single shared tensor
                    pass
                else:
                    # Log warning but don't fail - some tensors may have different structure
                    pass

            # For lists, only check if they represent per-sample data
            elif isinstance(value, list):
                # Only check length for lists that should be per-sample
                # Skip if the list contains packed data or metadata
                if len(value) == batch_size or len(value) == 0:
                    pass
                else:
                    # Log warning but don't fail
                    pass

        # [STAGE3_AUDIO_DIAG][FORWARD_INPUT] Pre-forward alignment diagnostics
        rank = 0
        try:
            import torch.distributed as torch_dist
            if torch_dist.is_available() and torch_dist.is_initialized():
                rank = torch_dist.get_rank()
        except:
            rank = 0

        if rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"\n[STAGE3_AUDIO_DIAG][FORWARD_INPUT] === Pre-forward alignment check ===", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] input_ids.shape={input_ids.shape}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] attention_mask.shape={attention_mask.shape}", flush=True)
            input_ids_batch = input_ids.size(0)
            attn_mask_batch = attention_mask.size(0)
            print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] input_ids batch={input_ids_batch}, attention_mask batch={attn_mask_batch}", flush=True)
            if input_ids.numel() > 0:
                min_id = input_ids.min().item()
                max_id = input_ids.max().item()
                print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] input_ids min={min_id}, max={max_id}", flush=True)
                vocab_size = getattr(model.config, 'vocab_size', 'UNKNOWN')
                print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] model.config.vocab_size={vocab_size}", flush=True)
                if isinstance(vocab_size, int) and max_id >= vocab_size:
                    print(f"[STAGE3_AUDIO_DIAG][MISMATCH] token id out of vocab: max_id={max_id} >= vocab_size={vocab_size}", flush=True)
                if min_id < 0:
                    print(f"[STAGE3_AUDIO_DIAG][MISMATCH] negative token id: min_id={min_id}", flush=True)
            if 'input_features' in custom_multimodal_inputs and custom_multimodal_inputs['input_features'] is not None:
                input_feat = custom_multimodal_inputs['input_features']
                feat_batch = input_feat.size(0)
                print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] input_features.shape={input_feat.shape}, batch={feat_batch}", flush=True)
                if feat_batch != input_ids_batch:
                    print(f"[STAGE3_AUDIO_DIAG][MISMATCH] input_ids batch != input_features batch: {input_ids_batch} != {feat_batch}", flush=True)
            if 'feature_attention_mask' in custom_multimodal_inputs and custom_multimodal_inputs['feature_attention_mask'] is not None:
                feat_attn_mask = custom_multimodal_inputs['feature_attention_mask']
                feat_attn_batch = feat_attn_mask.size(0)
                print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] feature_attention_mask.shape={feat_attn_mask.shape}, batch={feat_attn_batch}", flush=True)
                if feat_attn_batch != input_ids_batch:
                    print(f"[STAGE3_AUDIO_DIAG][MISMATCH] input_ids batch != feature_attention_mask batch: {input_ids_batch} != {feat_attn_batch}", flush=True)
                feat_sum = feat_attn_mask.sum(dim=-1)
                print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] feature_attention_mask.sum(dim=-1)={feat_sum.tolist()}", flush=True)
            if 'pixel_values_videos' in custom_multimodal_inputs and custom_multimodal_inputs['pixel_values_videos'] is not None:
                pv_videos = custom_multimodal_inputs['pixel_values_videos']
                print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] pixel_values_videos.shape={pv_videos.shape}", flush=True)
            if 'video_grid_thw' in custom_multimodal_inputs and custom_multimodal_inputs['video_grid_thw'] is not None:
                vg_thw = custom_multimodal_inputs['video_grid_thw']
                print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] video_grid_thw.shape={vg_thw.shape}", flush=True)
            use_audio_in_video = custom_multimodal_inputs.get('use_audio_in_video', 'MISSING')
            print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] use_audio_in_video={use_audio_in_video}", flush=True)
            if attention_mask.shape != input_ids.shape:
                print(f"[STAGE3_AUDIO_DIAG][MISMATCH] attention_mask shape != input_ids shape: {attention_mask.shape} != {input_ids.shape}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] === End pre-forward alignment check ===\n", flush=True)

        # [GRAD_CONTROL] Use context manager to control gradient computation
        if require_grad:
            context = torch.enable_grad()
        else:
            context = torch.no_grad()

        # [GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] Check forward context before model forward
        if rank == 0 and os.environ.get("GRPO_AUDIO_GRAD_DIAG", "0") == "1":
            print(f"\n[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] === Forward Context Check ===", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] require_grad={require_grad}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] torch.is_grad_enabled()={torch.is_grad_enabled()}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] model.training={self.model.training}", flush=True)

            # Try to get audio_tower
            try:
                unwrapped_model = self.model
                if hasattr(unwrapped_model, 'module'):
                    unwrapped_model = unwrapped_model.module
                if hasattr(unwrapped_model, 'audio_tower'):
                    print(f"[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] audio_tower.training={unwrapped_model.audio_tower.training}", flush=True)
            except:
                pass

            if 'input_features' in custom_multimodal_inputs and custom_multimodal_inputs['input_features'] is not None:
                input_feat = custom_multimodal_inputs['input_features']
                print(f"[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] input_features.requires_grad={input_feat.requires_grad}", flush=True)
                print(f"[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] input_features.grad_fn={input_feat.grad_fn}", flush=True)

            print(f"[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] === End Forward Context Check ===\n", flush=True)

        # [STAGE3_AUDIO_DIAG][LEN_TRACE] Before model forward
        if rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"[STAGE3_AUDIO_DIAG][LEN_TRACE] input_ids.shape={input_ids.shape}, attention_mask.shape={attention_mask.shape}", flush=True)
            if input_ids.shape[1] != attention_mask.shape[1]:
                print(f"[STAGE3_AUDIO_DIAG][LEN_TRACE][MISMATCH_BEFORE_FORWARD] LENGTH MISMATCH: input_ids seq={input_ids.shape[1]} != attention_mask seq={attention_mask.shape[1]}, diff={input_ids.shape[1] - attention_mask.shape[1]}", flush=True)

        # [STAGE3_AUDIO_DIAG][AUDIO_SCATTER] Comprehensive audio field diagnostics before forward
        if rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"\n[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] === Audio scatter diagnostics ===", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 1. input_ids.shape={input_ids.shape}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 2. attention_mask.shape={attention_mask.shape}", flush=True)

            # Audio features
            if 'input_features' in custom_multimodal_inputs and custom_multimodal_inputs['input_features'] is not None:
                input_feat = custom_multimodal_inputs['input_features']
                print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 3. input_features.shape={input_feat.shape}", flush=True)
                print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 3b. input_features.numel()={input_feat.numel()}", flush=True)
            else:
                print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 3. input_features=MISSING", flush=True)

            # Feature attention mask
            if 'feature_attention_mask' in custom_multimodal_inputs and custom_multimodal_inputs['feature_attention_mask'] is not None:
                feat_attn_mask = custom_multimodal_inputs['feature_attention_mask']
                print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 4. feature_attention_mask.shape={feat_attn_mask.shape}", flush=True)
                feat_sum = feat_attn_mask.sum(dim=-1)
                print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 5. feature_attention_mask.sum(dim=-1)={feat_sum.tolist()}", flush=True)
            else:
                print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 4. feature_attention_mask=MISSING", flush=True)

            # Audio feature lengths
            if 'audio_feature_lengths' in custom_multimodal_inputs and custom_multimodal_inputs['audio_feature_lengths'] is not None:
                audio_feat_lens = custom_multimodal_inputs['audio_feature_lengths']
                if isinstance(audio_feat_lens, torch.Tensor):
                    print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 6. audio_feature_lengths.shape={audio_feat_lens.shape}, dtype={audio_feat_lens.dtype}, values={audio_feat_lens.tolist()}", flush=True)
                else:
                    print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 6. audio_feature_lengths={audio_feat_lens} (type={type(audio_feat_lens).__name__})", flush=True)
            else:
                print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 6. audio_feature_lengths=MISSING", flush=True)

            # Audio seqlens (if exists)
            if 'audio_seqlens' in custom_multimodal_inputs and custom_multimodal_inputs['audio_seqlens'] is not None:
                audio_seqlens = custom_multimodal_inputs['audio_seqlens']
                if isinstance(audio_seqlens, torch.Tensor):
                    print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 7. audio_seqlens.shape={audio_seqlens.shape}, dtype={audio_seqlens.dtype}, values={audio_seqlens.tolist()}", flush=True)
                else:
                    print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 7. audio_seqlens={audio_seqlens} (type={type(audio_seqlens).__name__})", flush=True)
            else:
                print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 7. audio_seqlens=MISSING", flush=True)

            # use_audio_in_video
            use_audio_in_video = custom_multimodal_inputs.get('use_audio_in_video', 'MISSING')
            print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 8. use_audio_in_video={use_audio_in_video}", flush=True)

            # Get tokenizer from trainer (not from model.config)
            tokenizer = None
            if hasattr(self, 'processing_class'):
                tokenizer = getattr(self.processing_class, 'tokenizer', self.processing_class)
            elif hasattr(self, 'tokenizer'):
                tokenizer = self.tokenizer
            elif hasattr(self, 'processor'):
                tokenizer = getattr(self.processor, 'tokenizer', self.processor)
            elif hasattr(self, 'vlm_module') and hasattr(self.vlm_module, 'processor'):
                tokenizer = getattr(self.vlm_module.processor, 'tokenizer', self.vlm_module.processor)

            # Audio special token ids
            audio_special_token_ids = {}
            if tokenizer is not None:
                for token_str in ['<|AUDIO|>', '<|audio_bos|>', '<|audio_eos|>', '<|audio|>', '<|AUDIO_BOS|>', '<|AUDIO_EOS|>']:
                    try:
                        if hasattr(tokenizer, 'convert_tokens_to_ids'):
                            token_id = tokenizer.convert_tokens_to_ids(token_str)
                            if token_id is not None and token_id != tokenizer.unk_token_id:
                                audio_special_token_ids[token_str] = token_id
                        elif hasattr(tokenizer, 'encode'):
                            encoded = tokenizer.encode(token_str, add_special_tokens=False)
                            if encoded and len(encoded) == 1:
                                audio_special_token_ids[token_str] = encoded[0]
                    except:
                        pass

            print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 9. audio_special_token_ids={audio_special_token_ids}", flush=True)

            # Count audio tokens in input_ids
            if audio_special_token_ids:
                for token_str, token_id in audio_special_token_ids.items():
                    total_count = (input_ids == token_id).sum().item()
                    print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 10. {token_str}(id={token_id}): total_count={total_count}", flush=True)
                    for batch_idx in range(input_ids.shape[0]):
                        sample_count = (input_ids[batch_idx] == token_id).sum().item()
                        if sample_count > 0:
                            print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 10b. sample[{batch_idx}] {token_str}={sample_count}", flush=True)
            else:
                print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 10. No audio special tokens found", flush=True)

            # Audio token index from QwenOmni config
            try:
                audio_token_index = getattr(model.config, 'audio_token_index', None)
                if audio_token_index is not None:
                    print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 11. model.config.audio_token_index={audio_token_index}", flush=True)
                else:
                    print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 11. model.config.audio_token_index=MISSING", flush=True)
            except Exception as e:
                print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 11. Error getting audio_token_index: {e}", flush=True)

            # Construction source of audio_feature_lengths
            print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] 12. audio_feature_lengths construction: from feature_attention_mask.sum(dim=1) if feature_attention_mask provided", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][AUDIO_SCATTER] === End audio scatter diagnostics ===\n", flush=True)

        print(f"\n[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] === Forward Context Check ===", flush=True)
        print(f"[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] require_grad={require_grad}", flush=True)
        print(f"[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] torch.is_grad_enabled()={torch.is_grad_enabled()}", flush=True)
        print(f"[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] model.training={model.training}", flush=True)
        try:
            unwrapped_model = model
            if hasattr(unwrapped_model, 'module'):
                unwrapped_model = unwrapped_model.module
            if hasattr(unwrapped_model, 'audio_tower'):
                print(f"[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] audio_tower.training={unwrapped_model.audio_tower.training}", flush=True)
        except:
            pass
        if 'input_features' in custom_multimodal_inputs and custom_multimodal_inputs['input_features'] is not None:
            input_feat = custom_multimodal_inputs['input_features']
            print(f"[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] input_features.requires_grad={input_feat.requires_grad}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] input_features.grad_fn={input_feat.grad_fn}", flush=True)
        print(f"[GRPO_AUDIO_GRAD_DIAG][FORWARD_CONTEXT] === End Forward Context Check ===\n", flush=True)

        # [GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_HOOK_REGISTER] Register forward hook for audio_tower
        # Only register training hooks when audio_tower_only mode AND require_grad=True
        verbose = os.environ.get("GRPO_AUDIO_GRAD_DIAG_VERBOSE", "0") == "1"
        if grpo_lora_train_scope == "audio_tower_only" and require_grad:
            if verbose:
                print(f"\n[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_HOOK_REGISTER] === Registering Audio Tower Forward Hook (training mode) ===", flush=True)

            # Find all possible audio_tower objects
            audio_tower_objects = []
            unwrapped_model = self.accelerator.unwrap_model(model)

            # Try model.module.audio_tower
            if hasattr(model, 'module') and hasattr(model.module, 'audio_tower'):
                audio_tower_objects.append(('model.module.audio_tower', model.module.audio_tower, id(model.module.audio_tower)))

            # Try unwrapped.audio_tower
            if hasattr(unwrapped_model, 'audio_tower'):
                audio_tower_objects.append(('unwrapped.audio_tower', unwrapped_model.audio_tower, id(unwrapped_model.audio_tower)))

            # Deduplicate by id
            seen_ids = set()
            unique_audio_tower_objects = []
            for source, obj, obj_id in audio_tower_objects:
                if obj_id not in seen_ids:
                    seen_ids.add(obj_id)
                    unique_audio_tower_objects.append((source, obj, obj_id))

            if verbose:
                print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_HOOK_REGISTER] Found {len(unique_audio_tower_objects)} unique audio_tower objects", flush=True)

            if len(unique_audio_tower_objects) == 0:
                print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_HOOK_REGISTER][FATAL] no audio_tower module found", flush=True)

            for source, audio_tower_module, obj_id in unique_audio_tower_objects:
                if verbose:
                    print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_HOOK_REGISTER] Registering hook for: {source}, id={obj_id}", flush=True)

                def make_audio_tower_hook(hook_source, hook_obj_id):
                    def audio_tower_forward_hook(module, input, output):
                        verbose_inner = os.environ.get("GRPO_AUDIO_GRAD_DIAG_VERBOSE", "0") == "1"
                        if verbose_inner:
                            print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_FORWARD_HOOK] source={hook_source}, id={hook_obj_id}", flush=True)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_FORWARD_HOOK] torch.is_grad_enabled={torch.is_grad_enabled()}", flush=True)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_FORWARD_HOOK] module.training={module.training}", flush=True)

                            # Diagnose module's internal LoRA parameters
                            module_lora_param_total = 0
                            module_lora_param_requires_grad_true = 0
                            module_lora_params = []
                            for param_name, param in module.named_parameters():
                                if 'lora_' in param_name:
                                    module_lora_param_total += 1
                                    if param.requires_grad:
                                        module_lora_param_requires_grad_true += 1
                                    if len(module_lora_params) < 20:
                                        module_lora_params.append((param_name, param.requires_grad, param.shape))

                            print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_FORWARD_HOOK] module_lora_param_total={module_lora_param_total}", flush=True)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_FORWARD_HOOK] module_lora_param_requires_grad_true={module_lora_param_requires_grad_true}", flush=True)
                            for i, (name, req_grad, shape) in enumerate(module_lora_params[:5]):
                                print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_FORWARD_HOOK] lora_param_{i}: {name}, requires_grad={req_grad}, shape={shape}", flush=True)

                            if isinstance(output, torch.Tensor):
                                print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_FORWARD_HOOK] output requires_grad: {output.requires_grad}", flush=True)
                                print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_FORWARD_HOOK] output grad_fn: {output.grad_fn}", flush=True)
                            elif hasattr(output, 'last_hidden_state'):
                                print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_FORWARD_HOOK] output.last_hidden_state requires_grad: {output.last_hidden_state.requires_grad}", flush=True)
                                print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_FORWARD_HOOK] output.last_hidden_state grad_fn: {output.last_hidden_state.grad_fn}", flush=True)
                    return audio_tower_forward_hook

                audio_tower_module.register_forward_hook(make_audio_tower_hook(source, obj_id))
                if verbose:
                    print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_HOOK_REGISTER] Hook registered for {source}", flush=True)

                # [LORA_LAYER_HOOK_REGISTER] Register forward hooks on individual LoRA Linear layers
                if verbose:
                    print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_HOOK_REGISTER] === Registering LoRA Layer Forward Hooks for {source} ===", flush=True)
                lora_layer_hook_count = 0
                for module_name, module in audio_tower_module.named_modules():
                    # Find LoRA Linear layers (typically have lora_A and lora_B attributes)
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and lora_layer_hook_count < 3:
                        if verbose:
                            has_lora_A = hasattr(module, 'lora_A')
                            has_lora_B = hasattr(module, 'lora_B')
                            module_id = id(module)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_HOOK_REGISTER] audio_tower_source={source}", flush=True)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_HOOK_REGISTER] module_name={module_name}", flush=True)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_HOOK_REGISTER] module_id={module_id}", flush=True)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_HOOK_REGISTER] has_lora_A={has_lora_A}", flush=True)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_HOOK_REGISTER] has_lora_B={has_lora_B}", flush=True)

                        def make_lora_layer_hook(layer_name):
                            def lora_layer_forward_hook(module, input, output):
                                verbose_inner = os.environ.get("GRPO_AUDIO_GRAD_DIAG_VERBOSE", "0") == "1"
                                if verbose_inner:
                                    print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_FORWARD_HOOK] module_name={layer_name}", flush=True)

                                    # Output requires_grad
                                    if isinstance(output, torch.Tensor):
                                        print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_FORWARD_HOOK] output_requires_grad={output.requires_grad}", flush=True)
                                        print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_FORWARD_HOOK] output_grad_fn={output.grad_fn}", flush=True)

                                    # Adapter state
                                    active_adapters = getattr(module, 'active_adapters', None)
                                    disable_adapters = getattr(module, 'disable_adapters', getattr(module, '_disable_adapters', None))
                                    merged = getattr(module, 'merged', None)
                                    print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_FORWARD_HOOK] active_adapters={active_adapters}", flush=True)
                                    print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_FORWARD_HOOK] disable_adapters={disable_adapters}", flush=True)
                                    print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_FORWARD_HOOK] merged={merged}", flush=True)
                            return lora_layer_forward_hook

                        module.register_forward_hook(make_lora_layer_hook(module_name))
                        lora_layer_hook_count += 1

                if verbose:
                    print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_HOOK_REGISTER] Registered {lora_layer_hook_count} LoRA layer hooks for {source}", flush=True)

            if verbose:
                print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_HOOK_REGISTER] === End Audio Tower Forward Hook Registration ===\n", flush=True)

        # [AUDIO_TOWER_LORA_ENABLE] Ensure LoRA adapter is enabled for audio_tower in audio_tower_only mode
        if grpo_lora_train_scope == "audio_tower_only" and require_grad:
            if verbose:
                print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_LORA_ENABLE] audio_tower_only mode detected with require_grad=True", flush=True)

            # Find audio_tower_module if not already found
            if 'audio_tower_module' not in locals():
                audio_tower_module = None
                for module_name, module in model.named_modules():
                    if module_name == 'audio_tower' or (module_name.endswith('.audio_tower') and not any(c in module_name[:-len('.audio_tower')] for c in ['lora_', 'base_'])):
                        audio_tower_module = module
                        break

            if audio_tower_module is not None:
                if verbose:
                    print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_LORA_ENABLE] Enabling LoRA adapter for audio_tower", flush=True)
                # Enable LoRA adapter if it exists
                if hasattr(audio_tower_module, 'enable_lora'):
                    audio_tower_module.enable_lora()
                    if verbose:
                        print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_LORA_ENABLE] Called enable_lora() on audio_tower", flush=True)
                # Verify LoRA parameters are trainable
                lora_params_enabled = 0
                for param_name, param in audio_tower_module.named_parameters():
                    if 'lora_' in param_name and param.requires_grad:
                        lora_params_enabled += 1
                if verbose:
                    print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_LORA_ENABLE] audio_tower LoRA params with requires_grad=True: {lora_params_enabled}", flush=True)

                # [LORA_LAYER_STATE] Print LoRA layer adapter state for first 5 LoRA layers
                if verbose:
                    print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_STATE] === LoRA Layer Adapter State ===", flush=True)
                    lora_layer_state_count = 0
                    for module_name, module in audio_tower_module.named_modules():
                        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and lora_layer_state_count < 5:
                            active_adapters = getattr(module, 'active_adapters', None)
                            disable_adapters = getattr(module, 'disable_adapters', getattr(module, '_disable_adapters', None))
                            merged = getattr(module, 'merged', None)
                            lora_A = getattr(module, 'lora_A', {})
                            lora_B = getattr(module, 'lora_B', {})
                            lora_A_keys = list(lora_A.keys()) if isinstance(lora_A, dict) else 'N/A'
                            lora_B_keys = list(lora_B.keys()) if isinstance(lora_B, dict) else 'N/A'

                            print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_STATE] module_name={module_name}", flush=True)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_STATE]   active_adapters={active_adapters}", flush=True)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_STATE]   disable_adapters={disable_adapters}", flush=True)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_STATE]   merged={merged}", flush=True)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_STATE]   lora_A_keys={lora_A_keys}", flush=True)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_STATE]   lora_B_keys={lora_B_keys}", flush=True)

                            # Check if active adapter exists in lora_A/lora_B
                            if active_adapters and isinstance(lora_A, dict):
                                for adapter in (active_adapters if isinstance(active_adapters, list) else [active_adapters]):
                                    adapter_exists = adapter in lora_A
                                    print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_STATE]   adapter '{adapter}' exists in lora_A: {adapter_exists}", flush=True)

                            # Check scale/scaling if accessible
                            scaling = getattr(module, 'scaling', None)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_STATE]   scaling={scaling}", flush=True)

                            lora_layer_state_count += 1
                    print(f"[GRPO_AUDIO_GRAD_DIAG][LORA_LAYER_STATE] === End LoRA Layer Adapter State ===\n", flush=True)
            else:
                if verbose:
                    print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_TOWER_LORA_ENABLE] [WARNING] audio_tower_module is None, cannot enable LoRA", flush=True)

        # [ZERO3_GATHER_DIAG] Diagnose ZeRO-3 gather status before forward
        if grpo_lora_train_scope == "audio_tower_only" and require_grad and verbose:
            print(f"[GRPO_AUDIO_GRAD_DIAG][ZERO3_GATHER_DIAG] === ZeRO-3 Gather Diagnosis Before Forward ===", flush=True)
            try:
                from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

                # Find audio_tower_module if not already found
                if 'audio_tower_module' not in locals():
                    audio_tower_module = None
                    for module_name, module in model.named_modules():
                        if module_name == 'audio_tower' or (module_name.endswith('.audio_tower') and not any(c in module_name[:-len('.audio_tower')] for c in ['lora_', 'base_'])):
                            audio_tower_module = module
                            break

                if audio_tower_module is not None:
                    not_available_count = 0
                    for param_name, param in audio_tower_module.named_parameters():
                        if 'lora_' in param_name:
                            param_status = getattr(param, "ds_status", None)
                            if param_status == ZeroParamStatus.NOT_AVAILABLE:
                                not_available_count += 1
                    print(f"[GRPO_AUDIO_GRAD_DIAG][ZERO3_GATHER_DIAG] audio_tower LoRA params with NOT_AVAILABLE status: {not_available_count}", flush=True)
            except ImportError:
                print(f"[GRPO_AUDIO_GRAD_DIAG][ZERO3_GATHER_DIAG] ZeroParamStatus not available for gather diagnosis", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][ZERO3_GATHER_DIAG] === End ZeRO-3 Gather Diagnosis ===", flush=True)

        # [GRPO_LORA_SCOPE_REAPPLY] Apply LoRA train scope before policy forward
        if grpo_lora_train_scope == "audio_tower_only" and require_grad:
            self._apply_lora_train_scope(model, reason="before_policy_logprob_forward")

        # [ACTUAL_FORWARD_MODEL] Diagnose actual model objects before forward
        if grpo_lora_train_scope == "audio_tower_only" and require_grad and verbose:
            print(f"\n[GRPO_AUDIO_GRAD_DIAG][ACTUAL_FORWARD_MODEL] === Actual Forward Model Diagnosis ===", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][ACTUAL_FORWARD_MODEL] type(model)={type(model)}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][ACTUAL_FORWARD_MODEL] type(self.model)={type(self.model)}", flush=True)

            unwrapped_model = self.accelerator.unwrap_model(model)
            print(f"[GRPO_AUDIO_GRAD_DIAG][ACTUAL_FORWARD_MODEL] type(unwrapped)={type(unwrapped_model)}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][ACTUAL_FORWARD_MODEL] hasattr(model, 'module')={hasattr(model, 'module')}", flush=True)

            if hasattr(model, 'module'):
                print(f"[GRPO_AUDIO_GRAD_DIAG][ACTUAL_FORWARD_MODEL] type(model.module)={type(model.module)}", flush=True)
                print(f"[GRPO_AUDIO_GRAD_DIAG][ACTUAL_FORWARD_MODEL] hasattr(model.module, 'audio_tower')={hasattr(model.module, 'audio_tower')}", flush=True)
                if hasattr(model.module, 'audio_tower'):
                    print(f"[GRPO_AUDIO_GRAD_DIAG][ACTUAL_FORWARD_MODEL] id(model.module.audio_tower)={id(model.module.audio_tower)}", flush=True)

            print(f"[GRPO_AUDIO_GRAD_DIAG][ACTUAL_FORWARD_MODEL] hasattr(unwrapped, 'audio_tower')={hasattr(unwrapped_model, 'audio_tower')}", flush=True)
            if hasattr(unwrapped_model, 'audio_tower'):
                print(f"[GRPO_AUDIO_GRAD_DIAG][ACTUAL_FORWARD_MODEL] id(unwrapped.audio_tower)={id(unwrapped_model.audio_tower)}", flush=True)

            print(f"[GRPO_AUDIO_GRAD_DIAG][ACTUAL_FORWARD_MODEL] === End Actual Forward Model Diagnosis ===\n", flush=True)

        # [AUDIO_CHECKPOINT_DIAG] Diagnose gradient checkpointing status before forward
        if grpo_lora_train_scope == "audio_tower_only" and require_grad:
            # Always print INPUT_FEATURES_GRAD_FIX status (not verbose-gated)
            print(f"\n[GRPO_AUDIO_GRAD_DIAG][AUDIO_CHECKPOINT_DIAG] === Gradient Checkpointing Diagnosis ===", flush=True)

            if verbose:
                # Check audio_tower gradient_checkpointing attribute
                try:
                    unwrapped_model = self.accelerator.unwrap_model(model)
                    if hasattr(unwrapped_model, 'audio_tower'):
                        audio_tower_gc = getattr(unwrapped_model.audio_tower, 'gradient_checkpointing', 'UNKNOWN')
                        print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_CHECKPOINT_DIAG] audio_tower.gradient_checkpointing={audio_tower_gc}", flush=True)
                    else:
                        print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_CHECKPOINT_DIAG] audio_tower.gradient_checkpointing=UNKNOWN (no audio_tower)", flush=True)
                except Exception as e:
                    print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_CHECKPOINT_DIAG] audio_tower.gradient_checkpointing=UNKNOWN (error: {e})", flush=True)

                # Check model gradient_checkpointing attribute
                try:
                    model_gc = getattr(model, 'gradient_checkpointing', 'UNKNOWN')
                    print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_CHECKPOINT_DIAG] model.gradient_checkpointing={model_gc}", flush=True)
                except Exception as e:
                    print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_CHECKPOINT_DIAG] model.gradient_checkpointing=UNKNOWN (error: {e})", flush=True)

            # Check input_features properties (always print for INPUT_FEATURES_GRAD_FIX)
            if 'input_features' in custom_multimodal_inputs and custom_multimodal_inputs['input_features'] is not None:
                input_feat = custom_multimodal_inputs['input_features']
                print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_CHECKPOINT_DIAG] input_features.requires_grad_before={input_feat.requires_grad}", flush=True)
                if verbose:
                    print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_CHECKPOINT_DIAG] input_features.is_leaf={input_feat.is_leaf}", flush=True)
                    print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_CHECKPOINT_DIAG] input_features.dtype={input_feat.dtype}", flush=True)
                    print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_CHECKPOINT_DIAG] input_features.shape={input_feat.shape}", flush=True)
            else:
                print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_CHECKPOINT_DIAG] input_features=MISSING", flush=True)

            print(f"[GRPO_AUDIO_GRAD_DIAG][AUDIO_CHECKPOINT_DIAG] === End Gradient Checkpointing Diagnosis ===\n", flush=True)

        # [INPUT_FEATURES_GRAD_FIX] Enable gradient for input_features when training audio_tower LoRA
        # This is required because audio_tower uses gradient checkpointing, which needs at least one
        # grad-enabled input to build the autograd graph for LoRA parameters
        # Applies to both audio_tower_only and joint modes
        if grpo_lora_train_scope in {"audio_tower_only", "joint"} and require_grad:
            if 'input_features' in custom_multimodal_inputs and custom_multimodal_inputs['input_features'] is not None:
                input_feat = custom_multimodal_inputs['input_features']
                if isinstance(input_feat, torch.Tensor) and not input_feat.requires_grad:
                    print(f"\n[GRPO_AUDIO_GRAD_DIAG][INPUT_FEATURES_GRAD_FIX] === Enabling input_features gradient ===", flush=True)
                    print(f"[GRPO_AUDIO_GRAD_DIAG][INPUT_FEATURES_GRAD_FIX] enabled=True", flush=True)
                    print(f"[GRPO_AUDIO_GRAD_DIAG][INPUT_FEATURES_GRAD_FIX] GRPO_LORA_TRAIN_SCOPE={grpo_lora_train_scope}", flush=True)
                    print(f"[GRPO_AUDIO_GRAD_DIAG][INPUT_FEATURES_GRAD_FIX] before_requires_grad={input_feat.requires_grad}", flush=True)

                    # Detach and enable gradient - this creates a new leaf tensor with requires_grad=True
                    # without changing the numerical values or adding it to the optimizer
                    custom_multimodal_inputs['input_features'] = input_feat.detach().requires_grad_(True)

                    print(f"[GRPO_AUDIO_GRAD_DIAG][INPUT_FEATURES_GRAD_FIX] after_requires_grad={custom_multimodal_inputs['input_features'].requires_grad}", flush=True)
                    print(f"[GRPO_AUDIO_GRAD_DIAG][INPUT_FEATURES_GRAD_FIX] reason=audio_tower gradient checkpointing requires at least one grad-enabled input", flush=True)
                    print(f"[GRPO_AUDIO_GRAD_DIAG][INPUT_FEATURES_GRAD_FIX] === End input_features gradient fix ===\n", flush=True)

        # [GRPO_REF_LOGPS_MEMFIX] Split execution path based on require_grad
        if require_grad:
            # [GRPO_POLICY_RAW_FORWARD] Policy model path: call unwrapped model to avoid DDP _post_forward clone OOM
            print(f"\n[GRPO_POLICY_RAW_FORWARD] === Policy path: raw model forward to avoid DDP clone ===", flush=True)
            print(f"[GRPO_POLICY_RAW_FORWARD] enabled = True", flush=True)
            print(f"[GRPO_POLICY_RAW_FORWARD] wrapped model type = {type(model)}", flush=True)

            # [GRPO_POLICY_LMHEAD_PATCH] Policy model path: patch lm_head to compute only completion logits
            print(f"[GRPO_POLICY_LMHEAD_PATCH] enabled = True", flush=True)
            print(f"[GRPO_POLICY_LMHEAD_PATCH] input_ids.shape = {input_ids.shape}", flush=True)
            print(f"[GRPO_POLICY_LMHEAD_PATCH] real_completion_len = {real_completion_len}", flush=True)

            # Unwrap model to find raw_policy_model and lm_head
            raw_policy_model = model
            if hasattr(self, "accelerator"):
                try:
                    import inspect
                    unwrap_sig = inspect.signature(self.accelerator.unwrap_model)
                    if 'keep_fp32_wrapper' in unwrap_sig.parameters:
                        raw_policy_model = self.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
                        print(f"[GRPO_POLICY_RAW_FORWARD] unwrapped via accelerator with keep_fp32_wrapper=False", flush=True)
                    else:
                        raw_policy_model = self.accelerator.unwrap_model(model)
                        print(f"[GRPO_POLICY_RAW_FORWARD] unwrapped via accelerator", flush=True)
                except TypeError:
                    raw_policy_model = self.accelerator.unwrap_model(model)
                    print(f"[GRPO_POLICY_RAW_FORWARD] unwrapped via accelerator (TypeError fallback)", flush=True)
                except Exception as e:
                    print(f"[GRPO_POLICY_RAW_FORWARD][WARN] unwrap failed: {e}", flush=True)

            if hasattr(raw_policy_model, "module"):
                raw_policy_model = raw_policy_model.module
                print(f"[GRPO_POLICY_RAW_FORWARD] unwrapped .module", flush=True)

            print(f"[GRPO_POLICY_RAW_FORWARD] raw_policy_model type = {type(raw_policy_model)}", flush=True)
            print(f"[GRPO_POLICY_RAW_FORWARD] calling raw_policy_model forward, not DDP-wrapped model", flush=True)

            # Find lm_head
            lm_head = None
            if hasattr(raw_policy_model, 'lm_head'):
                lm_head = raw_policy_model.lm_head
                print(f"[GRPO_POLICY_LMHEAD_PATCH] using raw_policy_model.lm_head", flush=True)
            elif hasattr(raw_policy_model, 'get_base_model'):
                base = raw_policy_model.get_base_model()
                if hasattr(base, 'lm_head'):
                    lm_head = base.lm_head
                    print(f"[GRPO_POLICY_LMHEAD_PATCH] using base.lm_head", flush=True)
            elif hasattr(raw_policy_model, 'base_model') and hasattr(raw_policy_model.base_model, 'model'):
                base = raw_policy_model.base_model.model
                if hasattr(base, 'lm_head'):
                    lm_head = base.lm_head
                    print(f"[GRPO_POLICY_LMHEAD_PATCH] using base_model.model.lm_head", flush=True)

            if lm_head is None:
                raise RuntimeError(
                    "[GRPO_POLICY_LMHEAD_PATCH][FATAL] cannot find lm_head\n"
                    "  Cannot patch lm_head for completion-only logits"
                )

            print(f"[GRPO_POLICY_LMHEAD_PATCH] lm_head type = {type(lm_head)}", flush=True)

            # Save original lm_head forward
            orig_lm_head_forward = lm_head.forward

            # Define patched forward that only processes completion positions
            def completion_only_lm_head_forward(hidden_states):
                # Validate sufficient length for autoregressive shift
                if hidden_states.shape[1] < real_completion_len + 1:
                    raise RuntimeError(
                        f"[GRPO_POLICY_LMHEAD_PATCH][FATAL] insufficient hidden states\n"
                        f"  hidden_states.shape[1] = {hidden_states.shape[1]}\n"
                        f"  real_completion_len + 1 = {real_completion_len + 1}"
                    )
                # Slice to keep only last (real_completion_len + 1) positions
                # This includes positions needed for autoregressive prediction
                sliced_hidden = hidden_states[:, -(real_completion_len + 1):, :]
                print(f"[GRPO_POLICY_LMHEAD_PATCH] sliced_hidden.shape = {sliced_hidden.shape}", flush=True)
                return orig_lm_head_forward(sliced_hidden)

            # Temporarily patch lm_head.forward
            lm_head.forward = completion_only_lm_head_forward

            try:
                # Call raw_policy_model forward (not DDP-wrapped model) with patched lm_head
                with context:
                    outputs = raw_policy_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **custom_multimodal_inputs
                    )
                    logits = outputs.logits

                print(f"[GRPO_POLICY_LMHEAD_PATCH] logits.shape = {logits.shape}", flush=True)

                # Validate logits shape: should be [batch, real_completion_len + 1, vocab]
                if logits.shape[1] != real_completion_len + 1:
                    raise RuntimeError(
                        f"[GRPO_POLICY_LMHEAD_PATCH][FATAL] logits shape mismatch\n"
                        f"  logits.shape[1] = {logits.shape[1]}\n"
                        f"  real_completion_len + 1 = {real_completion_len + 1}"
                    )

                # Apply autoregressive shift: logits[:, :-1] predicts input_ids[:, -real_completion_len:]
                shift_logits = logits[:, :-1, :]  # (B, completion_len, vocab)
                target_ids = input_ids[:, -real_completion_len:]  # (B, completion_len)

                print(f"[GRPO_POLICY_LMHEAD_PATCH] shift_logits.shape = {shift_logits.shape}", flush=True)
                print(f"[GRPO_POLICY_LMHEAD_PATCH] target_ids.shape = {target_ids.shape}", flush=True)

                # Validate shapes
                if shift_logits.shape[1] != real_completion_len:
                    raise RuntimeError(
                        f"[GRPO_POLICY_LMHEAD_PATCH][FATAL] shift_logits shape mismatch\n"
                        f"  shift_logits.shape[1] = {shift_logits.shape[1]}\n"
                        f"  real_completion_len = {real_completion_len}"
                    )

                if target_ids.shape[1] != real_completion_len:
                    raise RuntimeError(
                        f"[GRPO_POLICY_LMHEAD_PATCH][FATAL] target_ids shape mismatch\n"
                        f"  target_ids.shape[1] = {target_ids.shape[1]}\n"
                        f"  real_completion_len = {real_completion_len}"
                    )

                # Compute log probabilities
                per_token_logps = []
                for logits_row, target_ids_row in zip(shift_logits, target_ids):
                    log_probs = logits_row.log_softmax(dim=-1)
                    token_log_prob = torch.gather(log_probs, dim=1, index=target_ids_row.unsqueeze(1)).squeeze(1)
                    per_token_logps.append(token_log_prob)
                    del log_probs, token_log_prob

                result = torch.stack(per_token_logps)
                del per_token_logps, shift_logits, logits, outputs

                print(f"[GRPO_POLICY_LMHEAD_PATCH] per_token_logps.shape = {result.shape}", flush=True)

            finally:
                # Restore original lm_head forward
                lm_head.forward = orig_lm_head_forward
                print(f"[GRPO_POLICY_LMHEAD_PATCH] lm_head.forward restored = True", flush=True)

            # [GRPO_POLICY_RAW_FORWARD] Add zero-touch to ensure all trainable LoRA parameters enter autograd graph
            # This prevents DDP unused parameter errors when calling raw_policy_model instead of DDP-wrapped model
            zero_touch = None
            zero_touch_param_count = 0
            for name, p in raw_policy_model.named_parameters():
                if p.requires_grad:
                    z = p.sum() * 0.0
                    zero_touch = z if zero_touch is None else zero_touch + z
                    zero_touch_param_count += 1

            print(f"[GRPO_POLICY_RAW_FORWARD] zero_touch_trainable_param_count = {zero_touch_param_count}", flush=True)

            if zero_touch is not None:
                result = result + zero_touch
                print(f"[GRPO_POLICY_RAW_FORWARD] zero_touch_added = True", flush=True)
            else:
                print(f"[GRPO_POLICY_RAW_FORWARD] zero_touch_added = False (no trainable params)", flush=True)

            # [GRPO_POLICY_LMHEAD_PATCH] Validate output shape
            if result.shape[1] != real_completion_len:
                raise RuntimeError(
                    f"[GRPO_POLICY_LMHEAD_PATCH][FATAL] per_token_logps shape mismatch\n"
                    f"  result.shape[1] = {result.shape[1]}\n"
                    f"  real_completion_len = {real_completion_len}\n"
                    f"  These must match"
                )

            # [GRPO_POLICY_LMHEAD_PATCH] Validate gradient flow
            if not result.requires_grad:
                raise RuntimeError(
                    "[GRPO_POLICY_LMHEAD_PATCH][FATAL] per_token_logps.requires_grad=False\n"
                    "  Gradient flow is broken, LoRA parameters will not be updated"
                )

            print(f"[GRPO_POLICY_LMHEAD_PATCH] per_token_logps.requires_grad = {result.requires_grad}", flush=True)
            print(f"[GRPO_POLICY_RAW_FORWARD] === End policy path ===\n", flush=True)

            # [GRPO_POLICY_RAW_FORWARD] Early return to prevent fallback to old DDP-wrapped model(...) path
            # This ensures require_grad=True policy path NEVER reaches any old model(...) calls below
            return result

        else:
            # Ref model path: use PeftModel wrapper with adapters disabled
            print(f"\n[GRPO_REF_WRAPPER_BASE] === Using PeftModel wrapper ref path ===", flush=True)
            print(f"[GRPO_REF_WRAPPER_BASE] enabled = True", flush=True)
            print(f"[GRPO_REF_WRAPPER_BASE] using PeftModel wrapper with adapters disabled", flush=True)
            print(f"[GRPO_REF_WRAPPER_BASE] wrapped model type = {type(model)}", flush=True)

            # Unwrap model from Accelerate wrapper
            raw_model = model
            if hasattr(self, "accelerator"):
                try:
                    # Try to unwrap with keep_fp32_wrapper=False if supported
                    import inspect
                    unwrap_sig = inspect.signature(self.accelerator.unwrap_model)
                    if 'keep_fp32_wrapper' in unwrap_sig.parameters:
                        raw_model = self.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
                        print(f"[GRPO_REF_WRAPPER_BASE] unwrapped via accelerator with keep_fp32_wrapper=False", flush=True)
                    else:
                        raw_model = self.accelerator.unwrap_model(model)
                        print(f"[GRPO_REF_WRAPPER_BASE] unwrapped via accelerator (keep_fp32_wrapper not supported)", flush=True)
                except Exception as e:
                    print(f"[GRPO_REF_WRAPPER_BASE] accelerator unwrap failed: {e}", flush=True)

            if hasattr(raw_model, "module"):
                raw_model = raw_model.module
                print(f"[GRPO_REF_WRAPPER_BASE] unwrapped .module", flush=True)

            print(f"[GRPO_REF_WRAPPER_BASE] raw_model type = {type(raw_model)}", flush=True)

            # [GRPO_REF_WRAPPER_BASE] Use PeftModel wrapper as ref_forward_model
            # Adapters are already disabled via outer disable_adapter() context manager
            # This ensures ref = base model while preserving input processing
            ref_forward_model = raw_model
            print(f"[GRPO_REF_WRAPPER_BASE] do_not_use_get_base_model = True", flush=True)
            print(f"[GRPO_REF_WRAPPER_BASE] ref_forward_model type = {type(ref_forward_model)}", flush=True)
            print(f"[GRPO_REF_WRAPPER_BASE] input_ids.shape = {input_ids.shape}", flush=True)

            # Calculate total sequence length
            total_seq_len = input_ids.shape[1]
            print(f"[GRPO_REF_WRAPPER_BASE] total_seq_len = {total_seq_len}", flush=True)

            # [GRPO_REF_FORCE_VOCAB_CHUNK_LOGPS] Check if vocab-chunk logprob is forced for all ref samples
            force_vocab_chunk = int(os.environ.get("GRPO_REF_FORCE_VOCAB_CHUNK_LOGPS", "1"))
            print(f"[GRPO_REF_FORCE_VOCAB_CHUNK_LOGPS] enabled = {force_vocab_chunk}", flush=True)

            # Determine if this sample should use vocab-chunk logprob
            # Default: all ref samples use vocab-chunk logprob when GRPO_REF_FORCE_VOCAB_CHUNK_LOGPS=1
            use_vocab_chunk_logprob = (force_vocab_chunk == 1)

            print(f"\n[GRPO_REF_VOCAB_CHUNK_ROUTE] === Vocab chunk routing ===", flush=True)
            print(f"[GRPO_REF_VOCAB_CHUNK_ROUTE] require_grad = False", flush=True)
            print(f"[GRPO_REF_VOCAB_CHUNK_ROUTE] total_seq_len = {total_seq_len}", flush=True)
            print(f"[GRPO_REF_VOCAB_CHUNK_ROUTE] real_completion_len = {real_completion_len}", flush=True)
            print(f"[GRPO_REF_VOCAB_CHUNK_ROUTE] force_vocab_chunk = {force_vocab_chunk}", flush=True)
            print(f"[GRPO_REF_VOCAB_CHUNK_ROUTE] use_vocab_chunk_logprob = {use_vocab_chunk_logprob}", flush=True)

            if use_vocab_chunk_logprob:
                # [GRPO_REF_VOCAB_CHUNK_LOGPS] Use vocab-chunked exact logprob computation for all ref samples
                print(f"[GRPO_REF_VOCAB_CHUNK_ROUTE] using ref vocab-chunked logprob path", flush=True)
                print(f"\n[GRPO_REF_VOCAB_CHUNK_LOGPS] === Ref sample vocab-chunked logprob ===", flush=True)
                print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] enabled = True", flush=True)
                print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] input_ids.shape = {input_ids.shape}", flush=True)
                print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] real_completion_len = {real_completion_len}", flush=True)

                # Find ref lm_head from PeftModel wrapper
                ref_lm_head = None
                if hasattr(ref_forward_model, "lm_head"):
                    ref_lm_head = ref_forward_model.lm_head
                    print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] using ref_forward_model.lm_head", flush=True)
                elif hasattr(ref_forward_model, "base_model") and hasattr(ref_forward_model.base_model, "model"):
                    base = ref_forward_model.base_model.model
                    if hasattr(base, "lm_head"):
                        ref_lm_head = base.lm_head
                        print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] using ref_forward_model.base_model.model.lm_head", flush=True)
                elif hasattr(raw_model, "lm_head"):
                    ref_lm_head = raw_model.lm_head
                    print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] using raw_model.lm_head", flush=True)

                if ref_lm_head is None:
                    raise RuntimeError(
                        "[GRPO_REF_VOCAB_CHUNK_LOGPS][FATAL] cannot find lm_head\n"
                        "  Cannot patch lm_head for ref sample vocab-chunked logprob"
                    )

                print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] ref_lm_head type = {type(ref_lm_head)}", flush=True)

                # Save original lm_head forward
                orig_ref_lm_head_forward = ref_lm_head.forward

                # Prepare target_ids for vocab-chunked computation
                target_ids = input_ids[:, -real_completion_len:]
                print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] target_ids.shape = {target_ids.shape}", flush=True)

                # Get vocab chunk size from environment
                vocab_chunk_size = int(os.environ.get("GRPO_REF_VOCAB_CHUNK_SIZE", "8192"))
                print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] vocab_chunk_size = {vocab_chunk_size}", flush=True)

                # Define patched forward that computes vocab-chunked exact logprobs
                def ref_vocab_chunked_logprob_lm_head_forward(hidden_states):
                    # Validate sufficient length for autoregressive shift
                    if hidden_states.shape[1] < real_completion_len + 1:
                        raise RuntimeError(
                            f"[GRPO_REF_VOCAB_CHUNK_LOGPS][FATAL] insufficient hidden states\n"
                            f"  hidden_states.shape[1] = {hidden_states.shape[1]}\n"
                            f"  real_completion_len + 1 = {real_completion_len + 1}"
                        )

                    # Slice to keep only last (real_completion_len + 1) positions
                    sliced_hidden = hidden_states[:, -(real_completion_len + 1):, :]
                    print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] sliced_hidden.shape = {sliced_hidden.shape}", flush=True)

                    # Use positions [:-1] to predict target_ids (autoregressive shift)
                    predict_hidden = sliced_hidden[:, :-1, :]
                    print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] predict_hidden.shape = {predict_hidden.shape}", flush=True)

                    B, T, H = predict_hidden.shape
                    hidden_flat = predict_hidden.reshape(B * T, H)
                    target_flat = target_ids.reshape(B * T)

                    print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] hidden_flat.shape = {hidden_flat.shape}", flush=True)
                    print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] target_flat.shape = {target_flat.shape}", flush=True)

                    # Get lm_head weight and bias
                    weight = orig_ref_lm_head_forward.__self__.weight
                    bias = getattr(orig_ref_lm_head_forward.__self__, "bias", None)

                    vocab_size = weight.shape[0]
                    print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] vocab_size = {vocab_size}", flush=True)

                    # Compute target logits directly
                    target_weight = weight.index_select(0, target_flat)
                    target_logits = (hidden_flat * target_weight).sum(dim=-1)
                    if bias is not None:
                        target_logits = target_logits + bias.index_select(0, target_flat)

                    # Compute logsumexp over vocab in chunks
                    running_lse = None

                    for vocab_start in range(0, vocab_size, vocab_chunk_size):
                        vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
                        print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] vocab chunk start={vocab_start} end={vocab_end}", flush=True)

                        weight_chunk = weight[vocab_start:vocab_end]
                        logits_chunk = hidden_flat @ weight_chunk.t()
                        if bias is not None:
                            logits_chunk = logits_chunk + bias[vocab_start:vocab_end]

                        chunk_lse = torch.logsumexp(logits_chunk.float(), dim=-1)

                        if running_lse is None:
                            running_lse = chunk_lse
                        else:
                            running_lse = torch.logaddexp(running_lse, chunk_lse)

                        del logits_chunk, chunk_lse

                    # Compute per-token logprobs: target_logit - logsumexp(vocab)
                    per_token_logps_flat = target_logits.float() - running_lse
                    per_token_logps = per_token_logps_flat.view(B, T)

                    print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] per_token_logps.shape = {per_token_logps.shape}", flush=True)
                    print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] per_token_logps.requires_grad = {per_token_logps.requires_grad}", flush=True)

                    # Return per_token_logps instead of vocab logits
                    return per_token_logps

                # Temporarily patch ref_lm_head.forward
                ref_lm_head.forward = ref_vocab_chunked_logprob_lm_head_forward

                try:
                    # Call ref_forward_model (PeftModel wrapper) with patched lm_head
                    # Adapters are already disabled via outer context manager
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            outputs = ref_forward_model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                **custom_multimodal_inputs
                            )
                            # outputs.logits now contains per_token_logps, not vocab logits
                            result = outputs.logits

                    print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] result.shape = {result.shape}", flush=True)
                    print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] result.dim() = {result.dim()}", flush=True)

                    # Validate that we got per_token_logps, not vocab logits
                    # per_token_logps should be [batch, real_completion_len]
                    # vocab logits would be [batch, seq_len, vocab_size] with dim=3
                    if result.dim() == 3:
                        vocab_size = getattr(ref_forward_model.config, 'vocab_size', None)
                        if vocab_size is None and hasattr(ref_forward_model, 'base_model'):
                            vocab_size = getattr(ref_forward_model.base_model.config, 'vocab_size', None)
                        if vocab_size is not None and result.shape[-1] == vocab_size:
                            raise RuntimeError(
                                f"[GRPO_REF_WRAPPER_BASE][FATAL] ref path still returned vocab logits\n"
                                f"  result.shape = {result.shape}\n"
                                f"  result.dim() = {result.dim()}\n"
                                f"  vocab_size = {vocab_size}\n"
                                f"  Expected per_token_logps with shape [batch, real_completion_len]"
                            )

                    # Validate output shape
                    if result.shape[1] != real_completion_len:
                        raise RuntimeError(
                            f"[GRPO_REF_VOCAB_CHUNK_LOGPS][FATAL] per_token_logps shape mismatch\n"
                            f"  result.shape[1] = {result.shape[1]}\n"
                            f"  real_completion_len = {real_completion_len}"
                        )

                    # Validate gradient status
                    if result.requires_grad:
                        raise RuntimeError(
                            "[GRPO_REF_VOCAB_CHUNK_LOGPS][FATAL] per_token_logps.requires_grad=True\n"
                            "  Ref path should not have gradients"
                        )

                    print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] per_token_logps.requires_grad = {result.requires_grad}", flush=True)
                    print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] === End ref sample vocab-chunked logprob ===\n", flush=True)

                finally:
                    # Restore original lm_head forward
                    ref_lm_head.forward = orig_ref_lm_head_forward
                    print(f"[GRPO_REF_VOCAB_CHUNK_LOGPS] lm_head.forward restored = True", flush=True)

            else:
                # [GRPO_REF_DIRECT_BASE][FATAL] Old normal ref direct base path is disabled
                raise RuntimeError(
                    "[GRPO_REF_DIRECT_BASE][FATAL] old raw get_base_model ref path is disabled because it can produce audio_seqlens=None error\n"
                    f"  total_seq_len = {total_seq_len}\n"
                    f"  force_vocab_chunk = {force_vocab_chunk}\n"
                    f"  use_vocab_chunk_logprob = {use_vocab_chunk_logprob}\n"
                    "  Set GRPO_REF_FORCE_VOCAB_CHUNK_LOGPS=1 to use vocab-chunk logprob for all ref samples"
                )

        # [GRAD_CONTROL] Only detach if gradients are not required (already handled in ref path)
        if not require_grad:
            result = result.detach()

        # [GRPO_AUDIO_GRAD_DIAG][LOGPROB_GRAPH_FINAL] Check final result graph
        if rank == 0 and os.environ.get("GRPO_AUDIO_GRAD_DIAG", "0") == "1":
            print(f"\n[GRPO_AUDIO_GRAD_DIAG][LOGPROB_GRAPH_FINAL] === Final Per-Token Logps Graph ===", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][LOGPROB_GRAPH_FINAL] result.requires_grad={result.requires_grad}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][LOGPROB_GRAPH_FINAL] result.grad_fn={result.grad_fn}", flush=True)
            print(f"[GRPO_AUDIO_GRAD_DIAG][LOGPROB_GRAPH_FINAL] === End Final Per-Token Logps Graph ===\n", flush=True)

        return result


    def _prepare_inputs(self, inputs):
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                try:
                    inputs = self._generate_and_score_completions(inputs)
                    self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
                except Exception as e:
                    # Code errors should be raised directly, not treated as bad samples
                    code_error_types = (UnboundLocalError, NameError, SyntaxError, AttributeError)
                    if isinstance(e, code_error_types):
                        raise

                    if self.bad_sample_tracker.enable:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        tb_str = traceback.format_exc()
                        for idx, sample in enumerate(inputs):
                            self.bad_sample_tracker.record_bad_sample(
                                sample=sample,
                                error_type=error_type,
                                error_message=error_msg,
                                sample_index=idx,
                                tb_str=tb_str,
                            )
                        self.bad_sample_tracker.check_threshold()
                        raise RuntimeError(
                            f"All samples in batch failed during _generate_and_score_completions. "
                            f"Error: {error_msg}"
                        )
                    else:
                        raise
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            try:
                inputs = self._generate_and_score_completions(inputs)
            except Exception as e:
                # Code errors should be raised directly, not treated as bad samples
                code_error_types = (UnboundLocalError, NameError, SyntaxError, AttributeError)
                if isinstance(e, code_error_types):
                    raise

                if self.bad_sample_tracker.enable:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    tb_str = traceback.format_exc()
                    for idx, sample in enumerate(inputs):
                        self.bad_sample_tracker.record_bad_sample(
                            sample=sample,
                            error_type=error_type,
                            error_message=error_msg,
                            sample_index=idx,
                            tb_str=tb_str,
                        )
                    self.bad_sample_tracker.check_threshold()
                    raise RuntimeError(
                        f"All samples in batch failed during _generate_and_score_completions. "
                        f"Error: {error_msg}"
                    )
                else:
                    raise
        return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _compute_affective_gate(self, inputs, func_name):
        """
        Compute sample-level gate for affective rewards.

        Gate modes:
        - 'auto': Use metadata field 'emotion' or heuristic on problem text
        - 'always_on': All samples pass gate
        - 'off': No samples pass gate (affective rewards disabled)
        """
        batch_size = len(inputs)
        device = next(self.model.parameters()).device

        if self.affective_reward_gate_mode == "off":
            return torch.zeros(batch_size, device=device)
        elif self.affective_reward_gate_mode == "always_on":
            return torch.ones(batch_size, device=device)
        elif self.affective_reward_gate_mode == "auto":
            gate = torch.zeros(batch_size, device=device)
            for idx, example in enumerate(inputs):
                if "emotion" in example and example["emotion"]:
                    gate[idx] = 1.0
                elif "problem" in example:
                    problem_text = example["problem"].lower()
                    emotion_keywords = ["emotion", "feel", "happy", "sad", "angry", "fear", "surprise",
                                       "mood", "sentiment", "affective", "facial", "expression"]
                    if any(kw in problem_text for kw in emotion_keywords):
                        gate[idx] = 1.0
            return gate
        else:
            logger.warning(f"Unknown affective_reward_gate_mode: {self.affective_reward_gate_mode}, defaulting to 'auto'")
            return torch.ones(batch_size, device=device)

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        # [STAGE3_AUDIO_DIAG] Initialize diagnostic variables at method start (all ranks)
        stage3_audio_diag = os.environ.get("STAGE3_AUDIO_DIAG", "0") == "1" or os.environ.get("GRPO_STAGE3_AUDIO_DIAG", "0") == "1"
        max_diag_steps = int(os.environ.get("GRPO_STAGE3_AUDIO_DIAG_MAX_STEPS", "3"))
        stage3_diag_step_count = getattr(self, "_stage3_diag_step_count", 0)

        rank = 0
        try:
            import torch.distributed as torch_dist
            if torch_dist.is_available() and torch_dist.is_initialized():
                rank = torch_dist.get_rank()
        except:
            rank = 0
        device = self.accelerator.device

        # [DIST_CHECK] Static diagnostics for torch_dist availability
        if torch_dist.is_available() and torch_dist.is_initialized():
            rank = torch_dist.get_rank()
        else:
            rank = 0

        if rank == 0:
            print("[DIST_CHECK] torch_dist available:", torch_dist.is_available(), flush=True)
            print("[DIST_CHECK] torch_dist initialized:", torch_dist.is_initialized(), flush=True)

        # print(inputs)
        prompts = [x["prompt"] for x in inputs]
        num_prompts = len(prompts)

        # [TASK_AWARE] Identify task types for each sample
        task_types = []
        for sample in inputs:
            task_type = self.vlm_module.identify_task_type(sample)
            task_types.append(task_type)

        # Expand inputs before multimodal packing to support num_generations
        # Each original sample is replicated self.num_generations times
        # This ensures processor/packing generates consistent text + multimodal inputs for all generations
        if self.num_generations > 1:
            expanded_inputs = []
            expanded_task_types = []
            for sample, task_type in zip(inputs, task_types):
                for _ in range(self.num_generations):
                    expanded_inputs.append(sample)
                    expanded_task_types.append(task_type)
            inputs = expanded_inputs
            task_types = expanded_task_types
            prompts = [x["prompt"] for x in inputs]

        # [TASK_PROMPT] Apply task-aware prompt routing if enabled
        if self.task_aware_prompt_enabled:
            for i, (sample, task_type) in enumerate(zip(inputs, task_types)):
                if rank == 0 and i < 2:
                    print(f"[TASK_PROMPT] sample {i}: task_type={task_type}", flush=True)

        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)
        use_audio_in_video = False #inputs[0].get("use_audio_in_video", False)
        # print(prompts_text)
        images, videos, audios = [], [], []

        # Extract images/videos/audios from original (non-expanded) inputs only
        # When num_generations > 1, inputs are already expanded (each sample repeated num_generations times)
        # We need to extract from the ORIGINAL num_prompts samples only
        original_inputs = inputs[::self.num_generations] if self.num_generations > 1 else inputs

        for each in original_inputs:
            if each["images"] is not None:
                images.extend(each["images"])
            if each["audios"] is not None:
                audios.extend(each["audios"])
            if each["videos"] is not None:
                videos.extend(each["videos"])

        # Expand multimodal inputs to match prompts_text count
        # prompts_text has num_prompts * num_generations elements, so multimodal inputs must too
        if self.num_generations > 1:
            images = images * self.num_generations if images else None
            audios = audios * self.num_generations if audios else None
            videos = videos * self.num_generations if videos else None

        if images is not None and len(images) == 0: images = None
        if audios is not None and len(audios) == 0: audios = None
        if videos is not None and len(videos) == 0: videos = None
    

        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            audios,
            videos,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            use_audio_in_video=use_audio_in_video,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs["use_audio_in_video"] = use_audio_in_video

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # [GRPO_AUDIO_LENGTH_DIAG] Audio length diagnostics
        grpo_audio_length_diag = os.environ.get("GRPO_AUDIO_LENGTH_DIAG", "0") == "1"
        if grpo_audio_length_diag and rank == 0:
            try:
                print("\n" + "="*80, flush=True)
                print("[GRPO_AUDIO_LENGTH_DIAG] ===== Audio Length Diagnostics =====", flush=True)

                # Get current step
                current_step = getattr(self, 'state', None)
                if current_step and hasattr(current_step, 'global_step'):
                    print(f"[GRPO_AUDIO_LENGTH_DIAG] global_step = {current_step.global_step}", flush=True)

                # Batch info
                batch_size = prompt_ids.shape[0]
                print(f"[GRPO_AUDIO_LENGTH_DIAG] batch_size = {batch_size}", flush=True)
                print(f"[GRPO_AUDIO_LENGTH_DIAG] input_ids.shape = {prompt_ids.shape}", flush=True)
                print(f"[GRPO_AUDIO_LENGTH_DIAG] attention_mask.shape = {prompt_mask.shape}", flush=True)

                # Check for input_features and feature_attention_mask
                has_input_features = 'input_features' in prompt_inputs and prompt_inputs['input_features'] is not None
                has_feature_attention_mask = 'feature_attention_mask' in prompt_inputs and prompt_inputs['feature_attention_mask'] is not None

                print(f"[GRPO_AUDIO_LENGTH_DIAG] has_input_features = {has_input_features}", flush=True)
                print(f"[GRPO_AUDIO_LENGTH_DIAG] has_feature_attention_mask = {has_feature_attention_mask}", flush=True)

                if has_input_features:
                    input_features = prompt_inputs['input_features']
                    print(f"[GRPO_AUDIO_LENGTH_DIAG] input_features.shape = {input_features.shape}", flush=True)

                if has_feature_attention_mask:
                    feature_attention_mask = prompt_inputs['feature_attention_mask']
                    print(f"[GRPO_AUDIO_LENGTH_DIAG] feature_attention_mask.shape = {feature_attention_mask.shape}", flush=True)

                    # Per-sample diagnostics
                    for sample_idx in range(min(batch_size, len(inputs))):
                        sample = inputs[sample_idx]

                        # Get dataset source
                        dataset_source = sample.get('_dataset_source', 'unknown')

                        # Get media path
                        media_path = 'unknown'
                        if 'audios' in sample and sample['audios'] and len(sample['audios']) > 0:
                            media_path = sample['audios'][0] if isinstance(sample['audios'], list) else sample['audios']
                        elif 'videos' in sample and sample['videos'] and len(sample['videos']) > 0:
                            media_path = sample['videos'][0] if isinstance(sample['videos'], list) else sample['videos']

                        # Calculate feature_attention_mask sum for this sample
                        if sample_idx < feature_attention_mask.shape[0]:
                            feature_mask_sum = feature_attention_mask[sample_idx].sum().item()

                            print(f"\n[GRPO_AUDIO_LENGTH_DIAG] sample_idx = {sample_idx}", flush=True)
                            print(f"[GRPO_AUDIO_LENGTH_DIAG]   dataset_source = {dataset_source}", flush=True)
                            print(f"[GRPO_AUDIO_LENGTH_DIAG]   media_path = {media_path}", flush=True)
                            print(f"[GRPO_AUDIO_LENGTH_DIAG]   feature_attention_mask.sum() = {feature_mask_sum}", flush=True)

                            # Check for long audio
                            if feature_mask_sum > 32768 or (has_input_features and input_features.shape[1] > 32768):
                                print(f"[GRPO_AUDIO_LENGTH_DIAG][LONG_AUDIO_CANDIDATE] sample_idx = {sample_idx}", flush=True)
                                print(f"[GRPO_AUDIO_LENGTH_DIAG][LONG_AUDIO_CANDIDATE]   dataset_source = {dataset_source}", flush=True)
                                print(f"[GRPO_AUDIO_LENGTH_DIAG][LONG_AUDIO_CANDIDATE]   media_path = {media_path}", flush=True)
                                print(f"[GRPO_AUDIO_LENGTH_DIAG][LONG_AUDIO_CANDIDATE]   feature_mask_sum = {feature_mask_sum}", flush=True)

                                # Get problem text
                                problem_text = sample.get('problem', '')
                                if isinstance(problem_text, str):
                                    problem_preview = problem_text[:200] if len(problem_text) > 200 else problem_text
                                    print(f"[GRPO_AUDIO_LENGTH_DIAG][LONG_AUDIO_CANDIDATE]   problem_text_preview = {repr(problem_preview)}", flush=True)

                print("="*80 + "\n", flush=True)

            except Exception as e:
                print(f"[GRPO_AUDIO_LENGTH_DIAG] WARNING: diagnostic failed: {e}", flush=True)
                import traceback
                traceback.print_exc()

        # max_prompt_length is not supported yet
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_inputs["input_ids"] = prompt_ids
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        #     prompt_inputs["attention_mask"] = prompt_mask

        # [PHASE3_DEBUG] Log prompt before generation
        print(f"\n[PHASE3_GENERATION] Prompt text preview (first 300 chars): {repr(prompts_text[0][:300])}")
        print(f"[PHASE3_GENERATION] Prompt token length: {prompt_ids.shape[1]}")
        print(f"[PHASE3_GENERATION] Batch size: {prompt_ids.shape[0]}")
        print(f"[PHASE3_GENERATION] prompt_ids[0][:50]: {prompt_ids[0][:50]}")
        print(f"[PHASE3_GENERATION] prompt_ids[0][-50:]: {prompt_ids[0][-50:]}")

        # Generate completions
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            # [GRPO_GEN_DEBUG] Minimal diagnostic before generation
            grpo_gen_debug = os.environ.get("GRPO_GEN_DEBUG", "0") == "1"
            if grpo_gen_debug and self.accelerator.is_main_process:
                try:
                    rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0
                    if rank == 0:
                        print("\n" + "="*80)
                        print("[GRPO_GEN_DEBUG] PRE-GENERATION DIAGNOSTICS")
                        print("="*80)

                        # Get tokenizer safely
                        tokenizer = getattr(self.processing_class, "tokenizer", None)
                        if tokenizer is None:
                            tokenizer = self.processing_class

                        # Log generation config
                        print(f"[GRPO_GEN_DEBUG] generation_config: {self.generation_config}")
                        print(f"[GRPO_GEN_DEBUG] model.generation_config: {unwrapped_model.generation_config}")
                        print(f"[GRPO_GEN_DEBUG] tokenizer.eos_token_id: {getattr(tokenizer, 'eos_token_id', None)}")
                        print(f"[GRPO_GEN_DEBUG] tokenizer.pad_token_id: {getattr(tokenizer, 'pad_token_id', None)}")

                        # Log input shapes
                        print(f"[GRPO_GEN_DEBUG] prompt_ids.shape: {prompt_ids.shape}")
                        print(f"[GRPO_GEN_DEBUG] prompt_mask.shape: {prompt_mask.shape}")
                        print(f"[GRPO_GEN_DEBUG] prompt_ids[0][:20]: {prompt_ids[0][:20].tolist()}")
                        print(f"[GRPO_GEN_DEBUG] prompt_ids[0][-20:]: {prompt_ids[0][-20:].tolist()}")

                        # Log generate kwargs
                        gen_kwargs = {k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()}
                        print(f"[GRPO_GEN_DEBUG] generate() kwargs keys: {list(gen_kwargs.keys())}")
                        for k in ['input_ids', 'attention_mask', 'pixel_values', 'pixel_values_videos']:
                            if k in gen_kwargs:
                                v = gen_kwargs[k]
                                if isinstance(v, torch.Tensor):
                                    print(f"[GRPO_GEN_DEBUG]   {k}: shape={v.shape}, dtype={v.dtype}")
                                else:
                                    print(f"[GRPO_GEN_DEBUG]   {k}: {type(v)}")
                except Exception as e:
                    print(f"[GRPO_GEN_DEBUG] WARNING: pre-generation diagnostic failed: {e}")

            # [GEN_BOUNDARY] Print prompt boundary diagnostics before generation
            if torch_dist.is_available() and torch_dist.is_initialized():
                rank = torch_dist.get_rank()
            else:
                rank = 0
            if rank == 0:
                try:
                    logger.info("[GEN_BOUNDARY] ===== Generation Boundary Diagnostics =====")
                    logger.info(f"[GEN_BOUNDARY] input_ids.shape: {prompt_inputs['input_ids'].shape}")
                    logger.info(f"[GEN_BOUNDARY] input_ids[0][-30:]: {prompt_inputs['input_ids'][0][-30:].tolist()}")

                    # Decode prompt tail
                    tokenizer = getattr(self.processing_class, "tokenizer", None)
                    if tokenizer is None:
                        tokenizer = self.processing_class
                    if hasattr(self.processing_class, "batch_decode"):
                        try:
                            decoded_prompt = self.processing_class.batch_decode(
                                prompt_inputs['input_ids'][0:1], skip_special_tokens=False
                            )
                            prompt_tail = decoded_prompt[0][-200:] if len(decoded_prompt[0]) > 200 else decoded_prompt[0]
                            logger.info(f"[GEN_BOUNDARY] decoded_prompt_tail: {repr(prompt_tail)}")
                        except Exception as e:
                            logger.info(f"[GEN_BOUNDARY] Decode prompt failed: {e}")
                except Exception as e:
                    logger.info(f"[GEN_BOUNDARY] Pre-generation boundary diagnostic failed: {e}")

            # [DS3_GENERATION_PATH] DeepSpeed ZeRO-3 generation diagnostics and parameter gathering
            ds_zero_stage = "unknown"
            gather_for_gen = os.environ.get("GRPO_DS3_GATHER_FOR_GENERATION", "0")

            if self.is_world_process_zero():
                try:
                    rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0
                    if rank == 0:
                        # Get model type info
                        model_type = type(self.model).__name__
                        unwrapped_model_type = type(unwrapped_model).__name__

                        # Get accelerator distributed type
                        distributed_type = str(self.accelerator.distributed_type) if hasattr(self.accelerator, 'distributed_type') else "unknown"

                        # Get DeepSpeed zero stage
                        if hasattr(self.accelerator, 'state') and hasattr(self.accelerator.state, 'deepspeed_plugin'):
                            ds_plugin = self.accelerator.state.deepspeed_plugin
                            if ds_plugin and hasattr(ds_plugin, 'zero_stage'):
                                ds_zero_stage = str(ds_plugin.zero_stage)

                        # Check if using DeepSpeedEngine
                        using_ds_engine = hasattr(self.model, 'module') and hasattr(self.model.module, 'generate')
                        using_unwrapped_gen = hasattr(unwrapped_model, 'generate')

                        print(f"[DS3_GENERATION_PATH] model_type={model_type}", flush=True)
                        print(f"[DS3_GENERATION_PATH] unwrapped_model_type={unwrapped_model_type}", flush=True)
                        print(f"[DS3_GENERATION_PATH] accelerator.distributed_type={distributed_type}", flush=True)
                        print(f"[DS3_GENERATION_PATH] deepspeed_zero_stage={ds_zero_stage}", flush=True)
                        print(f"[DS3_GENERATION_PATH] GRPO_DS3_GATHER_FOR_GENERATION={gather_for_gen}", flush=True)
                        print(f"[DS3_GENERATION_PATH] using_DeepSpeedEngine.generate={using_ds_engine}", flush=True)
                        print(f"[DS3_GENERATION_PATH] using_unwrapped_model.generate={using_unwrapped_gen}", flush=True)
                except Exception as e:
                    print(f"[DS3_GENERATION_PATH] diagnostic failed: {e}", flush=True)

            # [GENERATION_BOUNDARY_DEBUG] Comprehensive boundary and multimodal alignment check
            gen_boundary_debug = os.environ.get("GRPO_GENERATION_BOUNDARY_DEBUG", "0") == "1"
            if gen_boundary_debug and self.is_world_process_zero():
                try:
                    rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0
                    if rank == 0:
                        print(f"\n[GENERATION_BOUNDARY_DEBUG] ===== Generation Boundary Check (Sample 0) =====", flush=True)

                        if 'input_ids' in prompt_inputs:
                            input_ids = prompt_inputs['input_ids']
                            attention_mask = prompt_inputs.get('attention_mask', None)
                            print(f"[GENERATION_BOUNDARY_DEBUG] input_ids.shape={input_ids.shape}", flush=True)
                            if attention_mask is not None:
                                valid_len = attention_mask[0].sum().item()
                                print(f"[GENERATION_BOUNDARY_DEBUG] attention_mask[0].sum()={valid_len}", flush=True)

                            tokenizer = getattr(self.processing_class, "tokenizer", None)
                            if tokenizer is None:
                                tokenizer = self.processing_class

                            assistant_token_id = None
                            if hasattr(tokenizer, 'convert_tokens_to_ids'):
                                try:
                                    assistant_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
                                except:
                                    pass

                            sample_ids = input_ids[0].tolist()
                            assistant_positions = []
                            if assistant_token_id is not None:
                                for idx, token_id in enumerate(sample_ids):
                                    if token_id == assistant_token_id:
                                        assistant_positions.append(idx)

                            if assistant_positions:
                                last_assistant_pos = assistant_positions[-1]
                                print(f"[GENERATION_BOUNDARY_DEBUG] last_assistant_header_pos={last_assistant_pos}", flush=True)
                            else:
                                print(f"[GENERATION_BOUNDARY_DEBUG] WARNING: no assistant header token found", flush=True)

                            if attention_mask is not None and hasattr(tokenizer, 'batch_decode'):
                                valid_len = int(attention_mask[0].sum().item())
                                tail_start = max(0, valid_len - 80)
                                tail_ids = input_ids[0, tail_start:valid_len]
                                try:
                                    decoded_tail = tokenizer.batch_decode(tail_ids.unsqueeze(0), skip_special_tokens=False)
                                    decoded_tail_str = decoded_tail[0] if decoded_tail else ""
                                    print(f"[GENERATION_BOUNDARY_DEBUG] last_80_valid_tokens: {repr(decoded_tail_str[:400])}", flush=True)
                                except Exception as e:
                                    print(f"[GENERATION_BOUNDARY_DEBUG] decode tail failed: {e}", flush=True)

                            video_token_id = None
                            if hasattr(tokenizer, 'convert_tokens_to_ids'):
                                try:
                                    video_token_id = tokenizer.convert_tokens_to_ids('<|VIDEO|>')
                                except:
                                    pass

                            if video_token_id is not None:
                                video_positions = [idx for idx, token_id in enumerate(sample_ids) if token_id == video_token_id]
                                if video_positions:
                                    video_start = video_positions[0]
                                    video_end = video_positions[-1]
                                    video_count = len(video_positions)
                                    print(f"[GENERATION_BOUNDARY_DEBUG] VIDEO_token_span=[{video_start}, {video_end}], count={video_count}", flush=True)

                            if 'video_grid_thw' in prompt_inputs and prompt_inputs['video_grid_thw'] is not None:
                                vg_thw = prompt_inputs['video_grid_thw']
                                if isinstance(vg_thw, torch.Tensor) and vg_thw.size(0) > 0:
                                    print(f"[GENERATION_BOUNDARY_DEBUG] video_grid_thw[0]={vg_thw[0].tolist()}", flush=True)
                                    total_patches = vg_thw[0].prod().item()
                                    print(f"[GENERATION_BOUNDARY_DEBUG] total_video_patches={total_patches}", flush=True)

                            if 'pixel_values_videos' in prompt_inputs and prompt_inputs['pixel_values_videos'] is not None:
                                pv_videos = prompt_inputs['pixel_values_videos']
                                if isinstance(pv_videos, torch.Tensor):
                                    print(f"[GENERATION_BOUNDARY_DEBUG] pixel_values_videos.shape={pv_videos.shape}", flush=True)

                            if assistant_positions:
                                last_assistant_pos = assistant_positions[-1]
                                if attention_mask is not None:
                                    valid_len = int(attention_mask[0].sum().item())
                                    distance_to_end = valid_len - last_assistant_pos - 1
                                    print(f"[GENERATION_BOUNDARY_DEBUG] distance_from_last_assistant_to_end={distance_to_end}", flush=True)

                                    if distance_to_end > 200:
                                        print(f"[GENERATION_BOUNDARY_DEBUG] WARNING: assistant header is {distance_to_end} tokens from end", flush=True)
                            else:
                                raise RuntimeError(
                                    "[GENERATION_BOUNDARY_DEBUG] CRITICAL: No assistant header token found in input_ids.\n"
                                    f"input_ids[0][:50]={input_ids[0][:50].tolist()}\n"
                                    f"input_ids[0][-50:]={input_ids[0][-50:].tolist()}"
                                )

                        print(f"[GENERATION_BOUNDARY_DEBUG] ===== End Boundary Check =====\n", flush=True)
                except Exception as e:
                    if isinstance(e, RuntimeError) and "CRITICAL" in str(e):
                        raise
                    print(f"[GENERATION_BOUNDARY_DEBUG] check failed: {e}", flush=True)

            # [PROMPT_BOUNDARY_GUARD] Verify prompt boundary before generation
            if self.is_world_process_zero():
                try:
                    rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0
                    if rank == 0:
                        tokenizer = getattr(self.processing_class, "tokenizer", None)
                        if tokenizer is None:
                            tokenizer = self.processing_class

                        if 'input_ids' in prompt_inputs and hasattr(tokenizer, 'batch_decode'):
                            input_ids = prompt_inputs['input_ids']
                            prompt_tail_ids = input_ids[0, max(0, input_ids.size(1)-120):]
                            try:
                                decoded_tail = tokenizer.batch_decode(prompt_tail_ids.unsqueeze(0), skip_special_tokens=False)
                                decoded_tail_str = decoded_tail[0] if decoded_tail else ""
                                print(f"[PROMPT_BOUNDARY_GUARD] prompt_tail_last120: {repr(decoded_tail_str[:300])}", flush=True)

                                if '<|im_start|>assistant' not in decoded_tail_str and '<|assistant|>' not in decoded_tail_str:
                                    raise RuntimeError(
                                        "[PROMPT_BOUNDARY_GUARD] ERROR: prompt_tail missing <|im_start|>assistant.\n"
                                        f"Decoded: {repr(decoded_tail_str)}"
                                    )
                                print(f"[PROMPT_BOUNDARY_GUARD] assistant_header_found=True", flush=True)
                            except RuntimeError:
                                raise
                            except Exception as e:
                                print(f"[PROMPT_BOUNDARY_GUARD] decode failed: {e}", flush=True)
                except Exception as e:
                    if isinstance(e, RuntimeError) and "PROMPT_BOUNDARY_GUARD" in str(e):
                        raise
                    print(f"[PROMPT_BOUNDARY_GUARD] check failed: {e}", flush=True)

            # [GRPO_GENERATE_PREFLIGHT] Pre-generation diagnostics for crash localization
            if self.is_world_process_zero():
                try:
                    rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0
                    if rank == 0:
                        print("[GRPO_GENERATE_PREFLIGHT] ===== Pre-Generation Diagnostics =====", flush=True)

                        if 'input_ids' in prompt_inputs:
                            print(f"[GRPO_GENERATE_PREFLIGHT] input_ids.shape = {prompt_inputs['input_ids'].shape}", flush=True)

                        if 'attention_mask' in prompt_inputs:
                            print(f"[GRPO_GENERATE_PREFLIGHT] attention_mask.shape = {prompt_inputs['attention_mask'].shape}", flush=True)

                        has_pixel_values_videos = 'pixel_values_videos' in prompt_inputs and prompt_inputs['pixel_values_videos'] is not None
                        print(f"[GRPO_GENERATE_PREFLIGHT] has_pixel_values_videos = {has_pixel_values_videos}", flush=True)

                        has_input_features = 'input_features' in prompt_inputs and prompt_inputs['input_features'] is not None
                        print(f"[GRPO_GENERATE_PREFLIGHT] has_input_features = {has_input_features}", flush=True)

                        if has_input_features:
                            input_features = prompt_inputs['input_features']
                            if isinstance(input_features, torch.Tensor):
                                print(f"[GRPO_GENERATE_PREFLIGHT] input_features.shape = {input_features.shape}", flush=True)
                                print(f"[GRPO_GENERATE_PREFLIGHT] input_features.device = {input_features.device}", flush=True)

                        if 'video_grid_thw' in prompt_inputs and prompt_inputs['video_grid_thw'] is not None:
                            video_grid_thw = prompt_inputs['video_grid_thw']
                            if isinstance(video_grid_thw, torch.Tensor):
                                print(f"[GRPO_GENERATE_PREFLIGHT] video_grid_thw.shape = {video_grid_thw.shape}", flush=True)

                        if 'input_ids' in prompt_inputs:
                            prompt_token_length = prompt_inputs['input_ids'].shape[1]
                            print(f"[GRPO_GENERATE_PREFLIGHT] prompt_token_length = {prompt_token_length}", flush=True)

                        # Determine generation backend
                        attn_impl = "unknown"
                        if hasattr(unwrapped_model, 'config') and hasattr(unwrapped_model.config, '_attn_implementation'):
                            attn_impl = unwrapped_model.config._attn_implementation
                        elif hasattr(unwrapped_model, 'config') and hasattr(unwrapped_model.config, 'attn_implementation'):
                            attn_impl = unwrapped_model.config.attn_implementation
                        print(f"[GRPO_GENERATE_PREFLIGHT] generation_backend = {attn_impl}", flush=True)

                        print("[GRPO_GENERATE_PREFLIGHT] ===== End Pre-Generation Diagnostics =====", flush=True)
                except Exception as e:
                    print(f"[GRPO_GENERATE_PREFLIGHT] diagnostic failed: {e}", flush=True)

            # [GEN_BOUNDARY_CHECK] Check assistant header position before generation
            if rank == 0:
                try:
                    tokenizer = getattr(self.processing_class, "tokenizer", None)
                    if tokenizer is None:
                        tokenizer = self.processing_class

                    # Find assistant header token
                    if hasattr(tokenizer, "convert_tokens_to_ids"):
                        assistant_header_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
                        if assistant_header_id is not None and assistant_header_id != tokenizer.unk_token_id:
                            # Check if assistant header exists in prompt
                            prompt_ids_sample = prompt_ids[0]
                            assistant_positions = (prompt_ids_sample == assistant_header_id).nonzero(as_tuple=True)[0]

                            if len(assistant_positions) > 0:
                                last_assistant_pos = assistant_positions[-1].item()
                                distance_to_end = prompt_ids_sample.shape[0] - last_assistant_pos - 1
                                logger.info(f"[GEN_BOUNDARY_CHECK] last_assistant_header_pos={last_assistant_pos}, distance_to_end={distance_to_end}")

                                if distance_to_end > 200:
                                    logger.warning(f"[GEN_BOUNDARY_CHECK] [GEN_BOUNDARY_SUSPECT] assistant header too far from end (distance={distance_to_end})")
                            else:
                                logger.warning(f"[GEN_BOUNDARY_CHECK] [GEN_BOUNDARY_SUSPECT] No assistant header found in prompt!")
                except Exception as e:
                    logger.info(f"[GEN_BOUNDARY_CHECK] assistant header check failed: {e}")

            # [GENERATION_MODE_CONTROL] Set model to eval mode and use no_grad for generation
            original_training_mode = self.model.training
            self.model.eval()

            original_gradient_checkpointing = None
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'gradient_checkpointing'):
                original_gradient_checkpointing = self.model.config.gradient_checkpointing
                self.model.config.gradient_checkpointing = False

            try:
                with torch.no_grad():
                    # [GRPO_SUPPRESS_MODAL_SPECIAL_TOKENS] Prepare suppress_tokens to prevent modal special tokens in completion
                    suppress_modal_tokens_env = os.environ.get("GRPO_SUPPRESS_MODAL_SPECIAL_TOKENS", "0")
                    suppress_modal_tokens_enabled = suppress_modal_tokens_env == "1"

                    suppress_token_ids = []
                    if suppress_modal_tokens_enabled:
                        if rank == 0:
                            print(f"\n[GRPO_SUPPRESS_MODAL_SPECIAL_TOKENS] enabled = True", flush=True)

                        # Get tokenizer safely
                        tokenizer = getattr(self.processing_class, "tokenizer", None)
                        if tokenizer is None:
                            tokenizer = self.processing_class

                        # List of modal special tokens to suppress
                        modal_special_tokens = [
                            '<|AUDIO|>',
                            '<|audio_bos|>',
                            '<|audio_eos|>',
                            '<|vision_start|>',
                            '<|vision_end|>',
                            '<|image_pad|>',
                            '<|video_pad|>',
                            '<|VIDEO|>',
                            '<|IMAGE|>',
                        ]

                        token_to_id_map = {}
                        if hasattr(tokenizer, 'convert_tokens_to_ids'):
                            for token_str in modal_special_tokens:
                                try:
                                    token_id = tokenizer.convert_tokens_to_ids(token_str)
                                    # Only add if valid (not None, not unk_token_id)
                                    if token_id is not None:
                                        unk_id = getattr(tokenizer, 'unk_token_id', None)
                                        if unk_id is None or token_id != unk_id:
                                            suppress_token_ids.append(token_id)
                                            token_to_id_map[token_str] = token_id
                                except Exception as e:
                                    if rank == 0:
                                        print(f"[GRPO_SUPPRESS_MODAL_SPECIAL_TOKENS] WARNING: failed to get id for {token_str}: {e}", flush=True)

                        if rank == 0:
                            print(f"[GRPO_SUPPRESS_MODAL_SPECIAL_TOKENS] token_to_id = {token_to_id_map}", flush=True)
                            print(f"[GRPO_SUPPRESS_MODAL_SPECIAL_TOKENS] suppress_ids = {suppress_token_ids}", flush=True)
                    else:
                        if rank == 0:
                            print(f"[GRPO_SUPPRESS_MODAL_SPECIAL_TOKENS] enabled = False", flush=True)

                    # Check if generation micro-batch is enabled
                    gen_micro_batch_size_env = os.environ.get("GRPO_GENERATION_MICRO_BATCH_SIZE", None)
                    gen_micro_batch_enabled = gen_micro_batch_size_env is not None and gen_micro_batch_size_env != "0"

                    if gen_micro_batch_enabled:
                        # Generation micro-batch path
                        gen_micro_batch_size = int(gen_micro_batch_size_env)
                        original_batch_size = prompt_ids.shape[0]

                        if rank == 0:
                            print(f"\n[GRPO_GENERATION_MICROBATCH] === Generation micro-batch enabled ===", flush=True)
                            print(f"[GRPO_GENERATION_MICROBATCH] enabled = True", flush=True)
                            print(f"[GRPO_GENERATION_MICROBATCH] total_generation_batch_size = {original_batch_size}", flush=True)
                            print(f"[GRPO_GENERATION_MICROBATCH] micro_batch_size = {gen_micro_batch_size}", flush=True)

                        # Calculate number of chunks
                        num_chunks = (original_batch_size + gen_micro_batch_size - 1) // gen_micro_batch_size

                        # Collect generated results from all chunks
                        all_generated_results = []

                        for chunk_idx in range(num_chunks):
                            start = chunk_idx * gen_micro_batch_size
                            end = min(start + gen_micro_batch_size, original_batch_size)

                            if rank == 0:
                                print(f"[GRPO_GENERATION_MICROBATCH] chunk {chunk_idx} start/end = {start}/{end}", flush=True)

                            # Slice prompt_inputs for this chunk
                            chunk_prompt_inputs = {}
                            for key, value in prompt_inputs.items():
                                if key in self.vlm_module.get_non_generate_params():
                                    continue
                                if isinstance(value, torch.Tensor):
                                    chunk_prompt_inputs[key] = value[start:end]
                                else:
                                    chunk_prompt_inputs[key] = value

                            # Add suppress_tokens if enabled
                            if suppress_modal_tokens_enabled and suppress_token_ids:
                                # Merge with existing suppress_tokens if any
                                existing_suppress = chunk_prompt_inputs.get('suppress_tokens', [])
                                if isinstance(existing_suppress, list):
                                    merged_suppress = list(set(existing_suppress + suppress_token_ids))
                                else:
                                    merged_suppress = suppress_token_ids
                                chunk_prompt_inputs['suppress_tokens'] = merged_suppress

                            # Generate for this chunk
                            chunk_generated = unwrapped_model.generate(
                                **chunk_prompt_inputs,
                                generation_config=self.generation_config
                            )

                            all_generated_results.append(chunk_generated)

                            if rank == 0:
                                print(f"[GRPO_GENERATION_MICROBATCH] chunk {chunk_idx} generated shape = {chunk_generated.shape}", flush=True)

                        # Concatenate all chunks
                        generate_returned_result = torch.cat(all_generated_results, dim=0)

                        if rank == 0:
                            print(f"[GRPO_GENERATION_MICROBATCH] final concatenated shape = {generate_returned_result.shape}", flush=True)
                            print(f"[GRPO_GENERATION_MICROBATCH] === End generation micro-batch ===\n", flush=True)

                    else:
                        # Standard generation (default path)
                        gen_kwargs = {k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()}

                        # Add suppress_tokens if enabled
                        if suppress_modal_tokens_enabled and suppress_token_ids:
                            # Merge with existing suppress_tokens if any
                            existing_suppress = gen_kwargs.get('suppress_tokens', [])
                            if isinstance(existing_suppress, list):
                                merged_suppress = list(set(existing_suppress + suppress_token_ids))
                            else:
                                merged_suppress = suppress_token_ids
                            gen_kwargs['suppress_tokens'] = merged_suppress

                        generate_returned_result = unwrapped_model.generate(
                            **gen_kwargs,
                            generation_config=self.generation_config
                        )
            finally:
                if original_training_mode:
                    self.model.train()
                if original_gradient_checkpointing is not None:
                    self.model.config.gradient_checkpointing = original_gradient_checkpointing

            # [GEN_BOUNDARY] Print first generated tokens after generation
            if rank == 0:
                try:
                    logger.info("[GEN_BOUNDARY] ===== First Generated Tokens =====")
                    prompt_length = prompt_inputs['input_ids'].size(1)
                    if generate_returned_result.shape[1] > prompt_length:
                        first_gen_token_id = generate_returned_result[0, prompt_length].item()
                        logger.info(f"[GEN_BOUNDARY] first_generated_token_id: {first_gen_token_id}")

                        tokenizer = getattr(self.processing_class, "tokenizer", None)
                        if tokenizer is None:
                            tokenizer = self.processing_class
                        if hasattr(tokenizer, "decode"):
                            try:
                                first_token_str = tokenizer.decode([first_gen_token_id])
                                logger.info(f"[GEN_BOUNDARY] first_generated_token: {repr(first_token_str)}")
                            except Exception as e:
                                logger.info(f"[GEN_BOUNDARY] Decode first token failed: {e}")

                        if generate_returned_result.shape[1] > prompt_length + 1:
                            second_gen_token_id = generate_returned_result[0, prompt_length + 1].item()
                            logger.info(f"[GEN_BOUNDARY] second_generated_token_id: {second_gen_token_id}")
                            if hasattr(tokenizer, "decode"):
                                try:
                                    second_token_str = tokenizer.decode([second_gen_token_id])
                                    logger.info(f"[GEN_BOUNDARY] second_generated_token: {repr(second_token_str)}")
                                except Exception as e:
                                    logger.info(f"[GEN_BOUNDARY] Decode second token failed: {e}")
                    logger.info("[GEN_BOUNDARY] ===== End Generation Boundary Diagnostics =====")
                except Exception as e:
                    logger.info(f"[GEN_BOUNDARY] Post-generation boundary diagnostic failed: {e}")

            # [GRPO_GEN_DEBUG] Minimal diagnostic after generation
            if grpo_gen_debug and self.accelerator.is_main_process:
                try:
                    rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0
                    if rank == 0:
                        print("\n" + "="*80)
                        print("[GRPO_GEN_DEBUG] POST-GENERATION DIAGNOSTICS")
                        print("="*80)
                        print(f"[GRPO_GEN_DEBUG] generate_returned_result.shape: {generate_returned_result.shape}")
                        print(f"[GRPO_GEN_DEBUG] generate_returned_result[0][:50]: {generate_returned_result[0][:50].tolist()}")

                        # Get tokenizer safely
                        tokenizer = getattr(self.processing_class, "tokenizer", None)
                        if tokenizer is None:
                            tokenizer = self.processing_class

                        # Decode first sample
                        if hasattr(self.processing_class, "batch_decode"):
                            try:
                                decoded = self.processing_class.batch_decode(
                                    generate_returned_result[0:1], skip_special_tokens=False
                                )
                                print(f"[GRPO_GEN_DEBUG] Decoded (first 300 chars): {repr(decoded[0][:300])}")
                            except Exception as e:
                                print(f"[GRPO_GEN_DEBUG] Decode failed: {e}")
                except Exception as e:
                    print(f"[GRPO_GEN_DEBUG] WARNING: post-generation diagnostic failed: {e}")
            prompt_length = prompt_ids.size(1)
            if not self.vlm_module.is_embeds_input():
                prompt_completion_ids = generate_returned_result
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
                print(f"\n[PHASE3_GENERATION] prompt_completion_ids[0][:50]: {prompt_completion_ids[0][:50]}")
                print(f"[PHASE3_GENERATION] completion_ids[0][:50]: {completion_ids[0][:50]}")
            else:
                # In this case, the input of the LLM backbone is the embedding of the combination of the image and text prompt
                # So the returned result of the `generate` method only contains the completion ids
                completion_ids = generate_returned_result

                # Align prompt-side tensors to match completion_ids batch dimension
                # Target batch size is determined by completion_ids (from generate with num_return_sequences)
                target_batch_size = completion_ids.size(0)

                # Expand prompt_ids if needed
                if prompt_ids.size(0) != target_batch_size:
                    current_batch_size = prompt_ids.size(0)
                    assert target_batch_size % current_batch_size == 0, \
                        f"target_batch_size ({target_batch_size}) must be divisible by current prompt_ids batch ({current_batch_size})"
                    repeat_factor = target_batch_size // current_batch_size
                    prompt_ids = prompt_ids.repeat_interleave(repeat_factor, dim=0)

                # Expand prompt_mask if needed (independently check, don't assume it matches prompt_ids)
                if prompt_mask.size(0) != target_batch_size:
                    current_batch_size = prompt_mask.size(0)
                    assert target_batch_size % current_batch_size == 0, \
                        f"target_batch_size ({target_batch_size}) must be divisible by current prompt_mask batch ({current_batch_size})"
                    repeat_factor = target_batch_size // current_batch_size
                    prompt_mask = prompt_mask.repeat_interleave(repeat_factor, dim=0)

                # Expand multimodal inputs in prompt_inputs to maintain consistency
                multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
                for key in multimodal_keywords:
                    if key in prompt_inputs and prompt_inputs[key] is not None:
                        if isinstance(prompt_inputs[key], torch.Tensor):
                            # Skip packed tensors like pixel_values_videos (handled separately)
                            if key not in {'pixel_values_videos', 'pixel_values', 'pixel_values_images'}:
                                if prompt_inputs[key].size(0) != target_batch_size:
                                    current_batch_size = prompt_inputs[key].size(0)
                                    assert target_batch_size % current_batch_size == 0, \
                                        f"target_batch_size ({target_batch_size}) must be divisible by {key} batch ({current_batch_size})"
                                    repeat_factor = target_batch_size // current_batch_size
                                    prompt_inputs[key] = prompt_inputs[key].repeat_interleave(repeat_factor, dim=0)

                # Final consistency check before concatenation
                assert prompt_ids.size(0) == target_batch_size, \
                    f"prompt_ids batch ({prompt_ids.size(0)}) != target ({target_batch_size})"
                assert prompt_mask.size(0) == target_batch_size, \
                    f"prompt_mask batch ({prompt_mask.size(0)}) != target ({target_batch_size})"

                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # [GRPO_MODAL_TOKEN_GUARD] Check if completion contains suppressed modal special tokens
        if suppress_modal_tokens_enabled and suppress_token_ids:
            if rank == 0:
                try:
                    found_modal_tokens = []
                    for token_id in suppress_token_ids:
                        if (completion_ids == token_id).any():
                            found_modal_tokens.append(token_id)

                    if found_modal_tokens:
                        print(f"\n[GRPO_MODAL_TOKEN_GUARD] WARNING: found_modal_special_tokens_in_completion = {found_modal_tokens}", flush=True)

                        # Get token strings for logging
                        tokenizer = getattr(self.processing_class, "tokenizer", None)
                        if tokenizer is None:
                            tokenizer = self.processing_class

                        if hasattr(tokenizer, 'convert_ids_to_tokens'):
                            try:
                                token_strs = tokenizer.convert_ids_to_tokens(found_modal_tokens)
                                print(f"[GRPO_MODAL_TOKEN_GUARD] found_modal_tokens_str = {token_strs}", flush=True)
                            except:
                                pass
                    else:
                        print(f"[GRPO_MODAL_TOKEN_GUARD] No modal special tokens found in completion (good)", flush=True)
                except Exception as e:
                    print(f"[GRPO_MODAL_TOKEN_GUARD] check failed: {e}", flush=True)

        # [GRPO_GEN_TRACE] Diagnostic trace for completion generation
        grpo_trace_enabled = os.environ.get("GRPO_GEN_TRACE", "0") == "1"
        if grpo_trace_enabled and self.accelerator.is_main_process:
            try:
                rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0
                if rank == 0:
                    print("\n" + "="*80)
                    print("[GRPO_GEN_TRACE] DIAGNOSTIC TRACE - COMPLETION GENERATION")
                    print("="*80)

                    # Safe tokenizer access
                    tokenizer = getattr(self.processing_class, "tokenizer", None)
                    if tokenizer is None:
                        tokenizer = self.processing_class

                    for sample_idx in range(min(2, completion_ids.size(0))):
                        print(f"\n[GRPO_GEN_TRACE] ===== Sample {sample_idx} =====")

                        # 1. tokenizer.special_tokens_map
                        special_tokens_map = getattr(tokenizer, "special_tokens_map", None)
                        print(f"[GRPO_GEN_TRACE] 1. special_tokens_map: {special_tokens_map}")

                        # 2. tokenizer.eos_token_id and tokenizer.pad_token_id
                        eos_token_id = getattr(tokenizer, "eos_token_id", None)
                        pad_token_id = getattr(tokenizer, "pad_token_id", None)
                        print(f"[GRPO_GEN_TRACE] 2. eos_token_id: {eos_token_id}, pad_token_id: {pad_token_id}")

                        # 3. input_ids.shape and generated_ids.shape
                        print(f"[GRPO_GEN_TRACE] 3. input_ids.shape: {prompt_ids.shape}, generated_ids (completion_ids).shape: {completion_ids.shape}")

                        # 4. Real prompt_len from attention_mask.sum()
                        real_prompt_len = prompt_mask[sample_idx].sum().item()
                        print(f"[GRPO_GEN_TRACE] 4. Real prompt_len (attention_mask.sum()): {real_prompt_len}")

                        # 5. Current code's prompt_length used for slicing
                        print(f"[GRPO_GEN_TRACE] 5. Current code prompt_length (fixed): {prompt_length}")

                        # 6. First 80 token ids of completion
                        comp_ids_80 = completion_ids[sample_idx, :80]
                        print(f"[GRPO_GEN_TRACE] 6. completion_ids[:80]: {comp_ids_80.tolist()}")

                        # 7. First 80 tokens after convert_ids_to_tokens
                        if hasattr(tokenizer, "convert_ids_to_tokens"):
                            try:
                                tokens_80 = tokenizer.convert_ids_to_tokens(comp_ids_80.tolist())
                                print(f"[GRPO_GEN_TRACE] 7. tokens[:80]: {tokens_80}")
                            except Exception as e:
                                print(f"[GRPO_GEN_TRACE] 7. tokens[:80]: (failed to convert: {type(e).__name__})")
                        else:
                            print(f"[GRPO_GEN_TRACE] 7. tokens[:80]: (convert_ids_to_tokens not available)")

                        # 8. Decoded completion with skip_special_tokens=False
                        comp_decoded_with_special = None
                        if hasattr(self.processing_class, "batch_decode"):
                            try:
                                decoded_batch = self.processing_class.batch_decode(
                                    completion_ids[sample_idx:sample_idx+1], skip_special_tokens=False
                                )
                                comp_decoded_with_special = decoded_batch[0] if decoded_batch else None
                            except Exception:
                                pass
                        if comp_decoded_with_special is None and hasattr(tokenizer, "batch_decode"):
                            try:
                                decoded_batch = tokenizer.batch_decode(
                                    completion_ids[sample_idx:sample_idx+1], skip_special_tokens=False
                                )
                                comp_decoded_with_special = decoded_batch[0] if decoded_batch else None
                            except Exception:
                                pass
                        if comp_decoded_with_special is not None:
                            print(f"[GRPO_GEN_TRACE] 8. Decoded (skip_special_tokens=False, first 200 chars): {repr(comp_decoded_with_special[:200])}")
                        else:
                            print(f"[GRPO_GEN_TRACE] 8. Decoded (skip_special_tokens=False): (decode not available)")

                        # 9. Decoded completion with skip_special_tokens=True
                        comp_decoded_no_special = None
                        if hasattr(self.processing_class, "batch_decode"):
                            try:
                                decoded_batch = self.processing_class.batch_decode(
                                    completion_ids[sample_idx:sample_idx+1], skip_special_tokens=True
                                )
                                comp_decoded_no_special = decoded_batch[0] if decoded_batch else None
                            except Exception:
                                pass
                        if comp_decoded_no_special is None and hasattr(tokenizer, "batch_decode"):
                            try:
                                decoded_batch = tokenizer.batch_decode(
                                    completion_ids[sample_idx:sample_idx+1], skip_special_tokens=True
                                )
                                comp_decoded_no_special = decoded_batch[0] if decoded_batch else None
                            except Exception:
                                pass
                        if comp_decoded_no_special is not None:
                            print(f"[GRPO_GEN_TRACE] 9. Decoded (skip_special_tokens=True, first 200 chars): {repr(comp_decoded_no_special[:200])}")
                        else:
                            print(f"[GRPO_GEN_TRACE] 9. Decoded (skip_special_tokens=True): (decode not available)")

                        # 10. Count of special tags in completion
                        if comp_decoded_no_special is not None:
                            comp_text = comp_decoded_no_special
                            context_open = comp_text.count("<context>")
                            context_close = comp_text.count("</context>")
                            think_open = comp_text.count("<think>")
                            think_close = comp_text.count("</think>")
                            answer_open = comp_text.count("<answer>")
                            answer_close = comp_text.count("</answer>")
                            print(f"[GRPO_GEN_TRACE] 10. Tag counts: <context>={context_open}, </context>={context_close}, <think>={think_open}, </think>={think_close}, <answer>={answer_open}, </answer>={answer_close}")
                        else:
                            print(f"[GRPO_GEN_TRACE] 10. Tag counts: (skipped - no decoded text)")

                    print("\n" + "="*80)
                    print("[GRPO_GEN_TRACE] END DIAGNOSTIC TRACE")
                    print("="*80 + "\n")
            except Exception as e:
                print(f"[GRPO_GEN_TRACE] WARNING: diagnostic trace failed: {type(e).__name__}: {str(e)}")

        # [STAGE3_AUDIO_DIAG] Diagnostic after generation
        if stage3_audio_diag and rank == 0 and stage3_diag_step_count <= max_diag_steps:
            print(f"\n[STAGE3_AUDIO_DIAG] === After generation ===", flush=True)
            print(f"[STAGE3_AUDIO_DIAG] completion_ids shape: {completion_ids.shape}", flush=True)

            # Decode first 3 completions for inspection
            for sample_idx in range(min(3, completion_ids.size(0))):
                try:
                    decoded = self.processing_class.decode(completion_ids[sample_idx], skip_special_tokens=False)
                    decoded_preview = decoded[:300] if len(decoded) > 300 else decoded
                    print(f"[STAGE3_AUDIO_DIAG] completion[{sample_idx}] (first 300 chars): {repr(decoded_preview)}", flush=True)

                    # Check for degenerate patterns
                    if "contextsystemsystem" in decoded.lower():
                        print(f"[STAGE3_AUDIO_DIAG] WARNING: completion[{sample_idx}] contains 'contextsystemsystem' pattern", flush=True)

                    # Count repeated tokens
                    token_ids = completion_ids[sample_idx].tolist()
                    if len(token_ids) > 1:
                        max_repeat = 1
                        current_repeat = 1
                        for i in range(1, len(token_ids)):
                            if token_ids[i] == token_ids[i-1]:
                                current_repeat += 1
                                max_repeat = max(max_repeat, current_repeat)
                            else:
                                current_repeat = 1
                        if max_repeat > 10:
                            print(f"[STAGE3_AUDIO_DIAG] WARNING: completion[{sample_idx}] has {max_repeat} repeated tokens", flush=True)

                    # Check for empty or pure punctuation
                    if len(decoded.strip()) == 0:
                        print(f"[STAGE3_AUDIO_DIAG] WARNING: completion[{sample_idx}] is empty", flush=True)
                    elif all(c in '.,!?;:\'"()[]{}' for c in decoded.strip()):
                        print(f"[STAGE3_AUDIO_DIAG] WARNING: completion[{sample_idx}] is pure punctuation", flush=True)

                    # Check for early EOS
                    eos_token_id = getattr(self.processing_class, 'eos_token_id', None)
                    if eos_token_id is not None and eos_token_id in token_ids[:-1]:
                        eos_pos = token_ids[:-1].index(eos_token_id)
                        if eos_pos < len(token_ids) * 0.3:
                            print(f"[STAGE3_AUDIO_DIAG] WARNING: completion[{sample_idx}] has early EOS at position {eos_pos}/{len(token_ids)}", flush=True)

                except Exception as e:
                    print(f"[STAGE3_AUDIO_DIAG] ERROR decoding completion[{sample_idx}]: {e}", flush=True)

            print(f"[STAGE3_AUDIO_DIAG] === End post-generation diagnostic ===\n", flush=True)

        # Fix completion_ids slicing: use real prompt_len per sample instead of fixed prompt_length
        # This handles padding correctly when padding_side="left"
        use_per_sample_prompt_len = os.environ.get("GRPO_FIX_COMPLETION_SLICE", "0") == "1"
        orig_completion_ids = completion_ids
        if use_per_sample_prompt_len and not self.vlm_module.is_embeds_input():
            real_prompt_lens = prompt_mask.sum(dim=1).long()
            max_real_prompt_len = real_prompt_lens.max().item()

            if grpo_trace_enabled and self.accelerator.is_main_process:
                print(f"[GRPO_FIX_COMPLETION_SLICE] Checking per-sample prompt_len for completion slicing")
                print(f"[GRPO_FIX_COMPLETION_SLICE] real_prompt_lens: {real_prompt_lens[:min(2, len(real_prompt_lens))]}")
                print(f"[GRPO_FIX_COMPLETION_SLICE] max_real_prompt_len: {max_real_prompt_len}")
                print(f"[GRPO_FIX_COMPLETION_SLICE] prompt_completion_ids.size(1): {prompt_completion_ids.size(1)}")
                print(f"[GRPO_FIX_COMPLETION_SLICE] completion_ids.size(1): {completion_ids.size(1)}")

            # Check if completion_ids is already completion-only (not prompt+completion)
            # If prompt_completion_ids.size(1) > max_real_prompt_len, we can safely slice from prompt_completion_ids
            # Otherwise, completion_ids is already the completion part, skip per-sample slicing
            if prompt_completion_ids.size(1) > max_real_prompt_len:
                if grpo_trace_enabled and self.accelerator.is_main_process:
                    print(f"[GRPO_FIX_COMPLETION_SLICE] Slicing from prompt_completion_ids (size {prompt_completion_ids.size(1)}) using real_prompt_lens")

                new_completion_ids = []
                for i in range(prompt_completion_ids.size(0)):
                    real_len = real_prompt_lens[i].item()
                    new_completion_ids.append(prompt_completion_ids[i, real_len:])

                if len(new_completion_ids) > 0 and any(c.size(0) > 0 for c in new_completion_ids):
                    max_comp_len = max(c.size(0) for c in new_completion_ids)
                    completion_ids = torch.stack([
                        torch.cat([c, torch.full((max_comp_len - c.size(0),), self.processing_class.pad_token_id,
                                                 dtype=c.dtype, device=c.device)])
                        for c in new_completion_ids
                    ], dim=0)
                    if grpo_trace_enabled and self.accelerator.is_main_process:
                        print(f"[GRPO_FIX_COMPLETION_SLICE] After slicing: completion_ids.shape = {completion_ids.shape}")
                else:
                    if grpo_trace_enabled and self.accelerator.is_main_process:
                        print(f"[GRPO_FIX_COMPLETION_SLICE] WARNING: slicing resulted in empty completions, reverting to original")
                    completion_ids = orig_completion_ids
            else:
                if grpo_trace_enabled and self.accelerator.is_main_process:
                    print(f"[GRPO_FIX_COMPLETION_SLICE] skip per-sample slicing because completion_ids already appears to be completion-only")
                    print(f"[GRPO_FIX_COMPLETION_SLICE] (prompt_completion_ids.size(1)={prompt_completion_ids.size(1)} <= max_real_prompt_len={max_real_prompt_len})")
                # CRITICAL FIX: Do NOT revert to orig_completion_ids here
                # completion_ids is already correctly sliced at line 2660, keep it as is
                pass

        # CRITICAL CHECK: Verify generation returned correct number of sequences
        actual_generated = completion_ids.size(0)
        expected_generated = num_prompts * self.num_generations
        if actual_generated != expected_generated:
            raise RuntimeError(
                f"Generation output mismatch immediately after generate():\n"
                f"  num_prompts: {num_prompts}\n"
                f"  self.num_generations: {self.num_generations}\n"
                f"  expected generated sequences: {expected_generated}\n"
                f"  actual completion_ids.size(0): {actual_generated}\n"
                f"  This indicates generate() did not respect num_return_sequences parameter."
            )

        # Note: inputs were already expanded before multimodal packing
        # So prompt_ids, prompt_mask, and multimodal_inputs are already aligned with completion_ids batch dimension
        # No late-stage expansion needed here

        # Safety check: ensure completion_ids has non-zero sequence length before EOS computation
        if completion_ids.size(1) == 0:
            print(f"[GRPO_GEN_TRACE] WARNING: completion_ids has zero sequence length after slicing, reverting to original")
            completion_ids = orig_completion_ids
            if completion_ids.size(1) == 0:
                raise RuntimeError(
                    f"completion_ids has zero sequence length even after revert:\n"
                    f"  orig_completion_ids.shape: {orig_completion_ids.shape}\n"
                    f"  This indicates a critical issue with completion generation or slicing logic."
                )

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # [STAGE3_AUDIO_DIAG][LEN_TRACE] After completion_mask construction
        if rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"[STAGE3_AUDIO_DIAG][LEN_TRACE][ASSIGN] completion_ids.shape={completion_ids.shape}, completion_mask.shape={completion_mask.shape}", flush=True)

        # [DEVICE_CONSISTENCY] Ensure completion_mask is on same device as completion_ids
        if completion_mask.device != completion_ids.device:
            completion_mask = completion_mask.to(completion_ids.device)
        if self.is_world_process_zero():
            print(f"[DEVICE_CONSISTENCY] completion_mask.device={completion_mask.device}, completion_ids.device={completion_ids.device}", flush=True)

        # CRITICAL: Align prompt-side tensors to completion_ids batch dimension before concatenation
        # This is the final alignment point before torch.cat operations
        target_batch_size = completion_ids.size(0)

        # Expand prompt_ids if needed
        if prompt_ids.size(0) != target_batch_size:
            current_batch_size = prompt_ids.size(0)
            assert target_batch_size % current_batch_size == 0, \
                f"target_batch_size ({target_batch_size}) must be divisible by prompt_ids batch ({current_batch_size})"
            repeat_factor = target_batch_size // current_batch_size
            prompt_ids = prompt_ids.repeat_interleave(repeat_factor, dim=0)

        # Expand prompt_mask if needed
        if prompt_mask.size(0) != target_batch_size:
            current_batch_size = prompt_mask.size(0)
            assert target_batch_size % current_batch_size == 0, \
                f"target_batch_size ({target_batch_size}) must be divisible by prompt_mask batch ({current_batch_size})"
            repeat_factor = target_batch_size // current_batch_size
            prompt_mask = prompt_mask.repeat_interleave(repeat_factor, dim=0)

        # Rebuild prompt_completion_ids after potential expansion of prompt_ids
        # [STAGE3_AUDIO_DIAG][LEN_TRACE] Before prompt_completion_ids concatenation
        if rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"[STAGE3_AUDIO_DIAG][LEN_TRACE][ASSIGN] Before prompt_completion_ids concat: prompt_ids.shape={prompt_ids.shape}, completion_ids.shape={completion_ids.shape}", flush=True)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # [STAGE3_AUDIO_DIAG][LEN_TRACE] After prompt_completion_ids concatenation
        if rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"[STAGE3_AUDIO_DIAG][LEN_TRACE][ASSIGN] After prompt_completion_ids concat: prompt_completion_ids.shape={prompt_completion_ids.shape}", flush=True)

        # [STAGE3_AUDIO_DIAG][MISMATCH] Post-generation mismatch diagnostics
        # [STAGE3_AUDIO_DIAG][LEN_TRACE] Before _get_per_token_logps call - CRITICAL FIX
        if rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"\n[STAGE3_AUDIO_DIAG][LEN_TRACE] === Before _get_per_token_logps ===", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][LEN_TRACE] prompt_completion_ids.shape={prompt_completion_ids.shape}", flush=True)
            # attention_mask not yet defined at this point, will be checked after construction

        if stage3_audio_diag and rank == 0 and stage3_diag_step_count <= max_diag_steps and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"\n[STAGE3_AUDIO_DIAG][MISMATCH] === Post-generation mismatch check ===", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][MISMATCH] prompt_ids.shape={prompt_ids.shape}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][MISMATCH] completion_ids.shape={completion_ids.shape}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][MISMATCH] prompt_completion_ids.shape={prompt_completion_ids.shape}", flush=True)

            # Check multimodal input batch sizes
            if 'input_features' in prompt_inputs and prompt_inputs['input_features'] is not None:
                input_feat = prompt_inputs['input_features']
                print(f"[STAGE3_AUDIO_DIAG][MISMATCH] input_features.size(0)={input_feat.size(0)}, prompt_completion_ids.size(0)={prompt_completion_ids.size(0)}", flush=True)
                if input_feat.size(0) != prompt_completion_ids.size(0):
                    print(f"[STAGE3_AUDIO_DIAG][MISMATCH] [MISMATCH] audio features not expanded with completions", flush=True)

            if 'feature_attention_mask' in prompt_inputs and prompt_inputs['feature_attention_mask'] is not None:
                feat_attn_mask = prompt_inputs['feature_attention_mask']
                print(f"[STAGE3_AUDIO_DIAG][MISMATCH] feature_attention_mask.size(0)={feat_attn_mask.size(0)}, prompt_completion_ids.size(0)={prompt_completion_ids.size(0)}", flush=True)
                if feat_attn_mask.size(0) != prompt_completion_ids.size(0):
                    print(f"[STAGE3_AUDIO_DIAG][MISMATCH] [MISMATCH] feature_attention_mask not expanded with completions", flush=True)

            print(f"[STAGE3_AUDIO_DIAG][MISMATCH] === End post-generation mismatch check ===\n", flush=True)

        # Expand multimodal inputs in prompt_inputs to maintain consistency
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        for key in multimodal_keywords:
            if key in prompt_inputs and prompt_inputs[key] is not None:
                if isinstance(prompt_inputs[key], torch.Tensor):
                    # Skip packed tensors like pixel_values_videos (handled separately)
                    if key not in {'pixel_values_videos', 'pixel_values', 'pixel_values_images'}:
                        if prompt_inputs[key].size(0) != target_batch_size:
                            current_batch_size = prompt_inputs[key].size(0)
                            assert target_batch_size % current_batch_size == 0, \
                                f"target_batch_size ({target_batch_size}) must be divisible by {key} batch ({current_batch_size})"
                            repeat_factor = target_batch_size // current_batch_size
                            prompt_inputs[key] = prompt_inputs[key].repeat_interleave(repeat_factor, dim=0)

        # Final assertion before concatenation
        assert prompt_ids.size(0) == target_batch_size, \
            f"prompt_ids batch ({prompt_ids.size(0)}) != target ({target_batch_size})"
        assert prompt_mask.size(0) == target_batch_size, \
            f"prompt_mask batch ({prompt_mask.size(0)}) != target ({target_batch_size})"
        assert completion_mask.size(0) == target_batch_size, \
            f"completion_mask batch ({completion_mask.size(0)}) != target ({target_batch_size})"

        # [STAGE3_AUDIO_DIAG][LEN_TRACE] Before attention_mask concatenation
        if rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"[STAGE3_AUDIO_DIAG][LEN_TRACE][ASSIGN] Before attention_mask concat: prompt_mask.shape={prompt_mask.shape}, completion_mask.shape={completion_mask.shape}", flush=True)

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        # [STAGE3_AUDIO_DIAG][LEN_TRACE] After attention_mask construction
        if rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"[STAGE3_AUDIO_DIAG][LEN_TRACE][ASSIGN] After attention_mask concat: attention_mask.shape={attention_mask.shape}", flush=True)

        # Get the multimodal inputs (already expanded in _generate_and_score_completions)
        # [LOGPROB_FIX] Exclude generation-only parameters from logprob forward
        # rope_deltas should only be used in generate(), not in model forward for logprob computation
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        logprob_exclude_keys = {'rope_deltas'}  # Parameters that should not be passed to model forward in logprob stage
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords if k not in logprob_exclude_keys}

        # [STAGE3_AUDIO_DIAG][LEN_TRACE] Final check before _get_per_token_logps with attention_mask now defined
        if rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"[STAGE3_AUDIO_DIAG][LEN_TRACE] Final: prompt_completion_ids.shape={prompt_completion_ids.shape}, attention_mask.shape={attention_mask.shape}", flush=True)
            if prompt_completion_ids.shape[1] != attention_mask.shape[1]:
                print(f"[STAGE3_AUDIO_DIAG][LEN_TRACE][CRITICAL] MISMATCH DETECTED: seq_len {prompt_completion_ids.shape[1]} != {attention_mask.shape[1]}", flush=True)

        # [STAGE3_AUDIO_DIAG][LOGPROB_FIX] Diagnostic for excluded parameters
        if rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            excluded_keys_present = [k for k in logprob_exclude_keys if k in prompt_inputs and prompt_inputs[k] is not None]
            if excluded_keys_present:
                print(f"[STAGE3_AUDIO_DIAG][LOGPROB_FIX] Excluded generation-only parameters from logprob forward: {excluded_keys_present}", flush=True)
            multimodal_keys_in_inputs = [k for k in multimodal_inputs.keys() if multimodal_inputs[k] is not None]
            print(f"[STAGE3_AUDIO_DIAG][LOGPROB_FIX] multimodal_inputs keys passed to logprob forward: {multimodal_keys_in_inputs}", flush=True)

        # [STAGE3_AUDIO_DIAG][FORWARD_INPUT] Pre-logprob forward alignment check
        if rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"\n[STAGE3_AUDIO_DIAG][FORWARD_INPUT] === Pre-logprob forward alignment ===", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] prompt_completion_ids.shape={prompt_completion_ids.shape}", flush=True)
            pc_batch = prompt_completion_ids.size(0)
            if 'input_features' in multimodal_inputs and multimodal_inputs['input_features'] is not None:
                input_feat = multimodal_inputs['input_features']
                feat_batch = input_feat.size(0)
                print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] input_features batch={feat_batch}, prompt_completion_ids batch={pc_batch}", flush=True)
                if feat_batch != pc_batch:
                    print(f"[STAGE3_AUDIO_DIAG][MISMATCH] batch mismatch: {feat_batch} != {pc_batch}", flush=True)
            if 'feature_attention_mask' in multimodal_inputs and multimodal_inputs['feature_attention_mask'] is not None:
                feat_attn_mask = multimodal_inputs['feature_attention_mask']
                feat_attn_batch = feat_attn_mask.size(0)
                print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] feature_attention_mask batch={feat_attn_batch}, prompt_completion_ids batch={pc_batch}", flush=True)
                if feat_attn_batch != pc_batch:
                    print(f"[STAGE3_AUDIO_DIAG][MISMATCH] batch mismatch: {feat_attn_batch} != {pc_batch}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][FORWARD_INPUT] === End pre-logprob forward alignment ===\n", flush=True)

        # [STAGE3_AUDIO_DIAG][MISMATCH] Pre-forward diagnostics
        if stage3_audio_diag and rank == 0 and stage3_diag_step_count <= max_diag_steps and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"\n[STAGE3_AUDIO_DIAG][MISMATCH] === Pre-forward mismatch check ===", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][MISMATCH] prompt_completion_ids.shape={prompt_completion_ids.shape}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][MISMATCH] attention_mask.shape={attention_mask.shape}", flush=True)

            # Check input_ids token range
            if prompt_completion_ids.numel() > 0:
                min_id = prompt_completion_ids.min().item()
                max_id = prompt_completion_ids.max().item()
                print(f"[STAGE3_AUDIO_DIAG][MISMATCH] input_ids min={min_id}, max={max_id}", flush=True)

                vocab_size = getattr(self.model.config, 'vocab_size', 'UNKNOWN')
                print(f"[STAGE3_AUDIO_DIAG][MISMATCH] model.config.vocab_size={vocab_size}", flush=True)

                if isinstance(vocab_size, int) and max_id >= vocab_size:
                    print(f"[STAGE3_AUDIO_DIAG][MISMATCH] [MISMATCH] token id out of vocab: max_id={max_id} >= vocab_size={vocab_size}", flush=True)
                if min_id < 0:
                    print(f"[STAGE3_AUDIO_DIAG][MISMATCH] [MISMATCH] negative token id: min_id={min_id}", flush=True)

            # Check multimodal inputs
            for key in ['input_features', 'feature_attention_mask', 'pixel_values_videos']:
                if key in multimodal_inputs and multimodal_inputs[key] is not None:
                    val = multimodal_inputs[key]
                    if isinstance(val, torch.Tensor):
                        print(f"[STAGE3_AUDIO_DIAG][MISMATCH] multimodal_inputs[{key}].shape={val.shape}", flush=True)
                        if val.size(0) != prompt_completion_ids.size(0):
                            print(f"[STAGE3_AUDIO_DIAG][MISMATCH] [MISMATCH] {key} batch size mismatch: {val.size(0)} != {prompt_completion_ids.size(0)}", flush=True)

            print(f"[STAGE3_AUDIO_DIAG][MISMATCH] === End pre-forward mismatch check ===\n", flush=True)

        # [AUDIO_LENGTH_FIX] Ensure audio_feature_lengths is present for logprob forward
        if 'feature_attention_mask' in multimodal_inputs and multimodal_inputs['feature_attention_mask'] is not None:
            if 'audio_feature_lengths' not in multimodal_inputs or multimodal_inputs['audio_feature_lengths'] is None:
                feat_attn_mask = multimodal_inputs['feature_attention_mask']
                audio_feature_lengths = feat_attn_mask.sum(dim=-1)
                multimodal_inputs['audio_feature_lengths'] = audio_feature_lengths
                if rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
                    print(f"[STAGE3_AUDIO_DIAG][AUDIO_LENGTH_FIX] Constructed audio_feature_lengths from feature_attention_mask.sum(dim=-1): {audio_feature_lengths.tolist()}", flush=True)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps_with_microbatch(
                    self.model, prompt_completion_ids, attention_mask,
                    completion_ids=completion_ids, completion_mask=completion_mask,
                    **multimodal_inputs
                )
                # No need to slice - already returns completion-only logps
                old_per_token_logps = old_per_token_logps.detach()
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps_with_microbatch(
                    self.ref_model, prompt_completion_ids, attention_mask,
                    completion_ids=completion_ids, completion_mask=completion_mask,
                    **multimodal_inputs
                )
                # No need to slice - already returns completion-only logps
                ref_per_token_logps = ref_per_token_logps.detach()
            else:
                with self._temporarily_disable_adapters(self.model):
                    ref_per_token_logps = self._get_per_token_logps_with_microbatch(
                        self.model, prompt_completion_ids, attention_mask,
                        completion_ids=completion_ids, completion_mask=completion_mask,
                        **multimodal_inputs
                    )
                # No need to slice - already returns completion-only logps
                ref_per_token_logps = ref_per_token_logps.detach()

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # [PHASE3_DEBUG] Log raw completion before post-processing
        if len(completions_text) > 0:
            print(f"\n[PHASE3_GENERATION] Raw completion preview (first 300 chars): {repr(completions_text[0][:300])}")
            print(f"[PHASE3_GENERATION] completion_ids shape: {completion_ids.shape}")
            print(f"[PHASE3_GENERATION] prompt_length: {prompt_length}")
            print(f"[PHASE3_GENERATION] completion token count: {completion_ids.shape[1]}")
            print(f"[PHASE3_GENERATION] Total completions: {len(completions_text)}")

        # Post-process: truncate at first </answer> for image+multiple-choice tasks
        # This ensures reward computation focuses on the valid answer portion
        for idx, example in enumerate(inputs):
            if example.get('data_type') == 'image' and example.get('problem_type') == 'multiple choice':
                # Find all corresponding completions for this input (num_generations per input)
                start_idx = idx * self.num_generations
                end_idx = start_idx + self.num_generations
                for comp_idx in range(start_idx, end_idx):
                    if comp_idx < len(completions_text):
                        text = completions_text[comp_idx]
                        answer_end = text.find('</answer>')
                        if answer_end != -1:
                            completions_text[comp_idx] = text[:answer_end + len('</answer>')]

        # [PHASE3_DEBUG] Log completion after post-processing
        if len(completions_text) > 0:
            print(f"[PHASE3_GENERATION] After post-processing (first 300 chars): {repr(completions_text[0][:300])}")

        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]

        # [PHASE3_DEBUG] Log final completion before reward
        if len(completions) > 0:
            print(f"[PHASE3_GENERATION] Final completion content (first 300 chars): {repr(completions[0][0]['content'][:300])}")

        # [MEM_FIX_5] Clear temporary tensors from generation phase
        del orig_completion_ids
        # CRITICAL: Keep prompt_completion_ids and attention_mask for compute_loss to use canonical fields
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Note: prompts are already expanded at Line 1008 after inputs expansion
        # No need to expand prompts again here

        # Verify length consistency before computing rewards
        expected_completions = num_prompts * self.num_generations
        actual_completions = len(completions)
        if actual_completions != expected_completions:
            raise RuntimeError(
                f"Generation length mismatch: expected {expected_completions} completions "
                f"({num_prompts} prompts * {self.num_generations} generations), "
                f"but got {actual_completions} completions."
            )

        # Compute the rewards
        # Now prompts and completions have matching lengths

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)

        # [STAGE3_AUDIO_DIAG] Build diagnostic kwargs dict for pre-generation diagnostic
        diag_generation_kwargs = {k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()} if hasattr(self, 'vlm_module') else {}

        # [STAGE3_AUDIO_DIAG] Diagnostic before generation (only rank 0 prints)
        if stage3_audio_diag and rank == 0 and stage3_diag_step_count < max_diag_steps:
            print(f"\n[STAGE3_AUDIO_DIAG] === Before generation (step {stage3_diag_step_count+1}/{max_diag_steps}) ===", flush=True)
            print(f"[STAGE3_AUDIO_DIAG] diag_generation_kwargs keys: {list(diag_generation_kwargs.keys())}", flush=True)

            # Check key tensors in diag_generation_kwargs
            for key in ['input_ids', 'attention_mask', 'pixel_values', 'pixel_values_videos', 'input_features', 'feature_attention_mask']:
                if key in diag_generation_kwargs:
                    val = diag_generation_kwargs[key]
                    if val is not None and isinstance(val, torch.Tensor):
                        print(f"[STAGE3_AUDIO_DIAG] {key}: shape={val.shape}, dtype={val.dtype}, device={val.device}", flush=True)
                else:
                    print(f"[STAGE3_AUDIO_DIAG] {key}: MISSING", flush=True)

            num_return_seq = diag_generation_kwargs.get('num_return_sequences', 'MISSING')
            max_new_tok = diag_generation_kwargs.get('max_new_tokens', 'MISSING')
            print(f"[STAGE3_AUDIO_DIAG] num_return_sequences={num_return_seq}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG] max_new_tokens={max_new_tok}", flush=True)

            # [STAGE3_AUDIO_DIAG][MISMATCH] Extended pre-generation diagnostics
            if os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
                print(f"[STAGE3_AUDIO_DIAG][MISMATCH] === Pre-generation mismatch check ===", flush=True)
                if 'input_ids' in diag_generation_kwargs:
                    input_ids = diag_generation_kwargs['input_ids']
                    print(f"[STAGE3_AUDIO_DIAG][MISMATCH] input_ids.shape={input_ids.shape}", flush=True)
                if 'attention_mask' in diag_generation_kwargs:
                    attn_mask = diag_generation_kwargs['attention_mask']
                    print(f"[STAGE3_AUDIO_DIAG][MISMATCH] attention_mask.shape={attn_mask.shape}", flush=True)
                if 'input_features' in diag_generation_kwargs:
                    input_feat = diag_generation_kwargs['input_features']
                    if input_feat is not None:
                        print(f"[STAGE3_AUDIO_DIAG][MISMATCH] input_features.shape={input_feat.shape}", flush=True)
                if 'feature_attention_mask' in diag_generation_kwargs:
                    feat_attn_mask = diag_generation_kwargs['feature_attention_mask']
                    if feat_attn_mask is not None:
                        print(f"[STAGE3_AUDIO_DIAG][MISMATCH] feature_attention_mask.shape={feat_attn_mask.shape}", flush=True)
                        feat_sum = feat_attn_mask.sum(dim=-1)
                        print(f"[STAGE3_AUDIO_DIAG][MISMATCH] feature_attention_mask.sum(dim=-1)={feat_sum.tolist()}", flush=True)
                print(f"[STAGE3_AUDIO_DIAG][MISMATCH] === End pre-generation mismatch check ===", flush=True)

            print(f"[STAGE3_AUDIO_DIAG] === End pre-generation diagnostic ===\n", flush=True)
            self._stage3_diag_step_count = stage3_diag_step_count + 1

        with torch.inference_mode():
            # CRITICAL: This is a second generate call for reference model logprobs
            # Do NOT overwrite completion_ids which was already correctly sliced at line 2663
            # Save this result separately as raw_prompt_completion_ids_ref for reference only
            raw_prompt_completion_ids_ref = self.model.generate(**{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()})

        # [STAGE3_AUDIO_DIAG] Diagnostic after generation
        if stage3_audio_diag and rank == 0 and stage3_diag_step_count <= max_diag_steps:
            print(f"\n[STAGE3_AUDIO_DIAG] === After generation ===", flush=True)
            print(f"[STAGE3_AUDIO_DIAG] completion_ids shape: {completion_ids.shape}", flush=True)

            # Decode first 3 completions for inspection
            for sample_idx in range(min(3, completion_ids.size(0))):
                try:
                    decoded = self.processing_class.decode(completion_ids[sample_idx], skip_special_tokens=False)
                    decoded_preview = decoded[:300] if len(decoded) > 300 else decoded
                    print(f"[STAGE3_AUDIO_DIAG] completion[{sample_idx}] (first 300 chars): {repr(decoded_preview)}", flush=True)

                    # Check for degenerate patterns
                    if "contextsystemsystem" in decoded.lower():
                        print(f"[STAGE3_AUDIO_DIAG] WARNING: completion[{sample_idx}] contains 'contextsystemsystem' pattern", flush=True)

                    # Count repeated tokens
                    token_ids = completion_ids[sample_idx].tolist()
                    if len(token_ids) > 1:
                        max_repeat = 1
                        current_repeat = 1
                        for i in range(1, len(token_ids)):
                            if token_ids[i] == token_ids[i-1]:
                                current_repeat += 1
                                max_repeat = max(max_repeat, current_repeat)
                            else:
                                current_repeat = 1
                        if max_repeat > 10:
                            print(f"[STAGE3_AUDIO_DIAG] WARNING: completion[{sample_idx}] has {max_repeat} repeated tokens", flush=True)

                    # Check for empty or pure punctuation
                    if len(decoded.strip()) == 0:
                        print(f"[STAGE3_AUDIO_DIAG] WARNING: completion[{sample_idx}] is empty", flush=True)
                    elif all(c in '.,!?;:\'"()[]{}' for c in decoded.strip()):
                        print(f"[STAGE3_AUDIO_DIAG] WARNING: completion[{sample_idx}] is pure punctuation", flush=True)

                    # Check for early EOS
                    eos_token_id = getattr(self.processing_class, 'eos_token_id', None)
                    if eos_token_id is not None and eos_token_id in token_ids[:-1]:
                        eos_pos = token_ids[:-1].index(eos_token_id)
                        if eos_pos < len(token_ids) * 0.3:
                            print(f"[STAGE3_AUDIO_DIAG] WARNING: completion[{sample_idx}] has early EOS at position {eos_pos}/{len(token_ids)}", flush=True)

                except Exception as e:
                    print(f"[STAGE3_AUDIO_DIAG] ERROR decoding completion[{sample_idx}]: {e}", flush=True)

            print(f"[STAGE3_AUDIO_DIAG] === End post-generation diagnostic ===\n", flush=True)

        # Compute the rewards
        # Now prompts and completions have matching lengths

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                # [MEM_FIX_6] Clear reward model inputs after use
                del reward_inputs, texts
            else:
                # Note: inputs are already expanded at Line 1002-1007
                # Extract all input columns (but "prompt" and "completion") directly from expanded inputs
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].append(example[key])

                # Extract original emotion question/options from conversation BEFORE short prompt replacement
                emotion_options_text_list = []
                for example in inputs:
                    emotion_options_text = ""
                    if "conversation" in example and example["conversation"]:
                        conv = example["conversation"]
                        if isinstance(conv, list):
                            for msg in conv:
                                if isinstance(msg, dict):
                                    content = msg.get("content", "")
                                    if isinstance(content, str) and "Options:" in content:
                                        emotion_options_text = content
                                        break
                                    elif isinstance(content, list):
                                        for item in content:
                                            if isinstance(item, dict):
                                                text = item.get("text", "")
                                                if isinstance(text, str) and "Options:" in text:
                                                    emotion_options_text = text
                                                    break
                        elif isinstance(conv, str) and "Options:" in conv:
                            emotion_options_text = conv
                    emotion_options_text_list.append(emotion_options_text)

                reward_kwargs["emotion_options_text"] = emotion_options_text_list

                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                # [MEM_FIX_7] Clear reward kwargs after use
                del reward_kwargs, output_reward_func

        # markov
        if rewards_per_func.size(1) ==4 and self.markov_reward: # format, acc, reason, evi
            print("using markov")
            not_valid_evidence_index = rewards_per_func[:, -1]<=0.4
            not_valid_reason_index = rewards_per_func[:, -2]<=0.4
            rewards_per_func[not_valid_evidence_index, 1] = 0
            rewards_per_func[not_valid_evidence_index, 2] = 0
            rewards_per_func[not_valid_reason_index, 1] = 0

            # not_valid_format_index = rewards_per_func[:, 1]<=0.2
            # rewards_per_func[not_valid_format_index, 0] = 0

        # if rewards_per_func.size(1) ==2: # format, acc, reason, evi
        #     print("using markov")
        # not_valid_format_index = rewards_per_func[:, 1]<=0.2

        # rewards_per_func[not_valid_format_index, :] = 0

        # [TASK_AWARE_ROUTING] Apply task-aware reward weights if enabled (default disabled)
        if self.task_aware_routing_enabled and rank == 0:
            # Define reward weights for each task type
            # Format: [format, accuracy, affective_context, emotion_consistency]
            task_reward_weights = {
                "emotion_video": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device),
                "general_video_qa": torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=device),
                "social_reasoning": torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=device),
                "image_math": torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=device),
                "unknown": torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=device),
            }

            # Create task-aware weight matrix [num_completions, num_reward_funcs]
            num_completions = rewards_per_func.shape[0]
            num_reward_funcs = rewards_per_func.shape[1]
            task_weights = torch.ones(num_completions, num_reward_funcs, dtype=torch.float32, device=device)

            # Apply task-specific weights to each completion
            for i, task_type in enumerate(task_types):
                if task_type in task_reward_weights:
                    weights = task_reward_weights[task_type]
                    # Ensure weights match num_reward_funcs
                    if weights.shape[0] >= num_reward_funcs:
                        task_weights[i, :] = weights[:num_reward_funcs]
                    else:
                        task_weights[i, :weights.shape[0]] = weights

            # Apply weights to rewards_per_func
            rewards_per_func = rewards_per_func * task_weights

            # Log task type distribution
            task_type_counts = {}
            for task_type in task_types:
                task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

            print(f"[TASK_AWARE_ROUTING] task_type distribution: {task_type_counts}", flush=True)
            print(f"[TASK_AWARE_ROUTING] reward weights applied by task_type", flush=True)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )
        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        
        # Sum the rewards from all reward functions
        # rewards = rewards_per_func.sum(dim=1)

        def compute_advantage(rewards):
            # Strict consistency check before reshaping
            if rewards.numel() % self.num_generations != 0:
                raise RuntimeError(
                    f"Reward count mismatch in compute_advantage:\n"
                    f"  rewards.shape: {rewards.shape}\n"
                    f"  rewards.numel(): {rewards.numel()}\n"
                    f"  self.num_generations: {self.num_generations}\n"
                    f"  num_prompts (original): {num_prompts}\n"
                    f"  expected total rewards: {num_prompts * self.num_generations}\n"
                    f"  Cannot reshape rewards into groups of {self.num_generations}."
                )

            # Compute grouped-wise rewards
            # Each group consists of num_generations completions for the same prompt
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards
            if self.args.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)
            return advantages, std_grouped_rewards

        full_rewards = []
        patial_rewards = []
        for i, reward_func in enumerate(self.reward_funcs):
            func_name = reward_func.__name__
            reward_value = rewards_per_func[:, i] * self.reward_weights[i].to(device)

            # Apply affective gate if this is an affective reward function
            if self.use_affective_rewards and ("affective" in func_name or "emotion_consistency" in func_name):
                gate_mask = self._compute_affective_gate(inputs, func_name)
                reward_value = reward_value * gate_mask
                logger.debug(f"[AFFECTIVE_GATE] {func_name}: gate_mode={self.affective_reward_gate_mode}, gated_samples={gate_mask.sum().item()}/{len(gate_mask)}")

            # Affective rewards and partial rewards go to patial_rewards for token-level masking
            if "patial" in func_name or "affective" in func_name or "emotion_consistency" in func_name:
                patial_rewards.append((func_name, reward_value))
            else:
                full_rewards.append(reward_value)



        # rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

         # Get only the local slice of advantages
         # Note: prompts has been expanded to num_prompts * self.num_generations
        process_slice = slice(
            self.accelerator.process_index * num_prompts * self.num_generations,
            (self.accelerator.process_index + 1) * num_prompts * self.num_generations,
        )
        # rewards_per_func = rewards_per_func[process_slice, :]

        # Combine full rewards and partial rewards for final aggregation
        all_rewards = full_rewards + [reward_value for _, reward_value in patial_rewards]
        if len(all_rewards) > 0:
            rewards = torch.stack(all_rewards, dim=-1).nansum(dim=1)
        else:
            rewards = torch.zeros(rewards_per_func.shape[0], device=device)

        advantages, std_grouped_rewards = compute_advantage(rewards)
        advantages = advantages[process_slice]

        patial_advantages = []

        if len(patial_rewards) > 0:
            for func_name, partial_reward in patial_rewards:
                partial_reward_advantage = compute_advantage(partial_reward)[0][process_slice]
                if "affective_context" in func_name:
                    mask = generate_2d_mask(completion_ids, [34528], [522, 2147])
                    logger.debug(f"[MASK] {func_name}: affective_context mask (context span) applied")
                elif "emotion_consistency" in func_name:
                    mask = generate_2d_mask(completion_ids, [13708, 766], [522, 26865])
                    logger.debug(f"[MASK] {func_name}: emotion_consistency mask (think span) applied")
                elif "context" in func_name:
                    mask = generate_2d_mask(completion_ids, [34528], [522, 2147])
                    logger.debug(f"[MASK] {func_name}: context mask applied")
                else:
                    mask = generate_2d_mask(completion_ids, [13708, 766], [522, 26865])
                    logger.debug(f"[MASK] {func_name}: logical mask applied")
                mask = mask.to(partial_reward_advantage.device).detach()
                patial_advantages.append({"name": func_name, "reward": partial_reward_advantage.detach(), "mask": mask})



        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]


        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).detach().cpu().item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)

            if self.use_affective_rewards and ("affective" in reward_func_name or "emotion_consistency" in reward_func_name):
                logger.debug(f"[AFFECTIVE_REWARD_LOG] {reward_func_name}: mean_reward={mean_rewards:.4f}, gate_mode={self.affective_reward_gate_mode}")

        self._metrics[mode]["reward"].append(rewards.detach().cpu().mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.detach().cpu().mean().item())

        # [GEN_QUALITY_CHECK] Generation quality statistics after each step
        if self.is_world_process_zero():
            try:
                rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0
                if rank == 0:
                    # Compute tag hit rates
                    context_hits = 0
                    think_hits = 0
                    answer_hits = 0
                    total_completions = len(completions_text)

                    for comp_text in completions_text:
                        if "<context>" in comp_text:
                            context_hits += 1
                        if "<think>" in comp_text:
                            think_hits += 1
                        if "<answer>" in comp_text:
                            answer_hits += 1

                    context_rate = context_hits / total_completions if total_completions > 0 else 0
                    think_rate = think_hits / total_completions if total_completions > 0 else 0
                    answer_rate = answer_hits / total_completions if total_completions > 0 else 0

                    # Get format_reward mean (first reward function if available)
                    format_reward_mean = 0.0
                    if rewards_per_func.shape[1] > 0:
                        format_reward_mean = torch.nanmean(rewards_per_func[:, 0]).detach().cpu().item()

                    final_reward_mean = rewards.detach().cpu().mean().item()
                    final_reward_std = rewards.detach().cpu().std().item()

                    # Get first 2 completions preview
                    comp_preview_1 = completions_text[0][:300] if len(completions_text) > 0 else ""
                    comp_preview_2 = completions_text[1][:300] if len(completions_text) > 1 else ""

                    print(f"[GEN_VALIDITY] context_rate={context_rate:.2%} think_rate={think_rate:.2%} answer_rate={answer_rate:.2%}", flush=True)
                    print(f"[GEN_VALIDITY] format_reward_mean={format_reward_mean:.4f}", flush=True)
                    print(f"[REWARD_AGG] final_rewards mean={final_reward_mean:.4f} std={final_reward_std:.4f}", flush=True)
                    print(f"[GEN_QUALITY_CHECK] step={self.state.global_step} completions={total_completions} "
                          f"context_rate={context_rate:.2%} think_rate={think_rate:.2%} answer_rate={answer_rate:.2%} "
                          f"format_reward_mean={format_reward_mean:.4f} final_reward_mean={final_reward_mean:.4f} "
                          f"final_reward_std={final_reward_std:.4f}", flush=True)
                    print(f"[GEN_QUALITY_CHECK] comp_0_preview: {repr(comp_preview_1)}", flush=True)
                    if comp_preview_2:
                        print(f"[GEN_QUALITY_CHECK] comp_1_preview: {repr(comp_preview_2)}", flush=True)
            except Exception as e:
                print(f"[GEN_QUALITY_CHECK] diagnostic failed: {e}", flush=True)

        # [BAD_GENERATION_GUARD] Smoke test guard for collapsed generation (default disabled, diagnostic only)
        bad_gen_guard_enabled = os.environ.get("GRPO_BAD_GENERATION_GUARD", "0") == "1"
        if bad_gen_guard_enabled and self.is_world_process_zero():
            try:
                rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0
                if rank == 0:
                    # [TASK_AWARE_GUARD] Per-task-type statistics if task-aware routing enabled
                    if self.task_aware_routing_enabled:
                        # Compute per-task-type statistics
                        task_type_stats = {}
                        for task_type in ["emotion_video", "general_video_qa", "social_reasoning", "image_math", "unknown"]:
                            task_type_stats[task_type] = {
                                "answer_rate": 0.0,
                                "format_reward_mean": 0.0,
                                "count": 0,
                            }

                        # Collect stats per task type
                        for i, (comp_text, task_type) in enumerate(zip(completions_text, task_types)):
                            if task_type in task_type_stats:
                                task_type_stats[task_type]["count"] += 1
                                if "<answer>" in comp_text:
                                    task_type_stats[task_type]["answer_rate"] += 1

                        # Compute rates and format rewards per task type
                        for task_type in task_type_stats:
                            count = task_type_stats[task_type]["count"]
                            if count > 0:
                                task_type_stats[task_type]["answer_rate"] /= count
                                # Get format reward mean for this task type
                                if rewards_per_func.shape[1] > 0:
                                    task_indices = [i for i, t in enumerate(task_types) if t == task_type]
                                    if task_indices:
                                        task_rewards = rewards_per_func[task_indices, 0]
                                        task_type_stats[task_type]["format_reward_mean"] = torch.nanmean(task_rewards).detach().cpu().item()

                        # Log per-task-type stats (diagnostic only, no hard stop)
                        print(f"[BAD_GENERATION_GUARD] per-task-type stats: {task_type_stats}", flush=True)

                        # Check for per-task-type collapse and log (diagnostic only)
                        for task_type, stats in task_type_stats.items():
                            if stats["count"] > 0 and (stats["answer_rate"] == 0 or stats["format_reward_mean"] == 0):
                                print(f"[BAD_GENERATION_GUARD] WARNING: task_type {task_type} has low quality: answer_rate={stats['answer_rate']:.2%} format_reward_mean={stats['format_reward_mean']:.4f}", flush=True)
                                # Print sample from this task type
                                for i, (comp_text, t) in enumerate(zip(completions_text, task_types)):
                                    if t == task_type:
                                        print(f"[BAD_GENERATION_GUARD]   sample_preview: {repr(comp_text[:300])}", flush=True)
                                        break
                        # Compute tag hit rates for detailed report
                        context_hits = sum(1 for comp_text in completions_text if "<context>" in comp_text)
                        think_hits = sum(1 for comp_text in completions_text if "<think>" in comp_text)
                        total_completions = len(completions_text)

                        context_rate = context_hits / total_completions if total_completions > 0 else 0
                        think_rate = think_hits / total_completions if total_completions > 0 else 0

                        comp_preview_1 = completions_text[0][:300] if len(completions_text) > 0 else ""
                        comp_preview_2 = completions_text[1][:300] if len(completions_text) > 1 else ""

                        final_reward_mean = rewards.detach().cpu().mean().item()

                        print(f"[BAD_GENERATION_GUARD] detected collapsed generation", flush=True)
                        print(f"[BAD_GENERATION_GUARD] step={self.state.global_step} prompt_len={prompt_ids.shape[1]} "
                              f"completion_len={completion_ids.shape[1]} total_completions={total_completions}", flush=True)
                        print(f"[BAD_GENERATION_GUARD] tag_rates: context={context_rate:.2%} think={think_rate:.2%} answer={answer_rate:.2%}", flush=True)
                        print(f"[BAD_GENERATION_GUARD] comp_0_preview: {repr(comp_preview_1)}", flush=True)
                        if comp_preview_2:
                            print(f"[BAD_GENERATION_GUARD] comp_1_preview: {repr(comp_preview_2)}", flush=True)
                        print(f"[BAD_GENERATION_GUARD] reward_mean={final_reward_mean:.4f}", flush=True)

                        # [GEN_COLLAPSE_DIAG] Enhanced diagnostics for collapsed generation
                        print(f"[GEN_COLLAPSE_DIAG] ===== Collapsed Generation Diagnostics =====", flush=True)
                        print(f"[GEN_COLLAPSE_DIAG] global_step={self.state.global_step}", flush=True)
                        print(f"[GEN_COLLAPSE_DIAG] answer_rate={answer_rate:.4f}", flush=True)
                        print(f"[GEN_COLLAPSE_DIAG] format_reward_mean={format_reward_mean:.4f}", flush=True)

                        # Check tag presence in each completion
                        print(f"[GEN_COLLAPSE_DIAG] Tag presence in completions (first 8):", flush=True)
                        for i, comp_text in enumerate(completions_text[:8]):
                            has_context = "<context>" in comp_text and "</context>" in comp_text
                            has_think = "<think>" in comp_text and "</think>" in comp_text
                            has_answer = "<answer>" in comp_text and "</answer>" in comp_text
                            print(f"[GEN_COLLAPSE_DIAG]   comp_{i}: context={has_context} think={has_think} answer={has_answer}", flush=True)

                        # Print first 300 chars of first 8 completions
                        print(f"[GEN_COLLAPSE_DIAG] Completion heads (first 300 chars, first 8):", flush=True)
                        for i, comp_text in enumerate(completions_text[:8]):
                            head = comp_text[:300]
                            print(f"[GEN_COLLAPSE_DIAG]   comp_{i}_head: {repr(head)}", flush=True)

                        # Print last 300 chars of first 8 completions
                        print(f"[GEN_COLLAPSE_DIAG] Completion tails (last 300 chars, first 8):", flush=True)
                        for i, comp_text in enumerate(completions_text[:8]):
                            tail = comp_text[-300:] if len(comp_text) > 300 else comp_text
                            print(f"[GEN_COLLAPSE_DIAG]   comp_{i}_tail: {repr(tail)}", flush=True)

                        # Print prompt tail (last 300 chars)
                        try:
                            tokenizer = getattr(self.processing_class, "tokenizer", None)
                            if tokenizer is None:
                                tokenizer = self.processing_class
                            if hasattr(tokenizer, "decode"):
                                prompt_tail_ids = prompt_ids[0, max(0, prompt_ids.size(1)-100):]
                                prompt_tail_text = tokenizer.decode(prompt_tail_ids, skip_special_tokens=False)
                                prompt_tail_last300 = prompt_tail_text[-300:] if len(prompt_tail_text) > 300 else prompt_tail_text
                                print(f"[GEN_COLLAPSE_DIAG] prompt_tail_last300: {repr(prompt_tail_last300)}", flush=True)
                        except Exception as e:
                            print(f"[GEN_COLLAPSE_DIAG] failed to decode prompt_tail: {e}", flush=True)

                        # Print tensor shapes and attention mask info
                        print(f"[GEN_COLLAPSE_DIAG] prompt_ids.shape={prompt_ids.shape}", flush=True)
                        print(f"[GEN_COLLAPSE_DIAG] completion_ids.shape={completion_ids.shape}", flush=True)

                        # Calculate real prompt lengths from attention mask
                        if hasattr(prompt_mask, 'sum'):
                            real_prompt_lens = prompt_mask.sum(dim=1)
                            print(f"[GEN_COLLAPSE_DIAG] real_prompt_lens (from attention_mask.sum): {real_prompt_lens.tolist()}", flush=True)

                        # Print generation config if available
                        if hasattr(self.model, 'generation_config'):
                            gen_config = self.model.generation_config
                            print(f"[GEN_COLLAPSE_DIAG] generation_config:", flush=True)
                            print(f"[GEN_COLLAPSE_DIAG]   do_sample={getattr(gen_config, 'do_sample', 'N/A')}", flush=True)
                            print(f"[GEN_COLLAPSE_DIAG]   temperature={getattr(gen_config, 'temperature', 'N/A')}", flush=True)
                            print(f"[GEN_COLLAPSE_DIAG]   top_p={getattr(gen_config, 'top_p', 'N/A')}", flush=True)
                            print(f"[GEN_COLLAPSE_DIAG]   top_k={getattr(gen_config, 'top_k', 'N/A')}", flush=True)
                            print(f"[GEN_COLLAPSE_DIAG]   max_new_tokens={getattr(gen_config, 'max_new_tokens', 'N/A')}", flush=True)
                            print(f"[GEN_COLLAPSE_DIAG]   eos_token_id={getattr(gen_config, 'eos_token_id', 'N/A')}", flush=True)
                            print(f"[GEN_COLLAPSE_DIAG]   pad_token_id={getattr(gen_config, 'pad_token_id', 'N/A')}", flush=True)

                        print(f"[GEN_COLLAPSE_DIAG] ===== End Collapsed Generation Diagnostics =====", flush=True)

                        # Decode prompt tail and completion head to verify slicing
                        try:
                            tokenizer = getattr(self.processing_class, "tokenizer", None)
                            if tokenizer is None:
                                tokenizer = self.processing_class

                            if hasattr(tokenizer, "batch_decode"):
                                # Decode last 100 tokens of prompt
                                prompt_tail_ids = prompt_ids[0, max(0, prompt_ids.size(1)-100):]
                                prompt_tail_decoded = tokenizer.batch_decode(prompt_tail_ids.unsqueeze(0), skip_special_tokens=False)
                                print(f"[BAD_GENERATION_GUARD] prompt_tail_last100: {repr(prompt_tail_decoded[0][:200])}", flush=True)

                                # Decode first 100 tokens of completion
                                comp_head_ids = completion_ids[0, :min(100, completion_ids.size(1))]
                                comp_head_decoded = tokenizer.batch_decode(comp_head_ids.unsqueeze(0), skip_special_tokens=False)
                                print(f"[BAD_GENERATION_GUARD] completion_head_first100: {repr(comp_head_decoded[0][:200])}", flush=True)
                        except Exception as e:
                            print(f"[BAD_GENERATION_GUARD] failed to decode prompt_tail/completion_head: {e}", flush=True)

                        raise RuntimeError(f"[BAD_GENERATION_GUARD] Collapsed generation detected at step {self.state.global_step}: "
                                         f"answer_rate={answer_rate:.2%}, format_reward_mean={format_reward_mean:.4f}")
            except RuntimeError:
                raise
            except Exception as e:
                print(f"[BAD_GENERATION_GUARD] diagnostic failed: {e}", flush=True)

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()
            
            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                # if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                #     import pandas as pd

                #     # For logging
                #     table = {
                #         "step": [str(self.state.global_step)] * len(rewards),
                #         "prompt": prompts_to_log,
                #         "completion": completions_to_log,
                #         "reward": rewards.tolist(),
                #     }
                #     df = pd.DataFrame(table)
                #     wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "prompt_completion_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs,
            "patial_advantages": patial_advantages
        }

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to support backward micro-batch when GRPO_LOGPROB_BACKWARD_MICRO_BATCH_SIZE is set.

        When backward micro-batch is enabled, we manually handle the backward pass inside this method
        and return a detached loss to prevent Trainer from calling backward again.
        """
        backward_micro_batch_size_env = os.environ.get("GRPO_LOGPROB_BACKWARD_MICRO_BATCH_SIZE", None)

        # If backward micro-batch is disabled, use parent class training_step
        if backward_micro_batch_size_env is None or backward_micro_batch_size_env == "0":
            return super().training_step(model, inputs, num_items_in_batch)

        # Backward micro-batch enabled - handle backward manually
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Compute loss with backward micro-batch (this will handle backward internally)
        loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        # loss is already detached in compute_loss when backward micro-batch is enabled
        # Return it for logging only
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Get the prepared inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]

        # CRITICAL FIX: Use canonical fields if available
        if "prompt_completion_ids" in inputs and "attention_mask" in inputs:
            input_ids = inputs["prompt_completion_ids"]
            attention_mask = inputs["attention_mask"]
            use_canonical_fields = True
        else:
            # Fallback: reconstruct from components (should not happen with fixed _generate_and_score_completions)
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            use_canonical_fields = False

        # [AUDIO_DISABLE_CONSISTENCY] Verify no audio inputs when GRPO_DISABLE_AUDIO_INPUTS=1
        disable_audio = os.environ.get("GRPO_DISABLE_AUDIO_INPUTS", "0") == "1"
        if disable_audio:
            if 'input_features' in multimodal_inputs and multimodal_inputs['input_features'] is not None:
                raise RuntimeError("[AUDIO_DISABLE_CONSISTENCY] input_features should be removed in prepare_model_inputs when GRPO_DISABLE_AUDIO_INPUTS=1")
            if self.is_world_process_zero():
                print(f"[AUDIO_DISABLE_CONSISTENCY] generation/logprob/compute_loss use same no-audio inputs", flush=True)

        # [STAGE3_AUDIO_DIAG][COMPUTE_LOSS_CANONICAL] Diagnostic before logprob forward
        rank = 0
        try:
            import torch.distributed as torch_dist
            if torch_dist.is_available() and torch_dist.is_initialized():
                rank = torch_dist.get_rank()
        except:
            rank = 0

        if rank == 0 and os.environ.get("GRPO_STAGE3_MISMATCH_DIAG", "0") == "1":
            print(f"\n[STAGE3_AUDIO_DIAG][COMPUTE_LOSS_CANONICAL] === Canonical field check ===", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][COMPUTE_LOSS_CANONICAL] use_canonical_fields={use_canonical_fields}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][COMPUTE_LOSS_CANONICAL] prompt_ids.shape={prompt_ids.shape}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][COMPUTE_LOSS_CANONICAL] completion_ids.shape={completion_ids.shape}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][COMPUTE_LOSS_CANONICAL] completion_mask.shape={completion_mask.shape}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][COMPUTE_LOSS_CANONICAL] input_ids.shape={input_ids.shape}", flush=True)
            print(f"[STAGE3_AUDIO_DIAG][COMPUTE_LOSS_CANONICAL] attention_mask.shape={attention_mask.shape}", flush=True)

            # Count audio tokens in completion_ids
            tokenizer = None
            if hasattr(self, 'processing_class'):
                tokenizer = getattr(self.processing_class, 'tokenizer', self.processing_class)
            elif hasattr(self, 'tokenizer'):
                tokenizer = self.tokenizer

            if tokenizer is not None:
                for token_str in ['<|AUDIO|>', '<|audio_bos|>', '<|audio_eos|>']:
                    try:
                        if hasattr(tokenizer, 'convert_tokens_to_ids'):
                            token_id = tokenizer.convert_tokens_to_ids(token_str)
                            if token_id is not None and token_id != tokenizer.unk_token_id:
                                completion_count = (completion_ids == token_id).sum().item()
                                input_count = (input_ids == token_id).sum().item()
                                print(f"[STAGE3_AUDIO_DIAG][COMPUTE_LOSS_CANONICAL] {token_str}: completion_count={completion_count}, input_count={input_count}", flush=True)
                    except:
                        pass

            print(f"[STAGE3_AUDIO_DIAG][COMPUTE_LOSS_CANONICAL] === End canonical field check ===\n", flush=True)

        # CRITICAL: Hard shape checks to catch mismatches
        if completion_ids.shape != completion_mask.shape:
            raise RuntimeError(
                "[STAGE3_AUDIO_DIAG][FATAL] completion_ids and completion_mask shape mismatch: "
                f"completion_ids={completion_ids.shape}, completion_mask={completion_mask.shape}. "
                "completion_ids is likely raw/unsliced generation output."
            )

        if input_ids.shape != attention_mask.shape:
            raise RuntimeError(
                "[STAGE3_AUDIO_DIAG][FATAL] input_ids and attention_mask shape mismatch before logprob: "
                f"input_ids={input_ids.shape}, attention_mask={attention_mask.shape}"
            )

        # Check if backward micro-batch is enabled
        backward_micro_batch_size_env = os.environ.get("GRPO_LOGPROB_BACKWARD_MICRO_BATCH_SIZE", None)
        backward_micro_batch_enabled = backward_micro_batch_size_env is not None and backward_micro_batch_size_env != "0"

        if backward_micro_batch_enabled:
            # Use backward micro-batch path
            return self._compute_loss_with_backward_microbatch(
                model, inputs, input_ids, attention_mask, completion_ids, completion_mask, multimodal_inputs, rank
            )

        # Original path: compute policy logprobs with forward micro-batch only
        # [GRAD_FLOW] Recompute current policy per_token_logps with gradients enabled
        # This is critical: we must compute current policy logps in compute_loss with require_grad=True
        # to ensure loss has gradients flowing back to model parameters
        per_token_logps = self._get_per_token_logps_with_microbatch(
            model, input_ids, attention_mask, require_grad=True,
            completion_ids=completion_ids, completion_mask=completion_mask,
            **multimodal_inputs
        )
        # No need to slice - already returns completion-only logps

        # Get the advantages from inputs
        advantages = inputs["advantages"]
        patial_advantages = inputs["patial_advantages"]

        for patial_advantage in patial_advantages:
            advantages = advantages + (patial_advantage["reward"].unsqueeze(-1) * patial_advantage["mask"]).sum(dim=1)


        mode = "eval" if self.control.should_evaluate else "train"
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        if self.num_iterations > 1:
            # [GRAD_FLOW] old_per_token_logps must be computed without gradients (require_grad=False)
            old_per_token_logps = self._get_per_token_logps_with_microbatch(
                self.model, input_ids, attention_mask, require_grad=False,
                completion_ids=completion_ids, completion_mask=completion_mask,
                **multimodal_inputs
            )
            # No need to slice - already returns completion-only logps
        else:
            old_per_token_logps = per_token_logps.detach()

        # [DEVICE_ALIGN_FIX] Ensure all tensors are on same device before loss computation
        # This fixes potential device mismatch issues
        loss_device = per_token_logps.device
        loss_dtype = per_token_logps.dtype

        # Move completion_mask to loss device and ensure compatible dtype
        if completion_mask.device != loss_device:
            completion_mask = completion_mask.to(loss_device)
        if completion_mask.dtype != loss_dtype:
            completion_mask = completion_mask.to(loss_dtype)

        # Move advantages to loss device
        if advantages.device != loss_device:
            advantages = advantages.to(loss_device)
        if advantages.dtype != loss_dtype:
            advantages = advantages.to(loss_dtype)

        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty if beta > 0
        if self.beta > 0:
            if self.state.global_step >(self.state.max_steps/2):
                beta = self.beta*0.25
            else:
                beta = self.beta*(1-0.75*self.state.global_step/(self.state.max_steps/2))
            # [GRAD_FLOW] ref_per_token_logps must be computed without gradients (require_grad=False)
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps_with_microbatch(
                    self.ref_model, input_ids, attention_mask, require_grad=False,
                    completion_ids=completion_ids, completion_mask=completion_mask,
                    **multimodal_inputs
                )
                # No need to slice - already returns completion-only logps
            else:
                with self._temporarily_disable_adapters(self.model):
                    ref_per_token_logps = self._get_per_token_logps_with_microbatch(
                        self.model, input_ids, attention_mask, require_grad=False,
                        completion_ids=completion_ids, completion_mask=completion_mask,
                        **multimodal_inputs
                    )
                # No need to slice - already returns completion-only logps

            # [KL_DEVICE_ALIGN] Ensure ref_per_token_logps is on same device as per_token_logps
            if ref_per_token_logps.device != loss_device:
                ref_per_token_logps = ref_per_token_logps.to(loss_device)

            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + beta * per_token_kl

            # Log KL divergence (convert to scalar before appending)
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().detach().cpu().item())

        # Compute final loss
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp_min(1.0)

        # [LOSS_SCALAR_CHECK] Diagnostic before return
        if self.is_world_process_zero():
            try:
                rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0
                print(f"[LOSS_SCALAR_CHECK] rank={rank} loss.shape={loss.shape} loss.dim()={loss.dim()}", flush=True)
                print(f"[LOSS_SCALAR_CHECK] completion_mask.shape={completion_mask.shape}", flush=True)
                print(f"[LOSS_SCALAR_CHECK] completion_mask.sum()={completion_mask.sum().item()}", flush=True)
            except Exception as e:
                print(f"[LOSS_SCALAR_CHECK] diagnostic failed: {e}", flush=True)

        # [LOSS_SCALAR_FIX] Ensure loss is 0-dimensional scalar tensor
        if loss.dim() != 0:
            if self.is_world_process_zero():
                print(f"[LOSS_SCALAR_FIX] WARNING: non-scalar loss detected, shape={loss.shape}, reducing to scalar", flush=True)
            loss = loss.mean()

        # Log clip ratio (convert to scalar before appending)
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().detach().cpu().item())

        # [LOSS_GRAD_CHECK] Save gradient flow info BEFORE deleting tensors
        per_token_loss_requires_grad = per_token_loss.requires_grad if per_token_loss is not None else False
        per_token_loss_shape = per_token_loss.shape if per_token_loss is not None else None

        # [MEM_TRACE] Optional memory diagnostics after loss computation
        if os.environ.get("GRPO_MEM_TRACE", "0") == "1":
            if self.is_world_process_zero():
                try:
                    rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                    print(f"[MEM_TRACE] rank={rank} step={self.state.global_step} allocated={allocated:.2f}GB reserved={reserved:.2f}GB max_allocated={max_allocated:.2f}GB", flush=True)
                    torch.cuda.reset_peak_memory_stats()
                except Exception as e:
                    print(f"[MEM_TRACE] ERROR: {e}", flush=True)

        # [MEM_FIX_9] Clear temporary tensors from loss computation
        del per_token_loss1, per_token_loss2, per_token_loss, coef_1, coef_2, is_clipped
        if self.beta > 0:
            del per_token_kl

        # [LOSS_FINAL_CHECK] Final assertion before return
        assert loss.dim() == 0, f"[LOSS_FINAL_CHECK] loss must be 0-dimensional scalar, got shape={loss.shape}, dim={loss.dim()}"
        assert loss.requires_grad, f"[LOSS_FINAL_CHECK] loss must retain gradients for backward pass"

        # [GRPO_AUDIO_GRAD_DIAG] Audio tower LoRA gradient diagnostics before return
        grpo_lora_train_scope = os.environ.get("GRPO_LORA_TRAIN_SCOPE", "model_layers_only")
        verbose_grad = os.environ.get("GRPO_AUDIO_GRAD_DIAG_VERBOSE", "0") == "1"
        if grpo_lora_train_scope == "audio_tower_only" and self.is_world_process_zero():
            try:
                rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0

                # Count trainable parameters
                trainable_audio_tower_lora = 0
                trainable_model_layers_lora = 0
                audio_tower_lora_params = []
                model_layers_lora_params = []

                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        if 'audio_tower' in name and 'lora_' in name:
                            trainable_audio_tower_lora += 1
                            audio_tower_lora_params.append((name, param))
                        if 'model.layers' in name and 'lora_' in name:
                            trainable_model_layers_lora += 1
                            model_layers_lora_params.append((name, param))

                # Summary statistics for all audio_tower LoRA parameters
                grad_none_count = sum(1 for _, p in audio_tower_lora_params if p.grad is None)
                grad_nonzero_count = sum(1 for _, p in audio_tower_lora_params if p.grad is not None and p.grad.norm().item() > 0)
                grad_zero_count = sum(1 for _, p in audio_tower_lora_params if p.grad is not None and p.grad.norm().item() == 0)
                grad_norm_sum = sum(p.grad.norm().item() for _, p in audio_tower_lora_params if p.grad is not None)

                # Always print summary (not verbose-gated)
                print(f"\n[GRPO_AUDIO_GRAD_DIAG] ===== Audio Tower LoRA Gradient Summary =====", flush=True)
                print(f"[GRPO_AUDIO_GRAD_DIAG] trainable audio_tower LoRA count = {trainable_audio_tower_lora}", flush=True)
                print(f"[GRPO_AUDIO_GRAD_DIAG] trainable model.layers LoRA count = {trainable_model_layers_lora}", flush=True)
                print(f"[GRPO_AUDIO_GRAD_DIAG] grad_norm_sum = {grad_norm_sum:.6f}", flush=True)
                print(f"[GRPO_AUDIO_GRAD_DIAG] grad_nonzero_count = {grad_nonzero_count}", flush=True)

                if verbose_grad:
                    print(f"[GRPO_AUDIO_GRAD_DIAG] GRPO_LORA_TRAIN_SCOPE = {grpo_lora_train_scope}", flush=True)
                    # Check first 10 audio_tower LoRA parameters for gradients
                    print(f"[GRPO_AUDIO_GRAD_DIAG] First 10 audio_tower LoRA parameters:", flush=True)
                    for i, (name, param) in enumerate(audio_tower_lora_params[:10]):
                        grad_is_none = param.grad is None
                        grad_dtype = param.grad.dtype if param.grad is not None else "N/A"
                        grad_device = param.grad.device if param.grad is not None else "N/A"
                        grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                        grad_abs_mean = param.grad.abs().mean().item() if param.grad is not None else 0.0
                        grad_max = param.grad.abs().max().item() if param.grad is not None else 0.0

                        print(f"[GRPO_AUDIO_GRAD_DIAG]   {i+1}. {name}", flush=True)
                        print(f"[GRPO_AUDIO_GRAD_DIAG]      requires_grad={param.requires_grad}, grad_is_none={grad_is_none}", flush=True)
                        print(f"[GRPO_AUDIO_GRAD_DIAG]      grad_dtype={grad_dtype}, grad_device={grad_device}", flush=True)
                        print(f"[GRPO_AUDIO_GRAD_DIAG]      grad_norm={grad_norm:.6f}, grad_abs_mean={grad_abs_mean:.6f}, grad_max={grad_max:.6f}", flush=True)

                    print(f"[GRPO_AUDIO_GRAD_DIAG]   grad_is_none_count = {grad_none_count}", flush=True)
                    print(f"[GRPO_AUDIO_GRAD_DIAG]   grad_norm == 0 count = {grad_zero_count}", flush=True)

                # Check model.layers LoRA should be frozen
                if trainable_model_layers_lora > 0:
                    print(f"[GRPO_AUDIO_GRAD_DIAG][WARNING] audio_tower_only mode but found {trainable_model_layers_lora} trainable model.layers LoRA params", flush=True)

                print(f"[GRPO_AUDIO_GRAD_DIAG] ===== End Audio Tower LoRA Gradient Summary =====\n", flush=True)
            except Exception as e:
                print(f"[GRPO_AUDIO_GRAD_DIAG] diagnostic failed: {e}", flush=True)

        # [LOSS_GRAD_CHECK] Gradient flow diagnostics (using saved info, not deleted tensors)
        if self.is_world_process_zero():
            try:
                rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0
                print(f"[LOSS_GRAD_CHECK] rank={rank} per_token_logps.requires_grad={per_token_logps.requires_grad}", flush=True)
                print(f"[LOSS_GRAD_CHECK] rank={rank} per_token_loss.requires_grad={per_token_loss_requires_grad}", flush=True)
                print(f"[LOSS_GRAD_CHECK] rank={rank} per_token_loss.shape={per_token_loss_shape}", flush=True)
                print(f"[LOSS_GRAD_CHECK] rank={rank} loss.requires_grad={loss.requires_grad}", flush=True)
                print(f"[LOSS_GRAD_CHECK] rank={rank} loss.grad_fn={loss.grad_fn}", flush=True)

                # Check first trainable parameter
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(f"[LOSS_GRAD_CHECK] rank={rank} first_trainable_param_name={name}", flush=True)
                        print(f"[LOSS_GRAD_CHECK] rank={rank} first_trainable_param_requires_grad={param.requires_grad}", flush=True)
                        break
            except Exception as e:
                print(f"[LOSS_GRAD_CHECK] diagnostic failed: {e}", flush=True)

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        # [GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] Optimizer param group diagnostics
        grpo_lora_train_scope = os.environ.get("GRPO_LORA_TRAIN_SCOPE", "model_layers_only")
        if grpo_lora_train_scope == "audio_tower_only" and self.is_world_process_zero() and mode == "train":
            try:
                if hasattr(self, 'optimizer') and self.optimizer is not None:
                    rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0
                    print(f"\n[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] ===== Optimizer Param Group Diagnostics =====", flush=True)
                    print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] GRPO_LORA_TRAIN_SCOPE = {grpo_lora_train_scope}", flush=True)

                    if hasattr(self, '_grpo_audio_optimizer_verified') and self._grpo_audio_optimizer_verified:
                        print(f"[GRPO_AUDIO_GRAD_DIAG][ZERO3_NOTE] create_optimizer verification already passed at initialization", flush=True)
                        print(f"[GRPO_AUDIO_GRAD_DIAG][ZERO3_NOTE] verified audio_tower LoRA count: {self._grpo_audio_optimizer_audio_lora_count}", flush=True)
                        print(f"[GRPO_AUDIO_GRAD_DIAG][ZERO3_NOTE] verified model.layers LoRA count: {self._grpo_audio_optimizer_model_lora_count}", flush=True)

                    optimizer_param_count = 0
                    optimizer_audio_tower_lora_count = 0
                    optimizer_model_layers_lora_count = 0
                    optimizer_base_layer_count = 0
                    optimizer_vision_count = 0

                    for param_group in self.optimizer.param_groups:
                        for param in param_group['params']:
                            optimizer_param_count += 1
                            param_name = None
                            for name, p in self.model.named_parameters():
                                if p is param:
                                    param_name = name
                                    break

                            if param_name:
                                if 'audio_tower' in param_name and 'lora_' in param_name:
                                    optimizer_audio_tower_lora_count += 1
                                elif 'model.layers' in param_name and 'lora_' in param_name:
                                    optimizer_model_layers_lora_count += 1
                                elif any(keyword in param_name for keyword in self.vision_modules_keywords):
                                    optimizer_vision_count += 1
                                else:
                                    optimizer_base_layer_count += 1

                    print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] optimizer param group count = {len(self.optimizer.param_groups)}", flush=True)
                    print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] optimizer total param count = {optimizer_param_count}", flush=True)
                    print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] optimizer audio_tower LoRA param count = {optimizer_audio_tower_lora_count}", flush=True)
                    print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] optimizer model.layers LoRA param count = {optimizer_model_layers_lora_count}", flush=True)
                    print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] optimizer base layer param count = {optimizer_base_layer_count}", flush=True)
                    print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] optimizer vision module param count = {optimizer_vision_count}", flush=True)

                    if hasattr(self, '_grpo_audio_optimizer_verified') and self._grpo_audio_optimizer_verified:
                        if optimizer_audio_tower_lora_count == 0:
                            print(f"[GRPO_AUDIO_GRAD_DIAG][ZERO3_NOTE] post-DeepSpeed param_groups shows zero audio_tower LoRA params, but create_optimizer verified {self._grpo_audio_optimizer_audio_lora_count} params", flush=True)
                            print(f"[GRPO_AUDIO_GRAD_DIAG][ZERO3_NOTE] This is expected under DeepSpeed ZeRO-3 partitioning; create_optimizer verification is used as source of truth", flush=True)
                        if optimizer_model_layers_lora_count > 0:
                            print(f"[GRPO_AUDIO_GRAD_DIAG][ZERO3_NOTE] WARNING: post-DeepSpeed shows {optimizer_model_layers_lora_count} model.layers LoRA params, but create_optimizer verified 0", flush=True)
                    else:
                        if optimizer_audio_tower_lora_count == 0:
                            logger.error("[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER][FATAL] audio_tower_only requested but optimizer has no audio_tower LoRA params")
                            raise RuntimeError(
                                "[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER][FATAL] audio_tower_only validation failed!\n"
                                f"optimizer audio_tower LoRA param count: {optimizer_audio_tower_lora_count}\n"
                                "Expected: > 0"
                            )

                        if optimizer_model_layers_lora_count > 0:
                            logger.error("[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER][FATAL] audio_tower_only requested but optimizer contains model.layers LoRA params")
                            raise RuntimeError(
                                "[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER][FATAL] audio_tower_only validation failed!\n"
                                f"optimizer model.layers LoRA param count: {optimizer_model_layers_lora_count}\n"
                                "Expected: 0"
                            )

                    print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] ===== End Optimizer Param Group Diagnostics =====\n", flush=True)
            except Exception as e:
                print(f"[GRPO_AUDIO_GRAD_DIAG][OPTIMIZER] diagnostic failed: {e}", flush=True)

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        # Ensure batch_size is at least 1 to avoid division by zero
        computed_batch_size = effective_batch_size // self.num_generations
        safe_batch_size = max(1, int(computed_batch_size))

        return RepeatRandomSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=safe_batch_size,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )


    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )
