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

"""Supervised fine-tuning script for decoder language models."""

import logging
import os
import sys
import re
from typing import List

import datasets
import torch
from torch.utils.data import Dataset
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from open_r1.configs import SFTConfig
from open_r1.utils.callbacks import get_callbacks
import yaml
import json
import math
import random
from PIL import Image

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import field
from qwen_vl_utils import process_vision_info
from qwen_omni_utils import process_mm_info
import av

# Adapter saving utilities
from transformers import TrainerCallback
from accelerate.utils import is_peft_model
from peft import LoraConfig, get_peft_model, TaskType, inject_adapter_in_model
try:
    from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
except ImportError:
    load_safetensors = None
    save_safetensors = None

try:
    import deepspeed
    from deepspeed import zero
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    deepspeed = None
    zero = None

# Keep this import for DeepSpeed version compatibility.
ZERO_PARAM_STATUS_AVAILABLE = False
ZeroParamStatus = None
if DEEPSPEED_AVAILABLE:
    try:
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
        ZERO_PARAM_STATUS_AVAILABLE = True
    except (ImportError, AttributeError):
        try:
            from deepspeed.runtime.zero import ZeroParamStatus
            ZERO_PARAM_STATUS_AVAILABLE = True
        except (ImportError, AttributeError):
            ZERO_PARAM_STATUS_AVAILABLE = False

# Track a few samples for video-structure diagnostics.
_videos_debug_count = 0
_videos_debug_max = 3

def safe_save_peft_adapter(model, output_dir, accelerator=None, global_step=None, tokenizer_or_processor=None):
    """Save PEFT adapters with distributed-safe state extraction."""
    # Determine whether the current process should emit diagnostics.
    is_main_process = True
    if accelerator is not None:
        is_main_process = accelerator.is_main_process
    else:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                is_main_process = dist.get_rank() == 0
        except Exception:
            is_main_process = True

    if is_main_process:
        print(f"\n{'='*80}")
        print(f"[SFT_SAVE_DIAG] ===== PEFT Adapter Saving Diagnostics =====")
        print(f"[SFT_SAVE_DIAG] output_dir: {output_dir}")
        print(f"[SFT_SAVE_DIAG] global_step: {global_step}")
        print(f"[SFT_SAVE_DIAG] Model type: {type(model)}")
        print(f"[SFT_SAVE_DIAG] Has peft_config: {hasattr(model, 'peft_config')}")

    if is_main_process and hasattr(model, 'peft_config'):
        peft_config_dict = model.peft_config
        print(f"[SFT_SAVE_DIAG] PEFT config keys: {list(peft_config_dict.keys())}")
        for adapter_name, config in peft_config_dict.items():
            print(f"[SFT_SAVE_DIAG]   Adapter '{adapter_name}': {config}")

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

    if is_main_process:
        print(f"[SFT_SAVE_DIAG] Total LoRA params: {lora_a_count + lora_b_count}")
        print(f"[SFT_SAVE_DIAG] LoRA trainable: {lora_trainable_count}, frozen: {lora_frozen_count}")

    module_lora_counts = {}
    for n, p in model.named_parameters():
        if 'lora_' in n:
            if 'model.layers' in n:
                module = 'model.layers'
            elif 'audio_tower' in n:
                module = 'audio_tower'
            elif 'visual' in n:
                module = 'visual'
            else:
                module = 'other'

            if module not in module_lora_counts:
                module_lora_counts[module] = 0
            module_lora_counts[module] += 1

    if is_main_process:
        print(f"[SFT_SAVE_DIAG] LoRA counts by module:")
        for module in ['model.layers', 'audio_tower', 'visual', 'other']:
            if module in module_lora_counts:
                print(f"[SFT_SAVE_DIAG]   {module}: {module_lora_counts[module]}")

        print(f"[SFT_SAVE_DIAG] First 30 LoRA parameters:")
        for i, (name, shape, requires_grad) in enumerate(lora_param_names[:30]):
            print(f"[SFT_SAVE_DIAG]   {i+1}. {name} | shape={shape} | requires_grad={requires_grad}")

        trainable_param_names = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                trainable_param_names.append((n, p.shape))

        print(f"[SFT_SAVE_DIAG] First 30 trainable parameters:")
        for i, (name, shape) in enumerate(trainable_param_names[:30]):
            print(f"[SFT_SAVE_DIAG]   {i+1}. {name} | shape={shape}")

    unwrapped_model = model
    if accelerator is not None:
        unwrapped_model = accelerator.unwrap_model(model)
    elif hasattr(model, 'module'):
        unwrapped_model = model.module

    has_lora_params = any('lora_' in n for n, _ in unwrapped_model.named_parameters())

    if not is_peft_model(unwrapped_model) and not has_lora_params:
        if is_main_process:
            print(f"[SFT_SAVE_DIAG][WARNING] Model is not a PEFT model and has no LoRA params, skipping adapter save")
            print(f"{'='*80}\n")
        return

    if is_main_process and not is_peft_model(unwrapped_model) and has_lora_params:
        print(f"[SFT_SAVE_DIAG] Model has in-place LoRA (not PeftModel wrapper), will save as adapter-only")

    if is_main_process:
        print(f"[SFT_SAVE_DIAG] Using accelerator.get_state_dict for distributed-safe extraction...")

    if accelerator is not None:
        full_state_dict = accelerator.get_state_dict(model)
    else:
        full_state_dict = unwrapped_model.state_dict()

    lora_state_dict = {k: v for k, v in full_state_dict.items() if 'lora_' in k}

    has_audio_tower = any('audio_tower' in k and 'lora_' in k for k in lora_state_dict.keys())
    has_visual = any(('visual' in k or 'vision' in k) and 'lora_' in k for k in lora_state_dict.keys())
    has_model_layers = any('model.layers' in k and 'lora_' in k for k in lora_state_dict.keys())

    has_base_model_prefix = any(k.startswith('base_model.model.') for k in lora_state_dict.keys())

    has_nested_visual = any('visual.base_model.model' in k for k in lora_state_dict.keys())
    has_nested_audio = any('audio_tower.base_model.model' in k for k in lora_state_dict.keys())

    is_multimodal_joint_inplace = (has_audio_tower or has_visual or has_model_layers) and not has_base_model_prefix
    is_multimodal_joint_nested = has_nested_visual or has_nested_audio

    if is_main_process:
        print(f"[SFT_SAVE_DIAG] LoRA state_dict key count (before standardization): {len(lora_state_dict)}")
        print(f"[SFT_SAVE_DIAG] Detected in-place multimodal_joint: {is_multimodal_joint_inplace}")
        print(f"[SFT_SAVE_DIAG] Detected nested multimodal_joint: {is_multimodal_joint_nested}")

        model_layers_keys = [k for k in lora_state_dict.keys() if 'model.layers' in k]
        audio_tower_keys = [k for k in lora_state_dict.keys() if 'audio_tower' in k]
        vision_keys = [k for k in lora_state_dict.keys() if 'visual' in k or 'vision' in k]
        lora_a_keys = [k for k in lora_state_dict.keys() if 'lora_A' in k]
        lora_b_keys = [k for k in lora_state_dict.keys() if 'lora_B' in k]

        print(f"[SFT_SAVE_DIAG] Keys with model.layers: {len(model_layers_keys)}")
        print(f"[SFT_SAVE_DIAG] Keys with audio_tower: {len(audio_tower_keys)}")
        print(f"[SFT_SAVE_DIAG] Keys with visual/vision: {len(vision_keys)}")
        print(f"[SFT_SAVE_DIAG] Keys with lora_A: {len(lora_a_keys)}")
        print(f"[SFT_SAVE_DIAG] Keys with lora_B: {len(lora_b_keys)}")

        print(f"[SFT_SAVE_DIAG] First 30 state_dict keys (before standardization):")
        for i, key in enumerate(list(lora_state_dict.keys())[:30]):
            print(f"[SFT_SAVE_DIAG]   {i+1}. {key}")

    if is_multimodal_joint_inplace or is_multimodal_joint_nested:
        if is_main_process:
            model_layers_keys = [k for k in lora_state_dict.keys() if 'model.layers' in k]
            audio_tower_keys = [k for k in lora_state_dict.keys() if 'audio_tower' in k]
            vision_keys = [k for k in lora_state_dict.keys() if 'visual' in k or 'vision' in k]

            if len(lora_state_dict) == 0:
                error_msg = "[SFT_SAVE_DIAG][ERROR] multimodal_joint has 0 LoRA keys!"
                print(error_msg)
                raise RuntimeError(error_msg)

            if len(audio_tower_keys) == 0:
                error_msg = "[SFT_SAVE_DIAG][ERROR] multimodal_joint has 0 audio_tower LoRA keys!"
                print(error_msg)
                raise RuntimeError(error_msg)

            if len(vision_keys) == 0:
                error_msg = "[SFT_SAVE_DIAG][ERROR] multimodal_joint has 0 visual/vision LoRA keys!"
                print(error_msg)
                raise RuntimeError(error_msg)

            if len(model_layers_keys) == 0:
                error_msg = "[SFT_SAVE_DIAG][ERROR] multimodal_joint has 0 model.layers LoRA keys!"
                print(error_msg)
                raise RuntimeError(error_msg)

            print(f"[SFT_SAVE_DIAG] Validation passed: audio_tower={len(audio_tower_keys)}, visual/vision={len(vision_keys)}, model.layers={len(model_layers_keys)}")

    # Standardize multimodal_joint keys before saving.
    if is_multimodal_joint_inplace or is_multimodal_joint_nested:
        if is_main_process:
            print(f"[SFT_SAVE_DIAG] Exporting in-place multimodal_joint LoRA as standard PEFT adapter")

        standardized_lora_state_dict = {}
        audio_tower_standardized = 0
        visual_standardized = 0
        model_layers_standardized = 0

        for key, value in lora_state_dict.items():
            new_key = key

            if 'visual.base_model.model' in key:
                new_key = key.replace('visual.base_model.model', 'visual')
            if 'audio_tower.base_model.model' in key:
                new_key = key.replace('audio_tower.base_model.model', 'audio_tower')

            if not new_key.startswith('base_model.model.'):
                new_key = 'base_model.model.' + new_key

            import re
            new_key = re.sub(r'\.lora_A\.[^.]+\.weight$', '.lora_A.weight', new_key)
            new_key = re.sub(r'\.lora_B\.[^.]+\.weight$', '.lora_B.weight', new_key)
            new_key = re.sub(r'\.lora_A\.[^.]+\.bias$', '.lora_A.bias', new_key)
            new_key = re.sub(r'\.lora_B\.[^.]+\.bias$', '.lora_B.bias', new_key)

            standardized_lora_state_dict[new_key] = value

            if 'audio_tower' in new_key:
                audio_tower_standardized += 1
            elif 'visual' in new_key or 'vision' in new_key:
                visual_standardized += 1
            elif 'model.layers' in new_key:
                model_layers_standardized += 1

            if is_main_process and new_key != key:
                if len(standardized_lora_state_dict) <= 10:
                    print(f"[SFT_SAVE_DIAG] Key mapping: {key} -> {new_key}")

        lora_state_dict = standardized_lora_state_dict

        if is_main_process:
            print(f"[SFT_SAVE_DIAG] standardized audio_tower keys = {audio_tower_standardized}")
            print(f"[SFT_SAVE_DIAG] standardized visual/vision keys = {visual_standardized}")
            print(f"[SFT_SAVE_DIAG] standardized model.layers keys = {model_layers_standardized}")
            print(f"[SFT_SAVE_DIAG] LoRA state_dict key count (after standardization): {len(lora_state_dict)}")

            keys_with_lora_A_default = [k for k in lora_state_dict.keys() if '.lora_A.default.' in k]
            keys_with_lora_B_default = [k for k in lora_state_dict.keys() if '.lora_B.default.' in k]
            keys_with_lora_A_weight = [k for k in lora_state_dict.keys() if '.lora_A.weight' in k]
            keys_with_lora_B_weight = [k for k in lora_state_dict.keys() if '.lora_B.weight' in k]

            print(f"[SFT_SAVE_DIAG] stripped adapter_name from LoRA keys for PEFT-compatible save")
            print(f"[SFT_SAVE_DIAG] keys_with_lora_A_default = {len(keys_with_lora_A_default)}")
            print(f"[SFT_SAVE_DIAG] keys_with_lora_B_default = {len(keys_with_lora_B_default)}")
            print(f"[SFT_SAVE_DIAG] keys_with_lora_A_weight = {len(keys_with_lora_A_weight)}")
            print(f"[SFT_SAVE_DIAG] keys_with_lora_B_weight = {len(keys_with_lora_B_weight)}")

            if keys_with_lora_A_default or keys_with_lora_B_default:
                error_msg = (
                    f"[SFT_SAVE_DIAG][ERROR] Adapter name stripping failed! Keys still contain .default:\n"
                    f"  keys_with_lora_A_default: {len(keys_with_lora_A_default)}\n"
                    f"  keys_with_lora_B_default: {len(keys_with_lora_B_default)}\n"
                    f"  Examples: {(keys_with_lora_A_default + keys_with_lora_B_default)[:5]}"
                )
                print(error_msg)
                raise RuntimeError(error_msg)

            print(f"[SFT_SAVE_DIAG] First 30 state_dict keys (after standardization):")
            for i, key in enumerate(list(lora_state_dict.keys())[:30]):
                print(f"[SFT_SAVE_DIAG]   {i+1}. {key}")

            remaining_nested_visual = [k for k in lora_state_dict.keys() if 'visual.base_model.model' in k]
            remaining_nested_audio = [k for k in lora_state_dict.keys() if 'audio_tower.base_model.model' in k]
            missing_prefix = [k for k in lora_state_dict.keys() if not k.startswith('base_model.model.')]

            if remaining_nested_visual or remaining_nested_audio:
                error_msg = (
                    f"[SFT_SAVE_DIAG][ERROR] Key standardization failed! Nested keys remain:\n"
                    f"  Nested visual keys: {len(remaining_nested_visual)}\n"
                    f"  Nested audio keys: {len(remaining_nested_audio)}\n"
                    f"  Examples: {(remaining_nested_visual + remaining_nested_audio)[:5]}"
                )
                print(error_msg)
                raise RuntimeError(error_msg)

            if missing_prefix:
                error_msg = (
                    f"[SFT_SAVE_DIAG][ERROR] Key standardization failed! Keys missing base_model.model prefix:\n"
                    f"  Count: {len(missing_prefix)}\n"
                    f"  Examples: {missing_prefix[:5]}"
                )
                print(error_msg)
                raise RuntimeError(error_msg)

    # Validate before saving.
    if len(lora_state_dict) == 0:
        error_msg = (
            f"[SFT_SAVE_DIAG][ERROR] Cannot save empty PEFT adapter!\n"
            f"  output_dir: {output_dir}\n"
            f"  global_step: {global_step}\n"
            f"  use_peft: True\n"
            f"  deepspeed_available: {DEEPSPEED_AVAILABLE}\n"
            f"  total_lora_params: {lora_a_count + lora_b_count}\n"
            f"  trainable_lora_params: {lora_trainable_count}"
        )
        if is_main_process:
            print(error_msg)
        raise ValueError(error_msg)

    if accelerator is not None:
        accelerator.wait_for_everyone()
    else:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
        except Exception:
            pass

    # Only the main process writes files.
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)

        adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
        print(f"[SFT_SAVE_DIAG] Saving adapter to: {adapter_file}")

        if save_safetensors is not None:
            save_safetensors(lora_state_dict, adapter_file)
        else:
            print(f"[SFT_SAVE_DIAG][WARNING] safetensors not available, using torch.save")
            torch.save(lora_state_dict, adapter_file)

        if is_multimodal_joint_inplace or is_multimodal_joint_nested:
            # Generate a custom adapter config for multimodal_joint.
            print(f"[SFT_SAVE_DIAG] Generating custom adapter_config.json for multimodal_joint scope...")

            sample_peft_config = None
            if hasattr(unwrapped_model, 'peft_config') and unwrapped_model.peft_config:
                sample_peft_config = list(unwrapped_model.peft_config.values())[0]
            elif hasattr(unwrapped_model, 'audio_tower') and hasattr(unwrapped_model.audio_tower, 'peft_config'):
                sample_peft_config = list(unwrapped_model.audio_tower.peft_config.values())[0]
            elif hasattr(unwrapped_model, 'visual') and hasattr(unwrapped_model.visual, 'peft_config'):
                sample_peft_config = list(unwrapped_model.visual.peft_config.values())[0]
            elif hasattr(unwrapped_model.model, 'visual') and hasattr(unwrapped_model.model.visual, 'peft_config'):
                sample_peft_config = list(unwrapped_model.model.visual.peft_config.values())[0]

            if sample_peft_config is not None:
                lora_r = sample_peft_config.r
                lora_alpha = sample_peft_config.lora_alpha
                lora_dropout = sample_peft_config.lora_dropout
                base_model_path = sample_peft_config.base_model_name_or_path if hasattr(sample_peft_config, 'base_model_name_or_path') else None
            else:
                lora_r = getattr(unwrapped_model.config, 'lora_r', 8)
                lora_alpha = getattr(unwrapped_model.config, 'lora_alpha', 16)
                lora_dropout = getattr(unwrapped_model.config, 'lora_dropout', 0.05)
                base_model_path = unwrapped_model.config._name_or_path if hasattr(unwrapped_model.config, '_name_or_path') else None

            # Keep q_proj, v_proj, q, and v in multimodal_joint exports.
            merged_target_modules = ['q_proj', 'v_proj', 'q', 'v']

            import json
            config_dict = {
                "peft_type": "LORA",
                "auto_mapping": None,
                "base_model_name_or_path": base_model_path,
                "revision": None,
                "task_type": "CAUSAL_LM",
                "inference_mode": False,
                "r": lora_r,
                "target_modules": merged_target_modules,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "fan_in_fan_out": False,
                "bias": "none",
                "use_rslora": False,
                "modules_to_save": None,
                "init_lora_weights": True,
                "layers_to_transform": None,
                "layers_pattern": None,
            }

            config_file = os.path.join(output_dir, "adapter_config.json")
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)

            print(f"[SFT_SAVE_DIAG] Saved custom adapter_config.json to: {config_file}")
            print(f"[SFT_SAVE_DIAG] adapter_config target_modules = {merged_target_modules}")

        elif hasattr(unwrapped_model, 'peft_config'):
            for adapter_name, peft_config in unwrapped_model.peft_config.items():
                config_file = os.path.join(output_dir, "adapter_config.json")
                peft_config.save_pretrained(output_dir)
                print(f"[SFT_SAVE_DIAG] Saved adapter config to: {config_file}")

        if tokenizer_or_processor is not None:
            try:
                tokenizer_or_processor.save_pretrained(output_dir)
                print(f"[SFT_SAVE_DIAG] Saved tokenizer/processor to: {output_dir}")
            except Exception as e:
                print(f"[SFT_SAVE_DIAG][WARNING] Failed to save tokenizer/processor: {e}")

        print(f"[SFT_SAVE_DIAG] Verifying saved adapter...")
        if load_safetensors is not None and os.path.exists(adapter_file):
            try:
                loaded_adapter = load_safetensors(adapter_file)
                print(f"[SFT_SAVE_DIAG] Verification: loaded {len(loaded_adapter)} keys from adapter")

                if len(loaded_adapter) == 0:
                    error_msg = (
                        f"[SFT_SAVE_DIAG][ERROR] Saved adapter is empty after verification!\n"
                        f"  adapter_file: {adapter_file}\n"
                        f"  output_dir: {output_dir}\n"
                        f"  global_step: {global_step}"
                    )
                    print(error_msg)
                    raise ValueError(error_msg)

                model_layers_loaded = [k for k in loaded_adapter.keys() if 'model.layers' in k]
                print(f"[SFT_SAVE_DIAG] Verification: {len(model_layers_loaded)} model.layers keys in saved adapter")

            except Exception as e:
                print(f"[SFT_SAVE_DIAG][ERROR] Failed to verify adapter: {e}")
                raise

        print(f"[SFT_SAVE_DIAG] ===== PEFT Adapter Saved Successfully =====")
        print(f"{'='*80}\n")

    if accelerator is not None:
        accelerator.wait_for_everyone()
    else:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
        except Exception:
            pass


class PEFTAdapterSaveCallback(TrainerCallback):
    """Callback to safely save PEFT adapter at each checkpoint."""

    def __init__(self, tokenizer_or_processor=None):
        self.tokenizer_or_processor = tokenizer_or_processor

    def on_save(self, args, state, control, **kwargs):
        """Called after a checkpoint is saved."""
        # Note: This callback is only invoked on the main process by Trainer
        # However, safe_save_peft_adapter internally handles multi-rank coordination
        output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(output_dir):
            try:
                safe_save_peft_adapter(
                    kwargs.get('model'),
                    output_dir,
                    accelerator=kwargs.get('accelerator'),
                    global_step=state.global_step,
                    tokenizer_or_processor=self.tokenizer_or_processor
                )
            except Exception as e:
                logger.error(f"Failed to save PEFT adapter at checkpoint {state.global_step}: {e}")
                raise


class MultimodalJointSFTTrainer(SFTTrainer):
    """Custom trainer for multimodal_joint scope that saves adapter-only checkpoints."""

    def __init__(self, *args, sft_lora_train_scope=None, model_args=None, script_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sft_lora_train_scope = sft_lora_train_scope
        self.model_args = model_args
        self.script_args = script_args

    def _save(self, output_dir=None, state_dict=None):
        """Override _save to save adapter-only for multimodal_joint."""
        if self.sft_lora_train_scope == "multimodal_joint":
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"[SFT_SAVE_DIAG] Saving multimodal_joint in-place LoRA as adapter-only checkpoint")
            logger.info(f"[SFT_SAVE_DIAG] output_dir = {output_dir}")

            safe_save_peft_adapter(
                model=self.model,
                output_dir=output_dir,
                accelerator=self.accelerator,
                global_step=self.state.global_step,
                tokenizer_or_processor=self.processing_class if hasattr(self, 'processing_class') else None,
            )

            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

            logger.info(f"[SFT_SAVE_DIAG] multimodal_joint adapter-only checkpoint saved to {output_dir}")
        else:
            super()._save(output_dir=output_dir, state_dict=state_dict)


class DebugSFTTrainer(SFTTrainer):
    """Debug trainer for diagnosing loss shape issues."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._debug_printed = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to print loss diagnostics."""
        if not self._debug_printed:
            logger.info("[SFT_LOSS_SHAPE_DEBUG] ===== Loss Shape Diagnostics =====")
            logger.info(f"[SFT_LOSS_SHAPE_DEBUG] inputs.keys() = {inputs.keys()}")

            if 'input_ids' in inputs:
                logger.info(f"[SFT_LOSS_SHAPE_DEBUG] input_ids.shape = {inputs['input_ids'].shape}")
            if 'labels' in inputs:
                logger.info(f"[SFT_LOSS_SHAPE_DEBUG] labels.shape = {inputs['labels'].shape}")
            if 'attention_mask' in inputs:
                logger.info(f"[SFT_LOSS_SHAPE_DEBUG] attention_mask.shape = {inputs['attention_mask'].shape}")
            if 'input_features' in inputs:
                logger.info(f"[SFT_LOSS_SHAPE_DEBUG] input_features.shape = {inputs['input_features'].shape}")
            if 'feature_attention_mask' in inputs:
                logger.info(f"[SFT_LOSS_SHAPE_DEBUG] feature_attention_mask.shape = {inputs['feature_attention_mask'].shape}")
            if 'pixel_values_videos' in inputs:
                logger.info(f"[SFT_LOSS_SHAPE_DEBUG] pixel_values_videos.shape = {inputs['pixel_values_videos'].shape}")
            if 'pixel_values' in inputs:
                logger.info(f"[SFT_LOSS_SHAPE_DEBUG] pixel_values.shape = {inputs['pixel_values'].shape}")

            # Call parent compute_loss to get loss and outputs
            try:
                loss, outputs = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                    num_items_in_batch=num_items_in_batch,
                )
            except TypeError:
                loss, outputs = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                )

            logger.info(f"[SFT_LOSS_SHAPE_DEBUG] outputs type = {type(outputs)}")
            if hasattr(outputs, 'keys'):
                logger.info(f"[SFT_LOSS_SHAPE_DEBUG] outputs.keys() = {outputs.keys()}")
            elif isinstance(outputs, dict):
                logger.info(f"[SFT_LOSS_SHAPE_DEBUG] outputs.keys() = {outputs.keys()}")

            logger.info(f"[SFT_LOSS_SHAPE_DEBUG] loss type = {type(loss)}")
            logger.info(f"[SFT_LOSS_SHAPE_DEBUG] loss.shape = {loss.shape}")
            logger.info(f"[SFT_LOSS_SHAPE_DEBUG] loss.dim() = {loss.dim()}")
            logger.info(f"[SFT_LOSS_SHAPE_DEBUG] loss.requires_grad = {loss.requires_grad}")

            if hasattr(outputs, 'logits'):
                logger.info(f"[SFT_LOSS_SHAPE_DEBUG] logits.shape = {outputs.logits.shape}")
            elif isinstance(outputs, dict) and 'logits' in outputs:
                logger.info(f"[SFT_LOSS_SHAPE_DEBUG] logits.shape = {outputs['logits'].shape}")

            logger.info("[SFT_LOSS_SHAPE_DEBUG] ===== End Diagnostics =====")
            self._debug_printed = True

            raise RuntimeError("[SFT_LOSS_SHAPE_DEBUG] stop after printing loss diagnostics")

        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

def check_if_video_has_audio(video_path):
    try:
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            return False
        return True
    except:
        return False

def resolve_media_path(path, image_root=None):
    """
    Resolve media path (image/video/audio) to absolute path.

    Args:
        path: Original path (can be absolute or relative)
        image_root: Root directory for media files (from IMAGE_ROOT env or script_args)
                   Should point to: os.environ.get("DATA_ROOT") + "/".../data/videos/videos

    Returns:
        Absolute path to the media file

    Strategy:
        1. If path is already absolute, return as-is
        2. If path starts with "videos/", strip it and join with image_root
           (because image_root already contains "videos/videos")
        3. If path starts with dataset name (e.g., "MER24/"), directly join with image_root
        4. Otherwise, directly join with image_root
    """
    if path is None:
        return None

    # If already absolute path, return as-is
    if os.path.isabs(path):
        return path

    # If no image_root is provided, return the relative path unchanged.
    if image_root is None:
        return path

    # Handle the "videos/" prefix used by dataset paths.
    # we need to strip the leading "videos/" to avoid duplication
    if path.startswith("videos/"):
        # Strip the "videos/" prefix and join with image_root.
        # result: os.environ.get("DATA_ROOT") + "/".../data/videos/videos/MER24/sample.mp4
        path = path[len("videos/"):]
        resolved = os.path.join(image_root, path)
        return resolved

    # Handle direct dataset paths like "MER24/..."
    # Directly join with image_root
    resolved = os.path.join(image_root, path)
    return resolved

logger = logging.getLogger(__name__)
from dataclasses import dataclass

@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: str = field(default=None, metadata={"help": "The root directory of the image."})
    use_audio_in_video: bool = field(default=True)

    # Minimum direct truncation parameters (processor layer)
    max_image_pixels: int = field(default=200704, metadata={"help": "Maximum pixels for image processing"})
    min_image_pixels: int = field(default=3136, metadata={"help": "Minimum pixels for image processing"})
    max_video_pixels: int = field(default=200704, metadata={"help": "Maximum pixels for video processing"})
    min_video_pixels: int = field(default=3136, metadata={"help": "Minimum pixels for video processing"})
    video_num_frames: int = field(default=16, metadata={"help": "Number of frames to sample from video"})
    max_text_length: int = field(default=1024, metadata={"help": "Maximum text length for truncation"})

@dataclass
class SFTModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    use_peft: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj"])

processor = None

SYSTEM_PROMPT = """You are a helpful assistant dedicated to multimodal emotion interpretation. Always ground your responses in concrete signals from images, videos, audio, and text.

Provide outputs in the fixed order <context></context>, <think></think>, <answer></answer>. In <context>, first list the following fields exactly in this order before adding any extra detail:
Speaker emotion:
Emotion intensity:
Visual evidence:
Acoustic evidence:
Textual evidence:
Emotion trajectory:
Target of emotion:
Social relation:
Possible masking/sarcasm/deception:

Do not give generic scene summaries. Prioritize summarizing emotional states and the supporting evidence from each modality.

Within <think>, reason only by explicitly referencing the emotional evidence noted in <context>; double-check coherence before concluding.

Within <answer>, give the final response that stays consistent with the earlier emotional evidence and reasoning.
"""




class LazySupervisedDataset(Dataset):
   

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "emer_ov": " Please provide the words to describe emotions within the  <answer> </answer> tags.",
        "emer_ov_mc": " Please provide only the single or multiple option letter (e.g., A for single option or A,E for multi option, etc.) within the <answer> </answer> tags.",


    }
    def __init__(self, data_path: str, script_args: ScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r", encoding="utf-8") as file:
                yaml_text = os.path.expandvars(file.read())
            if "${DATA_ROOT}" in yaml_text:
                raise RuntimeError(f"DATA_ROOT must be set before loading YAML config: {data_path}")
            yaml_data = yaml.safe_load(yaml_text)
            datasets = yaml_data.get("datasets")
            # file should be in the format of:
            # datasets:
            #   - json_path: xxxx1.json
            #     sampling_strategy: first:1000
            #   - json_path: xxxx2.json
            #     sampling_strategy: end:3000
            #   - json_path: xxxx3.json
            #     sampling_strategy: random:999
            #     data_root: xxxx/xx

            for data in datasets:
                json_path = data.get("json_path")
                sampling_strategy = data.get("sampling_strategy", "all")
                sampling_number = None

                if json_path.endswith(".jsonl"):
                    cur_data_dict = []
                    with open(json_path, "r") as json_file:
                        for line in json_file:
                            cur_data_dict.append(json.loads(line.strip()))
                elif json_path.endswith(".json"):
                    with open(json_path, "r") as json_file:
                        cur_data_dict = json.load(json_file)
                else:
                    raise ValueError(f"Unsupported file type: {json_path}")

                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                    else:
                        sampling_number = int(sampling_number)

                # Apply the sampling strategy
                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    random.shuffle(cur_data_dict)
                    cur_data_dict = cur_data_dict[:sampling_number]

                # Attach per-dataset data_root to each sample
                dataset_data_root = data.get("data_root", None)
                if dataset_data_root:
                    for sample in cur_data_dict:
                        sample["_dataset_data_root"] = dataset_data_root

                print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

        self.mel_size = 128

        # Read media limiting parameters from environment variables
        self.video_nframes = int(os.environ.get("HUMANOMNI_VIDEO_NFRAMES", "16"))
        self.video_max_pixels = int(os.environ.get("HUMANOMNI_VIDEO_MAX_PIXELS", "200704"))
        self.image_max_pixels = int(os.environ.get("HUMANOMNI_IMAGE_MAX_PIXELS", "200704"))
        self.frames_upbound = self.video_nframes

        # Print diagnostic info once on rank 0
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except:
            rank = 0

        if rank == 0 and not hasattr(LazySupervisedDataset, '_media_limit_printed'):
            print(f"[MEDIA_LIMIT] image_max_pixels={self.image_max_pixels}", flush=True)
            print(f"[MEDIA_LIMIT] video_nframes={self.video_nframes}", flush=True)
            print(f"[MEDIA_LIMIT] video_max_pixels={self.video_max_pixels}", flush=True)
            LazySupervisedDataset._media_limit_printed = True

    def __len__(self):
        return len(self.list_data_dict)


    

     

    def _make_conversation_image_and_video(self, example, use_audio_in_video=False):
        # Resolve media paths before constructing messages
        # Priority: per-dataset data_root > global image_root
        dataset_data_root = example.get("_dataset_data_root", None)
        image_root = dataset_data_root if dataset_data_root else getattr(self.script_args, 'image_root', None)

        # Resolve all media path fields uniformly
        if "path" in example and isinstance(example["path"], str):
            example["path"] = resolve_media_path(example["path"], image_root)

        if "video" in example and isinstance(example["video"], str):
            example["video"] = resolve_media_path(example["video"], image_root)

        if "image" in example and isinstance(example["image"], str):
            example["image"] = resolve_media_path(example["image"], image_root)

        if "audio" in example and isinstance(example["audio"], str):
            example["audio"] = resolve_media_path(example["audio"], image_root)

        if example["problem_type"] == 'multiple choice' or  example["problem_type"] == 'emer_ov_mc':
            question = example['problem'] + "Options:\n"
            # Handle options field which may be missing, list, or dict
            options = example.get("options") or example.get("choices") or []
            if isinstance(options, dict):
                # Convert dict to sorted list
                options = [f"{k}. {v}" for k, v in sorted(options.items())]
            if isinstance(options, list) and options:
                for op in options:
                    question += str(op) + "\n"
        else:
            question = example['problem']

        # Validate solution field
        solution = example.get("solution", "")
        if solution is None:
            solution = ""
        if not isinstance(solution, str):
            solution = str(solution)
        solution = solution.strip()

        if not solution:
            raise ValueError(
                f"Empty solution in SFT sample: sample_id={example.get('sample_id')}, "
                f"problem_type={example.get('problem_type')}, source_file={example.get('source_file')}"
            )

        # Do not assert <think>; teacher-anchored SFT may contain raw teacher responses.
        # Keep solution as assistant content regardless of format.

        text_prompt =  f"{question}\n" + self.TYPE_TEMPLATE[example['problem_type']]
        if use_audio_in_video:
            if isinstance(example['path'], str):
                video_audio_avaliable = check_if_video_has_audio(example['path']) and example['data_type'] == "video"
                if video_audio_avaliable:
                    msg =[{
                            "role": "user",
                            "content": [
                                {
                                    "type": example['data_type'],
                                    example['data_type']: example['path']
                                },
                                {
                                "type": "audio",
                                "audio": example['path']
                                },
                                {
                                    "type": "text",
                                    "text": f"Here is a {example['data_type']}, with the audio from the video.\n" + text_prompt
                                }
                                ]
                        }]
                    
                else:
                    msg =[{
                            "role": "user",
                            "content": [
                                {
                                    "type": example['data_type'],
                                    example['data_type']: example['path']
                                },
                                {
                                    "type": "text",
                                    "text": f"Here is the {example['data_type']}, and there is no audio information, you don't need to process the audio.\n" + text_prompt
                                }
                                ]
                        }]
            else:
                msg =[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": example['path']["image"]
                                },
                                {
                                    "type": "audio",
                                    "audio": example['path']["audio"]
                                },
                                {
                                    "type": "text",
                                    "text": f"Here is the image, with the coresponding audio.\n" + text_prompt
                                }
                                ]
                        }]
        else:
            # Build media content dict with appropriate parameters
            media_content = {
                "type": example['data_type'],
                example['data_type']: example['path'],
            }

            # Add max_pixels for both image and video
            if example['data_type'] == 'image':
                media_content["max_pixels"] = self.image_max_pixels
            elif example['data_type'] == 'video':
                media_content["nframes"] = self.video_nframes
                media_content["max_pixels"] = self.video_max_pixels

            msg =[{
                        "role": "user",
                        "content": [
                            media_content,
                            {
                                "type": "text",
                                "text": text_prompt
                            }
                            ]
                    }]
        msg.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": example['solution']  #example['answer']  #
                }
                ]
        })

        msg.insert(0, {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    }
                    ]
            })
        # print(msg)
        
        return msg

    def __getitem__(self, i):
        """Get item with retry for bad media samples only."""
        MAX_BAD_SAMPLE_RETRY_ATTEMPTS = 16

        try:
            if hasattr(i, 'item'):
                i = i.item()
            elif isinstance(i, str):
                if i.isdigit() or (i.startswith('-') and i[1:].isdigit()):
                    i = int(i)
                else:
                    raise TypeError(f"LazySupervisedDataset received non-numeric string index: '{i}'.")
            i = int(i)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Invalid index type for LazySupervisedDataset: {type(i).__name__} = {i}") from e

        original_idx = i
        for retry_attempt in range(MAX_BAD_SAMPLE_RETRY_ATTEMPTS):
            try:
                return self._get_item(i)
            except RuntimeError as e:
                if "Bad media sample" in str(e):
                    if retry_attempt < MAX_BAD_SAMPLE_RETRY_ATTEMPTS - 1:
                        next_idx = (i + 1) % len(self)
                        logger.warning(f"Skipping bad media sample idx={i}, retry {retry_attempt + 1}/{MAX_BAD_SAMPLE_RETRY_ATTEMPTS}, next_idx={next_idx}, error={e}")
                        i = next_idx
                    else:
                        logger.error(f"Failed to fetch a valid sample after {MAX_BAD_SAMPLE_RETRY_ATTEMPTS} retries. Last bad sample idx={original_idx}, error={e}")
                        raise RuntimeError(f"Failed to fetch a valid sample after {MAX_BAD_SAMPLE_RETRY_ATTEMPTS} retries starting from idx={original_idx}") from e
                else:
                    raise


        

    def _get_item(self, i):
        source = self.list_data_dict[i]


        messages  = self._make_conversation_image_and_video(source, use_audio_in_video=self.script_args.use_audio_in_video)
        try:
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        except Exception as e:
            video_path = "unknown"
            if isinstance(source, dict) and "video" in source:
                video_path = source.get("video", "unknown")
            elif isinstance(source, dict) and "image" in source:
                video_path = source.get("image", "unknown")

            error_msg = f"Bad media sample at idx={i}, video={video_path}, err={type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Debug: inspect videos structure for first few samples with videos
        global _videos_debug_count
        if videos is not None and len(videos) > 0 and _videos_debug_count < _videos_debug_max:
            _videos_debug_count += 1
            video_elem = videos[0]
            debug_info = f"[DEBUG videos] sample_idx={i}, type(videos)={type(videos).__name__}, len(videos)={len(videos)}, "
            debug_info += f"type(videos[0])={type(video_elem).__name__}"

            if isinstance(video_elem, str):
                debug_info += f", is_path={video_elem.startswith(('/', '.', 'http'))}, repr_prefix={repr(video_elem[:100])}"
            elif isinstance(video_elem, (list, tuple)):
                debug_info += f", len(videos[0])={len(video_elem)}"
                if len(video_elem) > 0:
                    debug_info += f", type(videos[0][0])={type(video_elem[0]).__name__}"
            elif hasattr(video_elem, 'shape'):
                debug_info += f", shape={video_elem.shape}, ndim={video_elem.ndim}"
            elif hasattr(video_elem, '__len__'):
                try:
                    debug_info += f", len={len(video_elem)}"
                except:
                    pass

            logger.info(debug_info)

        return {
            'images': images,
            'audios': audios,
            'videos': videos,

            'messages': messages,
        #
        }



def _truncate_messages_prefix(messages, max_tokens, tokenizer):
    """Truncate messages to keep only prefix up to max_tokens."""
    import copy
    truncated = copy.deepcopy(messages)
    token_count = 0

    for msg in truncated:
        if isinstance(msg, dict) and 'content' in msg:
            content = msg['content']
            if isinstance(content, str):
                tokens = tokenizer.encode(content, add_special_tokens=False)
                if token_count + len(tokens) <= max_tokens:
                    token_count += len(tokens)
                else:
                    remaining = max_tokens - token_count
                    if remaining > 0:
                        truncated_tokens = tokens[:remaining]
                        msg['content'] = tokenizer.decode(truncated_tokens)
                    else:
                        msg['content'] = ""
                    break
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        tokens = tokenizer.encode(item['text'], add_special_tokens=False)
                        if token_count + len(tokens) <= max_tokens:
                            token_count += len(tokens)
                        else:
                            remaining = max_tokens - token_count
                            if remaining > 0:
                                truncated_tokens = tokens[:remaining]
                                item['text'] = tokenizer.decode(truncated_tokens)
                            else:
                                item['text'] = ""
                            break

    return truncated


def _truncate_media_prefix(images, videos, audios, media_budget_ratio):
    """Truncate media with uniform temporal sampling for videos.

    For Tensor videos with shape [T, C, H, W], uniformly sample frames across time.
    media_budget_ratio: float between 0 and 1, indicating what fraction of frames to keep.
    """
    truncated_images = list(images) if images is not None else None
    truncated_videos = list(videos) if videos is not None else None
    truncated_audios = list(audios) if audios is not None else None

    # Uniform temporal sampling for 4D Tensor videos
    if truncated_videos is not None:
        for i, video in enumerate(truncated_videos):
            if isinstance(video, torch.Tensor) and video.ndim == 4:
                T = video.shape[0]
                keep_frames = max(1, int(T * media_budget_ratio))

                if keep_frames >= T:
                    truncated_videos[i] = video
                else:
                    # Uniformly sample keep_frames indices across [0, T-1]
                    indices = torch.linspace(0, T - 1, steps=keep_frames, device=video.device).long()
                    truncated_videos[i] = video[indices].contiguous()

    # Truncate images and audios by list length (fallback for non-Tensor types)
    if truncated_images is not None and isinstance(truncated_images, list):
        keep_count = max(1, int(len(truncated_images) * media_budget_ratio))
        truncated_images = truncated_images[:keep_count]

    if truncated_audios is not None and isinstance(truncated_audios, list):
        keep_count = max(1, int(len(truncated_audios) * media_budget_ratio))
        truncated_audios = truncated_audios[:keep_count]

    return truncated_images, truncated_videos, truncated_audios


def collate_fn_factory(processor, max_text_length, dataset=None):
    """Factory function to create collate_fn with sample-level text and media prefix truncation and resampling."""
    def collate_fn(examples):
        """Collate function with adaptive text and media budget truncation for overlong samples."""
        TEXT_BUDGET_SEQUENCE = [1024, 768, 512]
        MEDIA_BUDGET_SEQUENCE = [1.0, 0.75, 0.5]
        MAX_FINAL_LENGTH = 12000
        MAX_RESAMPLING_ATTEMPTS = 50

        processed_examples = []
        resampling_attempts = 0

        current_examples = list(examples)

        while len(processed_examples) == 0 and resampling_attempts < MAX_RESAMPLING_ATTEMPTS:
            if resampling_attempts > 0 and dataset is not None:
                # Resample new examples from dataset
                import random
                new_indices = [random.randint(0, len(dataset) - 1) for _ in range(len(examples))]
                current_examples = [dataset[idx] for idx in new_indices]
                logger.info(f"Resampling attempt {resampling_attempts}: got new examples from dataset")

            for example_idx, each in enumerate(current_examples):
                messages = each["messages"]
                images = each.get("images")
                videos = each.get("videos")
                audios = each.get("audios")

                final_sample = None
                best_config = None

                # Try text budgets first (with full media)
                for text_budget in TEXT_BUDGET_SEQUENCE:
                    try:
                        truncated_messages = _truncate_messages_prefix(messages, text_budget, processor.tokenizer)
                        texts = processor.apply_chat_template([truncated_messages], tokenize=False, add_generation_prompt=False)
                        has_multimodal = images is not None or videos is not None or audios is not None

                        if has_multimodal:
                            batch = processor(text=texts, images=images, audio=audios, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
                        else:
                            batch = processor(text=texts, return_tensors="pt", padding=True, max_length=text_budget, truncation=True, use_audio_in_video=False)

                        final_seq_len = batch["attention_mask"].sum(dim=1).item()
                        if final_seq_len <= MAX_FINAL_LENGTH:
                            logger.info(f"Sample {example_idx}: text_budget={text_budget}, media_budget=1.0, final_seq_len={final_seq_len} (accepted)")
                            final_sample = {'messages': truncated_messages, 'images': images, 'videos': videos, 'audios': audios}
                            best_config = {'text_budget': text_budget, 'media_budget': 1.0}
                            break
                    except Exception as e:
                        logger.warning(f"Sample {example_idx}: text_budget={text_budget}, media_budget=1.0 error: {e}")
                        continue

                # If text budgets alone don't work, try media budgets
                if final_sample is None:
                    for media_budget in MEDIA_BUDGET_SEQUENCE:
                        if media_budget == 1.0:
                            continue
                        for text_budget in TEXT_BUDGET_SEQUENCE:
                            try:
                                truncated_messages = _truncate_messages_prefix(messages, text_budget, processor.tokenizer)
                                truncated_images, truncated_videos, truncated_audios = _truncate_media_prefix(images, videos, audios, media_budget)
                                texts = processor.apply_chat_template([truncated_messages], tokenize=False, add_generation_prompt=False)
                                has_multimodal = truncated_images is not None or truncated_videos is not None or truncated_audios is not None

                                if has_multimodal:
                                    batch = processor(text=texts, images=truncated_images, audio=truncated_audios, videos=truncated_videos, return_tensors="pt", padding=True, use_audio_in_video=False)
                                else:
                                    batch = processor(text=texts, return_tensors="pt", padding=True, max_length=text_budget, truncation=True, use_audio_in_video=False)

                                final_seq_len = batch["attention_mask"].sum(dim=1).item()
                                if final_seq_len <= MAX_FINAL_LENGTH:
                                    log_msg = f"Sample {example_idx}: text_budget={text_budget}, media_budget={media_budget}, final_seq_len={final_seq_len}"
                                    log_msg += f", text_budget_sequence={TEXT_BUDGET_SEQUENCE}, accepted_under_limit={MAX_FINAL_LENGTH} (accepted)"
                                    logger.info(log_msg)

                                    final_sample = {'messages': truncated_messages, 'images': truncated_images, 'videos': truncated_videos, 'audios': truncated_audios}
                                    best_config = {'text_budget': text_budget, 'media_budget': media_budget}
                                    break
                            except Exception as e:
                                logger.warning(f"Sample {example_idx}: text_budget={text_budget}, media_budget={media_budget} error: {e}")
                                continue

                        if final_sample is not None:
                            break

                if final_sample is None:
                    # Try final fallback for this specific failing sample only
                    TEXT_BUDGET_FALLBACK = [384, 256]
                    MEDIA_BUDGET_FALLBACK = [0.125, 0.0625]
                    MAX_FINAL_LENGTH_FALLBACK = 6000

                    fallback_found = False
                    for text_budget in TEXT_BUDGET_FALLBACK:
                        if fallback_found:
                            break
                        try:
                            truncated_messages = _truncate_messages_prefix(messages, text_budget, processor.tokenizer)
                            texts = processor.apply_chat_template([truncated_messages], tokenize=False, add_generation_prompt=False)
                            has_multimodal = images is not None or videos is not None or audios is not None

                            if has_multimodal:
                                batch = processor(text=texts, images=images, audio=audios, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
                            else:
                                batch = processor(text=texts, return_tensors="pt", padding=True, max_length=text_budget, truncation=True, use_audio_in_video=False)

                            final_seq_len = batch["attention_mask"].sum(dim=1).item()
                            if final_seq_len <= MAX_FINAL_LENGTH_FALLBACK:
                                logger.info(f"Sample {example_idx}: fallback text_budget={text_budget}, media_budget=1.0, final_seq_len={final_seq_len} (accepted via fallback)")
                                final_sample = {'messages': truncated_messages, 'images': images, 'videos': videos, 'audios': audios}
                                fallback_found = True
                                break
                        except Exception as e:
                            logger.debug(f"Sample {example_idx}: fallback text_budget={text_budget} error: {e}")
                            continue

                    if not fallback_found:
                        for media_budget in MEDIA_BUDGET_FALLBACK:
                            if fallback_found:
                                break
                            for text_budget in TEXT_BUDGET_FALLBACK:
                                try:
                                    truncated_messages = _truncate_messages_prefix(messages, text_budget, processor.tokenizer)
                                    truncated_images, truncated_videos, truncated_audios = _truncate_media_prefix(images, videos, audios, media_budget)
                                    texts = processor.apply_chat_template([truncated_messages], tokenize=False, add_generation_prompt=False)
                                    has_multimodal = truncated_images is not None or truncated_videos is not None or truncated_audios is not None

                                    if has_multimodal:
                                        batch = processor(text=texts, images=truncated_images, audio=truncated_audios, videos=truncated_videos, return_tensors="pt", padding=True, use_audio_in_video=False)
                                    else:
                                        batch = processor(text=texts, return_tensors="pt", padding=True, max_length=text_budget, truncation=True, use_audio_in_video=False)

                                    final_seq_len = batch["attention_mask"].sum(dim=1).item()
                                    if final_seq_len <= MAX_FINAL_LENGTH_FALLBACK:
                                        logger.info(f"Sample {example_idx}: fallback text_budget={text_budget}, media_budget={media_budget}, final_seq_len={final_seq_len} (accepted via fallback)")
                                        final_sample = {'messages': truncated_messages, 'images': truncated_images, 'videos': truncated_videos, 'audios': truncated_audios}
                                        fallback_found = True
                                        break
                                except Exception as e:
                                    logger.debug(f"Sample {example_idx}: fallback text_budget={text_budget}, media_budget={media_budget} error: {e}")
                                    continue

                    if final_sample is None:
                        logger.warning(f"Sample {example_idx}: all budgets including fallback failed. Skipping this sample.")
                        continue

                processed_examples.append(final_sample)

            resampling_attempts += 1

        # If still no valid samples after resampling attempts
        if len(processed_examples) == 0:
            raise RuntimeError(f"All samples in batch failed truncation after {resampling_attempts} resampling attempts (normal budgets, fallback, and resampling). Unable to construct valid batch. Consider filtering dataset for extremely long samples.")
            messages = each["messages"]
            images = each.get("images")
            videos = each.get("videos")
            audios = each.get("audios")

            final_sample = None
            best_config = None

            # Try text budgets first (with full media)
            for text_budget in TEXT_BUDGET_SEQUENCE:
                try:
                    truncated_messages = _truncate_messages_prefix(messages, text_budget, processor.tokenizer)
                    texts = processor.apply_chat_template([truncated_messages], tokenize=False, add_generation_prompt=False)
                    has_multimodal = images is not None or videos is not None or audios is not None

                    if has_multimodal:
                        batch = processor(text=texts, images=images, audio=audios, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
                    else:
                        batch = processor(text=texts, return_tensors="pt", padding=True, max_length=text_budget, truncation=True, use_audio_in_video=False)

                    final_seq_len = batch["attention_mask"].sum(dim=1).item()
                    if final_seq_len <= MAX_FINAL_LENGTH:
                        logger.info(f"Sample {example_idx}: text_budget={text_budget}, media_budget=1.0, final_seq_len={final_seq_len} (accepted)")
                        final_sample = {'messages': truncated_messages, 'images': images, 'videos': videos, 'audios': audios}
                        best_config = {'text_budget': text_budget, 'media_budget': 1.0}
                        break
                except Exception as e:
                    logger.warning(f"Sample {example_idx}: text_budget={text_budget}, media_budget=1.0 error: {e}")
                    continue

            # If text budgets alone don't work, try media budgets
            if final_sample is None:
                for media_budget in MEDIA_BUDGET_SEQUENCE:
                    if media_budget == 1.0:
                        continue
                    for text_budget in TEXT_BUDGET_SEQUENCE:
                        try:
                            truncated_messages = _truncate_messages_prefix(messages, text_budget, processor.tokenizer)
                            truncated_images, truncated_videos, truncated_audios = _truncate_media_prefix(images, videos, audios, media_budget)
                            texts = processor.apply_chat_template([truncated_messages], tokenize=False, add_generation_prompt=False)
                            has_multimodal = truncated_images is not None or truncated_videos is not None or truncated_audios is not None

                            if has_multimodal:
                                batch = processor(text=texts, images=truncated_images, audio=truncated_audios, videos=truncated_videos, return_tensors="pt", padding=True, use_audio_in_video=False)
                            else:
                                batch = processor(text=texts, return_tensors="pt", padding=True, max_length=text_budget, truncation=True, use_audio_in_video=False)

                            final_seq_len = batch["attention_mask"].sum(dim=1).item()
                            if final_seq_len <= MAX_FINAL_LENGTH:
                                # Detect uniform temporal sampling for 4D Tensor videos
                                sampling_mode = None
                                original_num_frames = None
                                kept_frames = None
                                sampled_indices_head = None
                                sampled_indices_tail = None

                                if videos is not None and truncated_videos is not None:
                                    for orig_video, trunc_video in zip(videos, truncated_videos):
                                        if isinstance(orig_video, torch.Tensor) and orig_video.ndim == 4:
                                            original_num_frames = orig_video.shape[0]
                                            if isinstance(trunc_video, torch.Tensor) and trunc_video.ndim == 4:
                                                kept_frames = trunc_video.shape[0]
                                                if kept_frames < original_num_frames:
                                                    sampling_mode = "uniform_temporal"
                                                    # Reconstruct sampled indices for logging
                                                    indices = torch.linspace(0, original_num_frames - 1, steps=kept_frames, device=orig_video.device).long()
                                                    sampled_indices_head = indices[:3].tolist()
                                                    sampled_indices_tail = indices[-3:].tolist()
                                            break

                                log_msg = f"Sample {example_idx}: text_budget={text_budget}, media_budget={media_budget}, final_seq_len={final_seq_len}"
                                if sampling_mode:
                                    log_msg += f", sampling_mode={sampling_mode}, original_num_frames={original_num_frames}, kept_frames={kept_frames}, sampled_indices_head={sampled_indices_head}, sampled_indices_tail={sampled_indices_tail}"
                                log_msg += f", text_budget_sequence={TEXT_BUDGET_SEQUENCE}, accepted_under_limit={MAX_FINAL_LENGTH} (accepted)"
                                logger.info(log_msg)

                                final_sample = {'messages': truncated_messages, 'images': truncated_images, 'videos': truncated_videos, 'audios': truncated_audios}
                                best_config = {'text_budget': text_budget, 'media_budget': media_budget}
                                break
                        except Exception as e:
                            logger.warning(f"Sample {example_idx}: text_budget={text_budget}, media_budget={media_budget} error: {e}")
                            continue

                    if final_sample is not None:
                        break

            if final_sample is None:
                # Try final fallback for this specific failing sample only
                TEXT_BUDGET_FALLBACK = [384, 256]
                MEDIA_BUDGET_FALLBACK = [0.25, 0.125]
                MAX_FINAL_LENGTH_FALLBACK = 6000

                fallback_found = False
                for text_budget in TEXT_BUDGET_FALLBACK:
                    if fallback_found:
                        break
                    try:
                        truncated_messages = _truncate_messages_prefix(messages, text_budget, processor.tokenizer)
                        texts = processor.apply_chat_template([truncated_messages], tokenize=False, add_generation_prompt=False)
                        has_multimodal = images is not None or videos is not None or audios is not None

                        if has_multimodal:
                            batch = processor(text=texts, images=images, audio=audios, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
                        else:
                            batch = processor(text=texts, return_tensors="pt", padding=True, max_length=text_budget, truncation=True, use_audio_in_video=False)

                        final_seq_len = batch["attention_mask"].sum(dim=1).item()
                        if final_seq_len <= MAX_FINAL_LENGTH_FALLBACK:
                            logger.info(f"Sample {example_idx}: fallback text_budget={text_budget}, media_budget=1.0, final_seq_len={final_seq_len} (accepted via fallback)")
                            final_sample = {'messages': truncated_messages, 'images': images, 'videos': videos, 'audios': audios}
                            fallback_found = True
                            break
                    except Exception as e:
                        logger.debug(f"Sample {example_idx}: fallback text_budget={text_budget} error: {e}")
                        continue

                if not fallback_found:
                    for media_budget in MEDIA_BUDGET_FALLBACK:
                        if fallback_found:
                            break
                        for text_budget in TEXT_BUDGET_FALLBACK:
                            try:
                                truncated_messages = _truncate_messages_prefix(messages, text_budget, processor.tokenizer)
                                truncated_images, truncated_videos, truncated_audios = _truncate_media_prefix(images, videos, audios, media_budget)
                                texts = processor.apply_chat_template([truncated_messages], tokenize=False, add_generation_prompt=False)
                                has_multimodal = truncated_images is not None or truncated_videos is not None or truncated_audios is not None

                                if has_multimodal:
                                    batch = processor(text=texts, images=truncated_images, audio=truncated_audios, videos=truncated_videos, return_tensors="pt", padding=True, use_audio_in_video=False)
                                else:
                                    batch = processor(text=texts, return_tensors="pt", padding=True, max_length=text_budget, truncation=True, use_audio_in_video=False)

                                final_seq_len = batch["attention_mask"].sum(dim=1).item()
                                if final_seq_len <= MAX_FINAL_LENGTH_FALLBACK:
                                    logger.info(f"Sample {example_idx}: fallback text_budget={text_budget}, media_budget={media_budget}, final_seq_len={final_seq_len} (accepted via fallback)")
                                    final_sample = {'messages': truncated_messages, 'images': truncated_images, 'videos': truncated_videos, 'audios': truncated_audios}
                                    fallback_found = True
                                    break
                            except Exception as e:
                                logger.debug(f"Sample {example_idx}: fallback text_budget={text_budget}, media_budget={media_budget} error: {e}")
                                pass

                if final_sample is None:
                    logger.warning(f"Sample {example_idx}: all budgets including fallback failed. Skipping this sample.")
                else:
                    processed_examples.append(final_sample)

            resampling_attempts += 1

        # If still no valid samples after resampling attempts
        if len(processed_examples) == 0:
            raise RuntimeError(f"All samples in batch failed truncation after {resampling_attempts} resampling attempts (normal budgets, fallback, and resampling). Unable to construct valid batch. Consider filtering dataset for extremely long samples.")

        images, videos, audios, prompts = [], [], [], []
        for each in processed_examples:
            prompts.append(each["messages"])
            if each["images"] is not None:
                images.extend(each["images"])
            if each["audios"] is not None:
                audios.extend(each["audios"])
            if each["videos"] is not None:
                videos.extend(each["videos"])

        if len(images) == 0: images = None
        if len(audios) == 0: audios = None
        if len(videos) == 0: videos = None

        texts = processor.apply_chat_template(prompts, tokenize=False, add_generation_prompt=False)
        has_multimodal = images is not None or videos is not None or audios is not None

        if has_multimodal:
            batch = processor(text=texts, images=images, audio=audios, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        else:
            batch = processor(text=texts, return_tensors="pt", padding=True, max_length=max_text_length, truncation=True, use_audio_in_video=False)

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
        audio_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
        labels[labels == image_token_id] = -100
        labels[labels == video_token_id] = -100
        labels[labels == audio_token_id] = -100

        batch["labels"] = labels

        # Final length protection: safety fuse
        # This should rarely trigger if message-level and media-level truncation work correctly
        seq_lens = batch["attention_mask"].sum(dim=1)
        max_seq_len = seq_lens.max().item()
        if max_seq_len > 32768:
            logger.error(f"Unexpected overflow after message and media-level truncation: final_seq_len={max_seq_len} > 32768. This indicates multimodal features consumed more tokens than expected.")
            raise RuntimeError(f"Unexpected overflow after message and media-level truncation: final_seq_len={max_seq_len} > 32768")

        return batch
    return collate_fn


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    print(log_level, training_args)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    ################
    # Load datasets
    ################

    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower() or "omni" in model_args.model_name_or_path.lower():

        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")

        # Apply minimum direct truncation parameters at processor layer
        logger.info("*** Applying minimum direct truncation ***")

        if hasattr(processor, 'image_processor') and processor.image_processor is not None:
            processor.image_processor.max_pixels = script_args.max_image_pixels
            processor.image_processor.min_pixels = script_args.min_image_pixels
            logger.info(f"Image processor: max_pixels={script_args.max_image_pixels}, min_pixels={script_args.min_image_pixels}")

        if hasattr(processor, 'video_processor') and processor.video_processor is not None:
            processor.video_processor.max_pixels = script_args.max_video_pixels
            processor.video_processor.min_pixels = script_args.min_video_pixels
            if hasattr(processor.video_processor, 'max_frames'):
                processor.video_processor.max_frames = script_args.video_num_frames
            if hasattr(processor.video_processor, 'num_frames'):
                processor.video_processor.num_frames = script_args.video_num_frames
            logger.info(f"Video processor: max_pixels={script_args.max_video_pixels}, min_pixels={script_args.min_video_pixels}, num_frames={script_args.video_num_frames}")

        logger.info(f"Text truncation: max_length={script_args.max_text_length}")

    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        # use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs

    # Model detection: first try path string, then check config.json
    vision_modules_keywords = []
    model_detected = False

    # Try path-based detection first
    if "Qwen2-VL" in model_args.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
        vision_modules_keywords = ['visual']
        model_detected = True
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
        vision_modules_keywords = ['visual']
        model_detected = True
    elif "qwen" in model_args.model_name_or_path.lower() and "omni" in model_args.model_name_or_path.lower():
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        model.config.vocab_size = 152064
        vision_modules_keywords = ['visual','audio_tower']
        model_detected = True

    # If path doesn't match, check model config
    if not model_detected:
        config_path = os.path.join(model_args.model_name_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            model_type = config.get("model_type", "")
            architectures = config.get("architectures", [])

            if "omni" in model_type.lower() or any("omni" in arch.lower() for arch in architectures):
                model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **model_kwargs)
                model.config.vocab_size = 152064
                vision_modules_keywords = ['visual','audio_tower']
                model_detected = True
            elif "qwen2_5_vl" in model_type.lower() or any("qwen2_5_vl" in arch.lower() for arch in architectures):
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_args.model_name_or_path, **model_kwargs
                )
                vision_modules_keywords = ['visual']
                model_detected = True
            elif "qwen2vl" in model_type.lower() or any("qwen2vl" in arch.lower() for arch in architectures):
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_args.model_name_or_path, **model_kwargs
                )
                vision_modules_keywords = ['visual']
                model_detected = True

    if not model_detected:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")


    if model_args.freeze_vision_modules:
        logger.info("Freezing vision modules...")
        for n, p in model.named_parameters():
            if any(keyword in n for keyword in vision_modules_keywords):
                p.requires_grad = False

    # [SFT_LORA_TRAIN_SCOPE] Apply LoRA training scope control
    # Supports: joint (audio_tower + model.layers), model_layers_only, audio_tower_only, multimodal_joint (audio_tower + visual + model.layers)
    sft_lora_train_scope = os.environ.get("SFT_LORA_TRAIN_SCOPE", "joint")
    logger.info(f"[SFT_LORA_TRAIN_SCOPE] Using scope: {sft_lora_train_scope}")

    # [MULTIMODAL_JOINT] In-place LoRA injection approach
    if sft_lora_train_scope == "multimodal_joint":
        logger.info("[SFT_LORA_INJECTION] multimodal_joint scope: in-place LoRA injection into audio_tower, visual, and text model.layers")

        # [VISION_MODULE_DETECTION] Detect visual/vision module
        vision_module = None
        vision_module_name = None
        vision_module_candidates = ['visual', 'vision', 'vision_tower', 'vision_model']
        for candidate in vision_module_candidates:
            if hasattr(model, candidate):
                vision_module = getattr(model, candidate)
                vision_module_name = candidate
                logger.info(f"[VISION_MODULE_DETECTION] Found vision module: model.{candidate}")
                break
            elif hasattr(model.model, candidate):
                vision_module = getattr(model.model, candidate)
                vision_module_name = f'model.{candidate}'
                logger.info(f"[VISION_MODULE_DETECTION] Found vision module: model.model.{candidate}")
                break

        if vision_module is None:
            raise RuntimeError(f"[VISION_MODULE_DETECTION] multimodal_joint scope requires a vision module, but none found in {vision_module_candidates}")

        # [SFT_VISION_LORA_TARGETS] Get vision-specific LoRA target modules
        vision_lora_target_modules_env = os.environ.get("SFT_VISION_LORA_TARGET_MODULES", None)
        if vision_lora_target_modules_env:
            vision_lora_target_modules = vision_lora_target_modules_env.split()
        else:
            vision_lora_target_modules = ["q", "v"]
        logger.info(f"[SFT_VISION_LORA_TARGETS] selected vision target_modules = {vision_lora_target_modules}")

        # [SFT_LORA_TARGETS] Helper to normalize target_modules to list
        def _normalize_lora_target_modules(x):
            """Normalize target_modules to list[str]."""
            if x is None:
                return []
            if isinstance(x, str):
                if not x.strip():
                    return []
                # Split by comma if present, otherwise single element
                if ',' in x:
                    return [t.strip() for t in x.split(',') if t.strip()]
                else:
                    return [x.strip()]
            if isinstance(x, (list, tuple)):
                return list(x)
            return list(x)

        # [SFT_LORA_TARGETS] Normalize and merge base and vision target modules
        base_lora_target_modules = model_args.lora_target_modules
        base_lora_target_modules = _normalize_lora_target_modules(base_lora_target_modules)
        vision_lora_target_modules = _normalize_lora_target_modules(vision_lora_target_modules)

        # [SFT_LORA_TARGETS] For multimodal_joint, ensure both q_proj and v_proj are included for text/audio layers
        # This is required to match adapter checkpoints that include both q_proj and v_proj
        if 'q_proj' in base_lora_target_modules and 'v_proj' not in base_lora_target_modules:
            logger.info("[SFT_LORA_TARGETS] multimodal_joint scope: auto-adding v_proj to match adapter checkpoint structure")
            base_lora_target_modules.append('v_proj')
        elif 'v_proj' in base_lora_target_modules and 'q_proj' not in base_lora_target_modules:
            logger.info("[SFT_LORA_TARGETS] multimodal_joint scope: auto-adding q_proj to match adapter checkpoint structure")
            base_lora_target_modules.append('q_proj')

        full_target_modules = list(dict.fromkeys(base_lora_target_modules + vision_lora_target_modules))
        logger.info(f"[SFT_LORA_TARGETS] final target_modules = {full_target_modules}")

        # [SFT_LORA_INJECTION] In-place LoRA injection using inject_adapter_in_model
        full_lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=full_target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
        )
        logger.info(f"[SFT_LORA_INJECTION] using in-place PEFT LoRA injection, no PeftModel wrapper")
        model = inject_adapter_in_model(full_lora_config, model, adapter_name="default")
        logger.info(f"[SFT_LORA_INJECTION] in-place LoRA injected with r={model_args.lora_r}, alpha={model_args.lora_alpha}, target_modules={full_target_modules}")

        # [SFT_LORA_FREEZE] Freeze all parameters first, then selectively unfreeze LoRA parameters
        logger.info("[SFT_LORA_FREEZE] Freezing all parameters after in-place LoRA injection...")
        for name, param in model.named_parameters():
            param.requires_grad = False

        logger.info("[SFT_LORA_FREEZE] Unfreezing only LoRA parameters in audio_tower, visual/vision, and model.layers...")
        unfrozen_count = 0
        trainable_text_lora_layer_indices = []
        for name, param in model.named_parameters():
            if "lora_" not in name:
                continue
            if "audio_tower" in name:
                param.requires_grad = True
                unfrozen_count += 1
            elif "visual" in name or "vision" in name:
                param.requires_grad = True
                unfrozen_count += 1
            elif "model.layers" in name:
                param.requires_grad = True
                unfrozen_count += 1
                # Extract layer index for diagnostics
                import re
                match = re.search(r'model\.layers\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx not in trainable_text_lora_layer_indices:
                        trainable_text_lora_layer_indices.append(layer_idx)

        trainable_text_lora_layer_indices.sort()
        logger.info(f"[SFT_LORA_FREEZE] Unfroze {unfrozen_count} LoRA parameters")
        logger.info(f"[SFT_LORA_SCOPE_DIAG] trainable_text_lora_layer_indices = {trainable_text_lora_layer_indices}")

    # [LEGACY_SCOPES] Keep old logic for joint/audio_tower_only/model_layers_only
    elif sft_lora_train_scope in ["joint", "audio_tower_only"]:
        # Inject LoRA into audio_tower
        logger.info("[SFT_LORA_TRAIN_SCOPE] Injecting LoRA into audio_tower")
        audio_lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
        )
        model.audio_tower = get_peft_model(model.audio_tower, audio_lora_config)
        logger.info(f"[SFT_LORA_TRAIN_SCOPE] audio_tower LoRA injected with r={model_args.lora_r}, alpha={model_args.lora_alpha}, target_modules={model_args.lora_target_modules}")

    if sft_lora_train_scope in ["joint", "model_layers_only"]:
        # Inject LoRA into model.layers (last 8 layers: 20-27)
        logger.info("[SFT_LORA_TRAIN_SCOPE] Injecting LoRA into model.layers[20:28]")
        for layer_idx in range(20, 28):
            layer = model.model.layers[layer_idx]
            layer_lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules,
                lora_dropout=model_args.lora_dropout,
                bias="none",
            )
            model.model.layers[layer_idx] = get_peft_model(layer, layer_lora_config)
        logger.info(f"[SFT_LORA_TRAIN_SCOPE] model.layers[20:28] LoRA injected with r={model_args.lora_r}, alpha={model_args.lora_alpha}, target_modules={model_args.lora_target_modules}")

    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False

    # [SFT_TRAINER_PEFT_CONFIG] Handle peft_config based on scope
    if sft_lora_train_scope == "multimodal_joint":
        # multimodal_joint already applied in-place PEFT manually, disable trainer peft_config
        peft_config = None
        logger.info(f"[SFT_TRAINER_PEFT_CONFIG] peft_config passed to trainer = None for multimodal_joint because in-place LoRA has already been injected manually")
    else:
        # Legacy scopes use trainer's automatic PEFT injection
        peft_config = get_peft_config(model_args)
        logger.info(f"[SFT_TRAINER_PEFT_CONFIG] peft_config passed to trainer = {peft_config}")

    # Prepare callbacks with PEFT adapter save callback
    callbacks = get_callbacks(training_args, model_args)
    if peft_config is not None or sft_lora_train_scope == "multimodal_joint":
        # Add callback for both manual PEFT (multimodal_joint) and automatic PEFT (legacy scopes)
        callbacks.append(PEFTAdapterSaveCallback(tokenizer_or_processor=processor.tokenizer))

    # [SFT_LOSS_SHAPE_DEBUG] Check if debug mode is enabled
    sft_loss_shape_debug = os.environ.get("SFT_LOSS_SHAPE_DEBUG", "0") == "1"

    # Choose trainer class based on scope and debug mode
    if sft_lora_train_scope == "multimodal_joint" and not sft_loss_shape_debug:
        logger.info("[SFT_TRAINER_CLASS] Using MultimodalJointSFTTrainer for adapter-only checkpoint saving")
        trainer = MultimodalJointSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=None,
            processing_class=processor.tokenizer,
            data_collator=collate_fn_factory(processor, script_args.max_text_length, dataset),
            peft_config=peft_config,
            callbacks=callbacks,
            sft_lora_train_scope=sft_lora_train_scope,
            model_args=model_args,
            script_args=script_args,
        )
    elif sft_loss_shape_debug:
        logger.info("[SFT_LOSS_SHAPE_DEBUG] Debug mode enabled, using DebugSFTTrainer")
        trainer = DebugSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=None,
            processing_class=processor.tokenizer,
            data_collator=collate_fn_factory(processor, script_args.max_text_length, dataset),
            peft_config=peft_config,
            callbacks=callbacks,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=None,
            processing_class=processor.tokenizer,
            data_collator=collate_fn_factory(processor, script_args.max_text_length, dataset),
            peft_config=peft_config,
            callbacks=callbacks,
        )

    # [SFT_LORA_SCOPE_LEGACY] Apply legacy scope control for joint/audio_tower_only/model_layers_only
    audio_tower_disabled = 0
    text_tower_disabled = 0
    text_tower_enabled_layers = []

    if sft_lora_train_scope in ["joint", "audio_tower_only", "model_layers_only"]:
        for name, param in trainer.model.named_parameters():
            if "lora_" in name:
                # Determine which module this LoRA parameter belongs to
                is_audio_tower = "audio_tower" in name
                is_model_layers = "model.layers" in name

                if sft_lora_train_scope == "audio_tower_only":
                    # Only train audio_tower LoRA, freeze model.layers LoRA
                    if is_audio_tower:
                        param.requires_grad = True
                    elif is_model_layers:
                        param.requires_grad = False
                        text_tower_disabled += 1
                    else:
                        param.requires_grad = False
                elif sft_lora_train_scope == "joint":
                    # Train both audio_tower and model.layers LoRA (layers 20-27)
                    if is_audio_tower:
                        param.requires_grad = True
                    elif is_model_layers:
                        import re
                        match = re.search(r'model\.layers\.(\d+)', name)
                        if match:
                            layer_num = int(match.group(1))
                            if layer_num < 20 or layer_num > 27:
                                param.requires_grad = False
                                text_tower_disabled += 1
                            else:
                                param.requires_grad = True
                                if layer_num not in text_tower_enabled_layers:
                                    text_tower_enabled_layers.append(layer_num)
                    else:
                        param.requires_grad = False
                else:  # model_layers_only (default Stage 1 behavior)
                    # Only train model.layers LoRA (layers 20-27), freeze audio_tower LoRA
                    if is_audio_tower:
                        param.requires_grad = False
                        audio_tower_disabled += 1
                    elif is_model_layers:
                        import re
                        match = re.search(r'model\.layers\.(\d+)', name)
                        if match:
                            layer_num = int(match.group(1))
                            if layer_num < 20 or layer_num > 27:
                                param.requires_grad = False
                                text_tower_disabled += 1
                            else:
                                param.requires_grad = True
                                if layer_num not in text_tower_enabled_layers:
                                    text_tower_enabled_layers.append(layer_num)
                    else:
                        param.requires_grad = False

        logger.info(f"[SFT_LORA_SCOPE] LoRA module restrictions applied for scope={sft_lora_train_scope}:")
        logger.info(f"[SFT_LORA_SCOPE]   - audio_tower LoRA disabled: {audio_tower_disabled} parameters")
        logger.info(f"[SFT_LORA_SCOPE]   - text_tower LoRA disabled: {text_tower_disabled} parameters")
        logger.info(f"[SFT_LORA_SCOPE]   - text_tower LoRA enabled layers: {sorted(set(text_tower_enabled_layers))}")

    # Count trainable parameters by module
    trainable_audio_tower_lora_count = 0
    trainable_model_layers_lora_count = 0
    trainable_vision_lora_count = 0
    trainable_non_lora_audio_count = 0
    trainable_non_lora_vision_count = 0
    trainable_non_lora_base_count = 0
    trainable_lm_head_count = 0
    trainable_embed_tokens_count = 0

    for name, p in trainer.model.named_parameters():
        if p.requires_grad:
            if "audio_tower" in name and "lora_" in name:
                trainable_audio_tower_lora_count += 1
            elif "model.layers" in name and "lora_" in name:
                trainable_model_layers_lora_count += 1
            elif ("visual" in name or "vision" in name) and "lora_" in name:
                trainable_vision_lora_count += 1
            elif "audio_tower" in name and "lora_" not in name:
                trainable_non_lora_audio_count += 1
            elif ("visual" in name or "vision" in name) and "lora_" not in name:
                trainable_non_lora_vision_count += 1
            elif "lm_head" in name:
                trainable_lm_head_count += 1
            elif "embed_tokens" in name:
                trainable_embed_tokens_count += 1
            elif "lora_" not in name:
                trainable_non_lora_base_count += 1

    logger.info(f"[SFT_LORA_SCOPE_DIAG] Trainable parameter counts:")
    logger.info(f"[SFT_LORA_SCOPE_DIAG]   - audio_tower LoRA: {trainable_audio_tower_lora_count}")
    logger.info(f"[SFT_LORA_SCOPE_DIAG]   - visual/vision LoRA: {trainable_vision_lora_count}")
    logger.info(f"[SFT_LORA_SCOPE_DIAG]   - model.layers LoRA: {trainable_model_layers_lora_count}")
    logger.info(f"[SFT_LORA_SCOPE_DIAG]   - audio_tower non-LoRA: {trainable_non_lora_audio_count}")
    logger.info(f"[SFT_LORA_SCOPE_DIAG]   - visual/vision non-LoRA: {trainable_non_lora_vision_count}")
    logger.info(f"[SFT_LORA_SCOPE_DIAG]   - base model non-LoRA: {trainable_non_lora_base_count}")
    logger.info(f"[SFT_LORA_SCOPE_DIAG]   - lm_head: {trainable_lm_head_count}")
    logger.info(f"[SFT_LORA_SCOPE_DIAG]   - embed_tokens: {trainable_embed_tokens_count}")

    # [SFT_LORA_SCOPE_CHECK] Runtime validation for multimodal_joint scope
    if sft_lora_train_scope == "multimodal_joint":
        logger.info("[SFT_LORA_SCOPE_CHECK] Validating multimodal_joint scope requirements...")

        errors = []
        if trainable_audio_tower_lora_count == 0:
            errors.append(f"audio_tower LoRA count is 0, expected > 0")
        if trainable_vision_lora_count == 0:
            errors.append(f"visual/vision LoRA count is 0, expected > 0")
        if trainable_model_layers_lora_count == 0:
            errors.append(f"model.layers LoRA count is 0, expected > 0")
        if trainable_non_lora_audio_count > 0:
            errors.append(f"audio_tower non-LoRA count is {trainable_non_lora_audio_count}, expected 0")
        if trainable_non_lora_vision_count > 0:
            errors.append(f"visual/vision non-LoRA count is {trainable_non_lora_vision_count}, expected 0")
        if trainable_non_lora_base_count > 0:
            errors.append(f"base model non-LoRA count is {trainable_non_lora_base_count}, expected 0")
        if trainable_lm_head_count > 0:
            errors.append(f"lm_head count is {trainable_lm_head_count}, expected 0")
        if trainable_embed_tokens_count > 0:
            errors.append(f"embed_tokens count is {trainable_embed_tokens_count}, expected 0")

        if errors:
            error_msg = "[SFT_LORA_SCOPE_CHECK] multimodal_joint scope validation FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            logger.info("[SFT_LORA_SCOPE_CHECK] multimodal_joint scope validation PASSED")

        # [SFT_INIT_ADAPTER] Load initial adapter weights if specified
        # Support both SFT_INIT_ADAPTER_PATH and TEACHER_ANCHORED_SFT_LOAD_ADAPTER
        sft_init_adapter_path = os.environ.get("SFT_INIT_ADAPTER_PATH", "") or os.environ.get("TEACHER_ANCHORED_SFT_LOAD_ADAPTER", "")
        if sft_init_adapter_path:
            logger.info(f"[SFT_INIT_ADAPTER] Loading initial adapter from {sft_init_adapter_path}")

            adapter_file = os.path.join(sft_init_adapter_path, "adapter_model.safetensors")
            if not os.path.exists(adapter_file):
                raise RuntimeError(f"[SFT_INIT_ADAPTER][ERROR] adapter_model.safetensors not found at {adapter_file}")

            # Load adapter weights
            from safetensors import safe_open
            adapter_state_dict = {}
            with safe_open(adapter_file, framework='pt', device='cpu') as f:
                for key in f.keys():
                    adapter_state_dict[key] = f.get_tensor(key)

            logger.info(f"[SFT_INIT_ADAPTER] adapter file keys = {len(adapter_state_dict)}")

            # Map PEFT standard keys to in-place LoRA parameter names
            # PEFT key: base_model.model.audio_tower.layers.X.self_attn.q_proj.lora_A.weight
            # In-place key: audio_tower.layers.X.self_attn.q_proj.lora_A.default.weight
            model_state_dict = dict(model.named_parameters())

            loaded_keys = 0
            missing_keys = []
            shape_mismatch = []
            loaded_audio_tower_lora = 0
            loaded_visual_or_vision_lora = 0
            loaded_model_layers_lora = 0

            for adapter_key, adapter_tensor in adapter_state_dict.items():
                # Remove base_model.model. prefix
                if adapter_key.startswith('base_model.model.'):
                    in_place_key = adapter_key.replace('base_model.model.', '')
                else:
                    in_place_key = adapter_key

                # Add .default adapter name
                # .lora_A.weight -> .lora_A.default.weight
                # .lora_B.weight -> .lora_B.default.weight
                in_place_key = in_place_key.replace('.lora_A.weight', '.lora_A.default.weight')
                in_place_key = in_place_key.replace('.lora_B.weight', '.lora_B.default.weight')
                in_place_key = in_place_key.replace('.lora_A.bias', '.lora_A.default.bias')
                in_place_key = in_place_key.replace('.lora_B.bias', '.lora_B.default.bias')

                # Find matching parameter in model
                if in_place_key in model_state_dict:
                    param = model_state_dict[in_place_key]
                    if param.shape == adapter_tensor.shape:
                        # Copy weights
                        param.data.copy_(adapter_tensor.to(param.device).to(param.dtype))
                        loaded_keys += 1

                        # Count by module
                        if 'audio_tower' in in_place_key:
                            loaded_audio_tower_lora += 1
                        elif 'visual' in in_place_key or 'vision' in in_place_key:
                            loaded_visual_or_vision_lora += 1
                        elif 'model.layers' in in_place_key:
                            loaded_model_layers_lora += 1
                    else:
                        shape_mismatch.append((adapter_key, in_place_key, adapter_tensor.shape, param.shape))
                else:
                    missing_keys.append((adapter_key, in_place_key))

            logger.info(f"[SFT_INIT_ADAPTER] loaded_keys = {loaded_keys}")
            logger.info(f"[SFT_INIT_ADAPTER] loaded audio_tower LoRA keys = {loaded_audio_tower_lora}")
            logger.info(f"[SFT_INIT_ADAPTER] loaded visual/vision LoRA keys = {loaded_visual_or_vision_lora}")
            logger.info(f"[SFT_INIT_ADAPTER] loaded model.layers LoRA keys = {loaded_model_layers_lora}")
            logger.info(f"[SFT_INIT_ADAPTER] missing keys = {len(missing_keys)}")
            logger.info(f"[SFT_INIT_ADAPTER] shape mismatch = {len(shape_mismatch)}")

            # Strict validation
            if loaded_keys != 368:
                error_msg = f"[SFT_INIT_ADAPTER][ERROR] Expected 368 loaded keys, got {loaded_keys}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            if len(missing_keys) > 0:
                error_msg = f"[SFT_INIT_ADAPTER][ERROR] Found {len(missing_keys)} missing keys:\n"
                for adapter_key, in_place_key in missing_keys[:10]:
                    error_msg += f"  adapter_key={adapter_key} -> in_place_key={in_place_key}\n"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            if len(shape_mismatch) > 0:
                error_msg = f"[SFT_INIT_ADAPTER][ERROR] Found {len(shape_mismatch)} shape mismatches:\n"
                for adapter_key, in_place_key, adapter_shape, param_shape in shape_mismatch[:10]:
                    error_msg += f"  {adapter_key} -> {in_place_key}: adapter_shape={adapter_shape}, param_shape={param_shape}\n"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            if loaded_audio_tower_lora != 128:
                error_msg = f"[SFT_INIT_ADAPTER][ERROR] Expected 128 audio_tower LoRA keys, got {loaded_audio_tower_lora}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            if loaded_visual_or_vision_lora != 128:
                error_msg = f"[SFT_INIT_ADAPTER][ERROR] Expected 128 visual/vision LoRA keys, got {loaded_visual_or_vision_lora}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            if loaded_model_layers_lora != 112:
                error_msg = f"[SFT_INIT_ADAPTER][ERROR] Expected 112 model.layers LoRA keys, got {loaded_model_layers_lora}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.info("[SFT_INIT_ADAPTER] initial adapter loaded successfully")

    # Count optimizer parameters by module
    optimizer_audio_tower_lora_count = 0
    optimizer_model_layers_lora_count = 0
    optimizer_base_layer_count = 0
    optimizer_vision_count = 0

    if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
        for group in trainer.optimizer.param_groups:
            for p in group['params']:
                # Find parameter name by matching object
                param_name = None
                for name, param in trainer.model.named_parameters():
                    if param is p:
                        param_name = name
                        break

                if param_name:
                    if "audio_tower" in param_name and "lora_" in param_name:
                        optimizer_audio_tower_lora_count += 1
                    elif "model.layers" in param_name and "lora_" in param_name:
                        optimizer_model_layers_lora_count += 1
                    elif "lora_" not in param_name and "visual" in param_name:
                        optimizer_vision_count += 1
                    elif "lora_" not in param_name:
                        optimizer_base_layer_count += 1

    # Create compatibility aliases for legacy scope validation
    trainable_base_layer_count = trainable_non_lora_base_count
    trainable_vision_count = trainable_non_lora_vision_count

    # Log diagnostic counts (legacy format for joint/audio_tower_only compatibility)
    logger.info(f"[SFT_LORA_SCOPE] Trainable parameter counts:")
    logger.info(f"[SFT_LORA_SCOPE]   trainable_audio_tower_lora_count={trainable_audio_tower_lora_count}")
    logger.info(f"[SFT_LORA_SCOPE]   trainable_model_layers_lora_count={trainable_model_layers_lora_count}")
    logger.info(f"[SFT_LORA_SCOPE]   trainable_base_layer_count={trainable_base_layer_count}")
    logger.info(f"[SFT_LORA_SCOPE]   trainable_vision_count={trainable_vision_count}")

    logger.info(f"[SFT_LORA_SCOPE] Optimizer parameter counts:")
    logger.info(f"[SFT_LORA_SCOPE]   optimizer_audio_tower_lora_count={optimizer_audio_tower_lora_count}")
    logger.info(f"[SFT_LORA_SCOPE]   optimizer_model_layers_lora_count={optimizer_model_layers_lora_count}")
    logger.info(f"[SFT_LORA_SCOPE]   optimizer_base_layer_count={optimizer_base_layer_count}")
    logger.info(f"[SFT_LORA_SCOPE]   optimizer_vision_count={optimizer_vision_count}")

    # Fatal checks based on scope (skip for multimodal_joint, which has its own validation)
    if sft_lora_train_scope == "audio_tower_only":
        if trainable_audio_tower_lora_count == 0:
            raise RuntimeError(f"[SFT_LORA_SCOPE][FATAL] audio_tower_only mode requires trainable_audio_tower_lora_count > 0, got {trainable_audio_tower_lora_count}")
        if trainable_model_layers_lora_count != 0:
            raise RuntimeError(f"[SFT_LORA_SCOPE][FATAL] audio_tower_only mode requires trainable_model_layers_lora_count == 0, got {trainable_model_layers_lora_count}")
        if trainable_base_layer_count != 0:
            raise RuntimeError(f"[SFT_LORA_SCOPE][FATAL] audio_tower_only mode requires trainable_base_layer_count == 0, got {trainable_base_layer_count}")
        if trainable_vision_count != 0:
            raise RuntimeError(f"[SFT_LORA_SCOPE][FATAL] audio_tower_only mode requires trainable_vision_count == 0, got {trainable_vision_count}")
        logger.info(f"[SFT_LORA_SCOPE] audio_tower_only mode validation passed")
    elif sft_lora_train_scope == "joint":
        if trainable_audio_tower_lora_count == 0:
            raise RuntimeError(f"[SFT_LORA_SCOPE][FATAL] joint mode requires trainable_audio_tower_lora_count > 0, got {trainable_audio_tower_lora_count}")
        if trainable_model_layers_lora_count == 0:
            raise RuntimeError(f"[SFT_LORA_SCOPE][FATAL] joint mode requires trainable_model_layers_lora_count > 0, got {trainable_model_layers_lora_count}")
        if trainable_base_layer_count != 0:
            raise RuntimeError(f"[SFT_LORA_SCOPE][FATAL] joint mode requires trainable_base_layer_count == 0, got {trainable_base_layer_count}")
        if trainable_vision_count != 0:
            raise RuntimeError(f"[SFT_LORA_SCOPE][FATAL] joint mode requires trainable_vision_count == 0, got {trainable_vision_count}")
        logger.info(f"[SFT_LORA_SCOPE] joint mode validation passed")

    # Log trainable params AFTER PEFT wrapping
    total_trainable_params = 0
    for name, p in trainer.model.named_parameters():
        if p.requires_grad:
            logger.info(f'train param: {name}')
            total_trainable_params += p.numel()
    logger.info(f"Total trainable params: {total_trainable_params}")
    logger.info(f"Training configuration summary:")
    logger.info(f"  - LoRA target modules: {model_args.lora_target_modules}")
    logger.info(f"  - SFT_LORA_TRAIN_SCOPE: {sft_lora_train_scope}")
    logger.info(f"  - Total trainable parameters: {total_trainable_params}")

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")

    # Apply input requires_grad fix for gradient checkpointing
    # Only needed when training with gradient checkpointing enabled
    if training_args.gradient_checkpointing and sft_lora_train_scope in ["audio_tower_only", "joint", "multimodal_joint"]:
        logger.info(f"[SFT_GRAD_FLOW_FIX] gradient checkpointing input-requires-grad fix enabled for {sft_lora_train_scope}")

        # Enable input_require_grads for text embeddings (for text model.layers LoRA)
        if sft_lora_train_scope == "multimodal_joint":
            if hasattr(trainer.model, "enable_input_require_grads"):
                trainer.model.enable_input_require_grads()
                logger.info("[SFT_GRAD_FLOW_FIX] enabled model.enable_input_require_grads()")
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                trainer.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                logger.info("[SFT_GRAD_FLOW_FIX] registered embedding forward hook for input requires_grad")

        # Monkey-patch the model's forward to ensure modal inputs have requires_grad
        original_forward = trainer.model.forward

        def forward_with_modal_grad_fix(*args, **kwargs):
            # Enable requires_grad for floating-point modal inputs
            for key in ["input_features", "pixel_values", "pixel_values_videos"]:
                if key in kwargs and kwargs[key] is not None:
                    modal_input = kwargs[key]
                    if isinstance(modal_input, torch.Tensor) and torch.is_floating_point(modal_input) and not modal_input.requires_grad:
                        # Create a minimal operation that preserves gradient flow
                        # This does not change the values, only enables gradient tracking
                        kwargs[key] = modal_input + 0.0 * modal_input.requires_grad_(True)

            return original_forward(*args, **kwargs)

        trainer.model.forward = forward_with_modal_grad_fix
        logger.info("[SFT_GRAD_FLOW_FIX] will set floating modal inputs requires_grad=True for keys input_features, pixel_values, pixel_values_videos")

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    # Fix: dataset is LazySupervisedDataset, not DatasetDict - use len(dataset) directly
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save PEFT adapter with diagnostics
    if peft_config is not None:
        try:
            safe_save_peft_adapter(
                trainer.model,
                training_args.output_dir,
                accelerator=trainer.accelerator,
                global_step=trainer.state.global_step,
                tokenizer_or_processor=processor.tokenizer
            )
        except Exception as e:
            logger.error(f"Failed to save PEFT adapter: {e}")
            raise

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(script_args.dataset_name),
        "dataset_tags": list(script_args.dataset_name),
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        # trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)




if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, SFTModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    logger.info(script_args, training_args, model_args)
    main(script_args, training_args, model_args)
