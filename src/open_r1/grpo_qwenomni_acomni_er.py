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

import logging
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import pathlib


from PIL import Image
from torch.utils.data import Dataset

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from open_r1.vlm_modules.qwenomni_module import QwenOmniModule
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

import whisper
import librosa
from decord import VideoReader, cpu, AudioReader
import numpy as np

from open_r1.bad_sample_handler import BadSampleTracker, SafeDatasetWrapper

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
# NOTE: Commented out because it causes ImportError with qwen2_5_omni models
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
# from typing import Tuple
# import copy
from qwen_omni_utils import process_mm_info
import av

def check_if_video_has_audio(video_path):
    try:
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            return False
        return True
    except Exception as e:
        # Catch all exceptions including av.AVError (if it exists) and file not found errors
        # Return False for any error - file doesn't exist or can't be read
        return False



logger = logging.getLogger(__name__)



# def custom_forward(
#         self,
#         hidden_states: torch.Tensor,
#         cu_seqlens: torch.Tensor,
#         rotary_pos_emb: Optional[torch.Tensor] = None,
#         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#     ) -> torch.Tensor:
#         seq_length = hidden_states.shape[0]
#         q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
#         # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
#         if position_embeddings is None:
#             logger.warning_once(
#                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#                 "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
#                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
#                 "removed and `position_embeddings` will be mandatory."
#             )
#             emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
#             cos = emb.cos().float()
#             sin = emb.sin().float()
#         else:
#             cos, sin = position_embeddings
#             # Add this
#             cos = cos.to(torch.float)
#             sin = sin.to(torch.float)
#         q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
#         q = q.squeeze(0)
#         k = k.squeeze(0)
#
#         max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
#         attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
#             seq_length, -1
#         )
#         attn_output = self.proj(attn_output)
#         return attn_output
#
# Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
        processor_name_or_path (`Optional[str]`):
            Path to load processor from. If None, will try to load from model checkpoint
            and fallback to base model if needed.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "accuracy", "context", "reasoning"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    use_affective_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to enable affective reward functions (emotion-related auxiliary rewards)"},
    )
    affective_reward_gate_mode: str = field(
        default="auto",
        metadata={"help": "Gate mode for affective rewards: 'auto' (metadata-based), 'always_on', or 'off'"},
    )
    affective_context_weight: float = field(
        default=0.2,
        metadata={"help": "Weight for affective_context_reward auxiliary reward"},
    )
    emotion_consistency_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for emotion_consistency_reward auxiliary reward"},
    )

    def __post_init__(self):
        # Override reward_funcs from environment variable if set
        if "GRPO_REWARD_FUNCS" in os.environ:
            reward_funcs_str = os.environ["GRPO_REWARD_FUNCS"]
            if reward_funcs_str:
                self.reward_funcs = [func.strip() for func in reward_funcs_str.split(",") if func.strip()]
                print(f"[ENV_OVERRIDE] reward_funcs set to {self.reward_funcs} from GRPO_REWARD_FUNCS")
            else:
                print(f"[ENV_OVERRIDE] Warning: GRPO_REWARD_FUNCS is empty, using default {self.reward_funcs}")

        # Override affective_context_weight from environment variable if set
        if "GRPO_AFFECTIVE_CONTEXT_REWARD_WEIGHT" in os.environ:
            try:
                self.affective_context_weight = float(os.environ["GRPO_AFFECTIVE_CONTEXT_REWARD_WEIGHT"])
                print(f"[ENV_OVERRIDE] affective_context_weight set to {self.affective_context_weight} from GRPO_AFFECTIVE_CONTEXT_REWARD_WEIGHT")
            except ValueError:
                print(f"[ENV_OVERRIDE] Warning: Invalid GRPO_AFFECTIVE_CONTEXT_REWARD_WEIGHT value, using default {self.affective_context_weight}")

        # Override emotion_consistency_weight from environment variable if set
        if "GRPO_EMOTION_CONSISTENCY_REWARD_WEIGHT" in os.environ:
            try:
                self.emotion_consistency_weight = float(os.environ["GRPO_EMOTION_CONSISTENCY_REWARD_WEIGHT"])
                print(f"[ENV_OVERRIDE] emotion_consistency_weight set to {self.emotion_consistency_weight} from GRPO_EMOTION_CONSISTENCY_REWARD_WEIGHT")
            except ValueError:
                print(f"[ENV_OVERRIDE] Warning: Invalid GRPO_EMOTION_CONSISTENCY_REWARD_WEIGHT value, using default {self.emotion_consistency_weight}")

    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    use_audio_in_video: Optional[bool] = field(
        default=False,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    processor_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to load processor from. If None, auto-fallback to base model."},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False




SYSTEM_PROMPT = """You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from available various modalities (image, video, audio, text context) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio or other available context information, and then proceed with an in-depth analysis related to the question.

In reasoning, It is encouraged to incorporate self-reflection and verification into your reasoning process. You are encouraged to review the image, video, audio, or other context information to ensure the answer accuracy.

Provide your understanding of the image, video, and audio between the <context> </context> tags, detail the reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags.
"""

EMOTION_SHORT_SYSTEM_PROMPT = """You are an emotion recognition assistant.
Return only:
<context>max 6 words</context><think>max 5 words</think><answer>letters only</answer>

Example format only:
<context>smiling and teasing</context><think>choose A,D,F</think><answer>A,D,F</answer>

Rules:
Use only the given option letters.
For multiple emotions, use comma-separated letters.
Do not copy the example answer unless it is correct.
Stop immediately after </answer>.
"""

SYSTEM_PROMPT_IMAGE_SINGLE_CHOICE = """You are analyzing a single image to answer a multiple-choice question.

STRICT RULES:
- This task uses only a single image
- Do not mention video
- Do not mention audio
- In <context>, use only one short sentence
- In <think>, use exactly one short sentence
- You must end with <answer>X</answer>
- X must be exactly one letter from A, B, C, or D
- Do not output anything after </answer>
- If uncertain, still choose exactly one option

Example: <context>Brows furrowed, mouth downturned.</context><think>These cues fit sadness best.</think><answer>B</answer>
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
        "judge": " Please answer Yes or No within the <answer> </answer> tags.",


    }

    def __init__(self, data_path: str, script_args: GRPOScriptArguments, question_template: str):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.question_template = question_template
        self.use_audio_in_video = script_args.use_audio_in_video

        # Get image_root for resolving relative paths
        self.image_root = script_args.image_root if script_args.image_root else None

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

                    if data.get("data_root", None):
                        for each in cur_data_dict:
                            if "path" in each:
                                if isinstance(each["path"], str):
                                    each["path"] = os.path.join(data["data_root"], each["path"])
                                elif isinstance(each["path"], dict):
                                    for k in each["path"].keys():
                                        each["path"][k] = os.path.join(data["data_root"], each["path"][k])

                    # Track data source for each sample
                    for each in cur_data_dict:
                        each["_dataset_source"] = json_path

                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            if data_path.endswith(".jsonl"):
                cur_data_dict = []
                with open(data_path, "r") as json_file:
                    for line in json_file:
                        cur_data_dict.append(json.loads(line.strip()))
            elif data_path.endswith(".json"):
                with open(data_path, "r") as json_file:
                    cur_data_dict = json.load(json_file)
            self.list_data_dict = cur_data_dict

        self.mel_size = 128

        # Read media limiting parameters from environment variables
        # Priority: LAYER2_MAX_VIDEO_FRAMES > HUMANOMNI_VIDEO_NFRAMES > default 16
        self.video_nframes = int(os.environ.get("LAYER2_MAX_VIDEO_FRAMES", os.environ.get("HUMANOMNI_VIDEO_NFRAMES", "16")))
        # Priority: LAYER2_MAX_VIDEO_PIXELS > HUMANOMNI_VIDEO_MAX_PIXELS > default 200704
        self.video_max_pixels = int(os.environ.get("LAYER2_MAX_VIDEO_PIXELS", os.environ.get("HUMANOMNI_VIDEO_MAX_PIXELS", "200704")))
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
            print(f"[MEDIA_LIMIT] source: LAYER2_MAX_VIDEO_FRAMES={os.environ.get('LAYER2_MAX_VIDEO_FRAMES', 'not set')}", flush=True)
            print(f"[MEDIA_LIMIT] source: LAYER2_MAX_VIDEO_PIXELS={os.environ.get('LAYER2_MAX_VIDEO_PIXELS', 'not set')}", flush=True)
            LazySupervisedDataset._media_limit_printed = True

    def __len__(self):
        return len(self.list_data_dict)


  

    def _resolve_media_path(self, path):
        """Resolve relative media paths to absolute paths using image_root."""
        if path is None:
            return path

        # If already absolute, return as-is (don't double-join with image_root)
        if os.path.isabs(path):
            return os.path.normpath(path)

        # If relative and image_root is available, join them
        # But only if image_root is actually set (not empty string or None)
        if self.image_root and self.image_root.strip():
            resolved = os.path.join(self.image_root, path)
            return os.path.normpath(resolved)

        # Otherwise return as-is (will likely fail later with clear error)
        return path

    def _make_conversation_image_and_video(self, example, use_audio_in_video=False):
        # Resolve media paths before building conversation
        resolved_path = example['path']
        if isinstance(resolved_path, str):
            resolved_path = self._resolve_media_path(resolved_path)
        elif isinstance(resolved_path, dict):
            resolved_path = {k: self._resolve_media_path(v) for k, v in resolved_path.items()}

        # Check if options are already in the problem text
        problem_text = example['problem']
        has_options_in_problem = False
        if example["problem_type"] == 'multiple choice' or example["problem_type"] == 'emer_ov_mc':
            # Check if problem already contains option lines (e.g., "A. xxx")
            for line in problem_text.split('\n'):
                line = line.strip()
                if line and len(line) > 2 and line[0].isalpha() and line[1] == '.':
                    has_options_in_problem = True
                    break

        # Only append options if they're not already in the problem text
        if (example["problem_type"] == 'multiple choice' or example["problem_type"] == 'emer_ov_mc') and not has_options_in_problem:
            question = example['problem'] + " Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        text_prompt =  f"{question}\n" + self.TYPE_TEMPLATE[example['problem_type']]

        if use_audio_in_video:
            if isinstance(resolved_path, str):
                video_audio_avaliable = check_if_video_has_audio(resolved_path) and example['data_type'] == "video"
                if video_audio_avaliable:
                    # Build video content with nframes and max_pixels constraints
                    video_content = {
                        "type": example['data_type'],
                        example['data_type']: resolved_path
                    }
                    if example['data_type'] == 'video':
                        video_content["nframes"] = self.video_nframes
                        video_content["max_pixels"] = self.video_max_pixels
                    elif example['data_type'] == 'image':
                        video_content["max_pixels"] = self.image_max_pixels

                    msg =[{
                            "role": "user",
                            "content": [
                                video_content,
                                {
                                "type": "audio",
                                "audio": resolved_path
                                },
                                {
                                    "type": "text",
                                    "text": f"Here is a {example['data_type']}, with the audio from the video.\n" + text_prompt
                                }
                                ]
                        }]

                else:
                    # Build video content with nframes and max_pixels constraints
                    video_content = {
                        "type": example['data_type'],
                        example['data_type']: resolved_path
                    }
                    if example['data_type'] == 'video':
                        video_content["nframes"] = self.video_nframes
                        video_content["max_pixels"] = self.video_max_pixels
                    elif example['data_type'] == 'image':
                        video_content["max_pixels"] = self.image_max_pixels

                    msg =[{
                            "role": "user",
                            "content": [
                                video_content,

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
                                    "image": resolved_path["image"],
                                    "max_pixels": self.image_max_pixels
                                },
                                {
                                    "type": "audio",
                                    "audio": resolved_path["audio"]
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
                example['data_type']: resolved_path,
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



        # Use concise prompt for single-image multiple-choice tasks
        if example['data_type'] == 'image' and example['problem_type'] == 'multiple choice':
            system_prompt_text = SYSTEM_PROMPT_IMAGE_SINGLE_CHOICE
        else:
            # Check if emotion short prompt should be used
            emotion_short_prompt_enabled = os.environ.get("GRPO_EMOTION_SHORT_PROMPT", "0") == "1"
            if emotion_short_prompt_enabled:
                # Identify task type to determine if this is an emotion-only sample
                from open_r1.vlm_modules.qwenomni_module import QwenOmniModule
                task_type = QwenOmniModule.identify_task_type(example)
                if task_type == "emotion_video":
                    system_prompt_text = EMOTION_SHORT_SYSTEM_PROMPT
                else:
                    system_prompt_text = SYSTEM_PROMPT
            else:
                system_prompt_text = SYSTEM_PROMPT

        msg.insert(0, {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system_prompt_text
                            }
                            ]
                    })

        
        return msg

    def __getitem__(self, i):
        # Format into conversation
        import traceback
        import os

        # Get max retry count from environment
        max_retries = int(os.environ.get("GRPO_DATASET_MAX_RETRY", "10"))

        # Get bad sample blacklist from environment
        bad_sample_indices_str = os.environ.get("GRPO_BAD_SAMPLE_INDICES", "")
        bad_sample_indices = set()
        if bad_sample_indices_str:
            try:
                bad_sample_indices = set(int(x.strip()) for x in bad_sample_indices_str.split(",") if x.strip())
            except ValueError:
                print(f"[GRPO_BAD_SAMPLE_SKIP] Warning: failed to parse GRPO_BAD_SAMPLE_INDICES={bad_sample_indices_str}")

        original_idx = i
        last_exception = None

        # Check if original index is in blacklist
        if original_idx in bad_sample_indices:
            print(f"[GRPO_BAD_SAMPLE_SKIP] original_index={original_idx} is in blacklist, skipping")
            # Pick a random fallback immediately
            fallback_idx = random.choice(range(len(self)))
            while fallback_idx in bad_sample_indices and fallback_idx != original_idx:
                fallback_idx = random.choice(range(len(self)))
            print(f"[GRPO_BAD_SAMPLE_SKIP] original_index={original_idx} fallback_index={fallback_idx}")
            try:
                return self._get_item(fallback_idx)
            except Exception as e:
                print(f"[GRPO_BAD_SAMPLE_SKIP] fallback_index={fallback_idx} also failed: {e}")
                last_exception = e

        # Try original sample
        try:
            return self._get_item(i)
        except Exception as e:
            print(f"[GRPO_BAD_SAMPLE_SKIP] original_index={i} failed to load")

            # Print detailed diagnostics
            source = self.list_data_dict[i]
            json_path = source.get('json_path', '<unknown>')
            original_path = source.get('path', '<no path>')
            video_path = source.get('video', '<no video>')

            print(f"[GRPO_BAD_SAMPLE_SKIP] original_index={i}")
            print(f"[GRPO_BAD_SAMPLE_SKIP] json_path={json_path}")
            print(f"[GRPO_BAD_SAMPLE_SKIP] original_path={original_path}")
            print(f"[GRPO_BAD_SAMPLE_SKIP] video_path={video_path}")
            print(f"[GRPO_BAD_SAMPLE_SKIP] error_type={type(e).__name__}")
            print(f"[GRPO_BAD_SAMPLE_SKIP] error_message={str(e)}")

            traceback.print_exc()
            last_exception = e

        # Retry with random fallback samples
        for attempt_idx in range(max_retries):
            try:
                fallback_idx = random.choice(range(len(self)))

                # Skip blacklisted indices
                if fallback_idx in bad_sample_indices:
                    print(f"[GRPO_BAD_SAMPLE_SKIP] attempt={attempt_idx} fallback_index={fallback_idx} is in blacklist, retrying")
                    continue

                print(f"[GRPO_BAD_SAMPLE_SKIP] attempt={attempt_idx} original_index={original_idx} fallback_index={fallback_idx}")
                sample = self._get_item(fallback_idx)
                return sample
            except Exception as e:
                # Print diagnostics for failed fallback
                if fallback_idx < len(self.list_data_dict):
                    fallback_source = self.list_data_dict[fallback_idx]
                    fallback_json_path = fallback_source.get('json_path', '<unknown>')
                    fallback_original_path = fallback_source.get('path', '<no path>')
                    fallback_video_path = fallback_source.get('video', '<no video>')

                    print(f"[GRPO_BAD_SAMPLE_SKIP] attempt={attempt_idx} fallback_index={fallback_idx} failed")
                    print(f"[GRPO_BAD_SAMPLE_SKIP] fallback_json_path={fallback_json_path}")
                    print(f"[GRPO_BAD_SAMPLE_SKIP] fallback_original_path={fallback_original_path}")
                    print(f"[GRPO_BAD_SAMPLE_SKIP] fallback_video_path={fallback_video_path}")
                    print(f"[GRPO_BAD_SAMPLE_SKIP] fallback_error_type={type(e).__name__}")

                traceback.print_exc()
                print(f'[GRPO_BAD_SAMPLE_SKIP] attempt={attempt_idx} fallback_index={fallback_idx} exception: {e}')
                last_exception = e

        # All retries exhausted - throw clear exception
        source = self.list_data_dict[original_idx]
        original_path = source.get('path', '<no path>')
        resolved_path = original_path
        if isinstance(original_path, str):
            resolved_path = self._resolve_media_path(original_path)
        elif isinstance(original_path, dict):
            resolved_path = {k: self._resolve_media_path(v) for k, v in original_path.items()}

        raise RuntimeError(
            f"Failed to load sample after {max_retries} retries.\n"
            f"Original sample_idx: {original_idx}\n"
            f"Original path: {original_path}\n"
            f"Resolved path: {resolved_path}\n"
            f"image_root: {self.image_root}\n"
            f"Max retries: {max_retries}\n"
            f"Bad sample blacklist: {bad_sample_indices}\n"
            f"Last exception: {last_exception}"
        )


    def _normalize_sample(self, source):
        """
        Normalize sample format to support both old and new formats.

        Old format: path, video, data_type, problem, problem_type, solution
        New format: images, videos, audios, prompt, answer, completion

        Returns normalized sample with: path, data_type, problem, problem_type, solution
        """
        normalized = {}

        # Check if this is new format (has images/videos/audios fields)
        if "images" in source or "videos" in source or "audios" in source:
            # New format: convert to old format structure
            images = source.get("images", None)
            videos = source.get("videos", None)
            audios = source.get("audios", None)

            # Determine data_type and path
            if images and len(images) > 0:
                normalized["data_type"] = "image"
                normalized["path"] = images[0] if isinstance(images, list) else images
            elif videos and len(videos) > 0:
                normalized["data_type"] = "video"
                normalized["path"] = videos[0] if isinstance(videos, list) else videos
            elif audios and len(audios) > 0:
                normalized["data_type"] = "audio"
                normalized["path"] = audios[0] if isinstance(audios, list) else audios
            else:
                # No media, treat as text-only (shouldn't happen in this task)
                normalized["data_type"] = "text"
                normalized["path"] = None

            # Map prompt -> problem
            normalized["problem"] = source.get("prompt", "")

            # Map answer -> solution (wrap in answer tags if not already)
            answer = source.get("answer", "")
            completion = source.get("completion", "")
            if completion:
                normalized["solution"] = completion
            elif answer:
                normalized["solution"] = f"<answer>{answer}</answer>"
            else:
                normalized["solution"] = ""

            # Infer problem_type from prompt and extract options if present
            prompt = normalized["problem"]
            if "Options:" in prompt:
                normalized["problem_type"] = "multiple choice"
                # Extract options from prompt (format: "A. xxx\nB. yyy\n...")
                option_lines = []
                for line in prompt.split('\n'):
                    line = line.strip()
                    if line and len(line) > 2 and line[0].isalpha() and line[1] == '.':
                        option_lines.append(line)
                normalized["options"] = option_lines if option_lines else []
            else:
                normalized["problem_type"] = "free-form"
                normalized["options"] = []

            # Copy other fields
            for key in ["id", "emotion", "metadata"]:
                if key in source:
                    normalized[key] = source[key]
        else:
            # Old format: pass through as-is
            normalized = source.copy()

        return normalized

    def _get_item(self, i):
        source = self.list_data_dict[i]

        # Normalize sample format (old vs new)
        source = self._normalize_sample(source)

        if "path" in source and source["path"] is not None:
            try:
                conversation  = self._make_conversation_image_and_video(source, use_audio_in_video=self.use_audio_in_video)
                problem_type = source["problem_type"]
                audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            except Exception as e:
                raise RuntimeError(f"Failed to process multimodal info for sample {i}: {str(e)}") from e
        else:
            raise ValueError(f"Sample {i} missing 'path' field after normalization: {source}")

        solution = source["solution"]
        return {
            'images': images,
            'audios': audios,
            'videos': videos,
            'conversation': conversation,
            'prompt': conversation,
            'solution': solution,
            "problem_type": problem_type
        }


def get_vlm_module(model_name_or_path):
    import os

    print(f"[GET_VLM_MODULE] input model_name_or_path = {model_name_or_path}", flush=True)

    lowered = model_name_or_path.lower()
    if "qwen" in lowered and "omni" in lowered:
        print(f"[GET_VLM_MODULE] selected module = QwenOmniModule", flush=True)
        return QwenOmniModule

    raise ValueError("This public release only supports Qwen2.5-Omni for GRPO training.")

def normalize_reward_func_names(reward_funcs):
    """
    Normalize reward_funcs to a list of individual reward function names.

    Handles various input formats:
    - None -> empty list
    - String "format,accuracy,affective_context,emotion_consistency" -> ["format", "accuracy", "affective_context", "emotion_consistency"]
    - List ["format,accuracy,affective_context,emotion_consistency"] -> ["format", "accuracy", "affective_context", "emotion_consistency"]
    - List ["format", "accuracy", "affective_context", "emotion_consistency"] -> same
    - Mixed list ["format,accuracy", "affective_context", "emotion_consistency"] -> ["format", "accuracy", "affective_context", "emotion_consistency"]

    Returns a deduplicated list preserving original order.
    """
    if reward_funcs is None:
        return []

    # Convert to list if string
    if isinstance(reward_funcs, str):
        reward_funcs = [reward_funcs]

    # Ensure it's a list
    if not isinstance(reward_funcs, list):
        reward_funcs = list(reward_funcs)

    # Split each element by comma and flatten
    normalized = []
    for item in reward_funcs:
        if isinstance(item, str):
            # Split by comma and strip whitespace
            parts = [part.strip() for part in item.split(',')]
            # Add non-empty parts
            for part in parts:
                if part:
                    normalized.append(part)
        else:
            # Non-string items: convert to string
            item_str = str(item).strip()
            if item_str:
                normalized.append(item_str)

    # Remove duplicates while preserving order
    seen = set()
    deduplicated = []
    for name in normalized:
        if name not in seen:
            seen.add(name)
            deduplicated.append(name)

    return deduplicated

def main(script_args, training_args, model_args):
    # Print distributed training status early
    import torch.distributed as dist
    import os

    print("=" * 70)
    print("GRPO Training Entry Point - Distributed Status")
    print("=" * 70)
    print(f"torch.distributed.is_available(): {dist.is_available()}")
    print(f"torch.distributed.is_initialized(): {dist.is_initialized()}")

    if dist.is_initialized():
        print(f"World size: {dist.get_world_size()}")
        print(f"Rank: {dist.get_rank()}")
        print(f"Local rank: {os.environ.get('LOCAL_RANK', 'not set')}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
        print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
    else:
        print("WARNING: torch.distributed not initialized yet")
        print(f"LOCAL_RANK env: {os.environ.get('LOCAL_RANK', 'not set')}")
        print(f"RANK env: {os.environ.get('RANK', 'not set')}")
        print(f"WORLD_SIZE env: {os.environ.get('WORLD_SIZE', 'not set')}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    print("=" * 70)

    # Print entry arguments (rank 0 only)
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        print("[ENTRY_ARGS] ===== GRPO Training Entry Arguments =====", flush=True)
        print(f"[ENTRY_ARGS] model_name_or_path={model_args.model_name_or_path}", flush=True)
        print(f"[ENTRY_ARGS] processor_name_or_path={script_args.processor_name_or_path}", flush=True)
        print(f"[ENTRY_ARGS] dataset_name={script_args.dataset_name}", flush=True)
        print(f"[ENTRY_ARGS] max_steps={training_args.max_steps}", flush=True)
        print(f"[ENTRY_ARGS] max_completion_length={training_args.max_completion_length}", flush=True)
        print(f"[ENTRY_ARGS] num_generations={training_args.num_generations}", flush=True)
        print(f"[ENTRY_ARGS] reward_funcs={script_args.reward_funcs}", flush=True)
        print(f"[ENTRY_ARGS] logging_steps={training_args.logging_steps}", flush=True)
        print(f"[ENTRY_ARGS] save_strategy={training_args.save_strategy}", flush=True)
        print("[ENTRY_ARGS] ===== End Entry Arguments =====", flush=True)

    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)

    # Normalize reward_funcs to handle comma-separated strings
    raw_reward_funcs = script_args.reward_funcs
    normalized_reward_funcs = normalize_reward_func_names(raw_reward_funcs)
    script_args.reward_funcs = normalized_reward_funcs

    if rank == 0:
        print(f"[GRPO_REWARD_FUNCS_NORMALIZE] raw_reward_funcs = {raw_reward_funcs}", flush=True)
        print(f"[GRPO_REWARD_FUNCS_NORMALIZE] normalized_reward_funcs = {normalized_reward_funcs}", flush=True)

    # Public ACOmni training enables format, accuracy, affective_context, and
    # emotion_consistency. Legacy context/reasoning rewards remain available
    # for compatible upstream runs.
    reward_funcs_registry = {
        "accuracy": vlm_module_cls.accuracy_reward,
        "format": vlm_module_cls.format_reward,
        "reasoning": QwenOmniModule.patial_reasoning_reward,
        "logic": QwenOmniModule.patial_reasoning_reward,
        "logical": QwenOmniModule.patial_reasoning_reward,
        "context": QwenOmniModule.patial_context_reward,
        "affective_context": vlm_module_cls.affective_context_reward,
        "emotion_consistency": vlm_module_cls.emotion_consistency_reward,
        "b500_preference": vlm_module_cls.b500_preference_reward
    }

    # Check for unknown reward function names
    unknown_rewards = [func for func in script_args.reward_funcs if func not in reward_funcs_registry]
    if unknown_rewards:
        available_rewards = list(reward_funcs_registry.keys())
        error_msg = (
            f"Unknown reward function(s): {unknown_rewards}\n"
            f"Available reward functions: {available_rewards}\n"
            f"Please check your GRPO_REWARD_FUNCS environment variable or --reward_funcs argument."
        )
        raise ValueError(error_msg)

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Build reward weights list for trainer (must match reward_funcs order)
    reward_weights_list = []
    for func_name in script_args.reward_funcs:
        if func_name == "affective_context":
            reward_weights_list.append(script_args.affective_context_weight)
        elif func_name == "emotion_consistency":
            reward_weights_list.append(script_args.emotion_consistency_weight)
        else:
            reward_weights_list.append(1.0)  # Default weight for format, accuracy, etc.

    # Set reward_weights in training_args for trainer to use
    training_args.reward_weights = reward_weights_list

    # Add affective rewards if enabled
    if script_args.use_affective_rewards:
        reward_funcs.append(vlm_module_cls.affective_context_reward)
        reward_funcs.append(vlm_module_cls.emotion_consistency_reward)
        reward_weights_list.append(script_args.affective_context_weight)
        reward_weights_list.append(script_args.emotion_consistency_weight)
        training_args.reward_weights = reward_weights_list
        print(f"[AFFECTIVE_REWARDS] Enabled with gate_mode={script_args.affective_reward_gate_mode}")
        print(f"[AFFECTIVE_REWARDS] affective_context_weight={script_args.affective_context_weight}")
        print(f"[AFFECTIVE_REWARDS] emotion_consistency_weight={script_args.emotion_consistency_weight}")

    print(f"[GRPO_REWARD_FUNCS] final_reward_funcs = {script_args.reward_funcs}")
    print(f"[REWARD_FUNCS] Base reward functions: {script_args.reward_funcs}")
    print(f"[REWARD_FUNCS] Total reward functions: {len(reward_funcs)}")
    print(f"[REWARD_FUNCS] Functions: {reward_funcs}")

    # Print individual reward weights
    for func_name, weight in zip(script_args.reward_funcs, reward_weights_list):
        print(f"[GRPO_REWARD_WEIGHTS] {func_name} = {weight}")
    # import ipdb;ipdb.set_trace()

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args, question_template=vlm_module_cls.get_question_template(task_type="rec"))

    # Wrap dataset with bad sample handler if enabled
    bad_sample_tracker = BadSampleTracker()
    if bad_sample_tracker.enable:
        logger.info("[BadSampleHandler] Enabling bad sample skip mechanism")
        dataset = SafeDatasetWrapper(dataset, bad_sample_tracker)
        logger.info(f"[BadSampleHandler] Dataset wrapped. Log path: {bad_sample_tracker.log_path}")

    # [GRPO_GRAD_CKPT_RNG] Configure gradient checkpointing RNG strategy
    grpo_grad_ckpt_preserve_rng = os.environ.get("GRPO_GRAD_CKPT_PRESERVE_RNG_STATE", None)

    if rank == 0:
        print(f"[GRPO_GRAD_CKPT_RNG] env GRPO_GRAD_CKPT_PRESERVE_RNG_STATE = {grpo_grad_ckpt_preserve_rng}", flush=True)
        logger.info(f"[GRPO_GRAD_CKPT_RNG] env GRPO_GRAD_CKPT_PRESERVE_RNG_STATE = {grpo_grad_ckpt_preserve_rng}")

    if grpo_grad_ckpt_preserve_rng == "0":
        # User explicitly requests preserve_rng_state=False
        if not hasattr(training_args, 'gradient_checkpointing_kwargs') or training_args.gradient_checkpointing_kwargs is None:
            training_args.gradient_checkpointing_kwargs = {}

        # Update kwargs to disable RNG state preservation
        training_args.gradient_checkpointing_kwargs["use_reentrant"] = False
        training_args.gradient_checkpointing_kwargs["preserve_rng_state"] = False

        if rank == 0:
            print(f"[GRPO_GRAD_CKPT_RNG] final gradient_checkpointing_kwargs = {training_args.gradient_checkpointing_kwargs}", flush=True)
            logger.info(f"[GRPO_GRAD_CKPT_RNG] final gradient_checkpointing_kwargs = {training_args.gradient_checkpointing_kwargs}")
    else:
        if rank == 0:
            current_kwargs = getattr(training_args, 'gradient_checkpointing_kwargs', None)
            print(f"[GRPO_GRAD_CKPT_RNG] final gradient_checkpointing_kwargs = {current_kwargs} (unchanged)", flush=True)
            logger.info(f"[GRPO_GRAD_CKPT_RNG] final gradient_checkpointing_kwargs = {current_kwargs} (unchanged)")

    # [GRPO_GRAD_CKPT_DISABLE] Disable full gradient checkpointing if requested (legacy, may cause OOM)
    grpo_disable_grad_ckpt = os.environ.get("GRPO_DISABLE_GRADIENT_CHECKPOINTING", None)

    if rank == 0:
        print(f"[GRPO_GRAD_CKPT_DISABLE] env GRPO_DISABLE_GRADIENT_CHECKPOINTING = {grpo_disable_grad_ckpt}", flush=True)
        logger.info(f"[GRPO_GRAD_CKPT_DISABLE] env GRPO_DISABLE_GRADIENT_CHECKPOINTING = {grpo_disable_grad_ckpt}")

    if grpo_disable_grad_ckpt == "1":
        # User explicitly requests to disable gradient checkpointing (WARNING: may cause OOM)
        training_args.gradient_checkpointing = False
        training_args.gradient_checkpointing_kwargs = None

        if rank == 0:
            print(f"[GRPO_GRAD_CKPT_DISABLE] WARNING: disabling full model gradient checkpointing may cause OOM", flush=True)
            logger.warning(f"[GRPO_GRAD_CKPT_DISABLE] WARNING: disabling full model gradient checkpointing may cause OOM")
            print(f"[GRPO_GRAD_CKPT_DISABLE] final training_args.gradient_checkpointing = {training_args.gradient_checkpointing}", flush=True)
            logger.info(f"[GRPO_GRAD_CKPT_DISABLE] final training_args.gradient_checkpointing = {training_args.gradient_checkpointing}")
            print(f"[GRPO_GRAD_CKPT_DISABLE] final training_args.gradient_checkpointing_kwargs = {training_args.gradient_checkpointing_kwargs}", flush=True)
            logger.info(f"[GRPO_GRAD_CKPT_DISABLE] final training_args.gradient_checkpointing_kwargs = {training_args.gradient_checkpointing_kwargs}")
    else:
        if rank == 0:
            print(f"[GRPO_GRAD_CKPT_DISABLE] full model gradient checkpointing not disabled (env not set to 1)", flush=True)
            logger.info(f"[GRPO_GRAD_CKPT_DISABLE] full model gradient checkpointing not disabled (env not set to 1)")

    # [GRPO_DDP_FIX] Log DDP and gradient checkpointing configuration before trainer init
    if rank == 0:
        print("[GRPO_DDP_FIX] ===== Training Configuration =====", flush=True)
        print(f"[GRPO_DDP_FIX] training_args.ddp_find_unused_parameters = {training_args.ddp_find_unused_parameters}", flush=True)
        print(f"[GRPO_DDP_FIX] training_args.gradient_checkpointing = {training_args.gradient_checkpointing}", flush=True)
        if hasattr(training_args, 'gradient_checkpointing_kwargs'):
            print(f"[GRPO_DDP_FIX] training_args.gradient_checkpointing_kwargs = {training_args.gradient_checkpointing_kwargs}", flush=True)
        else:
            print(f"[GRPO_DDP_FIX] training_args.gradient_checkpointing_kwargs = None", flush=True)
        print("[GRPO_DDP_FIX] ===== End Training Configuration =====", flush=True)

    # Initialize the GRPO trainer
    trainer = VLMGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
        processor_name_or_path=script_args.processor_name_or_path,
        bad_sample_tracker=bad_sample_tracker,
    )

    # Print trainable parameter check after trainer initialization (rank 0 only)
    if rank == 0:
        print("[TRAINABLE_CHECK] ===== Trainable Parameter Check After Trainer Init =====", flush=True)
        trainable_lora_a_count = 0
        trainable_lora_b_count = 0
        trainable_base_layer_count = 0
        trainable_audio_lora_count = 0
        trainable_total_count = 0

        for name, param in trainer.model.named_parameters():
            if param.requires_grad:
                trainable_total_count += 1
                if 'lora_A' in name:
                    trainable_lora_a_count += 1
                if 'lora_B' in name:
                    trainable_lora_b_count += 1
                if 'base_layer' in name:
                    trainable_base_layer_count += 1
                if 'audio_tower' in name and 'lora_' in name:
                    trainable_audio_lora_count += 1

        print(f"[TRAINABLE_CHECK] trainable_lora_A_count={trainable_lora_a_count}", flush=True)
        print(f"[TRAINABLE_CHECK] trainable_lora_B_count={trainable_lora_b_count}", flush=True)
        print(f"[TRAINABLE_CHECK] trainable_base_layer_count={trainable_base_layer_count}", flush=True)
        print(f"[TRAINABLE_CHECK] trainable_audio_lora_count={trainable_audio_lora_count}", flush=True)
        print(f"[TRAINABLE_CHECK] trainable_total_count={trainable_total_count}", flush=True)

        if trainable_base_layer_count > 0:
            print("[TRAINABLE_CHECK][WARNING] trainable_base_layer_count > 0, base layers should be frozen!", flush=True)

        logger.info(f"[TRAINABLE_CHECK] trainable_lora_A_count={trainable_lora_a_count}")
        logger.info(f"[TRAINABLE_CHECK] trainable_lora_B_count={trainable_lora_b_count}")
        logger.info(f"[TRAINABLE_CHECK] trainable_base_layer_count={trainable_base_layer_count}")
        logger.info(f"[TRAINABLE_CHECK] trainable_audio_lora_count={trainable_audio_lora_count}")
        logger.info(f"[TRAINABLE_CHECK] trainable_total_count={trainable_total_count}")

        if trainable_base_layer_count > 0:
            logger.warning(f"[TRAINABLE_CHECK][WARNING] base_layer is trainable: {trainable_base_layer_count} params")
        if trainable_audio_lora_count > 0:
            logger.warning(f"[TRAINABLE_CHECK][WARNING] audio_tower LoRA is trainable: {trainable_audio_lora_count} params")

        logger.info("[TRAINABLE_CHECK] ===== End Trainable Parameter Check =====")

    # [GRPO_AUDIO_CKPT_DISABLE] Disable audio_tower gradient checkpointing if requested
    grpo_disable_audio_tower_grad_ckpt = os.environ.get("GRPO_DISABLE_AUDIO_TOWER_GRADIENT_CHECKPOINTING", None)

    if rank == 0:
        print(f"[GRPO_AUDIO_CKPT_DISABLE] env GRPO_DISABLE_AUDIO_TOWER_GRADIENT_CHECKPOINTING = {grpo_disable_audio_tower_grad_ckpt}", flush=True)
        logger.info(f"[GRPO_AUDIO_CKPT_DISABLE] env GRPO_DISABLE_AUDIO_TOWER_GRADIENT_CHECKPOINTING = {grpo_disable_audio_tower_grad_ckpt}")

    if grpo_disable_audio_tower_grad_ckpt == "1":
        # Only disable audio_tower gradient checkpointing, keep main model gradient checkpointing enabled
        audio_tower_found = False
        modules_with_gradient_checkpointing = []
        modules_disabled = []

        try:
            # Try to find audio_tower in the model
            audio_tower = None
            if hasattr(trainer.model, 'audio_tower'):
                audio_tower = trainer.model.audio_tower
                audio_tower_found = True
            elif hasattr(trainer.model, 'module') and hasattr(trainer.model.module, 'audio_tower'):
                audio_tower = trainer.model.module.audio_tower
                audio_tower_found = True
            elif hasattr(trainer.model, 'base_model') and hasattr(trainer.model.base_model, 'audio_tower'):
                audio_tower = trainer.model.base_model.audio_tower
                audio_tower_found = True

            if audio_tower_found and audio_tower is not None:
                # Disable gradient checkpointing on audio_tower and its submodules
                for module_name, module in audio_tower.named_modules():
                    if hasattr(module, 'gradient_checkpointing'):
                        modules_with_gradient_checkpointing.append(module_name if module_name else 'audio_tower')
                        module.gradient_checkpointing = False
                        modules_disabled.append(module_name if module_name else 'audio_tower')

                # If audio_tower itself has gradient_checkpointing_disable method, call it
                if hasattr(audio_tower, 'gradient_checkpointing_disable'):
                    audio_tower.gradient_checkpointing_disable()
                    if rank == 0:
                        print(f"[GRPO_AUDIO_CKPT_DISABLE] audio_tower.gradient_checkpointing_disable() called", flush=True)
                        logger.info(f"[GRPO_AUDIO_CKPT_DISABLE] audio_tower.gradient_checkpointing_disable() called")

                if rank == 0:
                    print(f"[GRPO_AUDIO_CKPT_DISABLE] audio_tower found = True", flush=True)
                    logger.info(f"[GRPO_AUDIO_CKPT_DISABLE] audio_tower found = True")
                    print(f"[GRPO_AUDIO_CKPT_DISABLE] modules_with_gradient_checkpointing = {len(modules_with_gradient_checkpointing)}", flush=True)
                    logger.info(f"[GRPO_AUDIO_CKPT_DISABLE] modules_with_gradient_checkpointing = {len(modules_with_gradient_checkpointing)}")
                    print(f"[GRPO_AUDIO_CKPT_DISABLE] modules_disabled = {len(modules_disabled)}", flush=True)
                    logger.info(f"[GRPO_AUDIO_CKPT_DISABLE] modules_disabled = {len(modules_disabled)}")
                    if len(modules_disabled) > 0:
                        print(f"[GRPO_AUDIO_CKPT_DISABLE] first 10 disabled modules: {modules_disabled[:10]}", flush=True)
                        logger.info(f"[GRPO_AUDIO_CKPT_DISABLE] first 10 disabled modules: {modules_disabled[:10]}")
            else:
                if rank == 0:
                    print(f"[GRPO_AUDIO_CKPT_DISABLE] WARNING: audio_tower not found in model", flush=True)
                    logger.warning(f"[GRPO_AUDIO_CKPT_DISABLE] WARNING: audio_tower not found in model")
                    print(f"[GRPO_AUDIO_CKPT_DISABLE] audio_tower found = False", flush=True)
                    logger.info(f"[GRPO_AUDIO_CKPT_DISABLE] audio_tower found = False")

        except Exception as e:
            if rank == 0:
                print(f"[GRPO_AUDIO_CKPT_DISABLE] WARNING: failed to disable audio_tower gradient checkpointing: {e}", flush=True)
                logger.warning(f"[GRPO_AUDIO_CKPT_DISABLE] WARNING: failed to disable audio_tower gradient checkpointing: {e}")

        # Log final training_args state (should remain unchanged)
        if rank == 0:
            print(f"[GRPO_AUDIO_CKPT_DISABLE] final training_args.gradient_checkpointing = {training_args.gradient_checkpointing}", flush=True)
            logger.info(f"[GRPO_AUDIO_CKPT_DISABLE] final training_args.gradient_checkpointing = {training_args.gradient_checkpointing}")
            final_kwargs = getattr(training_args, 'gradient_checkpointing_kwargs', None)
            print(f"[GRPO_AUDIO_CKPT_DISABLE] final training_args.gradient_checkpointing_kwargs = {final_kwargs}", flush=True)
            logger.info(f"[GRPO_AUDIO_CKPT_DISABLE] final training_args.gradient_checkpointing_kwargs = {final_kwargs}")
    else:
        if rank == 0:
            print(f"[GRPO_AUDIO_CKPT_DISABLE] audio_tower gradient checkpointing not disabled (env not set to 1)", flush=True)
            logger.info(f"[GRPO_AUDIO_CKPT_DISABLE] audio_tower gradient checkpointing not disabled (env not set to 1)")

    # [GRPO_GRAD_CKPT_DISABLE] Disable full model gradient checkpointing if requested (legacy, use with caution)
    grpo_disable_grad_ckpt = os.environ.get("GRPO_DISABLE_GRADIENT_CHECKPOINTING", None)

    if grpo_disable_grad_ckpt == "1":
        # Disable gradient checkpointing on the full model (may cause OOM)
        model_grad_ckpt_disabled = False
        try:
            if hasattr(trainer.model, 'gradient_checkpointing_disable'):
                trainer.model.gradient_checkpointing_disable()
                model_grad_ckpt_disabled = True
                if rank == 0:
                    print(f"[GRPO_GRAD_CKPT_DISABLE] model.gradient_checkpointing_disable() called", flush=True)
                    logger.info(f"[GRPO_GRAD_CKPT_DISABLE] model.gradient_checkpointing_disable() called")
            elif hasattr(trainer.model, 'config') and hasattr(trainer.model.config, 'use_cache'):
                # For models without gradient_checkpointing_disable, ensure use_cache is enabled
                trainer.model.config.use_cache = True
                if rank == 0:
                    print(f"[GRPO_GRAD_CKPT_DISABLE] model.config.use_cache set to True", flush=True)
                    logger.info(f"[GRPO_GRAD_CKPT_DISABLE] model.config.use_cache set to True")
        except Exception as e:
            if rank == 0:
                print(f"[GRPO_GRAD_CKPT_DISABLE] WARNING: failed to disable model gradient checkpointing: {e}", flush=True)
                logger.warning(f"[GRPO_GRAD_CKPT_DISABLE] WARNING: failed to disable model gradient checkpointing: {e}")

        if rank == 0:
            print(f"[GRPO_GRAD_CKPT_DISABLE] model gradient checkpointing disabled = {model_grad_ckpt_disabled}", flush=True)
            logger.info(f"[GRPO_GRAD_CKPT_DISABLE] model gradient checkpointing disabled = {model_grad_ckpt_disabled}")
    else:
        if rank == 0:
            print(f"[GRPO_GRAD_CKPT_DISABLE] model gradient checkpointing disabled = False (env not set to 1)", flush=True)
            logger.info(f"[GRPO_GRAD_CKPT_DISABLE] model gradient checkpointing disabled = False (env not set to 1)")

    # [GRPO_LORA_DROPOUT] Disable LoRA dropout if requested
    grpo_disable_lora_dropout = os.environ.get("GRPO_DISABLE_LORA_DROPOUT", "0")

    if rank == 0:
        print(f"[GRPO_LORA_DROPOUT] env GRPO_DISABLE_LORA_DROPOUT = {grpo_disable_lora_dropout}", flush=True)
        logger.info(f"[GRPO_LORA_DROPOUT] env GRPO_DISABLE_LORA_DROPOUT = {grpo_disable_lora_dropout}")

    if grpo_disable_lora_dropout == "1":
        modified_lora_dropout_count = 0
        old_p_values = []

        # Iterate through all modules to find LoRA dropout
        for module_name, module in trainer.model.named_modules():
            # Check if this is a LoRA dropout module
            if isinstance(module, torch.nn.Dropout) and 'lora_dropout' in module_name.lower():
                old_p = module.p
                old_p_values.append((module_name, old_p))
                module.p = 0.0
                modified_lora_dropout_count += 1

        if rank == 0:
            print(f"[GRPO_LORA_DROPOUT] modified_lora_dropout_count = {modified_lora_dropout_count}", flush=True)
            logger.info(f"[GRPO_LORA_DROPOUT] modified_lora_dropout_count = {modified_lora_dropout_count}")

            if len(old_p_values) > 0:
                print(f"[GRPO_LORA_DROPOUT] old_p_values = {old_p_values[:10]}", flush=True)
                logger.info(f"[GRPO_LORA_DROPOUT] old_p_values = {old_p_values[:10]}")
                print(f"[GRPO_LORA_DROPOUT] new_p = 0.0", flush=True)
                logger.info(f"[GRPO_LORA_DROPOUT] new_p = 0.0")
            else:
                print(f"[GRPO_LORA_DROPOUT] WARNING: no lora_dropout modules found", flush=True)
                logger.warning(f"[GRPO_LORA_DROPOUT] WARNING: no lora_dropout modules found")
    else:
        if rank == 0:
            print(f"[GRPO_LORA_DROPOUT] LoRA dropout not modified (env not set to 1)", flush=True)
            logger.info(f"[GRPO_LORA_DROPOUT] LoRA dropout not modified (env not set to 1)")

    # Determine resume_from_checkpoint argument
    resume_arg = None

    # Priority 1: training_args.resume_from_checkpoint
    if hasattr(training_args, 'resume_from_checkpoint') and training_args.resume_from_checkpoint:
        resume_arg = training_args.resume_from_checkpoint
        logger.info(f"[GRPO_RESUME] training_args.resume_from_checkpoint = {resume_arg}")
    # Priority 2: environment variable RESUME_FROM_CHECKPOINT
    elif os.environ.get("RESUME_FROM_CHECKPOINT"):
        resume_arg = os.environ.get("RESUME_FROM_CHECKPOINT")
        logger.info(f"[GRPO_RESUME] env RESUME_FROM_CHECKPOINT = {resume_arg}")
    else:
        logger.info("[GRPO_RESUME] training_args.resume_from_checkpoint = None")
        logger.info(f"[GRPO_RESUME] env RESUME_FROM_CHECKPOINT = {os.environ.get('RESUME_FROM_CHECKPOINT', 'None')}")

    # Validate resume_arg if provided
    if resume_arg is not None:
        if not os.path.exists(resume_arg):
            raise RuntimeError(f"[GRPO_RESUME] resume_from_checkpoint path does not exist: {resume_arg}")

        from pathlib import Path
        resume_path = Path(resume_arg)

        # Check for adapter files
        has_adapter_files = (
            (resume_path / "adapter_config.json").exists()
            and (resume_path / "adapter_model.safetensors").exists()
        )

        # Check for HuggingFace Trainer state files
        has_hf_trainer_state = (
            (resume_path / "trainer_state.json").exists()
            and (
                (resume_path / "optimizer.pt").exists()
                or (resume_path / "optimizer.bin").exists()
            )
            and (resume_path / "scheduler.pt").exists()
        )

        # Check for RNG state
        has_rng_state = (
            any(resume_path.glob("rng_state*.pth"))
            or (resume_path / "rng_state.pth").exists()
        )

        # Check for DeepSpeed training state files
        has_deepspeed_state = (
            any(resume_path.glob("global_step*"))
            or any(resume_path.glob("zero_pp_rank_*"))
            or any(resume_path.glob("mp_rank_*"))
            or any(resume_path.glob("**/*optim_states.pt"))
            or any(resume_path.glob("**/*model_states.pt"))
        )

        # Determine checkpoint type
        if has_hf_trainer_state:
            checkpoint_type = "hf_trainer"
        elif has_deepspeed_state:
            checkpoint_type = "deepspeed"
        elif has_adapter_files:
            checkpoint_type = "adapter_only"
        else:
            checkpoint_type = "unknown"

        # Log checkpoint analysis
        logger.info(f"[GRPO_RESUME] resume_path = {resume_arg}")
        logger.info(f"[GRPO_RESUME] has_adapter_files = {has_adapter_files}")
        logger.info(f"[GRPO_RESUME] has_hf_trainer_state = {has_hf_trainer_state}")
        logger.info(f"[GRPO_RESUME] has_rng_state = {has_rng_state}")
        logger.info(f"[GRPO_RESUME] has_deepspeed_state = {has_deepspeed_state}")
        logger.info(f"[GRPO_RESUME] resume checkpoint type = {checkpoint_type}")

        # Determine if resume is allowed
        allow_resume = has_hf_trainer_state or has_deepspeed_state

        logger.info(f"[GRPO_RESUME] allow resume_from_checkpoint = {allow_resume}")

        # Reject adapter-only checkpoints
        if not allow_resume:
            raise RuntimeError(
                f"[GRPO_RESUME] ERROR: resume_from_checkpoint points to an adapter-only checkpoint without training state.\n"
                f"  Path: {resume_arg}\n"
                f"  Checkpoint type: {checkpoint_type}\n"
                f"  has_adapter_files: {has_adapter_files}\n"
                f"  has_hf_trainer_state: {has_hf_trainer_state}\n"
                f"  has_deepspeed_state: {has_deepspeed_state}\n"
                f"  This is an adapter initialization checkpoint (SFT output), not a GRPO resume checkpoint.\n"
                f"  To start a new GRPO run from this adapter, pass it as model_name_or_path instead.\n"
                f"  To resume GRPO training, provide a checkpoint with HuggingFace Trainer or DeepSpeed state."
            )

        logger.info(f"[GRPO_RESUME] final resume_from_checkpoint = {resume_arg}")
    else:
        logger.info("[GRPO_RESUME] final resume_from_checkpoint = None")
        logger.info("[GRPO_RESUME] starting new GRPO run from initialized policy model")

    # Apply trusted local checkpoint torch.load safety bypass if enabled
    trust_local_resume_enabled = os.environ.get("GRPO_TRUST_LOCAL_RESUME_TORCH_LOAD", "0") == "1"
    patch_applied = False

    if trust_local_resume_enabled:
        logger.info("[GRPO_TRUSTED_RESUME] GRPO_TRUST_LOCAL_RESUME_TORCH_LOAD=1 detected")
        print("[GRPO_TRUSTED_RESUME] GRPO_TRUST_LOCAL_RESUME_TORCH_LOAD=1 detected", flush=True)

        if resume_arg is not None:
            from pathlib import Path
            resume_path = Path(resume_arg)

            # Check required files
            has_trainer_state = (resume_path / "trainer_state.json").exists()
            has_optimizer_pt = (resume_path / "optimizer.pt").exists()
            has_scheduler_pt = (resume_path / "scheduler.pt").exists()

            # Determine output directory
            output_dir = os.environ.get("OUTPUT_DIR", None)
            if output_dir is None:
                output_dir = training_args.output_dir

            # Check if resume_path is under output directory
            path_is_under_output_dir = False
            try:
                resume_path_abs = resume_path.resolve()
                # Check if path contains ACOmni-public/output
                if "ACOmni-public/output" in str(resume_path_abs):
                    path_is_under_output_dir = True
                # Check if path is under OUTPUT_DIR
                elif output_dir:
                    output_dir_abs = Path(output_dir).resolve()
                    try:
                        resume_path_abs.relative_to(output_dir_abs)
                        path_is_under_output_dir = True
                    except ValueError:
                        path_is_under_output_dir = False
            except Exception as e:
                logger.warning(f"[GRPO_TRUSTED_RESUME] failed to resolve paths: {e}")
                path_is_under_output_dir = False

            # Print diagnostics
            print(f"[GRPO_TRUSTED_RESUME] enabled_env = {trust_local_resume_enabled}", flush=True)
            print(f"[GRPO_TRUSTED_RESUME] resume_path = {resume_arg}", flush=True)
            print(f"[GRPO_TRUSTED_RESUME] has_trainer_state = {has_trainer_state}", flush=True)
            print(f"[GRPO_TRUSTED_RESUME] has_optimizer_pt = {has_optimizer_pt}", flush=True)
            print(f"[GRPO_TRUSTED_RESUME] has_scheduler_pt = {has_scheduler_pt}", flush=True)
            print(f"[GRPO_TRUSTED_RESUME] output_dir = {output_dir}", flush=True)
            print(f"[GRPO_TRUSTED_RESUME] path_is_under_output_dir = {path_is_under_output_dir}", flush=True)

            logger.info(f"[GRPO_TRUSTED_RESUME] enabled_env = {trust_local_resume_enabled}")
            logger.info(f"[GRPO_TRUSTED_RESUME] resume_path = {resume_arg}")
            logger.info(f"[GRPO_TRUSTED_RESUME] has_trainer_state = {has_trainer_state}")
            logger.info(f"[GRPO_TRUSTED_RESUME] has_optimizer_pt = {has_optimizer_pt}")
            logger.info(f"[GRPO_TRUSTED_RESUME] has_scheduler_pt = {has_scheduler_pt}")
            logger.info(f"[GRPO_TRUSTED_RESUME] output_dir = {output_dir}")
            logger.info(f"[GRPO_TRUSTED_RESUME] path_is_under_output_dir = {path_is_under_output_dir}")

            # Validate all conditions
            all_conditions_met = (
                has_trainer_state
                and has_optimizer_pt
                and has_scheduler_pt
                and path_is_under_output_dir
            )

            if not all_conditions_met:
                raise RuntimeError(
                    f"[GRPO_TRUSTED_RESUME] ERROR: GRPO_TRUST_LOCAL_RESUME_TORCH_LOAD=1 but conditions not met.\n"
                    f"  resume_path: {resume_arg}\n"
                    f"  has_trainer_state: {has_trainer_state}\n"
                    f"  has_optimizer_pt: {has_optimizer_pt}\n"
                    f"  has_scheduler_pt: {has_scheduler_pt}\n"
                    f"  path_is_under_output_dir: {path_is_under_output_dir}\n"
                    f"  output_dir: {output_dir}\n"
                    f"  All conditions must be True to enable trusted torch.load bypass."
                )

            # Apply monkey patch
            def _grpo_allow_local_torch_load():
                print("[GRPO_TRUSTED_RESUME] bypassing Transformers torch.load safety check for trusted local checkpoint", flush=True)
                return None

            import transformers.utils.import_utils as import_utils
            import transformers.trainer as hf_trainer_mod

            import_utils.check_torch_load_is_safe = _grpo_allow_local_torch_load
            hf_trainer_mod.check_torch_load_is_safe = _grpo_allow_local_torch_load

            patch_applied = True
            print(f"[GRPO_TRUSTED_RESUME] patch_applied = {patch_applied}", flush=True)
            logger.info(f"[GRPO_TRUSTED_RESUME] patch_applied = {patch_applied}")
        else:
            raise RuntimeError(
                "[GRPO_TRUSTED_RESUME] ERROR: GRPO_TRUST_LOCAL_RESUME_TORCH_LOAD=1 but resume_arg is None.\n"
                "  Cannot apply torch.load bypass without a resume checkpoint path."
            )
    else:
        print(f"[GRPO_TRUSTED_RESUME] patch_applied = {patch_applied}", flush=True)
        logger.info(f"[GRPO_TRUSTED_RESUME] patch_applied = {patch_applied}")

    # Train and push the model to the Hub
    trainer.train(resume_from_checkpoint=resume_arg)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
