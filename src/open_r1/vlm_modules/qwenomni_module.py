from transformers import Qwen2_5OmniThinkerForConditionalGeneration, AutoProcessor, Qwen2_5OmniProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from qwen_omni_utils import process_mm_info

from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import os
import re
import ast
import time
from open_r1.vlm_modules.vlm_module import VLMBaseModule
import requests
import re


url = os.environ.get("API", "")
token = os.environ.get("API_KEY", "")

def gpt_api(prompt, model_name):
    # Re-read environment variables inside function to avoid stale values
    url = os.environ.get("API", "")
    token = os.environ.get("API_KEY", "")

    if not url:
        raise RuntimeError("Environment variable 'API' is not set. Please set it before calling gpt_api.")
    if not token:
        raise RuntimeError("Environment variable 'API_KEY' is not set. Please set it before calling gpt_api.")

    messages = [
                {
                    "role": "user",
                    "content": prompt

                }
            ]
    success = False
    max_try = 20
    tries = 0
    response_message = ""
    response = None
    last_error = None

    while (not success and tries < max_try):
        try:
            data = {
                    "model": model_name or "qwen-plus",
                    "messages": messages,
                    # "n": 1
                }

            headers = {
                    "Content-Type": "application/json",
                    "Authorization": 'Bearer ' + token}
            response = requests.post(url, json=data, headers=headers, timeout=15)

            # Check HTTP status code
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                print(f'[gpt_api] Non-200 status: {last_error}')
                time.sleep(1)
                tries += 1
                continue

            # Parse JSON response
            response_json = response.json()

            # Safely extract message content
            if 'choices' not in response_json or not response_json['choices']:
                last_error = f"Missing 'choices' in response. Keys: {list(response_json.keys())}"
                print(f'[gpt_api] Invalid JSON structure: {last_error}')
                time.sleep(1)
                tries += 1
                continue

            if 'message' not in response_json['choices'][0]:
                last_error = f"Missing 'message' in choices[0]. Keys: {list(response_json['choices'][0].keys())}"
                print(f'[gpt_api] Invalid choices structure: {last_error}')
                time.sleep(1)
                tries += 1
                continue

            if 'content' not in response_json['choices'][0]['message']:
                last_error = f"Missing 'content' in message. Keys: {list(response_json['choices'][0]['message'].keys())}"
                print(f'[gpt_api] Invalid message structure: {last_error}')
                time.sleep(1)
                tries += 1
                continue

            response_message = response_json['choices'][0]['message']['content']
            success = True

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            if response is not None:
                try:
                    print(f'[gpt_api] Error after receiving response: status={response.status_code}, text={response.text[:500]}, error={last_error}')
                except:
                    print(f'[gpt_api] Error: {last_error}')
            else:
                print(f'[gpt_api] Error before receiving response: {last_error}')
            time.sleep(1)
            tries += 1

    if not success:
        raise RuntimeError(f"gpt_api failed after {max_try} attempts. Last error: {last_error}")

    return response_message

class QwenOmniModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        
        return Qwen2_5OmniThinkerForConditionalGeneration
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return Qwen2_5OmniProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual','audio_tower']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'video_second_per_grid', 'feature_attention_mask', 'input_features', 'audio_feature_lengths', 'use_audio_in_video', 'rope_deltas']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return ['max_pixels', 'min_pixels']
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        # [AUDIO_DISABLE] Remove audio content from conversation when GRPO_DISABLE_AUDIO_INPUTS=1
        disable_audio = os.environ.get("GRPO_DISABLE_AUDIO_INPUTS", "0") == "1"

        rank = 0
        try:
            import torch.distributed as torch_dist
            if torch_dist.is_available() and torch_dist.is_initialized():
                rank = torch_dist.get_rank()
        except:
            rank = 0

        if disable_audio:
            # Create a copy of inputs to avoid modifying original data
            inputs_cleaned = []
            audio_entries_removed = 0
            for example in inputs:
                example_copy = dict(example)

                # Remove audio content entries from conversation if present
                if "conversations" in example_copy and isinstance(example_copy["conversations"], list):
                    cleaned_conversations = []
                    for msg in example_copy["conversations"]:
                        if isinstance(msg, dict):
                            msg_copy = dict(msg)
                            # Remove audio content entries
                            if "content" in msg_copy and isinstance(msg_copy["content"], list):
                                original_len = len(msg_copy["content"])
                                msg_copy["content"] = [
                                    item for item in msg_copy["content"]
                                    if not (isinstance(item, dict) and item.get("type") == "audio")
                                ]
                                audio_entries_removed += original_len - len(msg_copy["content"])
                            cleaned_conversations.append(msg_copy)
                        else:
                            cleaned_conversations.append(msg)
                    example_copy["conversations"] = cleaned_conversations

                inputs_cleaned.append(example_copy)

            inputs = inputs_cleaned

            if rank == 0:
                print(f"[AUDIO_DISABLE] mode=drop_audio_for_noaudio_smoke", flush=True)
                print(f"[AUDIO_DISABLE] audio_content_removed=True", flush=True)

        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    @staticmethod
    def _move_media_to_cpu(x):
        """Recursively move media tensors to CPU to prevent GPU OOM during preprocessing."""
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                return x.detach().cpu()
            return x
        elif isinstance(x, dict):
            return {k: QwenOmniModule._move_media_to_cpu(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            result = [QwenOmniModule._move_media_to_cpu(item) for item in x]
            return type(x)(result)
        else:
            return x

    def prepare_model_inputs(self, processing_class, prompts_text, images, audios, videos, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False, use_audio_in_video=False):

        rank = 0
        try:
            import torch.distributed as torch_dist
            if torch_dist.is_available() and torch_dist.is_initialized():
                rank = torch_dist.get_rank()
        except:
            rank = 0

        # [AUDIO_DISABLE] Handle GRPO_DISABLE_AUDIO_INPUTS=1: strict video-text only
        disable_audio = os.environ.get("GRPO_DISABLE_AUDIO_INPUTS", "0") == "1"
        if disable_audio:
            if rank == 0:
                print(f"[AUDIO_DISABLE] mode=drop_audio_for_noaudio_smoke", flush=True)
            # Remove audio tokens from prompts_text
            audio_token = getattr(processing_class, 'audio_token', None)
            audio_bos_token = getattr(processing_class, 'audio_bos_token', None)
            audio_eos_token = getattr(processing_class, 'audio_eos_token', None)

            audio_tokens_to_check = []
            if audio_token:
                audio_tokens_to_check.append(audio_token)
            if audio_bos_token:
                audio_tokens_to_check.append(audio_bos_token)
            if audio_eos_token:
                audio_tokens_to_check.append(audio_eos_token)
            audio_tokens_to_check.extend(['<|audio_bos|>', '<|AUDIO|>', '<|audio_eos|>', '<|audio|>', '<audio>'])
            audio_tokens_to_check = list(set(audio_tokens_to_check))

            # Remove audio tokens from prompts_text
            prompts_text_cleaned = []
            for text in prompts_text:
                if isinstance(text, str):
                    cleaned_text = text
                    for token in audio_tokens_to_check:
                        cleaned_text = cleaned_text.replace(token, '')
                    cleaned_text = ' '.join(cleaned_text.split())
                    prompts_text_cleaned.append(cleaned_text)
                else:
                    prompts_text_cleaned.append(text)

            prompts_text = prompts_text_cleaned
            audios = None
            use_audio_in_video = False
            if rank == 0:
                print(f"[AUDIO_DISABLE] audio_tokens_removed=True", flush=True)
                print(f"[AUDIO_DISABLE] audios_removed=True", flush=True)
                print(f"[AUDIO_DISABLE] use_audio_in_video=False", flush=True)

        # [MEDIA_DEVICE_DIAG] Diagnostic before moving to CPU
        if rank == 0:
            print(f"[MEDIA_DEVICE_DIAG] Before CPU move:", flush=True)
            if images is not None:
                if isinstance(images, list) and len(images) > 0:
                    first_img = images[0]
                    if isinstance(first_img, torch.Tensor):
                        print(f"[MEDIA_DEVICE_DIAG]   images[0] device={first_img.device}, shape={first_img.shape}", flush=True)
                    else:
                        print(f"[MEDIA_DEVICE_DIAG]   images[0] type={type(first_img).__name__}", flush=True)
            if videos is not None:
                if isinstance(videos, list) and len(videos) > 0:
                    first_vid = videos[0]
                    if isinstance(first_vid, torch.Tensor):
                        print(f"[MEDIA_DEVICE_DIAG]   videos[0] device={first_vid.device}, shape={first_vid.shape}", flush=True)
                    else:
                        print(f"[MEDIA_DEVICE_DIAG]   videos[0] type={type(first_vid).__name__}", flush=True)
            if audios is not None:
                if isinstance(audios, list) and len(audios) > 0:
                    first_aud = audios[0]
                    if isinstance(first_aud, torch.Tensor):
                        print(f"[MEDIA_DEVICE_DIAG]   audios[0] device={first_aud.device}, shape={first_aud.shape}", flush=True)
                    else:
                        print(f"[MEDIA_DEVICE_DIAG]   audios[0] type={type(first_aud).__name__}", flush=True)

        # Move all media inputs to CPU before processing to prevent GPU OOM during stack operations
        if rank == 0:
            print(f"[MEDIA_DEVICE_DIAG] Moving media tensors to CPU before processor", flush=True)

        images = self._move_media_to_cpu(images)
        audios = self._move_media_to_cpu(audios)
        videos = self._move_media_to_cpu(videos)

        if rank == 0:
            print(f"[MEDIA_DEVICE_DIAG] After CPU move:", flush=True)
            if images is not None:
                if isinstance(images, list) and len(images) > 0:
                    first_img = images[0]
                    if isinstance(first_img, torch.Tensor):
                        print(f"[MEDIA_DEVICE_DIAG]   images[0] device={first_img.device}, shape={first_img.shape}", flush=True)
            if videos is not None:
                if isinstance(videos, list) and len(videos) > 0:
                    first_vid = videos[0]
                    if isinstance(first_vid, torch.Tensor):
                        print(f"[MEDIA_DEVICE_DIAG]   videos[0] device={first_vid.device}, shape={first_vid.shape}", flush=True)

        # Call processor with CPU-resident media inputs
        prompt_inputs = processing_class(
            text=prompts_text,
            images=images,
            audio=audios,
            videos=videos,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
            use_audio_in_video=use_audio_in_video)

        # [AUDIO_DISABLE] Remove audio-related outputs when GRPO_DISABLE_AUDIO_INPUTS=1
        if disable_audio:
            # Remove input_features
            if 'input_features' in prompt_inputs:
                del prompt_inputs['input_features']
                if rank == 0:
                    print(f"[AUDIO_DISABLE] input_features_removed=True", flush=True)

            # Remove feature_attention_mask
            if 'feature_attention_mask' in prompt_inputs:
                del prompt_inputs['feature_attention_mask']
                if rank == 0:
                    print(f"[AUDIO_DISABLE] feature_attention_mask_removed=True", flush=True)

            if rank == 0:
                print(f"[AUDIO_DISABLE] video_inputs_preserved=True", flush=True)
        else:
            # [STAGE3_AUDIO_DIAG] Diagnostic for real audio input when GRPO_DISABLE_AUDIO_INPUTS=0
            # Support both STAGE3_AUDIO_DIAG and GRPO_STAGE3_AUDIO_DIAG for compatibility
            stage3_audio_diag = os.environ.get("STAGE3_AUDIO_DIAG", "0") == "1" or os.environ.get("GRPO_STAGE3_AUDIO_DIAG", "0") == "1"
            if stage3_audio_diag and rank == 0:
                print(f"\n[STAGE3_AUDIO_DIAG] === Audio Input Diagnostic (after processor) ===", flush=True)

                # Check input_features
                if 'input_features' in prompt_inputs:
                    input_features = prompt_inputs['input_features']
                    if input_features is not None:
                        print(f"[STAGE3_AUDIO_DIAG] input_features present: shape={input_features.shape}, dtype={input_features.dtype}, device={input_features.device}", flush=True)
                        if input_features.numel() > 0:
                            print(f"[STAGE3_AUDIO_DIAG] input_features stats: min={input_features.min():.6f}, max={input_features.max():.6f}, mean={input_features.mean():.6f}, std={input_features.std():.6f}", flush=True)
                            zero_ratio = (input_features == 0).float().mean().item()
                            nan_count = torch.isnan(input_features).sum().item()
                            inf_count = torch.isinf(input_features).sum().item()
                            print(f"[STAGE3_AUDIO_DIAG] input_features quality: zero_ratio={zero_ratio:.4f}, nan_count={nan_count}, inf_count={inf_count}", flush=True)
                    else:
                        print(f"[STAGE3_AUDIO_DIAG] input_features is None", flush=True)
                else:
                    print(f"[STAGE3_AUDIO_DIAG] input_features NOT in prompt_inputs", flush=True)

                # Check feature_attention_mask
                if 'feature_attention_mask' in prompt_inputs:
                    feature_attention_mask = prompt_inputs['feature_attention_mask']
                    if feature_attention_mask is not None:
                        print(f"[STAGE3_AUDIO_DIAG] feature_attention_mask present: shape={feature_attention_mask.shape}, dtype={feature_attention_mask.dtype}", flush=True)
                        mask_sum = feature_attention_mask.sum().item()
                        per_sample_sum = feature_attention_mask.sum(dim=1)
                        print(f"[STAGE3_AUDIO_DIAG] feature_attention_mask stats: total_sum={mask_sum}, per_sample_sum={per_sample_sum[:min(3, len(per_sample_sum))]}", flush=True)
                        all_zero = (feature_attention_mask == 0).all().item()
                        print(f"[STAGE3_AUDIO_DIAG] feature_attention_mask all_zero={all_zero}", flush=True)
                    else:
                        print(f"[STAGE3_AUDIO_DIAG] feature_attention_mask is None", flush=True)
                else:
                    print(f"[STAGE3_AUDIO_DIAG] feature_attention_mask NOT in prompt_inputs", flush=True)

                # Check input_ids for audio tokens
                if 'input_ids' in prompt_inputs:
                    input_ids = prompt_inputs['input_ids']
                    print(f"[STAGE3_AUDIO_DIAG] input_ids shape={input_ids.shape}", flush=True)

                    # Try to get audio token IDs from tokenizer
                    audio_token_count = 0
                    audio_token_ids = []
                    audio_token_names = ['<|AUDIO|>', '<|audio_bos|>', '<|audio_eos|>', '<|audio|>']

                    try:
                        tokenizer = processing_class.tokenizer
                        for token_name in audio_token_names:
                            try:
                                token_id = tokenizer.convert_tokens_to_ids(token_name)
                                if token_id != tokenizer.unk_token_id:  # Not unknown token
                                    audio_token_ids.append(token_id)
                            except:
                                pass

                        if audio_token_ids:
                            for token_id in audio_token_ids:
                                audio_token_count += (input_ids == token_id).sum().item()
                            print(f"[STAGE3_AUDIO_DIAG] audio_token_count (from tokenizer)={audio_token_count}, token_ids={audio_token_ids}", flush=True)
                        else:
                            print(f"[STAGE3_AUDIO_DIAG] No audio tokens found in tokenizer", flush=True)
                            print(f"[STAGE3_AUDIO_DIAG] tokenizer.special_tokens_map={tokenizer.special_tokens_map}", flush=True)
                    except Exception as e:
                        print(f"[STAGE3_AUDIO_DIAG] Error getting audio tokens from tokenizer: {e}", flush=True)

                    # Print first and last 80 token IDs for inspection
                    if input_ids.numel() > 0:
                        first_80 = input_ids[0, :min(80, input_ids.size(1))].tolist()
                        last_80 = input_ids[0, max(0, input_ids.size(1)-80):].tolist()
                        print(f"[STAGE3_AUDIO_DIAG] input_ids[0] first_80_tokens={first_80}", flush=True)
                        print(f"[STAGE3_AUDIO_DIAG] input_ids[0] last_80_tokens={last_80}", flush=True)
                else:
                    print(f"[STAGE3_AUDIO_DIAG] input_ids NOT in prompt_inputs", flush=True)

                print(f"[STAGE3_AUDIO_DIAG] === End Audio Input Diagnostic ===\n", flush=True)

        # [MEDIA_DEVICE_DIAG] Diagnostic after processor
        if rank == 0:
            print(f"[MEDIA_DEVICE_DIAG] After processor:", flush=True)
            for key in ['input_ids', 'attention_mask', 'pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'input_features', 'feature_attention_mask']:
                if key in prompt_inputs:
                    val = prompt_inputs[key]
                    if isinstance(val, torch.Tensor):
                        print(f"[MEDIA_DEVICE_DIAG]   {key} device={val.device}, shape={val.shape}", flush=True)
                    elif isinstance(val, list):
                        print(f"[MEDIA_DEVICE_DIAG]   {key} is list of length {len(val)}", flush=True)

        # [GRPO_PROMPT_TEMPLATE_DEBUG] Generation input template diagnostics
        prompt_template_debug = os.environ.get("GRPO_PROMPT_TEMPLATE_DEBUG", "0") == "1"
        if prompt_template_debug and rank == 0:
            print(f"\n[GRPO_PROMPT_TEMPLATE_DEBUG] Generation input template structure:", flush=True)

            # Content type list
            content_types = []
            if images is not None and (isinstance(images, list) and len(images) > 0 or isinstance(images, torch.Tensor)):
                content_types.append("image")
            if videos is not None and (isinstance(videos, list) and len(videos) > 0 or isinstance(videos, torch.Tensor)):
                content_types.append("video")
            if audios is not None and (isinstance(audios, list) and len(audios) > 0 or isinstance(audios, torch.Tensor)):
                content_types.append("audio")
            else:
                content_types.append("audio_deleted" if disable_audio else "audio_absent")
            content_types.append("text")
            print(f"[GRPO_PROMPT_TEMPLATE_DEBUG] content_types: {content_types}", flush=True)

            # Prompt structure verification
            if len(prompts_text) > 0:
                prompt_first = prompts_text[0][:300] if len(prompts_text[0]) > 300 else prompts_text[0]
                prompt_last = prompts_text[0][-300:] if len(prompts_text[0]) > 300 else prompts_text[0]
                print(f"[GRPO_PROMPT_TEMPLATE_DEBUG] prompt_first_300: {repr(prompt_first)}", flush=True)
                print(f"[GRPO_PROMPT_TEMPLATE_DEBUG] prompt_last_300: {repr(prompt_last)}", flush=True)
                print(f"[GRPO_PROMPT_TEMPLATE_DEBUG] has_assistant_header: {'<|im_start|>assistant' in prompts_text[0] or '<|assistant|>' in prompts_text[0]}", flush=True)

            # Input tensor info
            if 'input_ids' in prompt_inputs:
                input_ids = prompt_inputs['input_ids']
                print(f"[GRPO_PROMPT_TEMPLATE_DEBUG] input_ids.shape: {input_ids.shape}", flush=True)
            if 'attention_mask' in prompt_inputs:
                attention_mask = prompt_inputs['attention_mask']
                real_prompt_lens = attention_mask.sum(dim=1)
                print(f"[GRPO_PROMPT_TEMPLATE_DEBUG] attention_mask.shape: {attention_mask.shape}", flush=True)
                print(f"[GRPO_PROMPT_TEMPLATE_DEBUG] real_prompt_lens (first 3): {real_prompt_lens[:min(3, len(real_prompt_lens))].tolist()}", flush=True)
            print(f"[GRPO_PROMPT_TEMPLATE_DEBUG] disable_audio: {disable_audio}", flush=True)
            print()

        return prompt_inputs
    
    @staticmethod
    def identify_task_type(example: dict) -> str:
        """Identify task type from dataset source or data filename.

        Priority:
        1. _dataset_source field (emer_rewrite.json) -> emotion_video
        2. emer, emotion, MER, affective -> emotion_video
        3. Social-IQ, social, conversation, dialogue, relationship -> social_reasoning
        4. Video-R1, video qa, video question -> general_video_qa
        5. math, geometry, triangle, angle, equation, formula, calculate, solve -> image_math
        6. unknown
        """
        # Check _dataset_source field first (highest priority)
        dataset_source = example.get("_dataset_source", "").lower()
        if "emer" in dataset_source:
            return "emotion_video"

        # Get source/filename indicators
        source = example.get("source", "").lower()
        filename = example.get("filename", "").lower()
        problem_text = example.get("problem", "").lower() + " " + example.get("question", "").lower()

        # Combine all text for matching
        all_text = f"{source} {filename} {problem_text}".lower()

        # Emotion video keywords
        emotion_keywords = ["emer", "emotion", "mer", "affective", "emotional"]
        if any(kw in all_text for kw in emotion_keywords):
            return "emotion_video"

        # Social reasoning keywords
        social_keywords = ["social-iq", "social", "conversation", "dialogue", "relationship"]
        if any(kw in all_text for kw in social_keywords):
            return "social_reasoning"

        # General video QA keywords
        video_keywords = ["video-r1", "video qa", "video question", "video"]
        if any(kw in all_text for kw in video_keywords):
            return "general_video_qa"

        # Image math keywords
        math_keywords = ["math", "geometry", "triangle", "angle", "equation", "formula", "calculate", "solve"]
        if any(kw in all_text for kw in math_keywords):
            return "image_math"

        return "unknown"

    @staticmethod
    def get_question_template(task_type: str = "unknown"):
        """Get prompt template based on task type.

        emotion_video: Full affective context-first prompt
        general_video_qa: Shorter video QA prompt
        social_reasoning: Social reasoning prompt
        image_math: Short math-focused prompt
        unknown: Generic short prompt
        """
        task_aware_prompt_enabled = os.environ.get("GRPO_TASK_AWARE_PROMPT", "0") == "1"
        emotion_short_prompt_enabled = os.environ.get("GRPO_EMOTION_SHORT_PROMPT", "0") == "1"

        # Short prompt for emotion-only samples (overrides task-aware prompt)
        if emotion_short_prompt_enabled and task_type == "emotion_video":
            return "{Question} In <context></context> tags, write one sentence about emotional cues. In <think></think> tags, write one sentence of reasoning. In <answer></answer> tags, write only the emotion category or option letter."

        if not task_aware_prompt_enabled:
            # Default baseline behavior
            if task_type == "rec":
                return "{Question} In <context></context> tags, briefly note key emotional cues. In <think></think> tags, output your reasoning. In <answer></answer> tags, give the final answer in JSON format."
            else:
                return "{Question} In <context></context> tags, briefly note key emotional cues. In <think></think> tags, output your reasoning. In <answer></answer> tags, give the final answer."

        # Task-aware templates
        if task_type == "emotion_video":
            return "{Question} In <context></context> tags, analyze emotional cues from visual evidence, acoustic evidence, textual evidence, and emotion trajectory. In <think></think> tags, reason through the emotional context. In <answer></answer> tags, give the final answer."
        elif task_type == "general_video_qa":
            return "{Question} In <context></context> tags, note key visual and textual information. In <think></think> tags, output your reasoning. In <answer></answer> tags, give the final answer."
        elif task_type == "social_reasoning":
            return "{Question} In <context></context> tags, identify social context and relationships. In <think></think> tags, reason through social dynamics. In <answer></answer> tags, give the final answer."
        elif task_type == "image_math":
            return "{Question} In <think></think> tags, show your calculation steps. In <answer></answer> tags, give the final numerical answer."
        else:  # unknown
            return "{Question} In <think></think> tags, output your reasoning. In <answer></answer> tags, give the final answer."
            
            
    @staticmethod
    def format_reward(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""

        # Relaxed pattern: allows for optional context, requires think and answer
        pattern = r"<think>[\s\S]*?</think>[\s\S]*?<answer>[\s\S]*?</answer>"

        # Use unified extraction method
        completion_contents = [QwenOmniModule._extract_text_from_completion(completion) for completion in completions]

        rewards = []
        for idx, content in enumerate(completion_contents):
            reward = 0.0
            matches = re.search(pattern, content, re.DOTALL)
            if matches is not None:
                reward = 1.0

            if idx < 2:
                print(f"\n[FORMAT_REWARD] Sample {idx}:")
                print(f"  content preview (first 300 chars): {repr(content[:300])}")
                print(f"  has <think>: {'<think>' in content}")
                print(f"  has <answer>: {'<answer>' in content}")
                print(f"  match is None: {matches is None}")
                print(f"  reward: {reward}")

            rewards.append(reward)

        return rewards
    


    @staticmethod
    def precision_reward(completions, solution, **kwargs):

        completions = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion, sol in zip(completions, solution):
            reward = 0.0
            # print(completion, sol)
            answer_tag_pattern = r'<answer>(.*?)</answer>'
            # Try symbolic verification first
            # try:
            content_answer_match = re.search(answer_tag_pattern, completion, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                words = content_answer.split(",")
                count = 0
                for each in sol:
                    if each.lower() in content_answer or each in content_answer:
                        count +=1

                reward = float(count)/len(sol)
                # bbox_match = re.search(bbox_pattern, content_answer)
            rewards.append(reward)
            # except Exception as e :
            #     pass  # Continue to next verification method if this fails
        # print(rewards)
        return rewards
      
        
    @staticmethod
    def recall_reward(completions, solution, **kwargs):
        import re
        completions = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion, sol in zip(completions, solution):
            reward = 0.0
            # print(completion, sol)
            answer_tag_pattern = r'<answer>(.*?)</answer>'
            # Try symbolic verification first
            # try:
            content_answer_match = re.search(answer_tag_pattern, completion, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                words = content_answer.split(",")
                count = 0
                for each in sol:
                    if each.lower() in content_answer or each in content_answer:
                        count +=1

                reward = float(count)/len(sol)
                # bbox_match = re.search(bbox_pattern, content_answer)
            rewards.append(reward)
            # except Exception as e :
            #     pass  # Continue to next verification method if this fails
        # print(rewards)
        return rewards

    @staticmethod
    def accuracy_reward(completions, solution, **kwargs):

        # Print module path to confirm which file is actually loaded
        print(f"\n[ACCURACY_REWARD] Module file: {__file__}")
        print(f"[ACCURACY_REWARD] Function called with {len(completions)} completions")

        def extract_answer(text):
            """Extract answer from text, handling multiple <answer> tags by taking the first one."""
            pattern = r'<answer>\s*(.*?)\s*</answer>'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                answer = match.group(1).strip()
                return answer
            return ""

        def normalize_number(num_str):
            try:
                num_str = num_str.replace(',', '')
                return float(num_str)
            except Exception as e:
                print(f"Error converting '{num_str}' to float: {e}")
                return None

        def wer(reference, hypothesis):
            ref_words = reference.split()
            hyp_words = hypothesis.split()
            m = len(ref_words)
            n = len(hyp_words)
            d = [[0]*(n+1) for _ in range(m+1)]
            for i in range(m+1):
                d[i][0] = i
            for j in range(n+1):
                d[0][j] = j
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if ref_words[i-1] == hyp_words[j-1]:
                        d[i][j] = d[i-1][j-1]
                    else:
                        d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
            return d[m][n] / max(1, m)


        def compute_rouge_score(reference, hypothesis, use_stemmer=True):
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
            scores = scorer.score(reference, hypothesis)
            average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
            return average_fmeasure

        def similarity(reference, hypothesis):
            prompt = f"""
            Analyze the consistency between the content of the two compared texts and assign a score based on the following criteria:

            Grading criteria description (content consistency):

            5 points: The core facts, details, and logical relationships in the two texts are entirely consistent, with no differences.
            3-4 points: The core content is consistent, but there are differences in non-critical details (such as expression, supplementary information, examples, etc.).
            1-2 points: Some content is consistent, but there are contradictions or differences in key information.
            0 points: The core content is inconsistent or completely irrelevant.

            Example analysis process:

            Extract the core information from both texts (time, place, people, events, data, conclusions, etc.).
            Compare whether key facts align (e.g., whether the times are the same, whether the data matches).
            Analyze the consistency of logical relationships (causal relationships, sequence, etc.).
            Determine whether the differences are merely expressive (such as synonym replacement, sentence adjustment) or substantive content differences.

            reference: {reference}
            hypothesis: {hypothesis}

            only return the score number:
            """
    
            try:
                reward = gpt_api(prompt=prompt, model_name="qwen-plus")
                reward = ast.literal_eval(reward) / 5.0
            except:
                return 0

            return reward
        
 
        
        def emer_ov_mc(reference, hypothesis):
            list_a = reference.split(",")
            list_b = hypothesis.split(",")
            true_positive = len(set(list_a) & set(list_b))
            precision = true_positive / len(list_a) if list_a else 0
            recall = true_positive / len(list_b) if list_b else 0
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0
            
            return f1_score
        
        def judge(reference, hypothesis):
            if "yes" in reference.lower()  and "yes" in hypothesis.lower():
                return 1
            elif "no" in reference.lower()  and "no" in hypothesis.lower():
                return 1
            else:
                return 0


        question_type = kwargs['problem_type'][0]
        print(f"[ACCURACY_REWARD] question_type: {question_type}")

        # Use unified extraction method for consistency with affective rewards
        contents = [QwenOmniModule._extract_text_from_completion(completion) for completion in completions]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        rewards = []

        # Debug counter to limit stdout output
        debug_counter = 0
        max_debug_samples = 2

        for content, sol in zip(contents, solution):

            try:
                output_ans = extract_answer(content)
                gt_ans = extract_answer(sol)

                # Print debug info for first 2 samples (before branching)
                if debug_counter < max_debug_samples:
                    print(f"\n[ACCURACY_REWARD] Sample {debug_counter}:")
                    print(f"  content preview (first 300 chars): {repr(content[:300])}")
                    print(f"  sol preview (first 300 chars): {repr(sol[:300])}")

                    # Check for <answer> tags
                    content_answer_count = content.count('<answer>')
                    sol_answer_count = sol.count('<answer>')
                    print(f"  content <answer> count: {content_answer_count}")
                    print(f"  sol <answer> count: {sol_answer_count}")

                    print(f"  output_ans (extracted): '{output_ans}'")
                    print(f"  gt_ans (extracted): '{gt_ans}'")

                if question_type == "multiple choice":
                    # Normalize both prediction and GT for robust comparison
                    def normalize_choice(text):
                        """Extract single letter choice (A/B/C/D) from various formats."""
                        if not text:
                            return ""
                        # Remove whitespace and convert to uppercase
                        text = text.strip().upper()
                        # Extract first letter if it's A/B/C/D
                        for char in text:
                            if char in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                                return char
                        return text

                    normalized_output = normalize_choice(output_ans)
                    normalized_gt = normalize_choice(gt_ans)
                    reward = 1.0 if normalized_output == normalized_gt else 0.0

                    # Print detailed comparison for first 2 samples
                    if debug_counter < max_debug_samples:
                        print(f"  normalized_output: '{normalized_output}'")
                        print(f"  normalized_gt: '{normalized_gt}'")
                        print(f"  reward: {reward}")
                        debug_counter += 1
                elif question_type == "numerical":
                    gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                    out_has_decimal = ("." in output_ans) or ("," in output_ans)
                    if gt_has_decimal != out_has_decimal:
                        reward = 0.0
                    else:
                        gt_number = normalize_number(gt_ans)
                        out_number = normalize_number(output_ans)
                        if gt_number is None or out_number is None:
                            reward = 0.0
                        else:
                            reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
                elif question_type == "OCR":
                    error_rate = wer(gt_ans, output_ans)
                    reward = 1 - error_rate
                    reward = max(0.0, min(1.0, reward))
                elif question_type == "free-form":
                    reward = similarity(gt_ans, output_ans)
                    # score = compute_rouge_score(gt_ans, output_ans)
                    # reward = max(0.0, min(1.0, score))
                elif question_type == "regression":
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                    rel_diff = min(1.0, max(0.0, rel_diff))
                    reward = 1 - rel_diff
                elif question_type == "emer_ov":
                    reward = emer_ov_gpt(gt_ans, output_ans)
                elif question_type == "emer_ov_mc":
                    reward = emer_ov_mc(gt_ans, output_ans)
                elif  question_type == "judge":
                    reward = judge(output_ans, gt_ans)
                else:
                    reward = 0.0
            except Exception as e:
                print(f"Error in reward_fn for question_type '{question_type}': {e}")
                reward = 0.0
        
            rewards.append(reward)
            
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
                
        return rewards

    @staticmethod
    def patial_context_reward(completions, solution, **kwargs):
    
        def extract_parts(text, pattern):

            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""

        def similarity(reference, hypothesis):

            prompt = \
f"""You are assessing how well the 'hypothesis' text covers the key information from the 'reference' text. Differences in wording or extra details in the 'hypothesis' are fine if the 'reference's' main points are included.:

Score based on this coverage:

5 points : Hypothesis clearly and accurately reflects significant core themes or key aspects of the reference. It demonstrates a good understanding of a substantial part of the reference material.
4 points : Hypothesis reflects some important themes or aspects of the reference. The connection is evident, though perhaps not as comprehensive or central as a 5.
2 points : Hypothesis shows a recognizable connection to themes or aspects of the reference, but it might be more superficial, focus on less central points, or only partially grasp a key aspect.
1 points : Hypothesis has a tenuous or very limited connection to the reference. It might touch on a peripheral detail or a heavily reinterpreted aspect, but largely misses the main substance.
0 points : Hypothesis does not reflect any significant themes or key aspects of the reference, or is on a completely different topic.

Example analysis process:

Identify main themes and key aspects in 'reference'.
Determine if 'hypothesis' connects to or discusses any of these themes/aspects from 'reference'.
Judge the strength and relevance of this connection. Is a core part of the 'reference' reflected?
Differences are expected; evaluate if the 'hypothesis' still meaningfully reflects some key part of the 'reference'.
Assign score based on how well a significant aspect is reflected.

reference: {reference}
hypothesis: {hypothesis}

only return the score number:"""
    
            try:
                reward = gpt_api(prompt=prompt, model_name="qwen-plus")
                reward = ast.literal_eval(reward) / 5.0
            except:
                return 0

            return reward
           


        question_type = kwargs['problem_type'][0]
        
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
        
            try:
                output_evidence = extract_parts(content, pattern=r'<context>\s*(.*?)\s*</context>')
                
                gt_evidence = extract_parts(sol, pattern=r'<context>\s*(.*?)\s*</context>')
                if len(gt_evidence)==0:
                    reward = 0.0
          
                else:
           
                    reward = similarity(gt_evidence, output_evidence)

            except Exception as e:
                print(f"Error in reward_fn for question_type '{question_type}': {e}")
                reward = 0.0
        
            rewards.append(reward)
            
     
        return rewards

    @staticmethod
    def patial_reasoning_reward(completions, solution, **kwargs):
    
        def extract_parts(text, pattern):
         
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""

        def rationality(reference, hypothesis):
            
           
            prompt = \
f"""Please analyze whether the reasoning text is derived from the evidence and context text based on the following criteria and give a score of 0-5:
Grading criteria description (relevance and rationality):

Integration of Clues (1 point): During the reasoning process, there is incorporation of clues from the video, image, or audio.

Reflection and Confirmation (1 point): The reasoning involves reflection or second confirmation of choices or answers, including revisiting video, image, or audio evidence.

Logical Reasoning (1 point): The thought process is clear, deriving conclusions through rigorous logical reasoning, analysis, or extension without additional assumptions or contradictions.

Problem Analysis (1 point): The reasoning process includes thorough analysis in conjunction with the problem at hand.

Overall Consistency (1 point): The reasoning text is based on visual or audio evidence and context information, presenting no extra assumptions or contradictions.

Assign one point for each criterion that is met, for a total possible score of five points. Verify that each criterion is addressed and reflect this in your scoring.

context: {reference}
reasoning path: {hypothesis}

only return the score number:
            """
            try:
                reward = gpt_api(prompt=prompt, model_name="qwen-plus")
                reward = ast.literal_eval(reward) / 5.0
            except:
                return 0

            return reward


        question_type = kwargs['problem_type'][0]
        
        contents = [completion[0]["content"] for completion in completions]

        rewards = []

        for content, sol in zip(contents, solution):
        
            try:
                evidence =  extract_parts(content, pattern=r'<context>\s*(.*?)\s*</context>')
                think_path = extract_parts(content, pattern=r'<think>\s*(.*?)\s*</think>')
                answer = extract_parts(content, pattern=r'<answer>\s*(.*?)\s*</answer>')
                
                if len(evidence)==0 or len(think_path)==0:
                    reward = 0.0
                    
                else:
                    # output_think = extract_parts(content, pattern=r'<think>\s*(.*?)\s*</think>')


                    reward = rationality(evidence, think_path)

            except Exception as e:
                print(f"Error in reward_fn for question_type '{question_type}': {e}")
                reward = 0.0

            rewards.append(reward)


        return rewards

    @staticmethod
    def _extract_text_from_completion(completion):
        """Extract text content from completion in various formats.

        Handles both str and list[dict] formats for backward compatibility.
        """
        if isinstance(completion, str):
            return completion
        elif isinstance(completion, list) and len(completion) > 0:
            if isinstance(completion[0], dict) and "content" in completion[0]:
                return completion[0]["content"]
            elif isinstance(completion[0], str):
                return completion[0]
        return ""

    @staticmethod
    def _get_first_metadata_value(kwargs, *keys):
        for key in keys:
            value = kwargs.get(key)
            if isinstance(value, (list, tuple)):
                value = value[0] if len(value) > 0 else None
            if value is not None:
                return value
        return None

    @staticmethod
    def _is_affective_problem_type(problem_type):
        if problem_type is None:
            return False
        problem_type = str(problem_type).lower()
        return any(keyword in problem_type for keyword in ("emotion", "deception", "why", "how", "emer", "affective", "sentiment"))

    @staticmethod
    def _has_preference_pair_fields(kwargs, idx=None):
        solution = kwargs.get("solution")
        rejected_answer_for_eval = kwargs.get("rejected_answer_for_eval")
        pair_type = kwargs.get("pair_type")

        if solution is None or rejected_answer_for_eval is None:
            return False

        if isinstance(solution, (list, tuple)):
            if idx is None or idx >= len(solution):
                return False
            solution_value = solution[idx]
        else:
            solution_value = solution

        if isinstance(rejected_answer_for_eval, (list, tuple)):
            if idx is None or idx >= len(rejected_answer_for_eval):
                return False
            rejected_value = rejected_answer_for_eval[idx]
        else:
            rejected_value = rejected_answer_for_eval

        if solution_value is None or rejected_value is None:
            return False

        if isinstance(solution_value, str) and not solution_value.strip():
            return False
        if isinstance(rejected_value, str) and not rejected_value.strip():
            return False

        # pair_type is optional here; keep it available for the original reward logic if present.
        if isinstance(pair_type, (list, tuple)) and idx is not None and idx < len(pair_type):
            _ = pair_type[idx]

        return True

    @staticmethod
    def affective_context_reward(completions, solution, **kwargs):
        """Reward function for affective context understanding.

        Uses a staged scoring strategy to avoid sparse rewards:
        - Base score: Has <context> tag (0.2)
        - Emotion keywords: Contains emotion-related words (0.2)
        - Evidence keywords: Contains evidence-related words (visual/acoustic/textual) (0.2)
        - Trajectory keywords: Contains trajectory/change-related words (0.2)
        - Semantic similarity: GPT-based affective similarity (0.2)

        MASK: Applied to <context> span only (token-level masking in trainer)
        GATE: Sample-level gate based on emotion metadata or problem text heuristics
        WEIGHT: Configurable via affective_context_weight (default 0.2)
        """
        def extract_parts(text, pattern):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""

        def has_emotion_keywords(text):
            """Check if text contains emotion-related keywords."""
            emotion_keywords = [
                'emotion', 'emotional', 'feeling', 'feel', 'felt',
                'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust',
                'joy', 'anxious', 'calm', 'excited', 'frustrated',
                'relaxed', 'tense', 'pleased', 'disappointed', 'satisfied',
                'mood', 'sentiment', 'affective', 'affect'
            ]
            text_lower = text.lower()
            return any(keyword in text_lower for keyword in emotion_keywords)

        def has_evidence_keywords(text):
            """Check if text contains evidence-related keywords."""
            evidence_keywords = [
                'visual', 'see', 'saw', 'observe', 'look', 'watch', 'facial', 'expression', 'face',
                'acoustic', 'hear', 'heard', 'voice', 'tone', 'sound', 'audio', 'speak', 'said',
                'textual', 'text', 'subtitle', 'word', 'caption', 'read',
                'evidence', 'cue', 'signal', 'indicator', 'sign'
            ]
            text_lower = text.lower()
            return any(keyword in text_lower for keyword in evidence_keywords)

        def has_trajectory_keywords(text):
            """Check if text contains trajectory/change-related keywords."""
            trajectory_keywords = [
                'trajectory', 'change', 'changed', 'shift', 'shifted', 'transition',
                'initially', 'first', 'then', 'later', 'finally', 'eventually',
                'from', 'to', 'become', 'became', 'transform', 'evolve',
                'progression', 'development', 'over time'
            ]
            text_lower = text.lower()
            return any(keyword in text_lower for keyword in trajectory_keywords)

        def affective_similarity(reference, hypothesis):
            """GPT-based semantic similarity for affective content."""
            if len(hypothesis) < 10:  # Too short to be meaningful
                return 0.0

            prompt = f"""Evaluate how well the 'hypothesis' captures affective/emotional aspects from the 'reference'.

Score 0-5:
5: Accurately identifies key emotional states and affective tones
4: Captures most emotional aspects with minor gaps
3: Shows partial understanding of affective content
2: Limited connection to affective aspects
1: Barely touches on emotional content
0: No affective aspects reflected

reference: {reference}
hypothesis: {hypothesis}

Return only the score number:"""

            try:
                reward = gpt_api(prompt=prompt, model_name="qwen-plus")
                reward = ast.literal_eval(reward) / 5.0
            except:
                return 0.0
            return reward

        contents = [QwenOmniModule._extract_text_from_completion(completion) for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            try:
                output_context = extract_parts(content, pattern=r'<context>\s*(.*?)\s*</context>')
                gt_context = extract_parts(sol, pattern=r'<context>\s*(.*?)\s*</context>')

                # If no GT context, cannot evaluate
                if len(gt_context) == 0:
                    rewards.append(0.0)
                    continue

                # If no output context, give 0
                if len(output_context) == 0:
                    rewards.append(0.0)
                    continue

                # Staged scoring
                reward = 0.0

                # Stage 1: Has context tag (0.2)
                reward += 0.2

                # Stage 2: Has emotion keywords (0.2)
                if has_emotion_keywords(output_context):
                    reward += 0.2

                # Stage 3: Has evidence keywords (0.2)
                if has_evidence_keywords(output_context):
                    reward += 0.2

                # Stage 4: Has trajectory keywords (0.2)
                if has_trajectory_keywords(output_context):
                    reward += 0.2

                # Stage 5: Semantic similarity (0.2)
                semantic_score = affective_similarity(gt_context, output_context)
                reward += semantic_score * 0.2

                rewards.append(reward)

            except Exception as e:
                print(f"Error in affective_context_reward: {e}")
                rewards.append(0.0)

        return rewards

    @staticmethod
    def b500_preference_reward(completions, solution=None, rejected_answer_for_eval=None, pair_type=None, type=None, **kwargs):
        """Reward function for B500-centered preference optimization.

        Uses gain/protect pair structure to guide the model away from B500 errors
        and preserve B500 correct behaviors.

        Args:
            completions: Model completions
            solution: Ground truth solution (contains chosen answer in <answer> tags)
            rejected_answer_for_eval: The rejected answer (B500 wrong answer or candidate wrong answer)
            pair_type: "gain_pair", "protect_pair", or "style_anchor_pair"
            **kwargs: Additional arguments
        """
        def extract_answer(text):
            """Extract answer from <answer> tags."""
            pattern = r'<answer>\s*(.*?)\s*</answer>'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""

        def normalize_answer(text):
            """Normalize answer for comparison."""
            if not text:
                return ""
            text = text.strip().upper()
            # For yes/no answers
            if text.lower() in ("yes", "no"):
                return text.lower()
            # For single letter choices
            if len(text) == 1 and text.isalpha():
                return text
            # For numeric answers, strip commas
            if any(c.isdigit() for c in text):
                return text.replace(',', '')
            return text

        contents = [QwenOmniModule._extract_text_from_completion(completion) for completion in completions]
        rewards = []

        for idx, (content, sol) in enumerate(zip(contents, solution)):
            try:
                if not QwenOmniModule._has_preference_pair_fields(kwargs, idx=idx):
                    rewards.append(0.0)
                    continue

                # Extract prediction
                pred_raw = extract_answer(content)
                if not pred_raw:
                    rewards.append(0.0)
                    continue

                pred = normalize_answer(pred_raw)

                # Extract target (chosen answer)
                target_raw = extract_answer(sol) if isinstance(sol, str) else ""
                target = normalize_answer(target_raw)

                # Extract rejected answer
                rejected_list = rejected_answer_for_eval if isinstance(rejected_answer_for_eval, list) else [rejected_answer_for_eval] * len(completions)
                rejected = normalize_answer(rejected_list[idx] if idx < len(rejected_list) else "")
                if not rejected:
                    rewards.append(0.0)
                    continue

                # Base reward
                reward = 0.0
                if pred == target:
                    reward = 2.0
                elif pred == rejected:
                    reward = -2.0
                else:
                    reward = -0.5

                # Apply pair_type weighting
                pair_type_list = pair_type if isinstance(pair_type, list) else [pair_type] * len(completions)
                current_pair_type = pair_type_list[idx] if idx < len(pair_type_list) else None

                if current_pair_type == "protect_pair":
                    # Protect B500 correct behavior
                    if pred == target:
                        reward += 1.0
                    else:
                        reward -= 2.0
                elif current_pair_type == "gain_pair":
                    # Encourage learning from candidate
                    if pred == target:
                        reward += 1.0
                    elif pred == rejected:
                        reward -= 1.0

                rewards.append(reward)

            except Exception as e:
                print(f"[B500_PREFERENCE_REWARD] Error at sample {idx}: {e}")
                rewards.append(0.0)

        return rewards

    @staticmethod
    def emotion_consistency_reward(completions, solution=None, **kwargs):
        """Reward function for emotion consistency: compares predicted emotion with ground truth emotion.

        Supports multiple answer formats and emotion synonyms.

        Args:
            completions: Model completions
            solution: Ground truth solution (can be passed as positional or keyword arg)
            **kwargs: Additional arguments including solution field if not passed as positional arg
        """
        import os
        import re

        # Handle solution passed via kwargs if not provided as positional argument
        if solution is None:
            solution = kwargs.get("solution", [])

        debug_enabled = os.environ.get("GRPO_EMOTION_REWARD_DEBUG", "0") == "1"
        rank = 0
        try:
            import torch.distributed as torch_dist
            if torch_dist.is_available() and torch_dist.is_initialized():
                rank = torch_dist.get_rank()
        except:
            rank = 0

        emotion_synonyms = {
            "joy": ["joy", "happy", "happiness", "mocking", "proud"],
            "sad": ["sad", "sadness"],
            "anger": ["anger", "angry"],
            "fear": ["fear"],
            "disgust": ["disgust"],
            "surprise": ["surprise"],
            "neutral": ["neutral"],
        }

        synonym_to_category = {}
        for category, synonyms in emotion_synonyms.items():
            for syn in synonyms:
                synonym_to_category[syn.lower()] = category

        def normalize_emotion(text):
            if not text:
                return None
            text_lower = text.lower().strip()
            if text_lower in synonym_to_category:
                return synonym_to_category[text_lower]
            for syn, category in synonym_to_category.items():
                if syn in text_lower or text_lower in syn:
                    return category
            return None

        def extract_options_mapping_from_text(text):
            """Extract options mapping from text supporting both single-line and multi-line formats."""
            options_mapping = {}
            if not text:
                return options_mapping

            if 'Options:' not in text and 'options:' not in text:
                return options_mapping

            options_idx = text.lower().find('options:')
            if options_idx == -1:
                return options_mapping

            options_section = text[options_idx + 8:]

            # Try multi-line format first
            multiline_pattern = r'^\s*([A-Z])\s*[\.\)]\s*([^\n\r]+)'
            multiline_matches = re.findall(multiline_pattern, options_section, re.MULTILINE)

            if multiline_matches:
                for letter, emotion_text in multiline_matches:
                    emotion_clean = emotion_text.strip().lower()
                    emotion_clean = emotion_clean.split('\n')[0].split('\r')[0].strip()
                    if emotion_clean:
                        options_mapping[letter.upper()] = emotion_clean
                if options_mapping:
                    return options_mapping

            # Try single-line format
            singleline_pattern = r'([A-Z])\s*[\.\)]\s*([A-Za-z\s]+?)(?=\s+[A-Z]\s*[\.\)]|$)'
            singleline_matches = re.findall(singleline_pattern, options_section)

            if singleline_matches:
                for letter, emotion_text in singleline_matches:
                    emotion_clean = emotion_text.strip().lower()
                    if emotion_clean:
                        options_mapping[letter.upper()] = emotion_clean

            return options_mapping

        def extract_emotion_from_answer(answer_text, options_mapping=None, allow_letters=True):
            """Extract emotion category from answer text.

            Supports:
            - Single emotion: joy, happy, humor, mocking, etc.
            - Multiple emotions: A,E or A, E or A/E or A D E or ['A','D','F']
            - Direct emotion labels: Irony, Humor, Mocking
            - Letter-to-emotion mapping via options_mapping (from current prompt)
            - Lowercase letters: a,d,f
            - Prefixed answers: Answer: A,D,F
            - Natural language: "The answer is sad"
            - List format: ["sad"]
            - Dict format: {"answer": "F"} or {"label": "sad"}
            """
            if not answer_text:
                return None, set(), set()
            answer_text = answer_text.strip()

            if options_mapping is None:
                options_mapping = {}

            # Try direct emotion name match first (using fixed synonyms)
            emotion = normalize_emotion(answer_text)
            if emotion:
                return emotion, set(), set()

            # Try to extract from "X. Emotion" format
            parts = answer_text.split(".")
            if len(parts) >= 2:
                emotion = normalize_emotion(parts[-1])
                if emotion:
                    return emotion, set(), set()

            # Try to find emotion word in text
            words = answer_text.split()
            for word in words:
                emotion = normalize_emotion(word)
                if emotion:
                    return emotion, set(), set()

            # Try to map letters to emotions using options_mapping
            if allow_letters:
                # Handle multiple letter formats: A,E or A, E or A/E or A D E or ['A','D','F'] or a,d,f
                # Remove brackets, quotes, and common prefixes
                clean_text = answer_text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
                # Remove common prefixes like "Answer:", "Label:", etc.
                clean_text = re.sub(r'^(answer|label|option|choice|result|response)\s*[:=]\s*', '', clean_text, flags=re.IGNORECASE)
                clean_text = clean_text.strip()

                # Try to extract letters using multiple separators
                letters = []
                # First try comma separator
                if "," in clean_text:
                    letters = [l.strip().upper() for l in clean_text.split(",") if l.strip() and len(l.strip()) == 1 and l.strip().isalpha()]
                # Then try slash separator
                if not letters and "/" in clean_text:
                    letters = [l.strip().upper() for l in clean_text.split("/") if l.strip() and len(l.strip()) == 1 and l.strip().isalpha()]
                # Then try space separator
                if not letters and " " in clean_text:
                    letters = [l.strip().upper() for l in clean_text.split() if l.strip() and len(l.strip()) == 1 and l.strip().isalpha()]
                # Try single letter
                if not letters and len(clean_text) == 1 and clean_text.isalpha():
                    letters = [clean_text.upper()]

                if letters:
                    letters_set = set(letters)
                    if options_mapping:
                        # Map letters to emotions using options_mapping (from current prompt, NOT fixed synonyms)
                        emotions = []
                        for letter in letters:
                            if letter in options_mapping:
                                emotions.append(options_mapping[letter])
                        if emotions:
                            result = frozenset(emotions) if len(emotions) > 1 else emotions[0]
                            return result, letters_set, set(emotions)
                    # If no options_mapping available, return the letters as a frozenset
                    result = frozenset(letters) if len(letters) > 1 else letters[0]
                    return result, letters_set, set()

            return None, set(), set()

        def extract_ground_truth_from_solution(solution_obj, options_mapping=None):
            """Extract ground truth from solution object, supporting multiple field names and formats.

            Priority order for field names:
            1. answer (most common)
            2. solution, label, labels, gt, ground_truth, target, target_answer
            3. emotion, emotions, option, options, correct_answer, correct_option, correct_options

            Returns: (gt_emotion, gt_letters, gt_labels, gt_parse_failed_reason)
            """
            if not solution_obj:
                return None, set(), set(), "solution_obj_is_empty"

            # Flatten solution_obj to string first
            solution_text = flatten_text(solution_obj)
            if not solution_text:
                return None, set(), set(), "solution_text_is_empty"

            # Try to extract from <answer> tags first (standard format)
            answer_from_tags = extract_parts(solution_text, pattern=r'<answer>\s*(.*?)\s*</answer>')
            if answer_from_tags:
                gt_emotion, gt_letters, gt_labels = extract_emotion_from_answer(answer_from_tags, options_mapping)
                if gt_emotion is not None:
                    return gt_emotion, gt_letters, gt_labels, None

            # If solution_obj is dict, try multiple field names
            if isinstance(solution_obj, dict):
                # Priority field names
                field_names = [
                    "answer", "solution", "label", "labels", "gt", "ground_truth",
                    "target", "target_answer", "emotion", "emotions",
                    "option", "options", "correct_answer", "correct_option", "correct_options"
                ]

                for field_name in field_names:
                    if field_name in solution_obj:
                        field_value = solution_obj[field_name]
                        if field_value:
                            # Recursively flatten if it's nested
                            field_text = flatten_text(field_value)
                            if field_text:
                                gt_emotion, gt_letters, gt_labels = extract_emotion_from_answer(field_text, options_mapping)
                                if gt_emotion is not None:
                                    return gt_emotion, gt_letters, gt_labels, None

            # If solution_obj is list, try to extract from each element
            if isinstance(solution_obj, list):
                for item in solution_obj:
                    item_text = flatten_text(item)
                    if item_text:
                        gt_emotion, gt_letters, gt_labels = extract_emotion_from_answer(item_text, options_mapping)
                        if gt_emotion is not None:
                            return gt_emotion, gt_letters, gt_labels, None

            # Last resort: try to parse the flattened text directly
            gt_emotion, gt_letters, gt_labels = extract_emotion_from_answer(solution_text, options_mapping)
            if gt_emotion is not None:
                return gt_emotion, gt_letters, gt_labels, None

            return None, set(), set(), "no_valid_gt_found_in_any_field"

        def flatten_text(obj):
            """Recursively flatten complex nested structures (str, dict, list, nested list) into a single string."""
            if obj is None:
                return ""
            if isinstance(obj, str):
                return obj
            if isinstance(obj, dict):
                parts = []
                # Try common text field names first
                for key in ["text", "content", "question", "prompt", "value", "message"]:
                    if key in obj:
                        flattened = flatten_text(obj[key])
                        if flattened:
                            parts.append(flattened)
                # If no common keys found, flatten all values
                if not parts:
                    for v in obj.values():
                        flattened = flatten_text(v)
                        if flattened:
                            parts.append(flattened)
                return " ".join(parts)
            if isinstance(obj, (list, tuple)):
                parts = []
                for item in obj:
                    flattened = flatten_text(item)
                    if flattened:
                        parts.append(flattened)
                return " ".join(parts)
            return str(obj)

        def extract_parts(text, pattern):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""

        contents = [QwenOmniModule._extract_text_from_completion(completion) for completion in completions]
        rewards = []
        prompts_list = kwargs.get("prompts", [None] * len(completions))
        conversation_list = kwargs.get("conversation", [None] * len(completions))
        emotion_options_text_list = kwargs.get("emotion_options_text", [None] * len(completions))
        raw_user_question_list = kwargs.get("raw_user_question", [None] * len(completions))
        original_question_list = kwargs.get("original_question", [None] * len(completions))
        original_prompt_list = kwargs.get("original_prompt", [None] * len(completions))
        original_conversation_list = kwargs.get("original_conversation", [None] * len(completions))

        for idx, (content, sol) in enumerate(zip(contents, solution)):
            reward = 0.0
            try:
                # Try multiple fields in priority order to find Options mapping
                options_mapping = {}
                options_source_field = None
                options_source_text = None

                # Priority order for options extraction
                candidate_fields = [
                    ("raw_user_question", raw_user_question_list),
                    ("original_question", original_question_list),
                    ("original_prompt", original_prompt_list),
                    ("original_conversation", original_conversation_list),
                    ("emotion_options_text", emotion_options_text_list),
                    ("prompts", prompts_list),
                    ("conversation", conversation_list),
                ]

                for field_name, field_list in candidate_fields:
                    if idx < len(field_list):
                        field_value = field_list[idx]
                        if field_value:
                            field_text = flatten_text(field_value)
                            if field_text:
                                test_mapping = extract_options_mapping_from_text(field_text)
                                if test_mapping:
                                    options_mapping = test_mapping
                                    options_source_field = field_name
                                    options_source_text = field_text
                                    break

                output_answer = extract_parts(content, pattern=r'<answer>\s*(.*?)\s*</answer>')
                no_answer_tag = not output_answer
                if no_answer_tag:
                    pred_emotion = None
                    extracted_answer_display = ""
                    pred_letters = set()
                    pred_labels = set()
                else:
                    pred_emotion, pred_letters, pred_labels = extract_emotion_from_answer(output_answer, options_mapping)
                    extracted_answer_display = output_answer[:100]

                # Use new ground truth extraction function that supports multiple fields
                gt_emotion, gt_letters, gt_labels, gt_parse_failed_reason = extract_ground_truth_from_solution(sol, options_mapping)

                # Debug logging when parsed_ground_truth_labels is empty
                if debug_enabled and rank == 0 and idx < 2:
                    print(f"[EMOTION_REWARD_DEBUG] Sample {idx}:", flush=True)
                    print(f"[EMOTION_REWARD_DEBUG] options_source_field={options_source_field}", flush=True)
                    print(f"[EMOTION_REWARD_DEBUG] options_mapping={options_mapping}", flush=True)

                    # Additional debug info when gt_labels is empty
                    if not gt_labels:
                        raw_solution = flatten_text(sol)
                        solution_type = type(sol).__name__
                        solution_keys = list(sol.keys()) if isinstance(sol, dict) else "N/A"
                        gt_answer_type = type(extract_parts(sol, pattern=r'<answer>\s*(.*?)\s*</answer>')).__name__
                        available_gt_fields = list(sol.keys()) if isinstance(sol, dict) else "N/A"

                        print(f"[EMOTION_REWARD_DEBUG] raw_solution={repr(raw_solution[:200])}", flush=True)
                        print(f"[EMOTION_REWARD_DEBUG] solution_type={solution_type}", flush=True)
                        print(f"[EMOTION_REWARD_DEBUG] solution_keys={solution_keys}", flush=True)
                        gt_answer = extract_parts(sol, pattern=r'<answer>\s*(.*?)\s*</answer>')
                        print(f"[EMOTION_REWARD_DEBUG] gt_answer={repr(gt_answer[:100])}", flush=True)
                        print(f"[EMOTION_REWARD_DEBUG] gt_answer_type={gt_answer_type}", flush=True)
                        print(f"[EMOTION_REWARD_DEBUG] available_gt_fields={available_gt_fields}", flush=True)
                        print(f"[EMOTION_REWARD_DEBUG] gt_parse_failed_reason={gt_parse_failed_reason}", flush=True)

                    print(f"[EMOTION_REWARD_DEBUG] parsed_ground_truth_letters={gt_letters}", flush=True)
                    print(f"[EMOTION_REWARD_DEBUG] parsed_ground_truth_labels={gt_labels}", flush=True)
                    print(f"[EMOTION_REWARD_DEBUG] extracted_answer={repr(extracted_answer_display)}", flush=True)
                    print(f"[EMOTION_REWARD_DEBUG] no_answer_detected={no_answer_tag}", flush=True)
                    print(f"[EMOTION_REWARD_DEBUG] parsed_prediction_letters={pred_letters}", flush=True)
                    print(f"[EMOTION_REWARD_DEBUG] parsed_prediction_labels={pred_labels}", flush=True)

                if pred_emotion is None or gt_emotion is None:
                    reward = 0.0
                elif pred_emotion == gt_emotion:
                    reward = 1.0
                else:
                    reward = 0.0

                if debug_enabled and rank == 0 and idx < 2:
                    print(f"[EMOTION_REWARD_DEBUG] final_emotion_consistency_reward={reward}", flush=True)

            except Exception as e:
                reward = 0.0
                if debug_enabled and rank == 0:
                    print(f"[EMOTION_REWARD_DEBUG] Error in emotion_consistency_reward: {e}", flush=True)
                    import traceback
                    print(f"[EMOTION_REWARD_DEBUG] Traceback: {traceback.format_exc()}", flush=True)

            rewards.append(reward)

        return rewards

