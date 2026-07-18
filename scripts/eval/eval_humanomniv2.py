import os
import json
import re
import logging
import subprocess
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
import itertools

import argparse
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F

import av
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set.")
    return value


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_if_video_has_audio(video_path):
    try:
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            return False
        return True
    except:
        return False


def extract_audio_to_wav(video_path, audio_cache_dir):
    os.makedirs(audio_cache_dir, exist_ok=True)
    video_basename = os.path.basename(video_path)
    wav_filename = os.path.splitext(video_basename)[0] + ".wav"
    wav_path = os.path.join(audio_cache_dir, wav_filename)

    if os.path.exists(wav_path):
        return wav_path

    try:
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-y",
            wav_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        return wav_path
    except Exception as e:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        raise


def get_first_existing(data, keys, default=""):
    """Get first existing key from data, with nested fallback to 'raw' and 'meta'."""
    for key in keys:
        if key in data and data[key]:
            return data[key]
    for nested_key in ["raw", "meta"]:
        if nested_key in data and isinstance(data[nested_key], dict):
            for key in keys:
                if key in data[nested_key] and data[nested_key][key]:
                    return data[nested_key][key]
    return default

def get_world_answer(data, default=""):
    """Get WorldSense gold answer from multiple locations."""
    answer_keys = [
        "answer", "Answer", "ANSWER", "gt_answer", "gt", "label", "Label",
        "correct_answer", "correctAnswer", "correct_option", "correctOption",
        "correct", "solution", "Solution", "target", "Target"
    ]

    # Priority 1: Check task0.
    if isinstance(data, dict) and "task0" in data:
        task0 = data.get("task0", {})
        if isinstance(task0, dict):
            for key in answer_keys:
                if key in task0 and task0[key]:
                    return task0[key]

    # Priority 2: Check top-level data.
    for key in answer_keys:
        if key in data and data[key]:
            return data[key]

    # Priority 3: Check nested raw/meta (shallow).
    for nested_key in ["raw", "meta"]:
        if nested_key in data and isinstance(data[nested_key], dict):
            for key in answer_keys:
                if key in data[nested_key] and data[nested_key][key]:
                    return data[nested_key][key]

    return default

def get_batch_item(batch, key, index, default=""):
    """Safely get item from batch."""
    if key not in batch:
        return default
    value = batch[key]
    if isinstance(value, (list, tuple)):
        if index < len(value):
            return value[index]
        return default
    return value

def extract_think(output_str):
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return ""
    if len(matches) == 1:
        return matches[0].strip()
    first_answer = matches[0].strip()
    last_answer = matches[-1].strip()
    logger.warning(f"Multiple answer tags detected: multiple_answer_tags=True, answer_count={len(matches)}, first_answer={first_answer}, last_answer={last_answer}, using_first_answer=True")
    return first_answer

def extract_mc_answer(text, valid_letters=None):
    """Extract multiple choice answer from raw model output."""
    if valid_letters is None:
        valid_letters = ['A', 'B', 'C', 'D', 'E']

    # Priority 1: Match after </answer> tag.
    answer_tag_pattern = r'</answer>\s*([A-E])\s*(?:\n|$|[^a-zA-Z])'
    match = re.search(answer_tag_pattern, text)
    if match:
        letter = match.group(1).strip().upper()
        if letter in valid_letters:
            return letter, 'after_answer_tag'

    # Priority 2: Extract from inside <answer>...</answer> tags.
    answer_content_pattern = r'<answer>\s*([A-E])\s*(?:\.|,|\)|$)</answer>'
    match = re.search(answer_content_pattern, text, re.IGNORECASE)
    if match:
        letter = match.group(1).strip().upper()
        if letter in valid_letters:
            return letter, 'inside_answer_tag'

    # Also check for just a letter inside answer tags.
    answer_simple_pattern = r'<answer>\s*([A-E])\s*</answer>'
    match = re.search(answer_simple_pattern, text, re.IGNORECASE)
    if match:
        letter = match.group(1).strip().upper()
        if letter in valid_letters:
            return letter, 'inside_answer_tag'

    # Priority 3: Match a single option letter at end of output.
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line_stripped = line.strip()
        if line_stripped and len(line_stripped) <= 2:
            if line_stripped.upper() in valid_letters:
                return line_stripped.upper(), 'output_tail'

    # Priority 4: Match a letter near trigger phrases.
    trigger_patterns = [
        r'answer\s+is\s+([A-E])',
        r'the\s+answer\s+is\s+([A-E])',
        r'option\s+([A-E])',
        r'choose\s+([A-E])',
        r'i\s+choose\s+([A-E])',
        r'selected?\s+([A-E])',
        r'correct\s+(?:answer|option)\s+(?:is\s+)?([A-E])',
    ]
    for pattern in trigger_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            letter = match.group(1).strip().upper()
            if letter in valid_letters:
                return letter, 'trigger_phrase'

    # Priority 5: Fallback to extract_answer().
    fallback_result = extract_answer(text)
    if fallback_result:
        return fallback_result, 'fallback_extract_answer'

    return "", 'failed'

def normalize_mc_gold_answer(answer, choices=None):
    """Normalize multiple choice gold answer to a single letter."""
    if not answer:
        return "", answer

    answer_str = str(answer).strip()
    original = answer_str

    # Try to extract letter from various formats
    # Format: "A", "A.", "A)", "Answer: A", "Option A", etc.
    patterns = [
        r'^([A-E])\s*$',           # Just letter
        r'^([A-E])[.\)]\s*$',      # Letter with . or )
        r'[Aa]nswer\s*:\s*([A-E])',  # Answer: A
        r'[Oo]ption\s+([A-E])',    # Option A
        r'^([A-E])\s*[.\)]\s*',    # Letter with . or ) at start
    ]

    for pattern in patterns:
        match = re.search(pattern, answer_str)
        if match:
            letter = match.group(1).upper()
            return letter, original

    # Try to match numeric index (0-based: 0->A, 1->B, etc.)
    if answer_str.isdigit():
        idx = int(answer_str)
        if 0 <= idx <= 4:
            letter = chr(ord('A') + idx)
            return letter, original

    # Try to match complete option text against choices
    if choices and isinstance(choices, list):
        answer_lower = answer_str.lower()
        for i, choice in enumerate(choices):
            choice_str = str(choice).strip().lower()
            if choice_str == answer_lower:
                letter = chr(ord('A') + i)
                return letter, original

    # If no letter found, return empty string but keep original for debugging
    return "", original

def parse_all_answers(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    matches = re.findall(pattern, text, re.DOTALL)
    all_answers = [m.strip() for m in matches]
    first_answer = all_answers[0] if all_answers else ""
    last_answer = all_answers[-1] if all_answers else ""
    answer_count = len(all_answers)
    return {
        "all_answers": all_answers,
        "first_answer": first_answer,
        "last_answer": last_answer,
        "answer_count": answer_count
    }

def normalize_problem_type(data, dataset_name):
    """Normalize problem_type across datasets."""
    # Priority 1: Direct field lookup (lowercase and uppercase variants)
    for field_name in ["problem_type", "question_type", "task_type", "type", "Type", "category"]:
        if field_name in data and data[field_name]:
            return data[field_name]

    # Priority 2: Infer from options/choices fields (including Daily-Omni's "Choice")
    for field_name in ["options", "choices", "choice_list", "candidates", "Choice"]:
        if field_name in data and data[field_name]:
            return "multiple choice"

    # Priority 3: Check for option_A to option_D pattern
    if any(f"option_{chr(65+i)}" in data for i in range(4)):
        return "multiple choice"

    # Priority 4: Infer from problem text if it contains option markers
    problem_text = data.get("problem", "") or data.get("question", "") or data.get("Question", "")
    if problem_text and any(marker in problem_text for marker in ["A)", "B)", "C)", "D)", "A.", "B.", "C.", "D."]):
        return "multiple choice"

    # Default fallback
    available_keys = list(data.keys())
    logger.warning(
        f"Missing problem_type field: dataset_name={dataset_name}, "
        f"missing_problem_type=True, inferred_problem_type=free-form, "
        f"available_keys={available_keys}"
    )
    return "free-form"

def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception as e:
        return None
    
def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):

    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)
    
    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
    
    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
    
    conditions = rel_error < (1 - thresholds)  
    mra = conditions.float().mean()  
    return mra.item()


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

def reward_fn(output_ans, gt_ans, question_type, prediction_source=None):
    try:

        if question_type == "multiple choice":
            # For multiple choice, only accept if prediction came from valid sources
            # If it's a fallback or failed extraction, don't count as correct
            if prediction_source in ['fallback_extract_answer', 'failed']:
                # Only match if both are natural language and exactly equal
                if output_ans.strip() == gt_ans.strip():
                    return 1.0
                return 0.0
            # For valid MC sources (after_answer_tag, inside_answer_tag, output_tail, trigger_phrase)
            return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif question_type == "numerical":
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        elif question_type == "regression":
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            mra = mean_relative_accuracy(out_number, gt_number)
            return mra
        elif question_type == "emer_ov_mc":
            return emer_ov_mc(output_ans, gt_ans)
        elif  question_type == "judge":
            return judge(output_ans, gt_ans)

        else:
            return 0.0
    except Exception as e:
        return 0.0


def build_eval_key(sample, sample_index):
    """Build a unique evaluation key for a sample."""
    if isinstance(sample, dict):
        # Priority 1: world_sample_id for WorldSense expanded samples
        if 'world_sample_id' in sample and sample['world_sample_id']:
            return f"world_sample_id_{sample['world_sample_id']}"

        # Priority 2-5: Direct ID fields
        for id_field in ['sample_id', 'id', 'question_id', 'uid']:
            if id_field in sample and sample[id_field]:
                return f"{id_field}_{sample[id_field]}"

        # Priority 6-7: video_id/image_id + question
        for media_field in ['video_id', 'image_id']:
            if media_field in sample and sample[media_field]:
                question = sample.get('question', sample.get('problem', ''))
                if question:
                    # Use hash to keep key short
                    question_hash = hash(question) & 0x7FFFFFFF  # Positive 32-bit hash
                    return f"{media_field}_{sample[media_field]}_q{question_hash}"
                return f"{media_field}_{sample[media_field]}"

        # Priority 8-10: path/video/image + problem
        for media_field in ['path', 'video', 'image']:
            if media_field in sample and sample[media_field]:
                problem = sample.get('problem', sample.get('question', ''))
                if problem:
                    problem_hash = hash(problem) & 0x7FFFFFFF
                    media_basename = os.path.basename(str(sample[media_field]))
                    return f"{media_field}_{media_basename}_p{problem_hash}"
                media_basename = os.path.basename(str(sample[media_field]))
                return f"{media_field}_{media_basename}"

    # Benchmark-specific path selection.
    return ""



SYSTEM_PROMPT = """You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from available various modalities (image, video, audio, text context) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio or other available context information, and then proceed with an in-depth analysis related to the question. 

In reasoning, It is encouraged to incorporate self-reflection and verification into your reasoning process. You are encouraged to review the image, video, audio, or other context information to ensure the answer accuracy.

Provide your understanding of the image, video, and audio between the <context> </context> tags, detail the reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags.
"""

class MyDataset(Dataset):
    def __init__(self, dataset_name, data_path, processor, video_root, max_samples=None):
        super(MyDataset, self).__init__()
        self.dataset_name = dataset_name
        self.video_root = video_root
        self.audio_cache_dir = str(Path(require_env("OUTPUT_ROOT")) / "eval_audio_cache")

        data = []
        if data_path.endswith('.jsonl'):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        elif data_path.endswith('.json'):
            with open(data_path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, list):
                    data = loaded_data
                else:
                    if dataset_name == "world" and isinstance(loaded_data, dict):
                        # WorldSense: expand taskN fields into independent QA samples
                        original_video_count = len(loaded_data)
                        expanded_data = []

                        for video_id, video_sample in loaded_data.items():
                            # Extract video-level fields
                            video_level_fields = {
                                'video_id': video_sample.get('video_id', video_id),
                                'video_duration': video_sample.get('video_duration', ''),
                                'duration': video_sample.get('duration', ''),
                                'domain': video_sample.get('domain', ''),
                                'sub_category': video_sample.get('sub_category', ''),
                                'audio_class': video_sample.get('audio_class', ''),
                                'video_caption': video_sample.get('video_caption', '')
                            }

                            # Find all taskN fields
                            task_keys = sorted([k for k in video_sample.keys() if k.startswith('task') and k[4:].isdigit()],
                                             key=lambda x: int(x[4:]))

                            if not task_keys:
                                logger.warning(f"[WORLD_EXPAND] video_id={video_id} has no task fields, skipping")
                                continue

                            # Expand each taskN into an independent sample
                            for task_key in task_keys:
                                task_data = video_sample[task_key]

                                # Build expanded sample
                                expanded_sample = video_level_fields.copy()
                                expanded_sample['world_sample_id'] = f"{video_id}_{task_key}"
                                expanded_sample['world_original_video_id'] = video_id
                                expanded_sample['world_task_key'] = task_key
                                expanded_sample['question'] = task_data.get('question', '')
                                expanded_sample['answer'] = task_data.get('answer', '')
                                expanded_sample['candidates'] = task_data.get('candidates', [])
                                expanded_sample['task_domain'] = task_data.get('task_domain', '')
                                expanded_sample['task_type'] = task_data.get('task_type', '')

                                expanded_data.append(expanded_sample)

                        data = expanded_data
                        expanded_qa_count = len(data)

                        logger.info(f"[WORLD_EXPAND] original_video_count={original_video_count}")
                        logger.info(f"[WORLD_EXPAND] expanded_qa_count={expanded_qa_count}")
                        if data:
                            first_sample_keys = list(data[0].keys())
                            logger.info(f"[WORLD_EXPAND] first_expanded_sample_keys={first_sample_keys}")
                    else:
                        data = [loaded_data]
        else:
            raise ValueError("Input file must be .json or .jsonl")

        # Truncate data if max_samples is specified
        if max_samples is not None and max_samples > 0:
            original_size = len(data)
            data = data[:max_samples]
            logger.info(f"Dataset truncated: dataset_name={dataset_name}, original_size={original_size}, max_samples={max_samples}, truncated_size={len(data)}")

        self.processor = processor

      

        self.TYPE_TEMPLATE = {
            "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
            "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
            "free-form": " Please provide your text answer within the <answer> </answer> tags.",
            "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "emer_ov_mc": " Please provide only the single or multiple option letter (e.g., A for single option or A,E for multi option, etc.) within the <answer> </answer> tags.",
            "judge": " Please answer Yes or No within the <answer> </answer> tags.",


        }


       
        self.data = data

    def __getitem__(self, index):
            data = self.data[index]

            # For WorldSense, extract fields from expanded sample directly
            if self.dataset_name == "world":
                # WorldSense expanded samples have flat structure
                question = data.get('question', '')

                # Normalize candidates to standard list[str]
                candidates = data.get('candidates', [])

                # Ensure candidates is a list
                if not isinstance(candidates, list):
                    if isinstance(candidates, dict):
                        candidates = list(candidates.values())
                    elif isinstance(candidates, tuple):
                        candidates = list(candidates)
                    else:
                        candidates = [str(candidates)] if candidates else []

                # Filter out empty strings and convert to strings
                cleaned_candidates = []
                for cand in candidates:
                    cand_str = str(cand).strip()
                    if cand_str:
                        cleaned_candidates.append(cand_str)
                candidates = cleaned_candidates

                # Force multiple choice for WorldSense with candidates
                if candidates:
                    problem_type = "multiple choice"
                else:
                    problem_type = "free-form"
                    logger.warning(f"[WORLD_MC] sample_idx={index}, world_sample_id={data.get('world_sample_id', 'unknown')}, missing_candidates=True, fallback_to_free_form=True")

                # Get WorldSense answer and metadata
                world_answer = data.get('answer', '')
                world_sample_id = data.get('world_sample_id', '')
                world_task_key = data.get('world_task_key', '')
                world_domain = data.get('domain', '')
                world_sub_category = data.get('sub_category', '')
                world_audio_class = data.get('audio_class', '')
                world_task_domain = data.get('task_domain', '')
                world_task_type = data.get('task_type', '')

                options = candidates

                # Log WorldSense expanded sample on first 3 samples
                if index < 3:
                    logger.info(
                        f"[WORLD_MC] sample_idx={index}, "
                        f"world_sample_id={world_sample_id}, "
                        f"task_key={world_task_key}, "
                        f"problem_type={problem_type}, "
                        f"gt_answer={repr(world_answer)}, "
                        f"candidates_count={len(candidates)}, "
                        f"question_preview={question[:80] if question else 'EMPTY'}..."
                    )

                daily_type = None
            elif self.dataset_name == "daily":
                # Daily-Omni: read uppercase fields explicitly with case-insensitive fallback
                daily_question = get_first_existing(data, ["Question", "question", "QUESTION"], "")
                daily_choices = get_first_existing(data, ["Choice", "choice", "CHOICE", "choices"], [])
                daily_answer = get_first_existing(data, ["Answer", "answer", "ANSWER", "gt_answer", "label", "Label"], "")
                daily_type = get_first_existing(data, ["Type", "type", "TYPE", "task_type"], "")

                # Log Daily-Omni field resolution on first 3 samples with raw data inspection
                if index < 3:
                    raw_answer_value = data.get("Answer", data.get("answer", data.get("ANSWER", data.get("gt_answer", data.get("label", data.get("Label", "NOT_FOUND"))))))
                    logger.info(
                        f"Daily-Omni sample {index} raw data inspection: "
                        f"raw_keys={list(data.keys())}, "
                        f"raw_Answer_value={repr(raw_answer_value)}, "
                        f"data_has_Answer={('Answer' in data)}, "
                        f"data_has_answer={('answer' in data)}, "
                        f"data_has_ANSWER={('ANSWER' in data)}"
                    )
                    logger.info(
                        f"Daily-Omni sample {index} field mapping: "
                        f"raw_keys={list(data.keys())}, "
                        f"Question_non_empty={bool(daily_question)}, "
                        f"Choice_type={type(daily_choices).__name__}, "
                        f"Choice_len={len(daily_choices) if isinstance(daily_choices, (list, dict)) else 'N/A'}, "
                        f"Answer_resolved={repr(daily_answer)}, "
                        f"Type_resolved={repr(daily_type)}"
                    )

                question = daily_question
                choices = daily_choices

                # Normalize choices to list[str]
                if not isinstance(choices, list):
                    if isinstance(choices, dict):
                        choices = list(choices.values())
                    elif isinstance(choices, tuple):
                        choices = list(choices)
                    else:
                        choices = [str(choices)] if choices else []

                # Filter out empty strings and convert to strings
                cleaned_choices = []
                for choice in choices:
                    choice_str = str(choice).strip()
                    if choice_str:
                        cleaned_choices.append(choice_str)
                choices = cleaned_choices

                options = choices

                # Daily-Omni: force multiple choice if Choice or Answer exists
                if choices:
                    problem_type = "multiple choice"
                else:
                    problem_type = "free-form"

                # Log Daily-Omni field mapping on first sample
                if index == 0:
                    logger.info(
                        f"Daily-Omni field mapping: dataset_name={self.dataset_name}, "
                        f"resolved_question_field=Question, resolved_choice_field=Choice, "
                        f"resolved_answer_field=Answer, resolved_type_field=Type, "
                        f"forced_problem_type={problem_type}, daily_type={daily_type}, "
                        f"options_count={len(options)}"
                    )

                if index < 3:
                    logger.info(
                        f"Daily-Omni sample {index} after processing: "
                        f"problem_type={problem_type}, "
                        f"daily_type={daily_type}, "
                        f"gt_answer={repr(daily_answer)}"
                    )
            else:
                problem_type = normalize_problem_type(data, self.dataset_name)
                # Resolve question field with Daily-Omni support
                question = data.get('problem', data.get('question', data.get('Question', '')))

                # Resolve options/choices field with Daily-Omni support
                options = data.get("options", data.get("choices", data.get("Choice", [])))
                daily_type = None

            if problem_type == 'multiple choice' or problem_type == 'emer_ov_mc':
                question = question + " Options:\n"
                for op in options:
                    question += op + "\n"

            if "socialiq_trans" == self.dataset_name:
                question = question.split("The question is:\n")[-1]

            # Resolve video path with priority: video > path > video_id
            video_path = data.get('video', data.get("path", ""))

            if not video_path:
                # Try video_id as fallback (for WorldSense, this is always present)
                video_id = data.get('video_id', '')
                if video_id:
                    # Check if video_id already has extension
                    if any(video_id.endswith(ext) for ext in ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv']):
                        video_path = os.path.join(self.video_root, video_id)
                    else:
                        tried_nested_paths = []
                        tried_flat_paths = []

                        # For Daily-Omni: try nested layout first (videos/{video_id}/{video_id}_video.ext)
                        if self.dataset_name == "daily":
                            for ext in ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv']:
                                nested_candidate = os.path.join(self.video_root, video_id, f"{video_id}_video{ext}")
                                tried_nested_paths.append(nested_candidate)
                                if os.path.exists(nested_candidate):
                                    video_path = nested_candidate
                                    logger.info(
                                        f"Daily-Omni nested video found: dataset_name={self.dataset_name}, "
                                        f"sample_idx={index}, video_id={video_id}, "
                                        f"resolved_by=daily_nested_layout, resolved_video_path={video_path}"
                                    )
                                    break

                        # If not found in nested layout, try flat layout (videos/{video_id}.ext)
                        if not video_path:
                            for ext in ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv']:
                                flat_candidate = os.path.join(self.video_root, video_id + ext)
                                tried_flat_paths.append(flat_candidate)
                                if os.path.exists(flat_candidate):
                                    video_path = flat_candidate
                                    break

                        # If still not found, raise error with both tried paths
                        if not video_path:
                            error_msg = (
                                f"Video file not found: dataset_name={self.dataset_name}, "
                                f"sample_idx={index}, video_id={video_id}, "
                                f"tried_nested_paths={tried_nested_paths}, "
                                f"tried_flat_paths={tried_flat_paths}, "
                                f"final_reason=missing_video_file"
                            )
                            logger.error(error_msg)
                            raise FileNotFoundError(error_msg)

            if not os.path.isabs(video_path):
                video_path = os.path.join(self.video_root, video_path)

            # Ensure video_path is not a directory
            if os.path.isdir(video_path):
                error_msg = (
                    f"Video path is a directory, not a file: dataset_name={self.dataset_name}, "
                    f"sample_idx={index}, video_path={video_path}, final_reason=directory_instead_of_file"
                )
                logger.error(error_msg)
                raise IsADirectoryError(error_msg)

            # Log Daily-Omni field resolution on first encounter
            if self.dataset_name == "daily" and index == 0:
                logger.info(
                    f"Daily-Omni field mapping: dataset_name={self.dataset_name}, "
                    f"resolved_question_field=Question, resolved_choice_field=Choice, "
                    f"resolved_answer_field=Answer, resolved_video_field=video_id, "
                    f"resolved_video_path={video_path}"
                )

            video_audio_avaliable = check_if_video_has_audio(video_path)

            use_audio = True
            text_prompt = question + self.TYPE_TEMPLATE.get(problem_type, self.TYPE_TEMPLATE["free-form"])
            if video_audio_avaliable:
                try:
                    wav_path = extract_audio_to_wav(video_path, self.audio_cache_dir)
                    logger.info(f"Audio extracted with ffmpeg: dataset_name={self.dataset_name}, sample_idx={index}, video_path={video_path}, wav_path={wav_path}, use_audio=True")
                    message = [{
                        "role": "user",
                        "content": [
                            {
                                "type": data.get('data_type', 'video'),
                                data.get('data_type', 'video'): video_path
                            },
                            {
                                "type": "audio",
                                "audio": wav_path
                            },

                            {
                                "type": "text",
                                "text": f"Here is a {data.get('data_type', 'video')}, with the audio from the video.\n" + text_prompt
                            }
                        ]
                    }]
                except Exception as e:
                    logger.warning(f"Audio extraction failed, fallback_to_video_only=True: dataset_name={self.dataset_name}, sample_idx={index}, video_path={video_path}, err_type={type(e).__name__}, err_msg={str(e)}")
                    message = [{
                        "role": "user",
                        "content": [
                            {
                                "type": data.get('data_type', 'video'),
                                data.get('data_type', 'video'): video_path
                            },

                            {
                                "type": "text",
                                "text": text_prompt
                            }
                        ]
                    }]
                    use_audio = False
            else:
                 message = [{
                    "role": "user",
                    "content": [
                        {
                            "type": data.get('data_type', 'video'),
                            data.get('data_type', 'video'): video_path
                        },

                        {
                            "type": "text",
                            "text": text_prompt
                        }
                    ]
                }]
                 use_audio = False

            message.insert(0, {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    }
                    ]
            })

            try:
                audios, images, videos = process_mm_info(message, use_audio_in_video=False)
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                audio_failure_keywords = ["NoBackendError", "LibsndfileError", "Format not recognised", "Error opening audio", "audio extraction"]

                is_audio_failure = any(keyword in error_type or keyword in error_msg for keyword in audio_failure_keywords)

                if is_audio_failure:
                    logger.warning(f"Audio failure fallback: dataset_name={self.dataset_name}, sample_idx={index}, video_path={video_path}, err_type={error_type}, err_msg={error_msg}, fallback_to_video_only=True")

                    message = [{
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": SYSTEM_PROMPT
                            }
                        ]
                    }, {
                        "role": "user",
                        "content": [
                            {
                                "type": data.get('data_type', 'video'),
                                data.get('data_type', 'video'): video_path
                            },
                            {
                                "type": "text",
                                "text": text_prompt
                            }
                        ]
                    }]
                    use_audio = False

                    try:
                        audios, images, videos = process_mm_info(message, use_audio_in_video=False)
                    except Exception as e2:
                        logger.error(f"Video-only fallback also failed: dataset_name={self.dataset_name}, sample_idx={index}, video_path={video_path}, err_type={type(e2).__name__}, err_msg={str(e2)}")
                        raise
                else:
                    raise
            # Extract solution/answer with WorldSense and Daily-Omni support
            if self.dataset_name == "world":
                solution = world_answer
            elif self.dataset_name == "daily":
                # Daily-Omni: use case-insensitive Answer field lookup
                solution = get_first_existing(data, ["Answer", "answer", "ANSWER", "gt_answer", "label", "Label"], "")
            else:
                solution = data.get("solution", data.get("answer", data.get("Answer", "")))

            data_dict = {
                'images': images,
                'audios': audios,
                'videos': videos,
                'prompt': message,
                'solution': solution,
                "problem_type": problem_type,
                "daily_type": daily_type,
                "raw": data,
                'use_audio': use_audio
            }

            # For Daily-Omni, explicitly add all metadata fields for evaluation chain
            if self.dataset_name == "daily":
                # Use solution which already has case-insensitive lookup
                data_dict["sample_idx"] = index
                data_dict["dataset_name"] = self.dataset_name
                data_dict["gt_answer"] = solution
                data_dict["answer"] = solution
                data_dict["raw_answer"] = solution
                data_dict["task_type"] = daily_type
                data_dict["question"] = question
                data_dict["choices"] = options

                if index < 3:
                    logger.info(
                        f"Daily-Omni sample {index} final data_dict: "
                        f"sample_idx={index}, "
                        f"gt_answer={repr(solution)}, "
                        f"problem_type={problem_type}, "
                        f"daily_type={daily_type}, "
                        f"dict_keys={list(data_dict.keys())}"
                    )
            elif self.dataset_name == "world":
                # For WorldSense, add all metadata fields
                data_dict["sample_idx"] = index
                data_dict["dataset_name"] = self.dataset_name
                data_dict["world_sample_id"] = world_sample_id
                data_dict["world_task_key"] = world_task_key
                data_dict["gt_answer"] = solution
                data_dict["answer"] = solution
                data_dict["raw_answer"] = solution
                data_dict["world_domain"] = world_domain
                data_dict["world_sub_category"] = world_sub_category
                data_dict["world_audio_class"] = world_audio_class
                data_dict["world_task_domain"] = world_task_domain
                data_dict["world_task_type"] = world_task_type
                data_dict["question"] = question
                data_dict["choices"] = options

                if index < 3:
                    logger.info(
                        f"WorldSense sample {index} final data_dict: "
                        f"sample_idx={index}, "
                        f"world_sample_id={world_sample_id}, "
                        f"gt_answer={repr(solution)}, "
                        f"problem_type={problem_type}, "
                        f"world_domain={world_domain}, "
                        f"world_sub_category={world_sub_category}, "
                        f"dict_keys={list(data_dict.keys())}"
                    )
            else:
                # For non-Daily/non-World datasets, also add sample_idx for consistency
                data_dict["sample_idx"] = index
                data_dict["dataset_name"] = self.dataset_name

            return data_dict
          
    def __len__(self):
        return len(self.data)
        
def collate_fn(examples):
    """
    Custom collate function that preserves metadata fields for evaluation.

    For batch_size=1, returns list of dicts.
    For larger batches, groups model inputs and preserves metadata as lists.
    """
    # Metadata fields that must be preserved
    metadata_fields = [
        'sample_idx', 'dataset_name', 'problem_type', 'daily_type', 'task_type',
        'gt_answer', 'answer', 'raw_answer', 'question', 'choices', 'solution',
        'world_sample_id', 'world_task_key', 'world_domain', 'world_sub_category',
        'world_audio_class', 'world_task_domain', 'world_task_type'
    ]

    # For batch_size=1 (current use case), return examples as-is
    # This preserves all fields including metadata
    if len(examples) == 1:
        return examples

    # For larger batches, group model inputs and preserve metadata
    batch = {}

    # Collect model input fields (images, videos, audios, prompt, use_audio)
    model_input_fields = ['images', 'audios', 'videos', 'prompt', 'use_audio', 'raw']
    for field in model_input_fields:
        batch[field] = [ex.get(field) for ex in examples]

    # Preserve metadata fields as lists
    for field in metadata_fields:
        batch[field] = [ex.get(field) for ex in examples]

    return batch

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        # shard_size = total_size // world_size
        # left = total_size % world_size
        # shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        # begin = sum(shard_sizes[:rank])
        # end = min(sum(shard_sizes[:rank + 1]), total_size)
        # return range(begin, end)
        indices = []
        for i in range(total_size):
            if i % world_size == rank:
                indices.append(i)
        return indices

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


datasets = {
    "daily": {
        "gt_path": Path(require_env("DAILY_OMNI_ROOT")) / "qa_think.json",
        "video_root": Path(require_env("DAILY_OMNI_ROOT")) / "videos"
    },
    "world": {
        "gt_path": Path(require_env("WORLDSENSE_ROOT")) / "qa_think.json",
        "video_root": Path(require_env("WORLDSENSE_ROOT")) / "videos"
    },
    "ib": {
        "gt_path": Path(require_env("INTENTBENCH_ROOT")) / "qa.json",
        "video_root": Path(require_env("INTENTBENCH_ROOT")) / "videos"
    }
}



def main(args):

    file_name = args.file_name

    # Detect distributed environment
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))

    distributed = world_size > 1

    if distributed:
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(0)

    logger.info(f"[EVAL_DIST] distributed = {distributed}")
    logger.info(f"[EVAL_DIST] rank = {rank}")
    logger.info(f"[EVAL_DIST] local_rank = {local_rank}")
    logger.info(f"[EVAL_DIST] world_size = {world_size}")
    logger.info(f"[EVAL_DIST] cuda_device = {torch.cuda.current_device()}")

    eval_resume = os.getenv('EVAL_RESUME', '0') == '1'
    eval_save_every_sample = os.getenv('EVAL_SAVE_EVERY_SAMPLE', '1') == '1'

    if rank == 0:
        logger.info("Benchmark dataset paths:")
        logger.info(f"IntentBench root = {datasets['ib']['video_root']}")
        logger.info(f"Daily-Omni root = {datasets['daily']['video_root']}")
        logger.info(f"WorldSense root = {datasets['world']['video_root']}")

    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    adapter_model_path = os.path.join(args.model_path, "adapter_model.safetensors")
    is_adapter_only = os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path)

    logger.info(f"[EVAL_ADAPTER_LOAD] model_path = {args.model_path}")
    logger.info(f"[EVAL_ADAPTER_LOAD] is_adapter_only = {is_adapter_only}")

    if is_adapter_only:
        base_model_path = None

        if os.environ.get("EVAL_BASE_MODEL_PATH"):
            base_model_path = os.environ.get("EVAL_BASE_MODEL_PATH")
            logger.info(f"[EVAL_ADAPTER_LOAD] using EVAL_BASE_MODEL_PATH from env: {base_model_path}")

        elif os.environ.get("BASE_MODEL"):
            base_model_path = os.environ.get("BASE_MODEL")
            logger.info(f"[EVAL_ADAPTER_LOAD] using BASE_MODEL from env: {base_model_path}")

        # Priority 3: Default local base model
        else:
            base_model_path = os.environ.get("BASE_MODEL") or os.environ.get("MODEL_PATH")
            if not base_model_path:
                raise RuntimeError("BASE_MODEL or MODEL_PATH must be set for evaluation")
            logger.info(f"[EVAL_ADAPTER_LOAD] using default base model path: {base_model_path}")

        logger.info(f"[EVAL_ADAPTER_LOAD] base_model_path = {base_model_path}")
        logger.info(f"[EVAL_ADAPTER_LOAD] loading base model first")

        # Load base model
        from peft import PeftModel

        base_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
            local_files_only=True,
        )

        logger.info(f"[EVAL_ADAPTER_LOAD] loading adapter with PeftModel.from_pretrained")

        # Load adapter
        model = PeftModel.from_pretrained(
            base_model,
            args.model_path,
            is_trainable=False,
        )

        model.eval()

        logger.info(f"[EVAL_ADAPTER_LOAD] adapter loaded successfully")

        # Load processor from base model path
        processor_path = os.environ.get("EVAL_PROCESSOR_PATH") or os.environ.get("PROCESSOR_NAME_OR_PATH") or base_model_path
        logger.info(f"[EVAL_PROCESSOR_LOAD] processor_path = {processor_path}")
        processor = Qwen2_5OmniProcessor.from_pretrained(processor_path, local_files_only=True)

    else:
        # Load full model directly
        logger.info(f"[EVAL_ADAPTER_LOAD] loading full model from {args.model_path}")
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
            local_files_only=True,
        )

        # Load processor
        processor_path = os.environ.get("EVAL_PROCESSOR_PATH") or os.environ.get("PROCESSOR_NAME_OR_PATH") or args.model_path
        logger.info(f"[EVAL_PROCESSOR_LOAD] processor_path = {processor_path}")
        processor = Qwen2_5OmniProcessor.from_pretrained(processor_path, local_files_only=True)



    for dataset_name in args.dataset:
        logger.info(f"Processing file: {file_name}")
        output_root = Path(require_env("OUTPUT_ROOT")).expanduser().resolve()
        eval_root = output_root / "eval_results"
        OUTPUT_PATH = eval_root / f"{dataset_name}_{file_name}.json"
        gt_path = str(datasets[dataset_name]["gt_path"])
        video_root = str(datasets[dataset_name]["video_root"])
        dataset = MyDataset(dataset_name, gt_path, processor, video_root, max_samples=args.max_samples)

        rank_predictions_path = eval_root / f"{dataset_name}_{file_name}_predictions_rank{rank}.jsonl"
        rank_predictions_path.parent.mkdir(parents=True, exist_ok=True)

        # Load completed sample indices for resume.
        completed_sample_indices = set()
        duplicate_completed_count = 0

        if eval_resume:
            logger.info(f"[EVAL_RESUME] enabled = True")
            logger.info(f"[EVAL_RESUME] rank_predictions_path = {rank_predictions_path}")

            if os.path.exists(rank_predictions_path):
                logger.info(f"[EVAL_RESUME] loaded_existing = True")

                existing_records = []
                sample_index_count = {}

                with open(rank_predictions_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            existing_records.append(record)

                            if 'sample_index' in record and record['sample_index'] is not None:
                                sample_idx = int(record['sample_index'])
                                completed_sample_indices.add(sample_idx)
                                sample_index_count[sample_idx] = sample_index_count.get(sample_idx, 0) + 1
                        except Exception as e:
                            logger.warning(f"[EVAL_RESUME] WARNING: failed to parse line: {e}")
                            pass

                # Check for duplicates
                duplicate_indices = [idx for idx, count in sample_index_count.items() if count > 1]
                duplicate_completed_count = sum(count - 1 for count in sample_index_count.values() if count > 1)

                if duplicate_indices:
                    logger.warning(f"[EVAL_RESUME] duplicate_completed_sample_indices = {sorted(duplicate_indices)[:10]} (showing first 10)")
                    logger.warning(f"[EVAL_RESUME] duplicate_completed_count = {duplicate_completed_count}")

                    # Check if dedup is enabled
                    dedup_existing = os.getenv('EVAL_RESUME_DEDUP_EXISTING', '0') == '1'
                    if dedup_existing:
                        import time
                        timestamp = int(time.time())
                        backup_path = Path(f"{rank_predictions_path}.bak_{timestamp}")

                        # Backup original file
                        import shutil
                        shutil.copy2(str(rank_predictions_path), str(backup_path))
                        logger.info(f"[EVAL_RESUME] dedup_existing = True")
                        logger.info(f"[EVAL_RESUME] backup_path = {backup_path}")
                        logger.info(f"[EVAL_RESUME] dedup_before_lines = {len(existing_records)}")

                        # Deduplicate: keep last occurrence of each sample_index
                        unique_records = {}
                        for record in existing_records:
                            if 'sample_index' in record and record['sample_index'] is not None:
                                sample_idx = int(record['sample_index'])
                                unique_records[sample_idx] = record

                        # Rewrite jsonl with unique records
                        with open(rank_predictions_path, 'w', encoding='utf-8') as f:
                            for sample_idx in sorted(unique_records.keys()):
                                f.write(json.dumps(unique_records[sample_idx], ensure_ascii=False) + '\n')

                        logger.info(f"[EVAL_RESUME] dedup_after_unique = {len(unique_records)}")

                        # Update completed_sample_indices to deduplicated set
                        completed_sample_indices = set(unique_records.keys())
                    else:
                        logger.warning(f"[EVAL_RESUME] Set EVAL_RESUME_DEDUP_EXISTING=1 to deduplicate existing file")

                logger.info(f"[EVAL_RESUME] rank_completed_count = {len(completed_sample_indices)}")
                logger.info(f"[EVAL_RESUME] completed_sample_indices_head = {sorted(list(completed_sample_indices))[:10]}")
            else:
                logger.info(f"[EVAL_RESUME] loaded_existing = False")

        # Get all samples and filter by rank.
        all_samples = list(range(len(dataset)))
        rank_samples = [i for i in all_samples if i % world_size == rank]

        logger.info(f"[EVAL_DIST] total_count = {len(all_samples)}")
        logger.info(f"[EVAL_DIST] rank_count = {len(rank_samples)}")
        logger.info(f"[EVAL_DIST] rank_start_example_indices = {rank_samples[:5]}")

        # Filter remaining samples for this rank.
        remaining_rank_samples = [i for i in rank_samples if i not in completed_sample_indices]

        if eval_resume:
            logger.info(f"[EVAL_RESUME] rank_total_count = {len(rank_samples)}")
            logger.info(f"[EVAL_RESUME] rank_remaining_count = {len(remaining_rank_samples)}")
            logger.info(f"[EVAL_RESUME] remaining_sample_indices_head = {remaining_rank_samples[:10]}")
            logger.info(f"[EVAL_RESUME] skip_by_sample_index = True")

        # Create dataloader with InferenceSampler for distributed evaluation
        if distributed:
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                sampler=InferenceSampler(len(dataset)),
                batch_size=1,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )

        # import pdb; pdb.set_trace()

        final_output = []
        mean_acc = []
        mean_mra = []

        gts = []
        sources = []
        rets = []

        # if os.path.exists(OUTPUT_PATH):
        #     continue


        for inputs in tqdm(dataloader, desc=f"Processing batches (rank {rank})"):

            # import ipdb; ipdb.set_trace()

            # Check if this sample should be skipped in resume mode.
            sample = inputs[0]
            sample_index = sample.get('sample_idx', -1)

            # Skip by sample_index first (most reliable)
            if eval_resume and sample_index in completed_sample_indices:
                logger.info(f"[EVAL_RESUME] skip sample_index = {sample_index}")
                continue

            raw_sample = sample.get('raw', {})
            eval_key = build_eval_key(raw_sample, sample_index)

            images, videos, audios, prompts = [], [], [], []
            use_audio_batch = True
            for each in inputs:
                prompts.append(each["prompt"])
                use_audio_batch = each.get("use_audio", True)
                if each["images"] is not None:
                    images.extend(each["images"])
                if each["audios"] is not None and use_audio_batch:
                    audios.extend(each["audios"])
                if each["videos"] is not None:
                    videos.extend(each["videos"])
            if len(images) == 0: images = None
            if len(audios) == 0: audios = None
            if len(videos) == 0: videos = None

            text = processor.apply_chat_template(
                prompts,
                tokenize=False,
                add_generation_prompt=True,
            )
            # print(text)
            if use_audio_batch:
                model_inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
            else:
                logger.info(f"Processor input uses video-only mode: dataset_name={dataset_name}, sample_idx={inputs[0]['raw'].get('id', 'unknown')}, video_path={inputs[0]['raw'].get('video', inputs[0]['raw'].get('path', 'unknown'))}")
                model_inputs = processor(text=text, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)


            model_inputs = model_inputs.to(model.device).to(model.dtype)
            with torch.inference_mode():
                text_ids = model.generate(**model_inputs, use_audio_in_video=False, max_new_tokens=2048,
                # do_sample=True,
                # temperature=0.8,
                # top_p=0.9,
                )


            for j, (sample, model_output, input_text) in enumerate(zip(inputs, text_ids, model_inputs.input_ids)):
                model_output = processor.decode(model_output[input_text.size(0):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # model_output = model_output.replace(input_text, "")

                # Debug logging: only print sample info if --debug-print-samples is enabled
                if args.debug_print_samples > 0 and len(rets) < args.debug_print_samples:
                    sample_idx = sample.get('sample_idx', 'unknown')
                    logger.info(
                        f"[debug sample {len(rets)}] "
                        f"sample_idx={sample_idx}, "
                        f"dataset={dataset_name}, "
                        f"output_preview={model_output[:300]}"
                    )

                sample.pop("images")
                sample.pop("videos")
                sample.pop("audios")
                sample.pop("use_audio", None)

                rets.append(model_output)
                gts.append(sample["solution"])
                sources.append(sample)

                # Save every sample to rank-specific jsonl if enabled.
                if eval_save_every_sample:
                    sample_index = sample.get('sample_idx', -1)
                    raw_sample = sample.get('raw', {})
                    eval_key = build_eval_key(raw_sample, sample_index)

                    # Extract prediction and compute reward
                    problem_type = sample.get('problem_type', 'unknown')
                    is_mc_sample = (
                        problem_type in ["multiple choice", "multiple_choice", "multi-choice"]
                        or dataset_name in ["daily", "world"]
                    )

                    if is_mc_sample:
                        extracted_pred, prediction_source = extract_mc_answer(model_output)
                        final_ans = extracted_pred
                    else:
                        final_ans = extract_answer(model_output)
                        prediction_source = 'standard_extract_answer'

                    if final_ans == "":
                        final_ans = model_output

                    # Get ground truth
                    gt_ans_raw = sample.get("gt_answer", "")
                    if not gt_ans_raw:
                        gt_ans_raw = sample.get("answer", "")
                    if not gt_ans_raw:
                        gt_ans_raw = sample.get("raw_answer", "")
                    if not gt_ans_raw:
                        gt_ans_raw = extract_answer(sample["solution"])

                    choices = sample.get('choices', [])
                    normalized_gt_ans, gt_original = normalize_mc_gold_answer(gt_ans_raw, choices=choices)

                    if is_mc_sample:
                        comparison_pred = extracted_pred
                        comparison_gt = normalized_gt_ans
                    else:
                        comparison_pred = final_ans
                        comparison_gt = gt_ans_raw

                    reward = reward_fn(comparison_pred, comparison_gt, problem_type, prediction_source)

                    # Build record
                    record = {
                        "eval_key": eval_key,
                        "sample_index": sample_index,
                        "rank": rank,
                        "world_size": world_size,
                        "dataset": dataset_name,
                        "file_name": file_name,
                        "output": model_output,
                        "prediction": final_ans,
                        "gt": comparison_gt,
                        "reward": reward,
                        "problem_type": problem_type,
                        "source_sample": raw_sample
                    }

                    # Guard against duplicate append in same run
                    if sample_index in completed_sample_indices:
                        logger.warning(f"[EVAL_RESUME_GUARD] skip_duplicate_sample_index = {sample_index}")
                    else:
                        # Append to rank-specific jsonl
                        with open(rank_predictions_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')

                        # Add to completed set to prevent duplicate in this run
                        completed_sample_indices.add(sample_index)

                        logger.info(f"[EVAL_RESUME] append rank = {rank}")
                        logger.info(f"[EVAL_RESUME] append sample_index = {sample_index}")
                        logger.info(f"[EVAL_RESUME] append eval_key = {eval_key}")
                
                

        # Barrier to ensure all ranks finish evaluation.
        if distributed:
            torch.distributed.barrier()

        # Rank 0 merges all rank predictions.
        if rank == 0:
            logger.info(f"[EVAL_MERGE] rank0 merging predictions")

            # Find all rank prediction files.
            rank_files = []
            for r in range(world_size):
                rank_file = eval_root / f"{dataset_name}_{file_name}_predictions_rank{r}.jsonl"
                if rank_file.exists():
                    rank_files.append(rank_file)

            logger.info(f"[EVAL_MERGE] found_rank_files = {rank_files}")

            # Load all records from all ranks.
            merged_records = []
            seen_sample_indices = set()
            duplicate_count = 0

            for rank_file in rank_files:
                with open(rank_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            sample_idx = record.get('sample_index', -1)
                            # Deduplicate by sample_index
                            if sample_idx in seen_sample_indices:
                                duplicate_count += 1
                                continue
                            seen_sample_indices.add(sample_idx)
                            merged_records.append(record)
                        except:
                            pass

            # Sort by sample_index
            merged_records.sort(key=lambda x: x.get('sample_index', -1))

            logger.info(f"[EVAL_MERGE] raw_lines = {sum(1 for f in rank_files for _ in open(f))}")
            logger.info(f"[EVAL_MERGE] merged_count = {len(merged_records)}")
            logger.info(f"[EVAL_MERGE] duplicate_count = {duplicate_count}")
            logger.info(f"[EVAL_MERGE] unique_sample_index = {len(seen_sample_indices)}")
            logger.info(f"[EVAL_MERGE] dataset_len = {len(dataset)}")

            # Compute final accuracy from merged records.
            reward_sum = 0
            final_output = []

            for record in merged_records:
                reward_sum += record.get('reward', 0)

                # Reconstruct sample dict for final output
                sample_dict = record.get('source_sample', {})
                sample_dict['output'] = record.get('output', '')
                sample_dict['prediction'] = record.get('prediction', '')
                sample_dict['reward'] = record.get('reward', 0)
                sample_dict['problem_type'] = record.get('problem_type', 'unknown')

                final_output.append(sample_dict)

            final_acc = {'mean_acc': 0.0}
            if len(final_output) > 0:
                final_acc['mean_acc'] = float(reward_sum) / len(final_output)

            logger.info(f"Final accuracy: {final_acc}")

            # Write final result
            try:
                OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump({"results": final_output, "final_acc": [final_acc]}, f, indent=2, ensure_ascii=False)
                logger.info(f"[EVAL_MERGE] final_result_written = {OUTPUT_PATH}")
            except Exception as e:
                logger.error(f"Error writing final accuracy to output file: {e}")

            # Generate answer analysis from merged records.
            analysis_path = eval_root / f"{dataset_name}_{file_name}_answer_analysis.jsonl"
            with open(analysis_path, "w", encoding="utf-8") as analysis_file:
                for record in merged_records:
                    source_sample = record.get('source_sample', {})
                    output = record.get('output', '')
                    prediction = record.get('prediction', '')
                    gt = record.get('gt', '')
                    reward = record.get('reward', 0)
                    problem_type = record.get('problem_type', 'unknown')
                    sample_index = record.get('sample_index', -1)

                    answer_info = parse_all_answers(output)

                    analysis_record = {
                        "dataset_name": dataset_name,
                        "sample_idx": sample_index,
                        "problem_type": problem_type,
                        "gt_answer": gt,
                        "raw_output": output,
                        "all_answers": answer_info["all_answers"],
                        "answer_count": answer_info["answer_count"],
                        "first_answer": answer_info["first_answer"],
                        "last_answer": answer_info["last_answer"],
                        "chosen_answer_for_eval": prediction,
                        "is_correct": reward > 0.5,
                        "is_multi_answer": answer_info["answer_count"] > 1
                    }

                    analysis_file.write(json.dumps(analysis_record, ensure_ascii=False) + "\n")

            logger.info(f"Results saved to {OUTPUT_PATH}")
            logger.info(f"Answer analysis saved to {analysis_path}")

        else:
            # Non-rank0: just log completion.
            logger.info(f"[EVAL_DIST] rank {rank} evaluation completed")

        # Final barrier
        if distributed:
            torch.distributed.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation benchmark")
    parser.add_argument('--model-path', type=str, required=False, help="Path to the model")
    parser.add_argument('--file-name', type=str, required=False, help="Name of the file", default="debug")
    parser.add_argument('--dataset', default=['ib', 'daily', 'world'], nargs='+', type=str)
    parser.add_argument('--max-samples', type=int, default=None, help="Maximum number of samples to evaluate (for debugging)")
    parser.add_argument('--debug-print-samples', type=int, default=0, help="Number of samples to print debug info for (0=disabled, N=print first N samples)")
    args = parser.parse_args()

    main(args)
