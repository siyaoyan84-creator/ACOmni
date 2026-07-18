import os
import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# Custom exception for overlong prompts
class OverlongPromptError(ValueError):
    """Raised when prompt token length exceeds maximum allowed tokens."""
    pass


def extract_media_paths(sample: Any, max_depth: int = 10) -> Tuple[List[str], List[str], List[str]]:
    """
    Recursively extract media file paths from a sample.

    Returns:
        (video_paths, image_paths, audio_paths)
    """
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    audio_exts = {'.wav', '.mp3', '.flac', '.m4a', '.aac'}

    video_paths = []
    image_paths = []
    audio_paths = []

    def _recurse(obj, depth=0):
        if depth > max_depth:
            return

        if isinstance(obj, str):
            path_lower = obj.lower()
            ext = os.path.splitext(path_lower)[1]
            if ext in video_exts:
                video_paths.append(obj)
            elif ext in image_exts:
                image_paths.append(obj)
            elif ext in audio_exts:
                audio_paths.append(obj)
        elif isinstance(obj, dict):
            for key, val in obj.items():
                if key in {'video', 'image', 'audio', 'file', 'path', 'video_path', 'image_path', 'audio_path'}:
                    _recurse(val, depth + 1)
                else:
                    _recurse(val, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _recurse(item, depth + 1)

    _recurse(sample)
    return video_paths, image_paths, audio_paths


class BadSampleTracker:
    """Tracks and logs bad samples during training."""

    def __init__(self):
        self.enable = os.getenv('GRPO_SKIP_BAD_SAMPLES', '0') == '1'
        self.log_path = os.getenv(
            'GRPO_BAD_SAMPLE_LOG',
            'os.environ.get("OUTPUT_ROOT") + "/"logs/grpo_bad_samples.jsonl'
        )
        self.max_bad_samples = int(os.getenv('GRPO_MAX_BAD_SAMPLES', '100'))
        self.max_bad_ratio = float(os.getenv('GRPO_MAX_BAD_SAMPLE_RATIO', '0.05'))
        self.verbose = int(os.getenv('GRPO_BAD_SAMPLE_VERBOSE', '1')) == 1

        self.bad_count = 0
        self.seen_count = 0
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.pid = os.getpid()

        if self.enable and self.rank == 0:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def record_bad_sample(
        self,
        sample: Dict[str, Any],
        error_type: str,
        error_message: str,
        sample_index: int,
        dataset_name: Optional[str] = None,
        tb_str: Optional[str] = None,
    ):
        """Record a bad sample to the log file."""
        if not self.enable:
            return

        self.bad_count += 1
        self.seen_count += 1

        video_paths, image_paths, audio_paths = extract_media_paths(sample)

        question = ""
        if isinstance(sample, dict):
            if 'problem' in sample:
                question = str(sample['problem'])[:200]
            elif 'prompt' in sample:
                question = str(sample['prompt'])[:200]
            elif 'conversation' in sample:
                conv = sample['conversation']
                if isinstance(conv, list) and len(conv) > 0:
                    question = str(conv)[:200]

        source_file = sample.get('json_file', sample.get('source_file', ''))

        record = {
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            'pid': self.pid,
            'sample_index': sample_index,
            'dataset_name': dataset_name,
            'source_file': source_file,
            'video_path': video_paths,
            'image_path': image_paths,
            'audio_path': audio_paths,
            'question': question,
            'error_type': error_type,
            'error_message': error_message[:500],
            'traceback': (tb_str or '')[:2000],
            'bad_count': self.bad_count,
            'seen_count': self.seen_count,
            'bad_ratio': self.bad_count / max(1, self.seen_count),
        }

        if self.rank == 0:
            try:
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps(record) + '\n')
                    f.flush()
            except Exception as e:
                logger.warning(f"Failed to write bad sample log: {e}")

        if self.verbose:
            logger.warning(
                f"[BadSample] idx={sample_index} type={error_type} "
                f"bad={self.bad_count}/{self.seen_count} ratio={self.bad_count/max(1, self.seen_count):.3f}"
            )

    def check_threshold(self):
        """Check if bad sample threshold is exceeded."""
        if not self.enable:
            return

        bad_ratio = self.bad_count / max(1, self.seen_count)

        if self.bad_count >= self.max_bad_samples:
            raise RuntimeError(
                f"Bad sample count ({self.bad_count}) exceeded threshold ({self.max_bad_samples})"
            )

        if bad_ratio >= self.max_bad_ratio:
            raise RuntimeError(
                f"Bad sample ratio ({bad_ratio:.3f}) exceeded threshold ({self.max_bad_ratio})"
            )

    def increment_seen(self):
        """Increment seen count without recording a bad sample."""
        self.seen_count += 1


class SafeDatasetWrapper:
    """Wraps a dataset to skip bad samples."""

    def __init__(self, dataset, tracker: BadSampleTracker, max_retry: int = 20):
        self.dataset = dataset
        self.tracker = tracker
        self.max_retry = min(max_retry, len(dataset))
        self.dataset_name = getattr(dataset, '__class__.__name__', 'unknown')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if not self.tracker.enable:
            return self.dataset[idx]

        original_idx = idx

        for attempt in range(self.max_retry):
            try:
                sample = self.dataset[idx]
                self.tracker.increment_seen()
                return sample
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                tb_str = traceback.format_exc()

                self.tracker.record_bad_sample(
                    sample={},
                    error_type=error_type,
                    error_message=error_msg,
                    sample_index=idx,
                    dataset_name=self.dataset_name,
                    tb_str=tb_str,
                )

                self.tracker.check_threshold()

                if attempt < self.max_retry - 1:
                    idx = (idx + 1) % len(self.dataset)
                else:
                    raise RuntimeError(
                        f"Failed to load sample after {self.max_retry} retries. "
                        f"Original index: {original_idx}, Last error: {error_msg}"
                    )


class SafeCollatorWrapper:
    """Wraps a collator to skip bad samples during batch construction."""

    def __init__(self, collator, tracker: BadSampleTracker):
        self.collator = collator
        self.tracker = tracker

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.tracker.enable:
            return self.collator(features)

        if not features:
            return self.collator(features)

        good_features = []

        for i, feature in enumerate(features):
            try:
                good_features.append(feature)
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                tb_str = traceback.format_exc()

                self.tracker.record_bad_sample(
                    sample=feature,
                    error_type=error_type,
                    error_message=error_msg,
                    sample_index=i,
                    tb_str=tb_str,
                )

                self.tracker.check_threshold()

        if not good_features:
            raise RuntimeError(
                f"All {len(features)} samples in batch failed processing. "
                f"Total bad samples: {self.tracker.bad_count}"
            )

        try:
            return self.collator(good_features)
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            tb_str = traceback.format_exc()

            for idx, feature in enumerate(good_features):
                self.tracker.record_bad_sample(
                    sample=feature,
                    error_type=f"collator_{error_type}",
                    error_message=error_msg,
                    sample_index=idx,
                    tb_str=tb_str,
                )

            self.tracker.check_threshold()
            raise RuntimeError(
                f"Collator failed on batch of {len(good_features)} samples: {error_msg}"
            )
