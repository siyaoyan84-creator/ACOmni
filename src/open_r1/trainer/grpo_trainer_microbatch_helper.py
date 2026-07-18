"""Helper functions for GRPO micro-batch slicing."""

import torch


def slice_model_inputs_for_batch(model_inputs, start, end, rank=0):
    """Slice batch-first and packed multimodal inputs for micro-batch execution."""
    sliced_inputs = {}

    batch_first_keys = {
        "input_ids", "attention_mask", "completion_ids", "completion_mask",
        "labels", "target_ids", "input_features", "feature_attention_mask",
        "audio_feature_lengths", "audio_seqlens",
    }
    grid_keys = {"video_grid_thw", "image_grid_thw"}
    packed_keys = {"pixel_values_videos", "pixel_values", "pixel_values_images"}

    for key, value in model_inputs.items():
        if value is None:
            sliced_inputs[key] = None
            continue

        if key in batch_first_keys or key in grid_keys:
            sliced_inputs[key] = value[start:end] if isinstance(value, torch.Tensor) else value
            continue

        if key == "pixel_values_videos" and isinstance(value, torch.Tensor) and "video_grid_thw" in model_inputs:
            grid = model_inputs["video_grid_thw"]
            if isinstance(grid, torch.Tensor):
                patch_counts = [int(t) * int(h) * int(w) for t, h, w in grid.tolist()]
                cumsum = [0] + list(torch.tensor(patch_counts).cumsum(0).tolist())
                sliced_inputs[key] = value[cumsum[start]:cumsum[end]]
                continue

        if key in {"pixel_values", "pixel_values_images"} and isinstance(value, torch.Tensor) and "image_grid_thw" in model_inputs:
            grid = model_inputs["image_grid_thw"]
            if isinstance(grid, torch.Tensor):
                patch_counts = [int(t) * int(h) * int(w) for t, h, w in grid.tolist()]
                cumsum = [0] + list(torch.tensor(patch_counts).cumsum(0).tolist())
                sliced_inputs[key] = value[cumsum[start]:cumsum[end]]
                continue

        if isinstance(value, torch.Tensor) and value.dim() > 0:
            batch_size = model_inputs["input_ids"].shape[0] if "input_ids" in model_inputs else None
            sliced_inputs[key] = value[start:end] if batch_size is not None and value.shape[0] == batch_size else value
        else:
            sliced_inputs[key] = value

    return sliced_inputs
