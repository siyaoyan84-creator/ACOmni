#!/usr/bin/env bash
set -euo pipefail

: "${REPAIR_START_CHECKPOINT:?Set REPAIR_START_CHECKPOINT to ACOR-Affective}"
: "${BASE_MODEL:?Set BASE_MODEL}"
: "${REPAIR_DATA_CONFIG:?Set REPAIR_DATA_CONFIG}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
: "${LLM_JUDGE_API_BASE:?Set LLM_JUDGE_API_BASE}"
: "${LLM_JUDGE_API_KEY:?Set LLM_JUDGE_API_KEY}"

torchrun --nproc_per_node "${NPROC_PER_NODE:-1}" \
  src/open_r1/grpo_qwenomni_acor_er.py \
  --output_dir "${OUTPUT_ROOT}/acor_repair" \
  --model_name_or_path "$REPAIR_START_CHECKPOINT" \
  --processor_name_or_path "$BASE_MODEL" \
  --dataset_name "$REPAIR_DATA_CONFIG" \
  --image_root "$DATA_ROOT" \
  --max_prompt_length 2048 \
  --max_completion_length 768 \
  --num_generations 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-8 \
  --freeze_vision_modules true \
  --use_peft true \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj v_proj \
  --bf16 \
  --torch_dtype bfloat16 \
  --data_seed 42 \
  --report_to none \
  --scale_rewards false \
  --reward_funcs format accuracy affective_context emotion_consistency \
  --affective_context_weight 0.02 \
  --emotion_consistency_weight 0.01 \
  --use_audio_in_video true \
  --gradient_checkpointing true \
  --max_steps 50 \
  --save_strategy steps \
  --save_steps 10 \
  --save_total_limit 6
