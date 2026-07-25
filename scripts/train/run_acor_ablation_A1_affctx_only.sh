#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ACOR matched reward ablation A1: affective-context only.
# Required: BASE_MODEL, SFT_CHECKPOINT, DATA_ROOT, OUTPUT_ROOT, API.
# All A0–A3 training settings are matched except reward selection.

set -e

echo "=========================================="
echo "ACOR Reward Ablation A1 - Format + Accuracy + Affective Context"
echo "Start checkpoint: SFT checkpoint-2264"
echo "Reward funcs: format,accuracy,affective_context"
echo "Emotion consistency reward disabled"
echo "=========================================="

# Required external resources and environment variables
: "${SFT_CHECKPOINT:?Please set SFT_CHECKPOINT to the ACOR-SFT checkpoint path}"
: "${BASE_MODEL:?Please set BASE_MODEL to the base Qwen2.5-Omni model path}"
: "${OUTPUT_ROOT:?Please set OUTPUT_ROOT}"
: "${DATA_ROOT:?Please set DATA_ROOT}"
export START_CHECKPOINT="${SFT_CHECKPOINT}"
export STAGE1="$START_CHECKPOINT"

# Base model and processor
export GRPO_BASE_MODEL_PATH="$BASE_MODEL"
export GRPO_PROCESSOR_NAME_OR_PATH="$BASE_MODEL"

# Data configuration
export GRPO_DATASET_CONFIG="${PROJECT_ROOT}/configs/grpo/stage1_acor_er.yaml"

# Reward configuration
export GRPO_REWARD_FUNCS="format,accuracy,affective_context"

# Reward weights
export GRPO_AFFECTIVE_CONTEXT_REWARD_WEIGHT=1.0
export GRPO_EMOTION_CONSISTENCY_REWARD_WEIGHT=0.0

# API configuration
export LLM_JUDGE_API_KEY="${LLM_JUDGE_API_KEY:-}"

if [ -z "${LLM_JUDGE_API_BASE:-}" ]; then
    echo "ERROR: LLM_JUDGE_API_BASE environment variable is not set"
    echo "Please set API endpoint for context/reasoning reward LLM judge"
    exit 1
fi

# CUDA environment
module load CUDA/12.3 2>/dev/null || module load cuda/12.3 2>/dev/null || module load CUDA/12.2 2>/dev/null || module load cuda/12.2 2>/dev/null || true

if [ -z "${CUDA_HOME:-}" ]; then
  NVCC_PATH=$(command -v nvcc || true)
  if [ -n "$NVCC_PATH" ]; then
    export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
  elif [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
  elif [ -d /usr/local/cuda-12.3 ]; then
    export CUDA_HOME=/usr/local/cuda-12.3
  elif [ -d /usr/local/cuda-12.2 ]; then
    export CUDA_HOME=/usr/local/cuda-12.2
  fi
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export CUDA_PATH="$CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

TORCH_NVJITLINK_LIB="$CONDA_ENV/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
if [ -d "$TORCH_NVJITLINK_LIB" ]; then
  export LD_LIBRARY_PATH="$TORCH_NVJITLINK_LIB:${LD_LIBRARY_PATH:-}"
fi

export DS_BUILD_OPS=0

# Media configuration
export LAYER2_MAX_VIDEO_FRAMES=16
export LAYER2_MAX_VIDEO_PIXELS=200704
export HUMANOMNI_IMAGE_MAX_PIXELS=200704
export USE_AUDIO_IN_VIDEO=true
export GRPO_DISABLE_AUDIO_INPUTS=0
export GRPO_DISABLE_DEEPSPEED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export GRPO_SUPPRESS_MODAL_SPECIAL_TOKENS=1
export GRPO_LOGPROB_BACKWARD_MICRO_BATCH_SIZE=1
export GRPO_LOGPROB_FORWARD_MICRO_BATCH_SIZE=1
export GRPO_ATTN_IMPLEMENTATION=sdpa
export GRPO_REF_FORCE_VOCAB_CHUNK_LOGPS=1
export GRPO_REF_VOCAB_CHUNK_SIZE=8192
export GRPO_REF_LONG_SEQ_THRESHOLD=4096
export GRPO_DATASET_MAX_RETRY=20

# Training hyperparameters
export MAX_STEPS=500
export LEARNING_RATE=1e-7
export MAX_COMPLETION_LENGTH=768
export LOGGING_STEPS=1
export SAVE_STRATEGY="steps"
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=2

# Generation configuration
export NUM_GENERATIONS=8

# LoRA configuration
export GRPO_LORA_TRAIN_SCOPE="multimodal_joint"
export GRPO_LORA_R=8
export GRPO_LORA_ALPHA=16
export GRPO_LORA_DROPOUT=0.05

# Output directory
export OUTPUT_DIR="${OUTPUT_ROOT}/acor_reward_ablation_A1_affective_only_sft2264_500"

# Output directory check
if [ -d "$OUTPUT_DIR" ]; then
    echo "ERROR: OUTPUT_DIR already exists: $OUTPUT_DIR"
    echo "Please remove the existing directory or change OUTPUT_DIR to prevent overwriting"
    exit 1
fi

# Multi-GPU configuration
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$(python - <<'PYPORT'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PYPORT
)

# DeepSpeed configuration
unset DEEPSPEED_CFG

# Additional flags
export GRPO_GRAD_CKPT_PRESERVE_RNG_STATE=0
export GRPO_DISABLE_AUDIO_TOWER_GRADIENT_CHECKPOINTING=0
export GRPO_DISABLE_LORA_DROPOUT=0
export GRPO_TRUST_LOCAL_RESUME_TORCH_LOAD=0
export GRPO_DDP_FIND_UNUSED_PARAMETERS=false

export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=INFO


mkdir -p "$OUTPUT_DIR"

echo "========== Launching Training =========="
torchrun --nproc_per_node $NPROC_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    ${PROJECT_ROOT}/src/open_r1/grpo_qwenomni_acor_er.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "$START_CHECKPOINT" \
    --processor_name_or_path "$GRPO_PROCESSOR_NAME_OR_PATH" \
    --dataset_name "$GRPO_DATASET_CONFIG" \
    --image_root "$DATA_ROOT/videos/videos" \
    --max_prompt_length 2048 \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --num_generations $NUM_GENERATIONS \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate $LEARNING_RATE \
    --freeze_vision_modules true \
    --use_peft true \
    --lora_r $GRPO_LORA_R \
    --lora_alpha $GRPO_LORA_ALPHA \
    --lora_dropout $GRPO_LORA_DROPOUT \
    --lora_target_modules q_proj v_proj \
    --logging_steps $LOGGING_STEPS \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to none \
    --scale_rewards false \
    --reward_funcs $GRPO_REWARD_FUNCS \
    --affective_context_weight "$GRPO_AFFECTIVE_CONTEXT_REWARD_WEIGHT" \
    --emotion_consistency_weight "$GRPO_EMOTION_CONSISTENCY_REWARD_WEIGHT" \
    --use_audio_in_video true \
    --gradient_checkpointing true \
    --log_completions true \
    --attn_implementation $GRPO_ATTN_IMPLEMENTATION \
    --num_train_epochs 1 \
    --run_name acor_reward_ablation_A1_affective_only_sft2264_500 \
    --save_only_model false \
    --max_steps $MAX_STEPS \
    --save_strategy $SAVE_STRATEGY \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --ddp_find_unused_parameters $GRPO_DDP_FIND_UNUSED_PARAMETERS \
    2>&1 | tee "$OUTPUT_DIR/train.log"

exit ${PIPESTATUS[0]}
