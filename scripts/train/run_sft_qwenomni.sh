# ACOR SFT training entrypoint.
# Required: OUTPUT_ROOT.
# Project paths are resolved from this script location.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Priority: positional args > environment variables > defaults

# NNODES (world size)
if [ -n "$1" ]; then
    NNODES=$1
elif [ -n "$NNODES" ]; then
    NNODES=$NNODES
else
    NNODES=1
fi

if [ -n "$2" ]; then
    NPROC_PER_NODE=$2
elif [ -n "$NPROC_PER_NODE" ]; then
    NPROC_PER_NODE=$NPROC_PER_NODE
else
    NPROC_PER_NODE=1
fi

if [ -n "$NODE_RANK" ]; then
    NODE_RANK=$NODE_RANK
elif [ -n "$RANK" ]; then
    NODE_RANK=$RANK
else
    NODE_RANK=0
fi

if [ -n "$MASTER_ADDR" ]; then
    MASTER_ADDR=$MASTER_ADDR
else
    MASTER_ADDR="127.0.0.1"
fi

if [ -n "$MASTER_PORT" ]; then
    MASTER_PORT=$MASTER_PORT
else
    MASTER_PORT=16666
fi

RUN_NAME="qwenomni-sft"
: "${OUTPUT_ROOT:?Please set OUTPUT_ROOT}"
export LOG_PATH="${OUTPUT_ROOT}/debug_log_$RUN_NAME.txt"

# SFT LoRA scope: model_layers_only, audio_tower_only, or joint.
export SFT_LORA_TRAIN_SCOPE="${SFT_LORA_TRAIN_SCOPE:-model_layers_only}"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    VISIBLE_GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    if [ "$NPROC_PER_NODE" -gt "$VISIBLE_GPU_COUNT" ]; then
        exit 1
    fi
fi
mkdir -p "${OUTPUT_ROOT}/$RUN_NAME/"

torchrun --nproc_per_node="$NPROC_PER_NODE" --nnodes="$NNODES" --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    ${PROJECT_ROOT}/src/open_r1/sft.py \
    --deepspeed ${PROJECT_ROOT}/configs/deepspeed/zero2.json \
    --output_dir "${OUTPUT_ROOT}/$RUN_NAME" \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${PROJECT_ROOT}/configs/sft/stage1.yaml \
    --freeze_vision_modules true \
    --use_audio_in_video true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --learning_rate 2.0e-5 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --log_level info \
    --save_only_model true 2>&1 | tee "${OUTPUT_ROOT}/$RUN_NAME/train.log"

