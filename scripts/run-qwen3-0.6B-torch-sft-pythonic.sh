#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

# set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"


# --- 1. DYNAMIC HOST IP DETECTION (CRITICAL FOR SLURM) ---
# Don't hardcode IP. Get the actual IP of the current node.
# export HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
# echo "Detected Head Node IP: ${HEAD_NODE_IP}"

# # --- 2. PROXY CONFIGURATION ---
# # Ensure local traffic doesn't go through a corporate proxy
# export no_proxy="${HEAD_NODE_IP},localhost,127.0.0.1,0.0.0.0"
# export NO_PROXY="${HEAD_NODE_IP},localhost,127.0.0.1,0.0.0.0"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}
LOAD_SAVE_PATH="/fast/project/HFMI_SynergyUnit/tab_model/huggingface/shared_data/${RUN_ID}/checkpoints"

CKPT_ARGS=(
   --hf-checkpoint /fast/project/HFMI_SynergyUnit/tab_model/huggingface/Qwen3-0.6B
   --load /fast/project/HFMI_SynergyUnit/tab_model/huggingface/Qwen3-0.6B
   --ref-load /fast/project/HFMI_SynergyUnit/tab_model/huggingface/Qwen3-0.6B
)

SFT_ARGS=(
   --rollout-function-path miles.rollout.sft_rollout.generate_rollout
   --prompt-data /fast/project/HFMI_SynergyUnit/tab_model/huggingface/openhermes2_5.parquet
   --input-key messages
   --apply-chat-template
   --rollout-shuffle
   --num-epoch 3
   --rollout-batch-size 128
   --global-batch-size 128

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project crowd-pilot-miles
   --wandb-team instant-uv
   --wandb-group qwen3-0.6b-sft-torch
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.5
   --sglang-decode-log-interval 1000
   --sglang-chunked-prefill-size 4096
   --sglang-attention-backend fa3
)

TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_3
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
   --actor-num-gpus-per-node 2
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

# Build runtime env
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/fast/project/HFMI_SynergyUnit/mihir/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

python3 train.py \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${TRAIN_BACKEND_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}



