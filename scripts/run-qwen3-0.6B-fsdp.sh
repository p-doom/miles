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
export CUDA_VISIBLE_DEVICES=0,1

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"


# --- 1. DYNAMIC HOST IP DETECTION (CRITICAL FOR SLURM) ---
# Don't hardcode IP. Get the actual IP of the current node.
export HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
echo "Detected Head Node IP: ${HEAD_NODE_IP}"

# --- 2. PROXY CONFIGURATION ---
# Ensure local traffic doesn't go through a corporate proxy
export no_proxy="${HEAD_NODE_IP},localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="${HEAD_NODE_IP},localhost,127.0.0.1,0.0.0.0"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}
LOAD_SAVE_PATH="/fast/project/HFMI_SynergyUnit/tab_model/huggingface/shared_data/${RUN_ID}/checkpoints"

CKPT_ARGS=(
   --hf-checkpoint /fast/project/HFMI_SynergyUnit/tab_model/huggingface/Qwen3-0.6B
   --load /fast/project/HFMI_SynergyUnit/tab_model/huggingface/Qwen3-0.6B
   --ref-load /fast/project/HFMI_SynergyUnit/tab_model/huggingface/Qwen3-0.6B
)

ROLLOUT_ARGS=(
   --prompt-data /fast/project/HFMI_SynergyUnit/tab_model/huggingface/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8
   --global-batch-size 64
)

GRPO_ARGS=(
   --use-kl-loss
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project crowd-pilot-miles
   --wandb-team instant-uv
   --wandb-group qwen3-0.6b-grpo-fsdp
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
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 2
   --colocate
   --use-fault-tolerance
   --dump-details /fast/project/HFMI_SynergyUnit/tab_model/huggingface/shared_data/qwen3-600M-fsdp-1116-noref/dump_details
   # --fsdp-cpu-offload
)

# launch the master node of ray in container - 8 GPUs for training
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PYTHONPATH="/fast/project/HFMI_SynergyUnit/mihir/Megatron-LM/:${SCRIPT_DIR}"
python3 -m ray.scripts.scripts start --head \
    --node-ip-address=${HEAD_NODE_IP} \
    --num-gpus 2 \
    --num-cpus 8 \
    --object-store-memory 20000000000 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --port=6379


RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/fast/project/HFMI_SynergyUnit/mihir/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"


python3 -m ray.scripts.scripts job submit --address="http://${HEAD_NODE_IP}:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${TRAIN_BACKEND_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}



