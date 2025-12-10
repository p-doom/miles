    
#!/bin/bash

# --- CLEANUP PREVIOUS RUNS ---
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
# Double tap to be sure
pkill -9 ray
pkill -9 python

set -ex

# --- 1. DYNAMIC HOST IP DETECTION (CRITICAL FOR SLURM) ---
# Don't hardcode IP. Get the actual IP of the current node.
export HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
echo "Detected Head Node IP: ${HEAD_NODE_IP}"

# --- 2. PROXY CONFIGURATION ---
# Ensure local traffic doesn't go through a corporate proxy
export no_proxy="${HEAD_NODE_IP},localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="${HEAD_NODE_IP},localhost,127.0.0.1,0.0.0.0"

export PYTHONBUFFERED=16

# Check NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-0.6B.sh"


CKPT_ARGS=(
   --hf-checkpoint /fast/project/HFMI_SynergyUnit/tab_model/huggingface/Qwen3-0.6B/
   --ref-load /fast/project/HFMI_SynergyUnit/tab_model/huggingface/Qwen3-0.6B_torch_dist
   --load /fast/project/HFMI_SynergyUnit/tab_model/huggingface/Qwen3-0.6B_miles/
   --save /fast/project/HFMI_SynergyUnit/tab_model/huggingface/Qwen3-0.6B_miles/
   --save-interval 1000
)

SFT_ARGS=(
   --rollout-function-path miles.rollout.sft_rollout.generate_rollout
   --prompt-data /fast/project/HFMI_SynergyUnit/tab_model/huggingface/openhermes2_5.parquet
   --input-key conversations
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

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project crowd-pilot-miles
   --wandb-team instant-uv
   --wandb-group qwen3-4B-base-sft
   # --wandb-key ${WANDB_KEY}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# --- 3. START RAY HEAD ---
# Use the dynamic IP detected above
ray start --head \
    --node-ip-address=${HEAD_NODE_IP} \
    --num-gpus 2 \
    --num-cpus=4 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --port=6379

echo "Ray started. Waiting for Dashboard to be ready..."

# --- 4. WAIT FOR DASHBOARD (FIX FOR 504 ERROR) ---
# Loop until the dashboard port accepts connections
for i in {1..30}; do
    if curl -s "http://${HEAD_NODE_IP}:8265" > /dev/null; then
        echo "Dashboard is up!"
        break
    fi
    echo "Waiting for Ray Dashboard..."
    sleep 2
done
# Add a small safety buffer
sleep 5

# --- 5. SUBMIT JOB ---

# Build runtime env
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/fast/project/HFMI_SynergyUnit/mihir/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

# Submit using the dynamic IP
ray job submit --address="http://${HEAD_NODE_IP}:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}