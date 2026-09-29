#!/bin/bash

# Configs / caches
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache"
export HF_HUB_CACHE="/path/to/cache"
export HF_DATASETS_CACHE="/path/to/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true
export REQUESTS_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
export TMPDIR=/tmp
export NCCL_DEBUG=INFO



# Data safety: replace any out-of-vocab token ids instead of triggering a CUDA device-side assert.
export SANITIZE_OOV=1
export OOV_REPLACEMENT_ID=0
export OOV_LOG_LIMIT=5
nnodes=${SLURM_NNODES:-1}
node_rank=${SLURM_NODEID:-${SLURM_PROCID:-0}}
master_port=${MASTER_PORT:-12410}
master_addr=${MASTER_ADDR:-${SLURM_LAUNCH_NODE_IPADDR:-127.0.0.1}}
gpus_per_node_raw=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}
if [[ -n "${gpus_per_node_raw}" ]]; then
    nproc_per_node=$(echo "${gpus_per_node_raw}" | grep -oE '[0-9]+' | head -n 1)
fi
if [[ -z "${nproc_per_node:-}" ]]; then
    nproc_per_node=$(nvidia-smi -L | wc -l)
fi
echo "Distributed config: nnodes=${nnodes}, node_rank=${node_rank}, nproc_per_node=${nproc_per_node}, master_addr=${master_addr}, master_port=${master_port}"

# Ensure processes on this node and the library agree about how many local ranks there are
export LOCAL_WORLD_SIZE="${nproc_per_node}"
echo "LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"

# Activate env
source /path/to/venv/verify-ppt_training/bin/activate

# Change to the directory containing the training script
cd /path/to/verify-ppt-at-scale/external/nanotron

# Run the training script
python -m torch.distributed.run \
    --nnodes="${nnodes}" \
    --nproc_per_node="${nproc_per_node}" \
    --node_rank="${node_rank}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${master_addr}:${master_port}" \
    run_train.py --config-file /path/to/verify-ppt-at-scale/training/configs/ppt-smollm3_smollm3_500m.yaml
