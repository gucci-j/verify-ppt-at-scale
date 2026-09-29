#!/bin/bash
set -euo pipefail

: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:=29500}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
: "${SLURM_NNODES:?SLURM_NNODES must be set}"
: "${SLURM_PROCID:?SLURM_PROCID must be set}"

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
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

# Activate env
source /path/to/venv/verify-ppt_training/bin/activate

# Change to the directory containing the training script
cd /path/to/verify-ppt-at-scale/external/nanotron

# Run the training script
python -m torch.distributed.run \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --node_rank="${SLURM_PROCID}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    run_train.py --config-file /path/to/verify-ppt-at-scale/training/configs/ppt-olmo3_500m.yaml
