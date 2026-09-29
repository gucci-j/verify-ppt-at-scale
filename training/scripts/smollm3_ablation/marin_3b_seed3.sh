#!/bin/bash
set -euo pipefail

: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:=12403}"
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



# Data safety: replace any out-of-vocab token ids instead of triggering CUDA assert
export SANITIZE_OOV=1
export OOV_REPLACEMENT_ID=0
export OOV_LOG_LIMIT=5
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

# Activate environment inside container
source /path/to/venv/verify-ppt_training/bin/activate

# Change to nanotron directory
cd /path/to/verify-ppt/external/nanotron

# Run training
python -m torch.distributed.run \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --node_rank="${SLURM_PROCID}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    run_train.py --config-file /path/to/verify-ppt/training/configs/ablation/marin_3b_seed3.yaml
