#!/bin/bash

: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:=29500}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
: "${SLURM_NNODES:?SLURM_NNODES must be set}"
: "${SLURM_PROCID:?SLURM_PROCID must be set}"

checkpoint_path=$1
if [ -z "$checkpoint_path" ]; then
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi

# Set up the environment variables inside the container
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME=/path/to/cache
export HF_HUB_CACHE=/path/to/cache
export HF_DATASETS_CACHE=/path/to/cache
export HF_DATASETS_TRUST_REMOTE_CODE=true
export REQUESTS_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem

output_dir=${checkpoint_path}_hf
mkdir -p $output_dir

# Activate the Python environment
source /path/to/venv/verify-ppt_training/bin/activate

# Convert the Nanotron checkpoint to HuggingFace format
cd /path/to/verify-ppt/external/nanotron/examples/smollm3

python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=1 \
    --node_rank="${SLURM_PROCID}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
convert_nanotron_to_hf.py \
    --checkpoint_path ${checkpoint_path} \
    --save_path ${output_dir} \
    --tokenizer_name HuggingFaceTB/SmolLM3-3B
