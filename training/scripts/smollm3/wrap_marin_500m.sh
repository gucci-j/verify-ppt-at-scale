#!/bin/bash
#SBATCH --job-name=marin-500m
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=72:00:00

set -euo pipefail

SCRIPT_DIR="/path/to/verify-ppt-at-scale/training/scripts/smollm3"
TRAIN_SCRIPT="${SCRIPT_DIR}/marin_500m.sh"
CONTAINER=${CONTAINER:-/path/to/containers/vllm_26.01-py3.sif}

chmod +x "${TRAIN_SCRIPT}"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"

singularity exec \
    -B /etc/ssl/certs:/etc/ssl/certs \
    -B /etc/pki/ca-trust:/etc/pki/ca-trust \
    -B "$HOME:$HOME" \
    --nv "$CONTAINER" \
    /bin/bash "${TRAIN_SCRIPT}"
