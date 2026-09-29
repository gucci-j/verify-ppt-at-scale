#!/bin/bash
#SBATCH --job-name=fineweb_edu_3b
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="/path/to/verify-ppt-at-scale/training/scripts/smollm3_ablation"
TRAIN_SCRIPT="${SCRIPT_DIR}/fineweb_edu_3b.sh"
CONTAINER=${CONTAINER:-/path/to/containers/vllm_26.01-py3.sif}

chmod +x "${TRAIN_SCRIPT}"

export MASTER_ADDR=${MASTER_ADDR:-${SLURM_LAUNCH_NODE_IPADDR:-127.0.0.1}}
export MASTER_PORT=${MASTER_PORT:-12405}
echo "Using MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

srun --ntasks-per-node=1 singularity exec \
    -B /etc/ssl/certs:/etc/ssl/certs \
    -B /etc/pki/ca-trust:/etc/pki/ca-trust \
    -B "$HOME:$HOME" \
    --nv "$CONTAINER" \
    /bin/bash "${TRAIN_SCRIPT}"
