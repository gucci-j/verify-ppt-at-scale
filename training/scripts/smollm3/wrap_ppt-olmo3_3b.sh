#!/bin/bash
#SBATCH --job-name=ppt-olmo3_3b
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=3:00:00

set -euo pipefail

SCRIPT_DIR="/path/to/verify-ppt-at-scale/training/scripts/smollm3_ablation"
TRAIN_SCRIPT="${SCRIPT_DIR}/ppt-olmo3_3b.sh"
CONTAINER=${CONTAINER:-/path/to/containers/vllm_26.01-py3.sif}

chmod +x "${TRAIN_SCRIPT}"

export MASTER_ADDR=${MASTER_ADDR:-${SLURM_LAUNCH_NODE_IPADDR:-127.0.0.1}}
export MASTER_PORT=${MASTER_PORT:-12403}
echo "Using MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

srun --ntasks-per-node=1 singularity exec \
    -B /etc/ssl/certs:/etc/ssl/certs \
    -B /etc/pki/ca-trust:/etc/pki/ca-trust \
    -B "$HOME:$HOME" \
    --nv "$CONTAINER" \
    /bin/bash "${TRAIN_SCRIPT}"
