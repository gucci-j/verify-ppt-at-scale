#!/bin/bash
#SBATCH --job-name=ppt-c4-500m_v2
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="/path/to/verify-ppt/training/scripts/smollm3"
TRAIN_SCRIPT="${SCRIPT_DIR}/ppt_c4_500m_v2.sh"
CONTAINER=${CONTAINER:-/path/to/containers/vllm_26.01-py3.sif}
chmod +x "${TRAIN_SCRIPT}"

export MASTER_ADDR=${MASTER_ADDR:-${SLURM_LAUNCH_NODE_IPADDR:-127.0.0.1}}
export MASTER_PORT=${MASTER_PORT:-12410}
echo "Using MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

srun --ntasks-per-node=1 singularity exec \
    -B /etc/ssl/certs:/etc/ssl/certs \
    -B /etc/pki/ca-trust:/etc/pki/ca-trust \
    -B "$HOME:$HOME" \
    --nv "$CONTAINER" \
    /bin/bash "${TRAIN_SCRIPT}"
