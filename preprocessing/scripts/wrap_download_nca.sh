#!/bin/bash
#SBATCH --job-name=download_nca
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

SCRIPT_DIR="/path/to/verify-ppt/preprocessing/scripts"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/download_nca.sh"
CONTAINER=${CONTAINER:-/path/to/containers/vllm_26.01-py3.sif}

chmod +x "${DOWNLOAD_SCRIPT}"

singularity exec --nv \
    -B /etc/ssl/certs:/etc/ssl/certs \
    -B /etc/pki/ca-trust:/etc/pki/ca-trust \
    -B "$HOME:$HOME" \
    "$CONTAINER" \
    /bin/bash "${DOWNLOAD_SCRIPT}" "$@"
