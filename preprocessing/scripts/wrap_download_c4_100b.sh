#!/bin/bash
#SBATCH --job-name=download_c4_100b.sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=04:00:00

set -euo pipefail

SCRIPT_DIR="/path/to/verify-ppt/preprocessing/scripts"
PREPROCESS_SCRIPT="${SCRIPT_DIR}/download_c4_100b.sh"
CONTAINER=${CONTAINER:-/path/to/containers/vllm_26.01-py3.sif}

chmod +x "${PREPROCESS_SCRIPT}"

singularity exec \
    -B /etc/ssl/certs:/etc/ssl/certs \
    -B /etc/pki/ca-trust:/etc/pki/ca-trust \
    "$CONTAINER" \
    /bin/bash "${PREPROCESS_SCRIPT}" "$@"
