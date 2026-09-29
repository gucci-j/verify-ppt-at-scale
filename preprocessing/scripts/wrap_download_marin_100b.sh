#!/bin/bash
#SBATCH --job-name=download_marin_100b.sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="/path/to/verify-ppt-at-scale/preprocessing/scripts"
PREPROCESS_SCRIPT="${SCRIPT_DIR}/download_marin_100b.sh"
CONTAINER=${CONTAINER:-/path/to/containers/vllm_26.01-py3.sif}

chmod +x "${PREPROCESS_SCRIPT}"

singularity exec \
    "$CONTAINER" \
    /bin/bash "${PREPROCESS_SCRIPT}" "$@"
