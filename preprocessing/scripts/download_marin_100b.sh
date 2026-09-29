#!/bin/bash
set -euo pipefail

# Set configurations
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME=/path/to/cache
export HF_HUB_CACHE=/path/to/cache
export HF_DATASETS_CACHE=/path/to/cache
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TMPDIR=/tmp

# Activate the virtual environment
source /path/to/venv/verify-ppt_training/bin/activate

# Run the preprocessing script
cd /path/to/verify-ppt/preprocessing/src
python tokenize_marin.py \
    --output-root "" \
    --logging-root "" \
    --cache-root /path/to/cache \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --dclm-alias "${DCLM_ALIAS:-dclm_marin_103b}" \
    --dclm-max-samples "${DCLM_MAX_SAMPLES:-85000000}" \
    --predownload \
    --predownload-only
