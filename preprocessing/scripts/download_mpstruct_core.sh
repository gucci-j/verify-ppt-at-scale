#!/bin/bash
set -euo pipefail

# Set configurations
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME=/path/to/cache
export HF_HUB_CACHE=/path/to/cache
export HF_DATASETS_CACHE=/path/to/cache
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TMPDIR=/tmp
CACHE_DIR=${CACHE_DIR:-/path/to/cache}

# Activate the virtual environment
source /path/to/venv/verify-ppt_training/bin/activate

# Run the preprocessing script
cd /path/to/verify-ppt-at-scale/preprocessing/src
python tokenize_mpstruct_core.py \
    --output-root "" \
    --logging-root "" \
    --cache-root "${CACHE_DIR}" \
    --seq-length 2048 \
    --num-sentences 260000 \
    --k-struct 1 \
    --k-dep 4 \
    --use-head-diversity \
    --format ids \
    --seed 42 \
    --workers 8 \
    --predownload \
    --predownload-only "$@"
