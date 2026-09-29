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

# Run the GPU-accelerated generation script to produce raw NCA text file and cache tokenizer
cd /path/to/verify-ppt-at-scale/preprocessing/src
python tokenize_nca.py \
    --output-root "" \
    --logging-root "" \
    --cache-root /path/to/cache \
    --num-sentences 260000 \
    --seq-length 1406 \
    --grid 12 \
    --patch 2 \
    --num-colors 10 \
    --filter-threshold 0.50 \
    --filter-upper-bound 1.00 \
    --temperature 1e-4 \
    --init-rollout-steps 10 \
    --dt 1 \
    --gpu-batch-size 4096 \
    --cand-batch-size 8192 \
    --workers 8 \
    --predownload \
    --predownload-only

