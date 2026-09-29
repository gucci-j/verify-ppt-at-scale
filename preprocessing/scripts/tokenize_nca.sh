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

# Run the NCA data generation and tokenization script
cd /path/to/verify-ppt/preprocessing/src
python tokenize_nca.py \
    --output-root  /path/to/data/ppt \
    --logging-root /path/to/verify-ppt/preprocessing/logs \
    --cache-root   /path/to/cache \
    --num-sentences 260000 \
    --seq-length 1406 \
    --grid 12 \
    --patch 2 \
    --num-colors 10 \
    --filter-threshold 0.50 \
    --filter-upper-bound 1.00 \
    --temperature 1e-4 \
    --dt 1 \
    --gpu-batch-size 4096 \
    --cand-batch-size 8192 \
    --batch-size 5000 \
    --workers 16 \
    --tasks 128
