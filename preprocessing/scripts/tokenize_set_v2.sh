#!/bin/bash
set -euo pipefail

# Set configurations
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME=/path/to/cache
export HF_HUB_CACHE=/path/to/cache
export HF_DATASETS_CACHE=/path/to/cache
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TMPDIR=/tmp
# Set HF_HUB_OFFLINE=1 if tokenizer is already cached and node has no internet

# Activate the virtual environment
source /path/to/venv/verify-ppt_training/bin/activate

# Run the preprocessing script
cd /path/to/verify-ppt-at-scale/preprocessing/src
python tokenize_set_v2.py \
    --output-root  /path/to/data/ppt/set_v2 \
    --logging-root "/path/to/verify-ppt-at-scale/preprocessing/logs" \
    --cache-root   "/path/to/cache" \
    --workers 16 \
    --tasks 128
