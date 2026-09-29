#!/bin/bash
set -euo pipefail

WORKERS=${WORKERS:-16}
TASKS=${TASKS:-128}

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
cd /path/to/verify-ppt-at-scale/preprocessing/src
python tokenize_olmo3_baseline.py \
    --output-root  /path/to/data/ppt/olmo3 \
    --logging-root "/path/to/verify-ppt-at-scale/preprocessing/logs" \
    --cache-root   "/path/to/cache" \
    --workers "${WORKERS}" \
    --tasks "${TASKS}"
