#!/bin/bash
set -euo pipefail

# Set configurations
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME=/path/to/cache

export HF_HUB_CACHE=/path/to/cache
export HF_DATASETS_CACHE=/path/to/cache
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TMPDIR=/tmp
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-your_aws_access_key_id}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-your_aws_secret_access_key}"

# Activate the virtual environment
source /path/to/venv/verify-ppt_training/bin/activate

# Run the preprocessing script
cd /path/to/verify-ppt-at-scale/preprocessing/src
python tokenize_smollm3.py \
    --output-root "" \
    --logging-root "" \
    --cache-root /path/to/cache \
    --workers 4 \
    --predownload \
    --predownload-only
