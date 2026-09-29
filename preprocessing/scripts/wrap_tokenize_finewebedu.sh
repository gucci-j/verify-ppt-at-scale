#!/bin/bash
#SBATCH --job-name=tokenize_finewebedu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --array=0-31

set -euo pipefail

# --- Job-array sharding ------------------------------------------------------
# Each array task owns ranks [TASK_ID*LOCAL_TASKS, (TASK_ID+1)*LOCAL_TASKS) of
# EVERY source in build_sources(). TASKS must be divisible by the array size.
# NOTE: run the predownload step (wrap_download_marin.sh) to completion BEFORE
# launching this array — the tokenize step is now read-only and expects every
# `datasets--*-processed` dir to already exist.
TASKS="${TASKS:-256}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
if (( TASKS % ARRAY_COUNT != 0 )); then
  echo "ERROR: TASKS=${TASKS} not divisible by array size ${ARRAY_COUNT}" >&2
  exit 1
fi
export TASKS
export LOCAL_TASKS=$(( TASKS / ARRAY_COUNT ))
export LOCAL_RANK_OFFSET=$(( ARRAY_TASK_ID * LOCAL_TASKS ))
echo "[array] task ${ARRAY_TASK_ID}/${ARRAY_COUNT}: ranks [${LOCAL_RANK_OFFSET}, $((LOCAL_RANK_OFFSET + LOCAL_TASKS)))"
# -----------------------------------------------------------------------------

SCRIPT_DIR="/path/to/verify-ppt/preprocessing/scripts"
PREPROCESS_SCRIPT="${SCRIPT_DIR}/tokenize_finewebedu.sh"
CONTAINER=${CONTAINER:-/path/to/containers/vllm_26.01-py3.sif}

chmod +x "${PREPROCESS_SCRIPT}"

singularity exec \
    "$CONTAINER" \
    /bin/bash "${PREPROCESS_SCRIPT}" "$@"
