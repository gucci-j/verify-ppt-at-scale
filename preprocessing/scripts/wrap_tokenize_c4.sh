#!/bin/bash
#SBATCH --job-name=tokenize_c4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --array=0-63

set -euo pipefail

# --- Job-array sharding ------------------------------------------------------
# Total shards must match --tasks inside tokenize_c4.sh. Each array task owns a
# contiguous slice of ranks: [TASK_ID*LOCAL_TASKS, (TASK_ID+1)*LOCAL_TASKS).
# TASKS must be divisible by the array size (0-7 => 8 tasks => 128/8 = 16 each).
TASKS="${TASKS:-128}"
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
PREPROCESS_SCRIPT="${SCRIPT_DIR}/tokenize_c4.sh"
CONTAINER=${CONTAINER:-/path/to/containers/vllm_26.01-py3.sif}

chmod +x "${PREPROCESS_SCRIPT}"

singularity exec \
    "$CONTAINER" \
    /bin/bash "${PREPROCESS_SCRIPT}" "$@"
