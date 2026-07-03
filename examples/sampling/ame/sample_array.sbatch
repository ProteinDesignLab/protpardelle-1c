#!/bin/bash
#SBATCH --job-name=ame_samp
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/ame_samp-%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/ame_samp-%A_%a.err
#SBATCH --partition=bioe,possu,owners,gpu
#SBATCH --gpu_cmode=shared
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-7%4

# AME sampling array: chunks the 41 motifs in YAML across 8 array tasks (4 concurrent).
# Usage:
#   sbatch examples/sampling/ame/sample_array.sbatch <yaml_basename> [num_samples=100] [batch=8]
# YAML resolved relative to examples/sampling/ame/<yaml_basename>

set -eo pipefail

YAML_NAME="${1:?yaml name required}"
NUM_SAMPLES="${2:-100}"
BATCH_SIZE="${3:-8}"

REPO_DIR=/home/users/tianyulu/code/protpardelle-1c
YAML="${REPO_DIR}/examples/sampling/ame/${YAML_NAME}"

source /scratch/users/tianyulu/envs/protpardelle/bin/activate
export PROTPARDELLE_MODEL_PARAMS=/oak/stanford/groups/possu/tianyu/models/protpardelle-1c
export PROTPARDELLE_OUTPUT_DIR=/scratch/users/tianyulu/protpardelle-1c-sampling-results

cd "${REPO_DIR}"
echo "[$(date)] yaml=${YAML_NAME} array_id=${SLURM_ARRAY_TASK_ID}/8 num_samples=${NUM_SAMPLES}"

python -m protpardelle.sample \
    "${YAML}" \
    --motif-dir /home/users/tianyulu/code/protpardelle-1c/examples/motifs/ame/motifs \
    --num-samples "${NUM_SAMPLES}" \
    --num-mpnn-seqs 0 \
    --batch-size "${BATCH_SIZE}" \
    --array-id "${SLURM_ARRAY_TASK_ID}" \
    --num-arrays 8
