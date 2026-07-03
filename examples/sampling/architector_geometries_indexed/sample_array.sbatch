#!/bin/bash
#SBATCH --job-name=arch_idx
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/arch_idx-%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/arch_idx-%A_%a.err
#SBATCH --partition=bioe,possu,owners,gpu
#SBATCH --gpu_cmode=shared
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --array=0-15%4

# Indexed sampling array: one task per REE (16 total), 4 concurrent.
# Defaults: 64 samples/motif, batch 8, no MPNN/ESMFold.
# Usage:
#   sbatch examples/sampling/architector_geometries_indexed/sample_array.sbatch \
#       [num_samples=64] [num_mpnn_seqs=0] [batch_size=8]

set -eo pipefail  # NOTE: no -u (sourcing ~/.bashrc triggers unbound PS1)

NUM_SAMPLES="${1:-64}"
NUM_MPNN_SEQS="${2:-0}"
BATCH_SIZE="${3:-8}"

REES=(ce dy er eu gd ho la li lu nd pm pr sm tb tm yb)
REE="${REES[$SLURM_ARRAY_TASK_ID]}"

REE_UPPER=$(python3 -c "print('${REE}'.capitalize())")

source /scratch/users/tianyulu/envs/protpardelle/bin/activate

export PROTPARDELLE_MODEL_PARAMS=/oak/stanford/groups/possu/tianyu/models/protpardelle-1c
export PROTPARDELLE_OUTPUT_DIR=/scratch/users/tianyulu/protpardelle-1c-sampling-results

REPO_DIR="/home/users/tianyulu/code/protpardelle-1c"
YAML="${REPO_DIR}/examples/sampling/architector_geometries_indexed/${REE}.yaml"
MOTIF_DIR="/home/users/tianyulu/code/rhaven/data/architector_geometries/${REE_UPPER}"

echo "[$(date)] REE=${REE} YAML=${YAML} MOTIF_DIR=${MOTIF_DIR}"

cd "${REPO_DIR}"
python -m protpardelle.sample \
    "${YAML}" \
    --motif-dir "${MOTIF_DIR}" \
    --num-samples "${NUM_SAMPLES}" \
    --num-mpnn-seqs "${NUM_MPNN_SEQS}" \
    --batch-size "${BATCH_SIZE}"
