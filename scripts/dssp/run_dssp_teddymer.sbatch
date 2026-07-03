#!/bin/bash
#SBATCH --output=dssp-%x.%A_%a.out
#SBATCH --error=dssp-%x.%A_%a.err
#SBATCH --partition=possu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-31

# Run with:
#   sbatch --job-name=dssp_teddymer scripts/dssp/run_dssp_teddymer.sbatch \
#       <keys-list> <pdb-dir> <output-prefix>
# Then after all array tasks finish, concatenate:
#   head -n1 <output-prefix>_shard_0000.csv > <output-prefix>.csv
#   for f in <output-prefix>_shard_*.csv; do tail -n +2 "$f"; done >> <output-prefix>.csv

set -euo pipefail

KEYS_LIST=${1:?missing keys-list}
PDB_DIR=${2:?missing pdb-dir}
OUTPUT_PREFIX=${3:?missing output-prefix}

NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-32}
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}

source /scratch/users/tianyulu/envs/protpardelle/bin/activate

PROJECT_ROOT_DIR=$(python -c "from protpardelle.env import PROJECT_ROOT_DIR; print(PROJECT_ROOT_DIR)")

python "${PROJECT_ROOT_DIR}/scripts/dssp/run_dssp_teddymer.py" \
    --keys-list "${KEYS_LIST}" \
    --pdb-dir "${PDB_DIR}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --shard-id "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}"
