#!/bin/bash
# Sharded P-SEA DSSP assignment for genie3 AFDB or Pinder datasets.
#
# Usage:
#   sbatch --job-name=dssp_afdb_genie3 scripts/dssp/run_dssp_genie3.sbatch \
#       /scratch/users/tianyulu/datasets/genie3/afdb_l512_plddt70_names.list \
#       /scratch/users/tianyulu/datasets/genie3/afdb_l512_plddt70/pdbs \
#       /scratch/users/tianyulu/datasets/genie3/afdb_l512_plddt70_dssp \
#       '{id}.pdb.gz'
#
#   sbatch --job-name=dssp_pinder_genie3 scripts/dssp/run_dssp_genie3.sbatch \
#       /scratch/users/tianyulu/datasets/genie3/pinder_2024-02_ids.list \
#       /scratch/users/tianyulu/datasets/genie3/pinder_2024-02/filtered_pdbs \
#       /scratch/users/tianyulu/datasets/genie3/pinder_2024-02_dssp \
#       '{id}.pdb'
#
# After all tasks finish, concatenate:
#   head -n1 <output-prefix>_shard_0000.csv > <output-prefix>.csv
#   for f in <output-prefix>_shard_*.csv; do tail -n +2 "$f"; done >> <output-prefix>.csv
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/dssp-%x.%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/dssp-%x.%A_%a.err
#SBATCH --partition=possu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-63

set -euo pipefail

IDS_LIST=${1:?missing ids-list}
PDB_DIR=${2:?missing pdb-dir}
OUTPUT_PREFIX=${3:?missing output-prefix}
FILENAME_TEMPLATE=${4:-'{id}.pdb'}

NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-64}
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}

source /scratch/users/tianyulu/envs/protpardelle/bin/activate

PROJECT_ROOT_DIR=$(python -c "from protpardelle.env import PROJECT_ROOT_DIR; print(PROJECT_ROOT_DIR)")

python "${PROJECT_ROOT_DIR}/scripts/dssp/run_dssp_genie3.py" \
    --ids-list "${IDS_LIST}" \
    --pdb-dir "${PDB_DIR}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --filename-template "${FILENAME_TEMPLATE}" \
    --shard-id "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}"
