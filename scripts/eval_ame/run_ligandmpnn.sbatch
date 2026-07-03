#!/bin/bash
#SBATCH --job-name=ligmpnn
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/ligmpnn-%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/ligmpnn-%A_%a.err
#SBATCH --partition=bioe,possu,owners,gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --array=0-7%4

# Run LigandMPNN on a manifest of sampled PDBs, designing 8 sequences per backbone
# with motif residues fixed (using scaffold_motif_resnums from the manifest).
#
# Usage:
#   sbatch scripts/eval_ame/run_ligandmpnn.sbatch <manifest_csv> <out_dir> [num_seqs=8] [batch_size=8]

set -eo pipefail

MANIFEST="${1:?manifest csv required}"
OUT_DIR="${2:?out dir required}"
NUM_SEQS="${3:-8}"
BATCH_SIZE="${4:-8}"

mkdir -p "${OUT_DIR}"

source /scratch/users/tianyulu/envs/ligandmpnn/bin/activate

LMPNN_DIR=/home/users/tianyulu/code/LigandMPNN
cd "${LMPNN_DIR}"

# Chunk the manifest across array tasks. Each task processes 1/8 of the rows.
python /home/users/tianyulu/code/protpardelle-1c/scripts/eval_ame/_run_ligandmpnn_chunk.py \
    --manifest "${MANIFEST}" \
    --out-dir "${OUT_DIR}" \
    --num-seqs "${NUM_SEQS}" \
    --batch-size "${BATCH_SIZE}" \
    --array-id "${SLURM_ARRAY_TASK_ID}" \
    --num-arrays 8 \
    --lmpnn-dir "${LMPNN_DIR}"
