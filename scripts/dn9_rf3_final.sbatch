#!/usr/bin/env bash
#SBATCH -p bioe,possu
#SBATCH --job-name=dn9rf3f
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/dn9_rf3f_%j.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/dn9_rf3f_%j.err

set -euo pipefail
source /scratch/users/tianyulu/envs/rf3/bin/activate
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real

INPUT_JSON=/scratch/users/tianyulu/protpardelle-1c-sampling-results/dn9/rf3_final/inputs.json
OUTPUT_DIR=/scratch/users/tianyulu/protpardelle-1c-sampling-results/dn9/rf3_final/predictions

mkdir -p "$OUTPUT_DIR"
rf3 fold inputs="$INPUT_JSON" out_dir="$OUTPUT_DIR" early_stopping_plddt_threshold=0.0
