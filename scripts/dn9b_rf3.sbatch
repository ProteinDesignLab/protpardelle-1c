#!/usr/bin/env bash
#SBATCH -p bioe,possu
#SBATCH --job-name=dn9b_rf3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/dn9b_rf3_%j.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/dn9b_rf3_%j.err

set -euo pipefail
MODEL=$1
source /scratch/users/tianyulu/envs/rf3/bin/activate
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real

INPUT=/scratch/users/tianyulu/protpardelle-1c-sampling-results/dn9b/rf3_${MODEL}/inputs.json
OUT=/scratch/users/tianyulu/protpardelle-1c-sampling-results/dn9b/rf3_${MODEL}/predictions
mkdir -p "$OUT"
rf3 fold inputs="$INPUT" out_dir="$OUT" early_stopping_plddt_threshold=0.0
