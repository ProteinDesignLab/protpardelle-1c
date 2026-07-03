#!/usr/bin/env bash
#SBATCH -p bioe,possu
#SBATCH --job-name=dn9_rf3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/dn9_rf3_app_%j.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/dn9_rf3_app_%j.err

# Designed to be wrapped by launch_rf3_sbatch_apptainer.sh; runs INSIDE container.
set -euo pipefail

# Activate RF3 environment
ENV_DIR=/scratch/users/tianyulu/envs
source ${ENV_DIR}/rf3/bin/activate
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real

# Input: JSON file with all dn9 sequences (single chain, apo)
INPUT_JSON=/scratch/users/tianyulu/protpardelle-1c-sampling-results/dn9/rf3/dn9_all_apo.json
OUTPUT_DIR=/scratch/users/tianyulu/protpardelle-1c-sampling-results/dn9/rf3/predictions

mkdir -p "$OUTPUT_DIR"

rf3 fold inputs="$INPUT_JSON" out_dir="$OUTPUT_DIR" early_stopping_plddt_threshold=0.0
