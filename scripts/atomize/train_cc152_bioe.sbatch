#!/bin/bash
#SBATCH --job-name=cc152
#SBATCH --partition=bioe
#SBATCH --gpus-per-node=4
#SBATCH --gpu_cmode=shared
#SBATCH --cpus-per-task=24
#SBATCH --mem=160GB
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/%x_%j.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/%x_%j.err
source /scratch/users/tianyulu/envs/protpardelle/bin/activate
cd /home/users/tianyulu/code/protpardelle-1c
model_name=${1:-cc152}
output_dir=${2:-/scratch/users/tianyulu/protpardelle-1c-experiments}
WANDB__SERVICE_WAIT=300 python -m protpardelle.train \
    --config examples/training/${model_name}.yaml \
    --out-dir "${output_dir}" \
    --project protpardelle-1c \
    --wandb-id tyl \
    --exp-name "${model_name}" \
    --use-dataparallel \
    --num-workers 12
