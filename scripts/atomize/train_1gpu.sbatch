#!/bin/bash
#SBATCH --job-name=cc15x
#SBATCH --partition=bioe
#SBATCH --gpus-per-node=1
#SBATCH --gpu_cmode=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/%x_%j.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/%x_%j.err
source /scratch/users/tianyulu/envs/protpardelle/bin/activate
cd /home/users/tianyulu/code/protpardelle-1c
model_name=$1
WANDB__SERVICE_WAIT=300 python -u -m protpardelle.train \
    --config examples/training/${model_name}.yaml \
    --out-dir /scratch/users/tianyulu/protpardelle-1c-experiments \
    --project protpardelle-1c --wandb-id tyl \
    --exp-name "${model_name}" --num-workers 6
