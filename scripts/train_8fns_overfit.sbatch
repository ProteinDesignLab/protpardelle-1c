#!/bin/bash
#SBATCH --job-name=r4x-8fns
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=train-%x.%j.out
#SBATCH --error=train-%x.%j.err

source ~/.bashrc
source /scratch/users/tianyulu/envs/protpardelle/bin/activate

cd /home/users/tianyulu/code/protpardelle-1c

WANDB__SERVICE_WAIT=300 python -u -m protpardelle.train \
    --config examples/training/cc89-overfit.yaml \
    --out-dir /scratch/users/tianyulu/rhaven/cc89_overfit \
    --project cc89-overfit \
    --wandb-id tyl \
    --exp-name cc89-overfit-8fns \
    --overfit 1 \
    --num-workers 4
