#!/bin/bash
#SBATCH --output=train-%x.%j.out
#SBATCH --error=train-%x.%j.err
#SBATCH --partition=possu
#SBATCH --nodelist=sh04-10n01
#SBATCH --gpus-per-node=1
#SBATCH --gpu_cmode=shared
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00

source ~/.bashrc

source /scratch/users/tianyulu/envs/protpardelle/bin/activate

model_name=$1
debug=$2
config_path="/home/users/tianyulu/code/protpardelle-1c/examples/training/${model_name}.yaml"
project_name="protpardelle-1c"
wandb_id="tyl"

WANDB__SERVICE_WAIT=300 python -m protpardelle.train \
    --config "${config_path}" \
    --out-dir "/scratch/users/tianyulu/protpardelle-1c-experiments" \
    --project "${project_name}" \
    --wandb-id "${wandb_id}" \
    --exp-name "${model_name}" \
    --num-workers 8 \
    ${debug:+$debug}