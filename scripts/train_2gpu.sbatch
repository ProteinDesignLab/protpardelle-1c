#!/bin/bash
#SBATCH --output=train-%x.%j.out
#SBATCH --error=train-%x.%j.err
#SBATCH --partition=possu
#SBATCH --nodelist=sh04-04n01
#SBATCH --gpus-per-node=2
#SBATCH --gpu_cmode=shared
#SBATCH --cpus-per-task=16
#SBATCH --time=4-00:00:00

#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/job_%A_%a.err

source ~/.bashrc

source /scratch/users/tianyulu/envs/protpardelle/bin/activate

PROJECT_ROOT_DIR=$(python -c "from protpardelle.env import PROJECT_ROOT_DIR; print(PROJECT_ROOT_DIR)")

model_name=$1
output_dir=$2
debug=$3
overfit=$4
config_path="${PROJECT_ROOT_DIR}/examples/training/${model_name}.yaml"
project_name="protpardelle-1c"
wandb_id="tyl"

WANDB__SERVICE_WAIT=300 python -m protpardelle.train \
    --config "${config_path}" \
    --out-dir "${output_dir}" \
    --project "${project_name}" \
    --wandb-id "${wandb_id}" \
    --exp-name "${model_name}" \
    --num-workers 8 \
    --use-dataparallel \
    ${debug:+$debug} \
    ${overfit:+--overfit $overfit}
