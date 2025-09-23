#!/bin/bash
#SBATCH --job-name=protpardelle_training
#SBATCH --output=train-%x.%j.out
#SBATCH --error=train-%x.%j.err
#SBATCH --partition=your-partition-name
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00

if command -v module >/dev/null 2>&1; then
    module load gcc/12.4.0
fi

if [ -z "$ENV_DIR" ]; then
    conda activate protpardelle
else
    source "$ENV_DIR"/protpardelle/bin/activate
fi

PROJECT_ROOT_DIR=$(python -c "from protpardelle.env import PROJECT_ROOT_DIR; print(PROJECT_ROOT_DIR)")

model_name=$1
output_dir=$2

if [[ "${2:-}" == "--debug" ]]; then
    debug="--debug"
fi

config_path="${PROJECT_ROOT_DIR}/examples/training/${model_name}.yaml"
project_name="protpardelle-1c"
wandb_id="your-wandb-id"

WANDB__SERVICE_WAIT=300 python -m protpardelle.train \
    --config-path "${config_path}" \
    --output-dir "${output_dir}" \
    --project-name "${project_name}" \
    --wandb-id "${wandb_id}" \
    --exp-name "${model_name}" \
    --num-workers 8 \
    ${debug:+$debug}
