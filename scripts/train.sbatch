#!/bin/bash
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH -p your-partition-name
#SBATCH --gpus-per-node=1
#SBATCH --gpu_cmode=shared
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00


source $ENV_DIR/protpardelle/bin/activate

BASE_DIR=/your/path/to/protpardelle-1c
MODEL_NAME=$1
DEBUG=$2
PROTPARDELLE_EXPERIMENT_DIR=/your/path/to/protpardelle-1c-experiments

# Set num_workers depending on DEBUG
if [[ "$DEBUG" == *"--debug"* ]]; then
    NUM_WORKERS=0
else
    NUM_WORKERS=8
fi

# Run commands
cd ${BASE_DIR}

WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES="0" python3 src/protpardelle/train.py \
    --config examples/training/${MODEL_NAME}.yaml \
    --project protpardelle-1c \
    --wandb_id tyl \
    --exp_name ${MODEL_NAME} \
    --train \
    --use_dataparallel \
    --num_workers ${NUM_WORKERS} \
    --out_dir ${PROTPARDELLE_EXPERIMENT_DIR} \
    ${DEBUG}