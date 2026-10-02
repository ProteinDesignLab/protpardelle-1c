#!/bin/bash
# 4x H100 training for cc152 / cc153 on possu (high_p priority on our own H100s).
#
# The job re-reads its own checkpoint directory at startup and resumes from the
# newest epoch rather than restarting from the cc89 init, so a preemption or a
# requeue costs at most one epoch. Walltime is possu's 7-day maximum; chain
# successive jobs with --dependency=afterany:<jobid> to run beyond that.
#SBATCH --job-name=cc15x
#SBATCH --partition=possu
#SBATCH --qos=high_p
#SBATCH --gpus-per-node=4
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/%x_%j.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/%x_%j.err
set -uo pipefail

source /scratch/users/tianyulu/envs/protpardelle/bin/activate
cd /home/users/tianyulu/code/protpardelle-1c

model_name=$1
OUT=/scratch/users/tianyulu/protpardelle-1c-experiments
CKDIR=$OUT/protpardelle-1c/${model_name}/checkpoints
BASE_CFG=examples/training/${model_name}.yaml
RUN_CFG=/scratch/users/tianyulu/run_configs/${model_name}_${SLURM_JOB_ID}.yaml
mkdir -p "$(dirname "$RUN_CFG")"

# Resume from the newest epoch checkpoint if this job has run before.
python scripts/atomize/prepare_run_config.py "$BASE_CFG" "$RUN_CFG" "$CKDIR"

echo "node=$(hostname) gpus=$(nvidia-smi -L | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4

WANDB__SERVICE_WAIT=300 python -u -m protpardelle.train \
    --config "$RUN_CFG" \
    --out-dir "$OUT" \
    --project protpardelle-1c \
    --wandb-id tyl \
    --exp-name "${model_name}" \
    --use-dataparallel \
    --num-workers 16
