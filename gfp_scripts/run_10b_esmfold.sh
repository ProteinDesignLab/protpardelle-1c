#!/bin/bash
#SBATCH -p bioe
#SBATCH --nodelist=sh03-16n15
#SBATCH --array=0-62%4
#SBATCH --job-name=gfpsce
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/job_%A_%a.err

source ~/.bashrc

# Activate environment
ENV_DIR=/scratch/users/tianyulu/envs
source ${ENV_DIR}/esmfold/bin/activate

PARENT_DIR=$1

SEQ_DES_CSV="${PARENT_DIR}/seq_des_outputs_part_${SLURM_ARRAY_TASK_ID}.csv"

# Run ESMFold prediction
python3 10b_esmfold.py $SEQ_DES_CSV
