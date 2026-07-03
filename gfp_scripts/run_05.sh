#!/bin/bash
#SBATCH --output=train-%x.%j.out
#SBATCH --error=train-%x.%j.err
#SBATCH --partition=bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --gpu_cmode=shared
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/job_%A_%a.err

source ~/.bashrc

source /scratch/users/tianyulu/envs/protpardelle/bin/activate

python3 05_gfp_scaffold_filter.py
