#!/bin/bash

#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH -p bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/job_%A_%a.err

source ~/.bashrc

source /scratch/users/tianyulu/envs/protpardelle/bin/activate

INPUT_CSV=$1

python3 chroma_design.py ${INPUT_CSV}
