#!/bin/bash
#SBATCH --output=sample-%x.%j.out
#SBATCH --error=sample-%x.%j.err
#SBATCH --partition=possu
#SBATCH --gpu_cmode=shared
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

source ~/.bashrc

source /scratch/users/tianyulu/envs/protpardelle/bin/activate

python3 md.py
