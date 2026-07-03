#!/bin/bash
#SBATCH --partition=bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --gpu_cmode=shared
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/tm08-%x.%j.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/tm08-%x.%j.err

source ~/.bashrc

source /scratch/users/tianyulu/envs/allatom_design/bin/activate

/scratch/users/tianyulu/gfp/denovo/foldseek/bin/foldseek easy-cluster \
  /scratch/users/tianyulu/gfp/denovo/samples_flat \
  /scratch/users/tianyulu/gfp/denovo/foldseek_08 \
  /scratch/users/tianyulu/gfp/denovo/tmp_08 \
  --tmscore-threshold 0.8 \
  --threads 8 \
  -v 3
