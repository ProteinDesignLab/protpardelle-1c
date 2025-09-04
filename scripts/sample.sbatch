#!/bin/bash
#SBATCH --output=sample-%x.%j.out
#SBATCH --error=sample-%x.%j.err
#SBATCH --partition=possu
#SBATCH --gpu_cmode=shared
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-16:00:00

source ~/.bashrc

source /scratch/users/tianyulu/envs/protpardelle/bin/activate

run_yaml=$1
motif_dir=$2
num_samples=$3
num_mpnn_seqs=$4

python -m protpardelle.sample \
    "${run_yaml}" \
    --motif-dir "${motif_dir}" \
    --num-samples "${num_samples}" \
    --num-mpnn-seqs "${num_mpnn_seqs}"
