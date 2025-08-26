#!/bin/bash
#SBATCH --job-name=protpardelle_sampling
#SBATCH --output=sample-%x.%j.out
#SBATCH --error=sample-%x.%j.err
#SBATCH --partition=your-partition-name
#SBATCH --gpu_cmode=shared
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00

if command -v module >/dev/null 2>&1; then
    module load gcc/12.4.0
fi

if [ -z "$ENV_DIR" ]; then
    conda activate protpardelle
else
    source "$ENV_DIR"/protpardelle/bin/activate
fi

run_name=$1
motif_dir=$2
num_samples=$3
num_mpnn_seqs=$4
use_wandb=$5

python -m protpardelle.sample \
    "${run_name}" \
    --motif-dir "${motif_dir}" \
    --num-samples "${num_samples}" \
    --num-mpnn-seqs "${num_mpnn_seqs}" \
    "${use_wandb}"
