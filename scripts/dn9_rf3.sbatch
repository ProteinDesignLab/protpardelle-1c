#!/bin/bash
#SBATCH --job-name=dn9_rf3
#SBATCH --partition=bioe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/dn9_rf3_%j.out
#SBATCH --export=NONE

INPUT_FASTA=$1
OUT_DIR=$2
if [ -z "$INPUT_FASTA" ] || [ -z "$OUT_DIR" ]; then
    echo "Usage: sbatch dn9_rf3.sbatch <input_fasta_or_dir> <out_dir>"
    exit 1
fi

mkdir -p "$OUT_DIR"
/scratch/users/tianyulu/envs/rf3/bin/rf3 fold \
    "inputs=$INPUT_FASTA" \
    "out_dir=$OUT_DIR" \
    annotate_b_factor_with_plddt=true
