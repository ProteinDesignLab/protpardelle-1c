#!/bin/bash
#SBATCH --job-name=dn9b
#SBATCH --partition=bioe,possu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:30:00
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/dn9b_%j.out
#SBATCH --export=NONE

CONFIG=$1
if [ -z "$CONFIG" ]; then
    echo "Usage: sbatch dn9_bulk.sbatch <config.yaml>"
    exit 1
fi

source /scratch/users/tianyulu/envs/protpardelle/bin/activate
export TMPDIR=/scratch/users/tianyulu/tmp
mkdir -p $TMPDIR

cd /home/users/tianyulu/code/protpardelle-1c

python -m protpardelle.sample \
    "$CONFIG" \
    --motif-dir examples/motifs/dn9_variants \
    --num-samples 4 \
    --num-mpnn-seqs 4 \
    --batch-size 4 \
    --no-use-wandb
