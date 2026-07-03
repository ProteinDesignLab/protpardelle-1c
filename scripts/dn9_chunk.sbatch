#!/bin/bash
#SBATCH --job-name=dn9
#SBATCH --partition=bioe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:30:00
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/dn9_chunk_%j.out
#SBATCH --export=NONE

# Arg: $1 = chunk YAML path
CHUNK_YAML=$1
if [ -z "$CHUNK_YAML" ]; then
    echo "Usage: sbatch dn9_chunk.sbatch <chunk_yaml>"
    exit 1
fi

source /scratch/users/tianyulu/envs/protpardelle/bin/activate
export TMPDIR=/scratch/users/tianyulu/tmp
mkdir -p $TMPDIR

cd /home/users/tianyulu/code/protpardelle-1c

python -m protpardelle.sample \
    "$CHUNK_YAML" \
    --motif-dir examples/motifs/dn9_variants \
    --num-samples 8 \
    --num-mpnn-seqs 4 \
    --batch-size 8 \
    --no-use-wandb
