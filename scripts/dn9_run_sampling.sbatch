#!/bin/bash
#SBATCH --job-name=dn9_samp
#SBATCH --partition=bioe
#SBATCH --gres=gpu:1
#SBATCH --constraint='GPU_BRAND:NVIDIA&GPU_MEM:80GB'
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/dn9_%A_%a.out
#SBATCH --array=0-3
#SBATCH --export=NONE

# Arguments: $1=model_name (cc148 or cc149)
MODEL=${1:-cc148}
NUM_ARRAYS=4
OUT_BASE=/scratch/users/tianyulu/protpardelle-1c-sampling-results/dn9/raw_samples

source /scratch/users/tianyulu/envs/protpardelle/bin/activate
export TMPDIR=/scratch/users/tianyulu/tmp
mkdir -p $TMPDIR

cd /home/users/tianyulu/code/protpardelle-1c

python -m protpardelle.sample \
    /scratch/users/tianyulu/protpardelle-1c-sampling-results/dn9/configs/dn9_${MODEL}.yaml \
    --motif-dir examples/motifs/dn9_variants \
    --num-samples 4 \
    --num-mpnn-seqs 0 \
    --batch-size 4 \
    --no-use-wandb \
    --array-id $SLURM_ARRAY_TASK_ID \
    --num-arrays $NUM_ARRAYS
