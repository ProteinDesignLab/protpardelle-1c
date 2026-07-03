#!/bin/bash
#SBATCH --job-name=dn9b
#SBATCH --partition=bioe,possu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/dn9b_%j.out
#SBATCH --export=NONE

MODEL=$1
NUM_SAMPLES=${2:-20}
BATCH=${3:-8}
if [ -z "$MODEL" ]; then
    echo "Usage: sbatch dn9b_sweep.sbatch <model_name> [num_samples=20] [batch=8]"
    exit 1
fi

source /scratch/users/tianyulu/envs/protpardelle/bin/activate
export TMPDIR=/scratch/users/tianyulu/tmp
mkdir -p $TMPDIR

cd /home/users/tianyulu/code/protpardelle-1c

python -m protpardelle.sample \
    /scratch/users/tianyulu/protpardelle-1c-sampling-results/dn9b/configs/sweep_${MODEL}.yaml \
    --motif-dir examples/motifs/dn9_variants \
    --num-samples $NUM_SAMPLES \
    --num-mpnn-seqs 4 \
    --batch-size $BATCH \
    --no-use-wandb
