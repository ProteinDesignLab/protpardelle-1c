#!/bin/bash
#SBATCH --output=sample-%x.%j.out
#SBATCH --error=sample-%x.%j.err
#SBATCH --array=0-6
#SBATCH --partition=bioe,possu
#SBATCH --gpu_cmode=shared
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00

source ~/.bashrc

source /scratch/users/tianyulu/envs/protpardelle/bin/activate

cd /home/users/tianyulu/code/protpardelle-1c

RUN_YAMLS=(
    12162025-lanm-416
    12162025-lanm-521
    12162025-lanm-564
    12162025-lanm-595
    12162025-lanm-649
    12162025-lanm-777
    12162025-lanm-838
)

python -m protpardelle.sample \
    examples/sampling/${RUN_YAMLS[$SLURM_ARRAY_TASK_ID]}.yaml \
    --motif-dir ./examples/motifs/hrep \
    --num-samples 1280 \
    --num-mpnn-seqs 0
