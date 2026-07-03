#!/bin/bash
#SBATCH --job-name=dsmr-pfg
#SBATCH --partition=possu,bioe
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/c08_rmsd_%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/c08_rmsd_%A_%a.err

source /scratch/users/tianyulu/envs/gfp/bin/activate

cd /home/users/tianyulu/code/protpardelle-1c/gfp_scripts

python 19b_chroma08_dual_rmsd_filter.py \
    --chunk-id $SLURM_ARRAY_TASK_ID \
    --num-chunks 10 \
    --workers 16
