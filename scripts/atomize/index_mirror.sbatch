#!/bin/bash
#SBATCH --job-name=idx_pdb
#SBATCH -p owners
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=8GB
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH -o /scratch/users/tianyulu/slurm_log_files/%x_%A_%a.out
#SBATCH -e /scratch/users/tianyulu/slurm_log_files/%x_%A_%a.err
set -uo pipefail
source /scratch/users/tianyulu/envs/protpardelle/bin/activate
cd /home/users/tianyulu/code/protpardelle-1c
python scripts/atomize/index_pdb_mirror.py --shard ${SLURM_ARRAY_TASK_ID} --n-shards 4 --procs 16
