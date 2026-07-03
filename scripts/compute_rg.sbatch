#!/bin/bash
#SBATCH --job-name=trg
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/trg-%j.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/trg-%j.err
#SBATCH --partition=bioe,possu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00

source ~/.bashrc
source /scratch/users/tianyulu/envs/protpardelle/bin/activate

cd /home/users/tianyulu/code/protpardelle-1c

python scripts/compute_rg.py \
    --pdb-dir /scratch/users/tianyulu/datasets/teddymer/pulchra_teddymer \
    --meta /scratch/users/tianyulu/datasets/teddymer/filtered_metadata.tsv \
    --out /scratch/users/tianyulu/datasets/teddymer/filtered_metadata_rg.tsv \
    --workers ${SLURM_CPUS_PER_TASK:-16}
