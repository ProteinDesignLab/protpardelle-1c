#!/bin/bash

#SBATCH --job-name=ch_mta
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/chroma_met_to_arg_%j.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/chroma_met_to_arg_%j.err
#SBATCH -p bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

echo "Starting ChromaDesign with MET->ARG override"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source ~/.bashrc
source /scratch/users/tianyulu/envs/protpardelle/bin/activate

INPUT_CSV=/scratch/users/tianyulu/gfp/denovo/chroma_05/design_input_sasa_lt_50.csv
SCRIPT_DIR=/home/users/tianyulu/code/protpardelle-1c/gfp_scripts

echo ""
echo "Input CSV: ${INPUT_CSV}"
echo "Number of structures: $(tail -n +2 ${INPUT_CSV} | wc -l)"
echo ""

cd ${SCRIPT_DIR}

python3 chroma_design_met_to_arg.py ${INPUT_CSV} --num-samples 8

echo ""
echo "Done!"
echo "Time: $(date)"
