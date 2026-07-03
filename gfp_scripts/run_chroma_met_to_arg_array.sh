#!/bin/bash

#SBATCH --job-name=ch_mta
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/chroma_met_to_arg_%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/chroma_met_to_arg_%A_%a.err
#SBATCH -p bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-5

echo "Starting ChromaDesign with MET->ARG override"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source ~/.bashrc
source /scratch/users/tianyulu/envs/protpardelle/bin/activate

INPUT_CSV=/scratch/users/tianyulu/gfp/denovo/chroma_08/design_input_part${SLURM_ARRAY_TASK_ID}.csv
SCRIPT_DIR=/home/users/tianyulu/code/protpardelle-1c/gfp_scripts

echo ""
echo "Input CSV: ${INPUT_CSV}"
echo "Number of structures: $(tail -n +2 ${INPUT_CSV} | wc -l)"
echo ""

cd ${SCRIPT_DIR}

python3 chroma_design_met_to_arg.py ${INPUT_CSV} --num-samples 1

echo ""
echo "Done with task ${SLURM_ARRAY_TASK_ID}!"
echo "Time: $(date)"
