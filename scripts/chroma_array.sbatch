#!/bin/bash

#SBATCH --job-name=cd8fnsce
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/chroma_%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/chroma_%A_%a.err
#SBATCH -p bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-5

source ~/.bashrc
source /scratch/users/tianyulu/envs/protpardelle/bin/activate

# Configuration
INPUT_DIR="/scratch/users/tianyulu/allatom-design-sampling-results/8fns_conformations"
OUTPUT_DIR="/scratch/users/tianyulu/allatom-design-sampling-results/8fns_conformations/chroma_design_output"
SCRIPT_DIR="/home/users/tianyulu/code/protpardelle-1c/scripts"
NUM_SAMPLES=32
NUM_JOBS=6

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Get all PDB files and split into chunks
ALL_PDBS=($(ls ${INPUT_DIR}/*.pdb | sort))
TOTAL_FILES=${#ALL_PDBS[@]}
CHUNK_SIZE=$(( (TOTAL_FILES + NUM_JOBS - 1) / NUM_JOBS ))  # ceiling division

# Calculate start and end indices for this array task
START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE ))
if [ ${END_IDX} -gt ${TOTAL_FILES} ]; then
    END_IDX=${TOTAL_FILES}
fi

echo "Job ${SLURM_ARRAY_TASK_ID}: Processing files ${START_IDX} to $((END_IDX - 1)) (total: $((END_IDX - START_IDX)) files)"

# Create input CSV for this chunk
INPUT_CSV="${OUTPUT_DIR}/input_chunk_${SLURM_ARRAY_TASK_ID}.csv"
echo "pdb_path,fix_pos,pdb_stem" > ${INPUT_CSV}

for (( i=START_IDX; i<END_IDX; i++ )); do
    PDB_PATH="${ALL_PDBS[$i]}"
    PDB_STEM=$(basename "${PDB_PATH}" .pdb)
    # Empty fix_pos means design all positions
    echo "${PDB_PATH},,${PDB_STEM}" >> ${INPUT_CSV}
done

echo "Created input CSV: ${INPUT_CSV}"
echo "Number of structures in this chunk: $((END_IDX - START_IDX))"

# Run chroma design
python3 ${SCRIPT_DIR}/chroma_design_array.py ${INPUT_CSV} --num-samples ${NUM_SAMPLES}

echo "Job ${SLURM_ARRAY_TASK_ID} completed"
