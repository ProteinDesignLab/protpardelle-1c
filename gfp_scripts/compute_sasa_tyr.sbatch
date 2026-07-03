#!/bin/bash
#SBATCH --job-name=sat
#SBATCH --output=/scratch/users/tianyulu/gfp/denovo/sasa_tyr_%j.out
#SBATCH --error=/scratch/users/tianyulu/gfp/denovo/sasa_tyr_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=possu
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G

# Activate environment
source /scratch/users/tianyulu/envs/gfp/bin/activate

# Change to script directory
cd /home/users/tianyulu/code/protpardelle-1c/gfp_scripts

# Remove old SASA file to force recomputation (chroma_05 only)
rm -f /scratch/users/tianyulu/gfp/denovo/chroma_05/design_input_with_sasa.csv

# Run SASA computation with TYR packing
echo "Starting SASA computation with TYR packing..."
echo "Time: $(date)"

python compute_sasa_tyr.py

echo "Done!"
echo "Time: $(date)"
