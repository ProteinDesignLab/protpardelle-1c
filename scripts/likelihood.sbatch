#!/bin/bash
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH -p possu
#SBATCH --gpu_cmode=shared
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

module load gcc/12.4.0
source $ENV_DIR/protpardelle/bin/activate

MODEL=$1
EPOCH=$2
PDB_PATH=$3

python -m protpardelle.likelihood --model-name ${MODEL} --epoch ${EPOCH} --pdb-path ${PDB_PATH}
