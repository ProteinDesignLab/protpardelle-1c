#!/bin/bash
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH -p your-partition-name
#SBATCH --gpu_cmode=shared
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00

module load gcc/12.4.0
source $ENV_DIR/protpardelle/bin/activate

RUN_NAME=$1
MOTIF_DIR=$2
N_SAMPLES=$3
N_MPNN_SEQS=$4
DEBUG=$5

python -m protpardelle.sample ${RUN_NAME} --motif-dir ${MOTIF_DIR} --n-samples ${N_SAMPLES} --num-mpnn-seqs ${N_MPNN_SEQS} ${DEBUG}
