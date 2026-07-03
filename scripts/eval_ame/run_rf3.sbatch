#!/bin/bash
#SBATCH --job-name=rf3_ame
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/rf3_ame-%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/rf3_ame-%A_%a.err
#SBATCH --partition=bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00

# Inner sbatch run by run_rf3_dispatch.sbatch / launch_rf3_sbatch_apptainer.sh.
# The dispatcher hardcodes JSON_INPUT and OUT_DIR before submission.
#
# Pattern matches rhaven/scripts/structure_prediction/rf3/rf3_predict_DN1.sh,
# which is the proven-working invocation (python inference.py with hydra args).

set -eo pipefail

JSON_INPUT="${JSON_INPUT:?JSON_INPUT must be set by dispatcher}"
OUT_DIR="${OUT_DIR:?OUT_DIR must be set by dispatcher}"

ENV_DIR=/scratch/users/tianyulu/envs
source ${ENV_DIR}/rf3/bin/activate

export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real

mkdir -p "${OUT_DIR}"

cd /home/users/tianyulu/code/foundry

echo "[$(date)] rf3 inference: ${JSON_INPUT} -> ${OUT_DIR}"

python models/rf3/src/rf3/inference.py \
    inputs="${JSON_INPUT}" \
    ckpt_path=/oak/stanford/groups/possu/tianyu/models/foundry/rf3_foundry_01_24_latest_remapped.ckpt \
    out_dir="${OUT_DIR}" \
    one_model_per_file=true \
    annotate_b_factor_with_plddt=true \
    skip_existing=true
