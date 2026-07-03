#!/bin/bash
#SBATCH --job-name=rf3_disp
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/rf3_disp-%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/rf3_disp-%A_%a.err
#SBATCH --partition=normal,owners
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# Outer dispatch: each array task picks one RF3 input JSON from a list file
# and submits an inner job (apptainer-wrapped python rf3 inference) for it.
#
# Usage:
#   N=$(($(wc -l < rf3_inputs/rf3_input_jsons.txt) - 1))
#   sbatch --array=0-${N}%4 scripts/eval_ame/run_rf3_dispatch.sbatch \
#       rf3_inputs/rf3_input_jsons.txt rf3_outputs/

set -eo pipefail

JSON_LIST="${1:?json list file required}"
OUT_ROOT="${2:?output root dir required}"

JSON_INPUT=$(sed -n "$((${SLURM_ARRAY_TASK_ID} + 1))p" "${JSON_LIST}")
[[ -z "${JSON_INPUT}" ]] && { echo "no JSON for task ${SLURM_ARRAY_TASK_ID}"; exit 0; }

# Derive a per-JSON output subdir from the JSON basename (no extension)
JSON_BASE=$(basename "${JSON_INPUT}" .json)
OUT_DIR="${OUT_ROOT%/}/${JSON_BASE}"

LAUNCHER=/home/users/tianyulu/code/rhaven/scripts/structure_prediction/rf3/launch_rf3_sbatch_apptainer.sh
INNER=/home/users/tianyulu/code/protpardelle-1c/scripts/eval_ame/run_rf3.sbatch

echo "[$(date)] dispatch task ${SLURM_ARRAY_TASK_ID}: ${JSON_INPUT} -> ${OUT_DIR}"

# Make a per-task copy of the inner script with JSON_INPUT/OUT_DIR baked in.
TMP_INNER=$(mktemp /scratch/users/tianyulu/tmp/rf3_inner_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.XXXX.sh)
{
  head -n 1 "${INNER}"  # shebang
  # Slurm directives from the inner template, kept intact
  grep -E '^[[:space:]]*#SBATCH' "${INNER}"
  echo ""
  echo "# Variables baked in by run_rf3_dispatch.sbatch"
  echo "export JSON_INPUT=\"${JSON_INPUT}\""
  echo "export OUT_DIR=\"${OUT_DIR}\""
  echo ""
  # Body of the inner script (skip shebang + #SBATCH lines)
  grep -vE '^[[:space:]]*#SBATCH|^#!' "${INNER}"
} > "${TMP_INNER}"
chmod +x "${TMP_INNER}"

bash "${LAUNCHER}" "${TMP_INNER}"
