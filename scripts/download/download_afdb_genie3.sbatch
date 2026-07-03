#!/bin/bash
# Download AFDB PDBs (gzipped) for the genie3 afdbreps_l-512_plddt-70 set.
#
# The genie3 info.csv names entries as `AF-{uniprot}-F1-model_v4`, but AFDB
# has since shipped v5/v6 and many v4 URLs now 404 (~97% in our last run).
# This script tries `_v4` first, then `_v6` and `_v5` as fallbacks. Curl's
# exit code is checked via PIPESTATUS (not the gzip exit code) so failures
# are not silently written as empty 20-byte gzip files. Missing IDs are
# recorded to a per-shard fail log under $SCRATCH/slurm_log_files/.
#
# Usage:
#   sbatch --job-name=afdb_genie3 scripts/download/download_afdb_genie3.sbatch \
#       /scratch/users/tianyulu/datasets/genie3/afdb_l512_plddt70_names.list \
#       /scratch/users/tianyulu/datasets/genie3/afdb_l512_plddt70/pdbs
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/afdb_dl-%x.%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/afdb_dl-%x.%A_%a.err
#SBATCH --partition=possu
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-31

set -euo pipefail

NAMES_LIST=${1:?missing names-list}
OUT_DIR=${2:?missing out-dir}

NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-32}
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}

mkdir -p "${OUT_DIR}"
FAIL_LOG="/scratch/users/tianyulu/slurm_log_files/afdb_dl-${SLURM_JOB_NAME:-afdb_genie3}-${SLURM_ARRAY_JOB_ID:-$$}_${SHARD_ID}.fail"
: > "${FAIL_LOG}"

shard_file=$(mktemp)
awk -v sid="${SHARD_ID}" -v ns="${NUM_SHARDS}" 'NR % ns == sid' "${NAMES_LIST}" > "${shard_file}"
n_total=$(wc -l < "${shard_file}")
echo "[shard ${SHARD_ID}/${NUM_SHARDS}] ${n_total} entries -> ${OUT_DIR}"
echo "[shard ${SHARD_ID}/${NUM_SHARDS}] fail log: ${FAIL_LOG}"

fetch_one() {
    local name="$1"
    local out="${OUT_DIR}/${name}.pdb.gz"
    # Resume-safe: skip non-empty existing files. (20-byte empties were cleaned
    # up before this rerun.)
    if [[ -s "${out}" ]]; then
        return 0
    fi

    # genie3's manifest names entries with `_v4`. AFDB now serves v6/v5 for
    # many entries (v4 was retired). The genie3 IDs are just the uniprot
    # accession plus version suffix, so we substitute and retry.
    local base="${name%_v*}"  # e.g. AF-A0A4P6L4I8-F1-model
    local versions=(v4 v6 v5 v3)
    local tmp="${out}.tmp"
    rm -f "${tmp}"

    for v in "${versions[@]}"; do
        local url="https://alphafold.ebi.ac.uk/files/${base}_${v}.pdb"
        # Capture curl's exit status independently of the gzip pipeline so
        # 404s/timeouts are not silently turned into 20-byte empty gzips.
        set +e
        curl -sSL --fail --retry 3 --retry-delay 2 --max-time 60 -o "${tmp}.pdb" "${url}"
        local rc=$?
        set -e
        if [[ ${rc} -eq 0 && -s "${tmp}.pdb" ]]; then
            gzip -c "${tmp}.pdb" > "${tmp}"
            rm -f "${tmp}.pdb"
            mv "${tmp}" "${out}"
            return 0
        fi
        rm -f "${tmp}.pdb"
    done

    # All versions 404'd / errored.
    echo "${name}" >> "${FAIL_LOG}"
    return 1
}
export -f fetch_one
export OUT_DIR FAIL_LOG

cat "${shard_file}" | xargs -n1 -P16 -I{} bash -c 'fetch_one "$@" || true' _ {}

rm -f "${shard_file}"
n_fail=$(wc -l < "${FAIL_LOG}")
echo "[shard ${SHARD_ID}/${NUM_SHARDS}] done. failures=${n_fail}/${n_total}"
