#!/bin/bash
# Download Pinder 2024-02 dimer PDBs from gs://pinder for the genie3 dataset.
#
# Layout written:
#     <out-dir>/pdbs/<2char-prefix>/<id>.pdb
#
# A flat <out-dir>/filtered_pdbs/ with 1.5M files hit Lustre "Disk quota
# exceeded" on Sherlock (with overall block + inode quota well below limits),
# so we shard by first two chars of <id>. dataset.py's pinder_2024-02 branch
# resolves files via this sharded layout.
#
# Usage:
#   sbatch --job-name=pinder_genie3 scripts/download/download_pinder_genie3.sbatch \
#       /scratch/users/tianyulu/datasets/genie3/data/train/pinder/2024-02/info.csv \
#       /scratch/users/tianyulu/datasets/genie3/pinder_2024-02
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/pinder_dl-%x.%j.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/pinder_dl-%x.%j.err
#SBATCH --partition=possu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00

set -euo pipefail

INFO_CSV=${1:?missing info.csv}
OUT_DIR=${2:?missing out-dir}

mkdir -p "${OUT_DIR}/pdbs"

# Lustre striping: round-robin all OSTs so a 1.5M-file directory tree doesn't
# concentrate on a few OSTs (which is what we believe caused the previous
# extract to fail with "Disk quota exceeded" mid-stream).
if command -v lfs >/dev/null 2>&1; then
    lfs setstripe -c -1 "${OUT_DIR}/pdbs" 2>/dev/null || true
fi

if command -v module >/dev/null 2>&1; then
    module load google-cloud-sdk
fi

source /scratch/users/tianyulu/envs/protpardelle/bin/activate

ZIP_PATH="${OUT_DIR}/pdbs.zip"
if [[ ! -s "${ZIP_PATH}" ]]; then
    echo "[pinder] downloading pdbs.zip from gs://pinder/2024-02/pdbs.zip ..."
    gsutil -o "GSUtil:parallel_thread_count=8" -m cp \
        gs://pinder/2024-02/pdbs.zip "${ZIP_PATH}"
else
    echo "[pinder] reusing existing ${ZIP_PATH} ($(du -h "${ZIP_PATH}" | cut -f1))"
fi

# slurm copies the sbatch script to /var/spool/slurmd, so $0 doesn't resolve
# back to the project tree. Use protpardelle.env.PROJECT_ROOT_DIR (same trick
# the dssp + train sbatch scripts use) to locate the sibling extractor.
PROJECT_ROOT_DIR=$(python -c "from protpardelle.env import PROJECT_ROOT_DIR; print(PROJECT_ROOT_DIR)")

echo "[pinder] streaming extract -> ${OUT_DIR}/pdbs/<2char>/<id>.pdb ..."
python "${PROJECT_ROOT_DIR}/scripts/download/extract_pinder_genie3.py" \
    --zip-path "${ZIP_PATH}" \
    --info-csv "${INFO_CSV}" \
    --out-dir "${OUT_DIR}/pdbs"

n_subdirs=$(find "${OUT_DIR}/pdbs" -maxdepth 1 -mindepth 1 -type d | wc -l)
n_files=$(find "${OUT_DIR}/pdbs" -maxdepth 2 -type f -name '*.pdb' | wc -l)
echo "[pinder] done. subdirs=${n_subdirs} files=${n_files}"
echo "[pinder] zip can be removed once verified: ${ZIP_PATH}"
