#!/bin/bash
#SBATCH --job-name=extract_pdb
#SBATCH -p normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH -o /scratch/users/tianyulu/slurm_log_files/%x_%j.out
#SBATCH -e /scratch/users/tianyulu/slurm_log_files/%x_%j.err
set -euo pipefail
DEST=/scratch/users/tianyulu/datasets/pdb_atomized
mkdir -p $DEST
SRC=/oak/stanford/groups/possu/tianyu/datasets
# 1) CCD (chemical component dictionary) for reference conformers/elements
if [ ! -s $DEST/components.cif.gz ]; then
  curl -sL -o $DEST/components.cif.gz https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
fi
echo "CCD done $(date)"
# 2) extract AF3-style assembly mmCIFs
for split in train val test; do
  if [ ! -d $DEST/${split}_mmcifs ]; then
    tar -xzf $SRC/${split}_mmcifs.tar.gz -C $DEST &
  fi
done
wait
echo "mmcif extract done $(date)"
find $DEST/train_mmcifs -name '*.cif' | wc -l
