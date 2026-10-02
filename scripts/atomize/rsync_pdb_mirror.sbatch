#!/bin/bash
#SBATCH --job-name=pdb_mirror
#SBATCH -p owners
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --array=0-7
#SBATCH -o /scratch/users/tianyulu/slurm_log_files/%x_%A_%a.out
#SBATCH -e /scratch/users/tianyulu/slurm_log_files/%x_%A_%a.err
set -uo pipefail
DEST=/scratch/users/tianyulu/datasets/pdb_mirror/mmCIF
mkdir -p $DEST
# 2-char hash dirs are [0-9a-z]{2}; split the 36 first-chars across 8 array tasks
CHARS=(0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z)
for i in "${!CHARS[@]}"; do
  if [ $(( i % 8 )) -ne ${SLURM_ARRAY_TASK_ID} ]; then continue; fi
  c=${CHARS[$i]}
  echo "=== rsync prefix $c $(date) ==="
  rsync -rlpt --port=33444 --include="${c}*/" --include="${c}*/*" --exclude="*" \
      rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ $DEST/ 2>&1 | tail -3
done
echo "DONE $(date)"
du -sh $DEST
