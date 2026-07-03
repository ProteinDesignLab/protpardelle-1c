#!/usr/bin/env python3
"""
Generate RF3 input JSON and shell scripts for chroma_08_top1000_rmsd sequences.
Creates CRO and no-CRO versions, split into 8 chunks (~1000 seqs each).
"""

import csv
import json
import re
import os
from pathlib import Path
import math

CSV_PATH = Path("/scratch/users/tianyulu/allatom-design-sampling-results/chroma_08_top1000_rmsd/chroma_08_top1000_rmsd/seq_des_outputs.csv")
INPUT_DIR = Path("/scratch/users/tianyulu/gfp/denovo/chroma_08/caliby/rf3_inputs")
RF3_OUTPUT_BASE = Path("/scratch/users/tianyulu/gfp/denovo/chroma_08/caliby")

NUM_CHUNKS = 8
CHROMOPHORE_PATTERN = re.compile(r'PTLVTTLTYG')

AA_MAP = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
}

all_cro_entries = []
all_no_cro_entries = []

print(f"Reading {CSV_PATH}")
skipped = 0
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        seq = row['seq']
        out_pdb = row['out_pdb']
        sample_name = Path(out_pdb).stem

        motif_match = CHROMOPHORE_PATTERN.search(seq)
        if not motif_match:
            print(f"  Warning: PTLVTTLTYG not found in {sample_name}, skipping")
            skipped += 1
            continue

        motif_start = motif_match.start()
        tyg_start = motif_start + 7  # 0-indexed position of T in TYG
        tyg_pos = tyg_start + 1      # 1-indexed

        residue_after_tyg = seq[tyg_start + 3]
        cro_seq = seq[:tyg_start] + "(CRO)" + seq[tyg_start + 3:]

        leu_pos = tyg_pos - 1
        cro_pos = tyg_pos
        next_pos = tyg_pos + 1
        next_aa_3letter = AA_MAP.get(residue_after_tyg, 'UNK')

        all_cro_entries.append({
            "name": f"{sample_name}_cro",
            "components": [{"seq": cro_seq, "chain_id": "A"}],
            "bonds": [
                [f"A/LEU/{leu_pos}/C", f"A/CRO/{cro_pos}/N1"],
                [f"A/CRO/{cro_pos}/C3", f"A/{next_aa_3letter}/{next_pos}/N"]
            ]
        })

        all_no_cro_entries.append({
            "name": f"{sample_name}_no_cro",
            "components": [{"seq": seq, "chain_id": "A"}]
        })

print(f"Processed {len(all_cro_entries)} sequences, skipped {skipped}")

# Split into chunks
seqs_per_chunk = math.ceil(len(all_cro_entries) / NUM_CHUNKS)
print(f"Splitting into {NUM_CHUNKS} chunks of ~{seqs_per_chunk} sequences")

cro_json_dir = INPUT_DIR / "cro_json"
no_cro_json_dir = INPUT_DIR / "no_cro_json"
cro_json_dir.mkdir(parents=True, exist_ok=True)
no_cro_json_dir.mkdir(parents=True, exist_ok=True)

for chunk_idx in range(NUM_CHUNKS):
    start = chunk_idx * seqs_per_chunk
    end = min(start + seqs_per_chunk, len(all_cro_entries))
    if start >= len(all_cro_entries):
        break

    with open(cro_json_dir / f"cro_{chunk_idx:03d}.json", 'w') as f:
        json.dump(all_cro_entries[start:end], f, indent=2)
    with open(no_cro_json_dir / f"no_cro_{chunk_idx:03d}.json", 'w') as f:
        json.dump(all_no_cro_entries[start:end], f, indent=2)

    print(f"  Chunk {chunk_idx:03d}: {end - start} sequences")

# Write array job scripts
for label, json_subdir, out_subdir in [("cro", "cro_json", "cro"), ("no_cro", "no_cro_json", "no_cro")]:
    script = f"""#!/bin/bash
#SBATCH --job-name=c08t_{label[:2]}
#SBATCH -p bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-{NUM_CHUNKS - 1}%4
#
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/c08t_{label[:2]}_%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/c08t_{label[:2]}_%A_%a.err

# Activate environment
ENV_DIR=/scratch/users/tianyulu/envs
source ${{ENV_DIR}}/rf3/bin/activate

export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real

CHUNK_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

INPUT_FILE={INPUT_DIR}/{json_subdir}/{label}_${{CHUNK_ID}}.json
OUTPUT_DIR={RF3_OUTPUT_BASE}/{out_subdir}/${{CHUNK_ID}}
mkdir -p ${{OUTPUT_DIR}}

cd /home/users/tianyulu/code/foundry

python models/rf3/src/rf3/inference.py \\
    inputs="${{INPUT_FILE}}" \\
    ckpt_path=/oak/stanford/groups/possu/tianyu/models/foundry/rf3_foundry_01_24_latest_remapped.ckpt \\
    out_dir="${{OUTPUT_DIR}}" \\
    one_model_per_file=true \\
    annotate_b_factor_with_plddt=true \\
    early_stopping_plddt_threshold=0.0
"""
    script_path = INPUT_DIR / f"rf3_{label}_array.sh"
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    print(f"Wrote {script_path}")

print("\nDone!")
