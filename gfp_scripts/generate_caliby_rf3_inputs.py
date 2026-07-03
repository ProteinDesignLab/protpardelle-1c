#!/usr/bin/env python3
"""
Generate RF3 input JSON and shell script files from denovo GFP design ensemble
seq_des_outputs.csv files. Creates both CRO and no-CRO versions for each design.
Splits into chunks of max 2000 sequences per JSON file, with at most 4 array
jobs per .sh script.
"""

import csv
import json
import re
import os
from pathlib import Path
import math

# Paths
BASE_DIR = Path("/scratch/users/tianyulu/allatom-design-sampling-results/denovo_gfp_design_ensemble")
INPUT_DIR = Path("/scratch/users/tianyulu/gfp/rf3_outputs/caliby/rf3_inputs")
RF3_OUTPUT_BASE = Path("/scratch/users/tianyulu/gfp/rf3_outputs/caliby")

# Max sequences per JSON file
MAX_SEQS_PER_FILE = 2000

# Max array jobs per .sh script
MAX_JOBS_PER_SCRIPT = 4

# Subfolders with their CSV files and name prefixes
SUBFOLDERS = [
    ("gt05", BASE_DIR / "tm_gt_05/denovo_gfp_design_ensemble/seq_des_outputs.csv"),
    ("lt05n11", BASE_DIR / "tm_lt_05_not_11_stranded/denovo_gfp_design_ensemble/seq_des_outputs.csv"),
    ("lt05_11s", BASE_DIR / "tm_lt_05_11_stranded/denovo_gfp_design_ensemble/seq_des_outputs.csv"),
]

# The conserved chromophore motif pattern
CHROMOPHORE_PATTERN = re.compile(r'PTLVTTLTYG')

# Amino acid 1-letter to 3-letter code mapping
AA_MAP = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
}

# Collect all entries
all_cro_entries = []
all_no_cro_entries = []

for prefix, csv_path in SUBFOLDERS:
    print(f"Processing {prefix}: {csv_path}")
    if not csv_path.exists():
        print(f"  WARNING: File not found, skipping")
        continue

    skipped = 0
    count = 0

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            example_id = row['example_id']
            seq = row['seq']
            out_pdb = row['out_pdb']

            # Extract sample name from the output path (e.g., d24_0_sample0)
            sample_name = Path(out_pdb).stem  # e.g., d24_0_sample0

            # Find the conserved chromophore motif PTLVTTLTYG
            motif_match = CHROMOPHORE_PATTERN.search(seq)
            if not motif_match:
                print(f"  Warning: PTLVTTLTYG not found in {sample_name}, skipping")
                skipped += 1
                continue

            motif_start = motif_match.start()
            tyg_start = motif_start + 7  # Position of T in TYG (0-indexed)
            tyg_pos = tyg_start + 1      # 1-indexed position

            if seq[tyg_start - 1] != 'L':
                print(f"  Warning: Expected L before TYG but found {seq[tyg_start - 1]} in {sample_name}")

            residue_after_tyg = seq[tyg_start + 3]
            cro_seq = seq[:tyg_start] + "(CRO)" + seq[tyg_start + 3:]

            leu_pos = tyg_pos - 1
            cro_pos = tyg_pos
            next_pos = tyg_pos + 1
            next_aa_3letter = AA_MAP.get(residue_after_tyg, 'UNK')

            design_name = f"{prefix}_{sample_name}"

            # CRO entry
            cro_entry = {
                "name": f"{design_name}_cro",
                "components": [{"seq": cro_seq, "chain_id": "A"}],
                "bonds": [
                    [f"A/LEU/{leu_pos}/C", f"A/CRO/{cro_pos}/N1"],
                    [f"A/CRO/{cro_pos}/C3", f"A/{next_aa_3letter}/{next_pos}/N"]
                ]
            }
            all_cro_entries.append(cro_entry)

            # No CRO entry
            no_cro_entry = {
                "name": f"{design_name}_no_cro",
                "components": [{"seq": seq, "chain_id": "A"}]
            }
            all_no_cro_entries.append(no_cro_entry)
            count += 1

    print(f"  Processed {count} designs, skipped {skipped}")

print(f"\nTotal entries: {len(all_cro_entries)} CRO, {len(all_no_cro_entries)} no-CRO")

# Split into chunks
num_chunks = math.ceil(len(all_cro_entries) / MAX_SEQS_PER_FILE)
print(f"Splitting into {num_chunks} chunks of max {MAX_SEQS_PER_FILE} sequences each")

# Create subdirectories for JSON files
cro_json_dir = INPUT_DIR / "cro_json"
no_cro_json_dir = INPUT_DIR / "no_cro_json"
cro_json_dir.mkdir(parents=True, exist_ok=True)
no_cro_json_dir.mkdir(parents=True, exist_ok=True)

for chunk_idx in range(num_chunks):
    start = chunk_idx * MAX_SEQS_PER_FILE
    end = min(start + MAX_SEQS_PER_FILE, len(all_cro_entries))

    cro_chunk = all_cro_entries[start:end]
    no_cro_chunk = all_no_cro_entries[start:end]

    cro_json_path = cro_json_dir / f"cro_{chunk_idx:03d}.json"
    no_cro_json_path = no_cro_json_dir / f"no_cro_{chunk_idx:03d}.json"

    with open(cro_json_path, 'w') as f:
        json.dump(cro_chunk, f, indent=2)

    with open(no_cro_json_path, 'w') as f:
        json.dump(no_cro_chunk, f, indent=2)

    print(f"  Chunk {chunk_idx:03d}: {len(cro_chunk)} sequences")

print(f"Wrote {num_chunks} CRO JSON files to {cro_json_dir}")
print(f"Wrote {num_chunks} no-CRO JSON files to {no_cro_json_dir}")

# Create array job scripts, at most MAX_JOBS_PER_SCRIPT jobs per script
num_scripts = math.ceil(num_chunks / MAX_JOBS_PER_SCRIPT)
print(f"\nCreating {num_scripts} script(s) per type (max {MAX_JOBS_PER_SCRIPT} jobs each)")

for script_idx in range(num_scripts):
    array_start = script_idx * MAX_JOBS_PER_SCRIPT
    array_end = min(array_start + MAX_JOBS_PER_SCRIPT, num_chunks) - 1

    suffix = f"_{script_idx}" if num_scripts > 1 else ""

    cro_script = f"""#!/bin/bash
#SBATCH --job-name=cal_cro{suffix}
#SBATCH -p bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array={array_start}-{array_end}
#
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/cal_cro_%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/cal_cro_%A_%a.err

# Activate environment
ENV_DIR=/scratch/users/tianyulu/envs
source ${{ENV_DIR}}/rf3/bin/activate

export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real

# Zero-pad the array task ID to match file naming
CHUNK_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

INPUT_FILE={cro_json_dir}/cro_${{CHUNK_ID}}.json
OUTPUT_DIR={RF3_OUTPUT_BASE}/cro/${{CHUNK_ID}}
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

    no_cro_script = f"""#!/bin/bash
#SBATCH --job-name=cal_nc{suffix}
#SBATCH -p bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array={array_start}-{array_end}
#
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/cal_nc_%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/cal_nc_%A_%a.err

# Activate environment
ENV_DIR=/scratch/users/tianyulu/envs
source ${{ENV_DIR}}/rf3/bin/activate

export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real

# Zero-pad the array task ID to match file naming
CHUNK_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

INPUT_FILE={no_cro_json_dir}/no_cro_${{CHUNK_ID}}.json
OUTPUT_DIR={RF3_OUTPUT_BASE}/no_cro/${{CHUNK_ID}}
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

    cro_script_path = INPUT_DIR / f"rf3_cro_array{suffix}.sh"
    no_cro_script_path = INPUT_DIR / f"rf3_no_cro_array{suffix}.sh"

    with open(cro_script_path, 'w') as f:
        f.write(cro_script)
    os.chmod(cro_script_path, 0o755)

    with open(no_cro_script_path, 'w') as f:
        f.write(no_cro_script)
    os.chmod(no_cro_script_path, 0o755)

    print(f"  Script {script_idx}: array {array_start}-{array_end}")
    print(f"    {cro_script_path}")
    print(f"    {no_cro_script_path}")

print(f"\nDone! To submit, use the launch_rf3_sbatch_apptainer.sh wrapper for each script.")
