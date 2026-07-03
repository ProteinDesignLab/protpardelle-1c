#!/usr/bin/env python3
"""Generate RF3 input JSON files for GFP designs.

For each design group:
1. Find the sequence with lowest energy
2. Create CRO version (TYG replaced with (CRO) + bond connectivity)
3. Create no-CRO version (original sequence)
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict


# 3-letter to 1-letter amino acid code
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# 1-letter to 3-letter amino acid code
AA_1TO3 = {v: k for k, v in AA_3TO1.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate RF3 input JSON files")
    parser.add_argument("--csv_file", required=True, help="Path to design output CSV")
    parser.add_argument("--partition_dir", required=True, help="Path to partitioned PDB directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for JSON and sh files")
    parser.add_argument("--rf3_output_base", default="/scratch/users/tianyulu/gfp/rf3_outputs",
                        help="Base directory for RF3 outputs")
    return parser.parse_args()


def find_chromophore_position(seq, fix_pos_list):
    """Find the position of TYG chromophore in the sequence.

    The fix_pos contains positions of fixed residues including the chromophore region.
    We need to find where T-Y-G appears in the sequence (the chromophore).

    Returns: (tyg_start_pos, prev_residue, prev_pos, next_residue, next_pos)
    where positions are 1-indexed.
    """
    # Sort fix_pos to find contiguous regions
    fix_pos = sorted(fix_pos_list)

    # Look for TYG pattern in the sequence
    # The chromophore TYG should be in the fix_pos region
    for i, pos in enumerate(fix_pos):
        if pos + 2 <= len(seq):
            # Check if we have TYG at this position (1-indexed)
            triplet = seq[pos-1:pos+2]  # Convert to 0-indexed
            if triplet == "TYG":
                # Found TYG at position pos, pos+1, pos+2
                prev_pos = pos - 1
                prev_residue = seq[prev_pos - 1] if prev_pos >= 1 else None
                next_pos = pos + 3
                next_residue = seq[next_pos - 1] if next_pos <= len(seq) else None
                return pos, prev_residue, prev_pos, next_residue, next_pos

    # If no exact TYG found, look for the LTYGVQ pattern
    pattern_idx = seq.find("LTYGVQ")
    if pattern_idx != -1:
        # TYG starts at pattern_idx + 1 (0-indexed) = pattern_idx + 2 (1-indexed)
        tyg_start = pattern_idx + 2  # 1-indexed position of T
        prev_pos = tyg_start - 1
        prev_residue = seq[prev_pos - 1]
        next_pos = tyg_start + 3
        next_residue = seq[next_pos - 1]
        return tyg_start, prev_residue, prev_pos, next_residue, next_pos

    # Broader search for TYG anywhere in sequence
    tyg_idx = seq.find("TYG")
    if tyg_idx != -1:
        tyg_start = tyg_idx + 1  # 1-indexed
        prev_pos = tyg_start - 1
        prev_residue = seq[prev_pos - 1] if prev_pos >= 1 else None
        next_pos = tyg_start + 3
        next_residue = seq[next_pos - 1] if next_pos <= len(seq) else None
        return tyg_start, prev_residue, prev_pos, next_residue, next_pos

    return None, None, None, None, None


def create_cro_sequence(seq, tyg_start):
    """Replace TYG with (CRO) in the sequence.

    tyg_start is 1-indexed position of T in TYG.
    """
    # Convert to 0-indexed
    idx = tyg_start - 1
    # Replace TYG (3 chars) with (CRO)
    cro_seq = seq[:idx] + "(CRO)" + seq[idx+3:]
    return cro_seq


def create_bonds(prev_residue, prev_pos, next_residue, next_pos, cro_pos):
    """Create bond connectivity for CRO.

    Bonds connect:
    - Previous residue's C atom to CRO's N1 atom
    - CRO's C3 atom to next residue's N atom

    Note: After replacing TYG with CRO, the positions shift.
    CRO takes position of T (tyg_start).
    Next residue (originally at tyg_start+3) is now at tyg_start+1.
    """
    prev_aa3 = AA_1TO3.get(prev_residue, prev_residue)
    next_aa3 = AA_1TO3.get(next_residue, next_residue)

    # In the CRO sequence, positions after CRO shift by -2
    # CRO is at position cro_pos
    # Next residue is at cro_pos + 1
    next_pos_in_cro_seq = cro_pos + 1

    bonds = [
        [f"A/{prev_aa3}/{prev_pos}/C", f"A/CRO/{cro_pos}/N1"],
        [f"A/CRO/{cro_pos}/C3", f"A/{next_aa3}/{next_pos_in_cro_seq}/N"]
    ]
    return bonds


def get_partition_from_path(pdb_path, partition_dir):
    """Determine which partition a design belongs to based on its location."""
    # The pdb_path from CSV might be original, we need to find where it was moved
    design_file = os.path.basename(pdb_path)

    # Check each partition
    partitions = [
        "tm_above_0.5",
        "tm_below_0.5/11_stranded",
        "tm_below_0.5/not_11_stranded"
    ]

    for partition in partitions:
        check_path = os.path.join(partition_dir, partition, design_file)
        if os.path.exists(check_path):
            return partition

    return None


def main():
    args = parse_args()

    # Read CSV and group by design group (cluster_rep)
    print("Reading CSV file...")
    design_groups = defaultdict(list)

    with open(args.csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster = row['cluster_rep']
            design_groups[cluster].append(row)

    print(f"Found {len(design_groups)} design groups")

    # For each group, find lowest energy design
    print("Finding lowest energy sequences...")
    lowest_energy_designs = {}

    for cluster, designs in design_groups.items():
        # Sort by energy and take lowest
        sorted_designs = sorted(designs, key=lambda x: float(x['energy']))
        lowest = sorted_designs[0]
        lowest_energy_designs[cluster] = lowest

    print(f"Selected {len(lowest_energy_designs)} lowest energy designs")

    # Group designs by partition
    print("Grouping by partition...")
    partition_designs = defaultdict(list)

    for cluster, design in lowest_energy_designs.items():
        # Get design file name (e.g., d0_0.pdb)
        design_file = design['design_pdb']
        # Extract base name (e.g., d0)
        base_match = re.match(r'^(d\d+)_\d\.pdb$', design_file)
        if not base_match:
            continue
        base_name = base_match.group(1)

        # Find which partition this design is in
        partition = get_partition_from_path(
            os.path.join(args.partition_dir, f"{base_name}_0.pdb"),
            args.partition_dir
        )

        if partition:
            partition_designs[partition].append({
                'name': base_name,
                'seq': design['seq'],
                'fix_pos': [int(x) for x in design['fix_pos'].split(',')],
                'energy': float(design['energy']),
                'cluster': cluster
            })

    # Report counts
    for partition, designs in partition_designs.items():
        print(f"  {partition}: {len(designs)} designs")

    # Create output directories and generate JSON files
    os.makedirs(args.output_dir, exist_ok=True)

    for partition, designs in partition_designs.items():
        partition_safe = partition.replace('/', '_')

        print(f"\nProcessing partition: {partition}")

        # Create JSON entries
        cro_entries = []
        no_cro_entries = []
        failed_entries = []

        for design in designs:
            seq = design['seq']
            fix_pos = design['fix_pos']
            name = design['name']

            # Find chromophore position
            tyg_start, prev_res, prev_pos, next_res, next_pos = find_chromophore_position(seq, fix_pos)

            if tyg_start is None:
                failed_entries.append(name)
                continue

            # Create CRO entry
            cro_seq = create_cro_sequence(seq, tyg_start)
            bonds = create_bonds(prev_res, prev_pos, next_res, next_pos, tyg_start)

            cro_entry = {
                "name": f"{name}_cro",
                "components": [{"seq": cro_seq, "chain_id": "A"}],
                "bonds": bonds
            }
            cro_entries.append(cro_entry)

            # Create no-CRO entry
            no_cro_entry = {
                "name": f"{name}_no_cro",
                "components": [{"seq": seq, "chain_id": "A"}]
            }
            no_cro_entries.append(no_cro_entry)

        if failed_entries:
            print(f"  Warning: Failed to find TYG in {len(failed_entries)} sequences")

        # Write JSON files
        cro_json_path = os.path.join(args.output_dir, f"{partition_safe}_cro.json")
        no_cro_json_path = os.path.join(args.output_dir, f"{partition_safe}_no_cro.json")

        with open(cro_json_path, 'w') as f:
            json.dump(cro_entries, f, indent=4)
        print(f"  Wrote {len(cro_entries)} CRO entries to {cro_json_path}")

        with open(no_cro_json_path, 'w') as f:
            json.dump(no_cro_entries, f, indent=4)
        print(f"  Wrote {len(no_cro_entries)} no-CRO entries to {no_cro_json_path}")

        # Generate shell scripts
        cro_output_dir = os.path.join(args.rf3_output_base, f"chroma_05_{partition_safe}_cro")
        no_cro_output_dir = os.path.join(args.rf3_output_base, f"chroma_05_{partition_safe}_no_cro")

        cro_sh_content = f'''#!/bin/bash
#SBATCH --job-name=rf3_{partition_safe[:8]}_c
#SBATCH -p bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/job_%A_%a.err

# Activate environment
ENV_DIR=/scratch/users/tianyulu/envs
source ${{ENV_DIR}}/rf3/bin/activate

export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real

INPUT_FILE={cro_json_path}
OUTPUT_DIR={cro_output_dir}
mkdir -p ${{OUTPUT_DIR}}

cd /home/users/tianyulu/code/foundry

python models/rf3/src/rf3/inference.py \\
    inputs="${{INPUT_FILE}}" \\
    ckpt_path=/oak/stanford/groups/possu/tianyu/models/foundry/rf3_foundry_01_24_latest_remapped.ckpt \\
    out_dir="${{OUTPUT_DIR}}" \\
    one_model_per_file=true \\
    annotate_b_factor_with_plddt=true \\
    early_stopping_plddt_threshold=0.0
'''

        no_cro_sh_content = f'''#!/bin/bash
#SBATCH --job-name=rf3_{partition_safe[:8]}_nc
#SBATCH -p bioe,possu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#
#SBATCH --output=/scratch/users/tianyulu/slurm_log_files/job_%A_%a.out
#SBATCH --error=/scratch/users/tianyulu/slurm_log_files/job_%A_%a.err

# Activate environment
ENV_DIR=/scratch/users/tianyulu/envs
source ${{ENV_DIR}}/rf3/bin/activate

export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real

INPUT_FILE={no_cro_json_path}
OUTPUT_DIR={no_cro_output_dir}
mkdir -p ${{OUTPUT_DIR}}

cd /home/users/tianyulu/code/foundry

python models/rf3/src/rf3/inference.py \\
    inputs="${{INPUT_FILE}}" \\
    ckpt_path=/oak/stanford/groups/possu/tianyu/models/foundry/rf3_foundry_01_24_latest_remapped.ckpt \\
    out_dir="${{OUTPUT_DIR}}" \\
    one_model_per_file=true \\
    annotate_b_factor_with_plddt=true \\
    early_stopping_plddt_threshold=0.0
'''

        cro_sh_path = os.path.join(args.output_dir, f"rf3_{partition_safe}_cro.sh")
        no_cro_sh_path = os.path.join(args.output_dir, f"rf3_{partition_safe}_no_cro.sh")

        with open(cro_sh_path, 'w') as f:
            f.write(cro_sh_content)
        os.chmod(cro_sh_path, 0o755)

        with open(no_cro_sh_path, 'w') as f:
            f.write(no_cro_sh_content)
        os.chmod(no_cro_sh_path, 0o755)

        print(f"  Created shell scripts: {os.path.basename(cro_sh_path)}, {os.path.basename(no_cro_sh_path)}")

    print("\nDone!")
    print(f"\nTo submit jobs:")
    for partition in partition_designs.keys():
        partition_safe = partition.replace('/', '_')
        print(f"  sbatch {os.path.join(args.output_dir, f'rf3_{partition_safe}_cro.sh')}")
        print(f"  sbatch {os.path.join(args.output_dir, f'rf3_{partition_safe}_no_cro.sh')}")


if __name__ == "__main__":
    main()
