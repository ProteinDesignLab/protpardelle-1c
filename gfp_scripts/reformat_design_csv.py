#!/usr/bin/env python3
"""Reformat design CSV to fixpos format and copy lowest energy PDBs."""

import argparse
import csv
import os
import re
import shutil
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Reformat design CSV to fixpos format")
    parser.add_argument("--input_csv", required=True, help="Input design CSV")
    parser.add_argument("--output_csv", required=True, help="Output fixpos CSV")
    parser.add_argument("--output_pdb_dir", required=True, help="Directory for lowest energy PDBs")
    parser.add_argument("--source_pdb_dir", required=True, help="Source directory with partitioned PDBs")
    return parser.parse_args()


def find_pdb_in_partitions(design_file, source_dir):
    """Find the PDB file in one of the partition subdirectories."""
    partitions = [
        "tm_above_0.5",
        "tm_below_0.5/11_stranded",
        "tm_below_0.5/not_11_stranded",
        ""  # Also check root directory
    ]

    for partition in partitions:
        check_path = os.path.join(source_dir, partition, design_file)
        if os.path.exists(check_path):
            return check_path

    return None


def main():
    args = parse_args()

    os.makedirs(args.output_pdb_dir, exist_ok=True)

    # Read CSV and group by design group
    print("Reading input CSV...")
    design_groups = defaultdict(list)

    with open(args.input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract base name from design_pdb (e.g., d0_0.pdb -> d0)
            design_pdb = row['design_pdb']
            match = re.match(r'^(d\d+)_\d\.pdb$', design_pdb)
            if match:
                base_name = match.group(1)
                design_groups[base_name].append(row)

    print(f"Found {len(design_groups)} design groups")

    # Find lowest energy design per group
    print("Finding lowest energy designs...")
    lowest_energy_designs = {}

    for base_name, designs in design_groups.items():
        sorted_designs = sorted(designs, key=lambda x: float(x['energy']))
        lowest_energy_designs[base_name] = sorted_designs[0]

    # Write output CSV and copy PDBs
    print("Writing output CSV and copying PDBs...")

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pdb_key', 'fixed_pos_seq', 'fixed_pos_scn', 'fixed_pos_override_seq', 'pos_restrict_aatype'])

        copied = 0
        skipped = 0

        for base_name in sorted(lowest_energy_designs.keys(), key=lambda x: int(x[1:])):
            design = lowest_energy_designs[base_name]

            # Parse fix_pos
            fix_pos = [int(x) for x in design['fix_pos'].split(',')]

            # The fix_pos already has CYS position missing (position 55 not in list)
            # Format: add chain prefix 'A'
            fixed_pos_seq = ','.join([f'A{pos}' for pos in fix_pos])

            # MET position is second to last in fix_pos, override to ARG
            met_pos = fix_pos[-2]  # Second to last position
            fixed_pos_override_seq = f'A{met_pos}:R'

            # Find and copy PDB
            design_pdb = design['design_pdb']
            source_path = find_pdb_in_partitions(design_pdb, args.source_pdb_dir)

            if source_path:
                # Keep original filename in pdbs subdirectory
                pdbs_subdir = os.path.join(args.output_pdb_dir, 'pdbs')
                os.makedirs(pdbs_subdir, exist_ok=True)
                dest_path = os.path.join(pdbs_subdir, design_pdb)
                shutil.copy2(source_path, dest_path)
                copied += 1

                # Use original filename (without .pdb) as pdb_key
                pdb_key = design_pdb.replace('.pdb', '')
                writer.writerow([pdb_key, fixed_pos_seq, '', fixed_pos_override_seq, ''])
            else:
                skipped += 1
                print(f"  Warning: Could not find {design_pdb}")

    print(f"\nDone!")
    print(f"  Copied {copied} PDB files to {args.output_pdb_dir}")
    print(f"  Skipped {skipped} files (not found)")
    print(f"  Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
