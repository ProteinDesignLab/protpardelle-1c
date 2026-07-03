#!/usr/bin/env python3
"""Partition PDB structures by TM-score against a reference and beta barrel classification.

This version:
- Uses multiprocessing for parallel TM-score and DSSP computation
- Handles 8 designs per input structure (filters based on design_idx=0)
- Splits into subfolders based on partition label without discarding
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from functools import partial
from multiprocessing import Pool, cpu_count

import biotite.structure.io.pdb as pdb_io
from biotite.application.dssp import DsspApp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Partition PDB structures by TM-score and beta barrel classification."
    )
    parser.add_argument("--pdb_dir", required=True, help="Directory containing PDB structures")
    parser.add_argument("--ref_pdb", required=True, help="Path to reference PDB (e.g. 2AWJ)")
    parser.add_argument("--tmalign_bin", default="TMalign", help="Path to TMalign binary")
    parser.add_argument("--mkdssp_bin", default="mkdssp", help="Path to mkdssp binary")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--designs_per_structure", type=int, default=8,
                        help="Number of sequence designs per input structure (default: 8)")
    parser.add_argument("--copy_instead_of_move", action="store_true",
                        help="Copy files instead of moving them")
    return parser.parse_args()


def compute_tmscore_worker(args):
    """Worker function for parallel TM-score computation."""
    query_pdb, ref_pdb, tmalign_bin = args
    try:
        result = subprocess.run(
            [tmalign_bin, query_pdb, ref_pdb],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return (os.path.basename(query_pdb), None, f"TMalign failed: {result.stderr.strip()}")

        tm_scores = re.findall(r"TM-score=\s*([\d.]+)", result.stdout)
        if len(tm_scores) < 2:
            return (os.path.basename(query_pdb), None, "Could not parse TM-scores")
        return (os.path.basename(query_pdb), float(tm_scores[1]), None)
    except subprocess.TimeoutExpired:
        return (os.path.basename(query_pdb), None, "TMalign timeout")
    except Exception as e:
        return (os.path.basename(query_pdb), None, str(e))


def count_beta_strands_worker(args):
    """Worker function for parallel DSSP computation."""
    pdb_path, mkdssp_bin = args
    fname = os.path.basename(pdb_path)
    try:
        pdb_file = pdb_io.PDBFile.read(pdb_path)
        structure = pdb_file.get_structure(model=1)

        app = DsspApp(structure, bin_path=mkdssp_bin)
        app.start()
        app.join()
        sse = app.get_sse()

        # Count contiguous runs of 'E'
        n_strands = 0
        in_strand = False
        for ss in sse:
            if ss == 'E':
                if not in_strand:
                    n_strands += 1
                    in_strand = True
            else:
                in_strand = False
        return (fname, n_strands, None)
    except Exception as e:
        return (fname, None, str(e))


def get_design_prefix(fname):
    """Extract design prefix (e.g., 'd123' from 'd123_0.pdb')."""
    match = re.match(r'(d\d+)_\d+\.pdb', fname)
    return match.group(1) if match else None


def get_related_designs(prefix, all_files, designs_per_structure):
    """Get all design files for a given prefix (e.g., d123_0.pdb through d123_7.pdb)."""
    related = []
    for i in range(designs_per_structure):
        expected = f"{prefix}_{i}.pdb"
        if expected in all_files:
            related.append(expected)
    return related


def main():
    args = parse_args()

    pdb_dir = os.path.abspath(args.pdb_dir)
    ref_pdb = os.path.abspath(args.ref_pdb)
    num_workers = args.num_workers or cpu_count()

    print(f"Using {num_workers} parallel workers")

    # Gather all PDB files
    all_pdb_files = sorted(f for f in os.listdir(pdb_dir) if f.endswith(".pdb"))
    if not all_pdb_files:
        print(f"No .pdb files found in {pdb_dir}")
        sys.exit(1)

    all_pdb_set = set(all_pdb_files)
    print(f"Found {len(all_pdb_files)} total PDB files in {pdb_dir}")

    # Get unique structure prefixes and their first design (design_idx=0)
    first_designs = sorted(f for f in all_pdb_files if f.endswith("_0.pdb"))
    print(f"Found {len(first_designs)} unique structures (design_idx=0)")

    # Step 1: Compute TM-scores in parallel (only for first designs)
    print(f"\n--- Step 1: Computing TM-scores for {len(first_designs)} structures ---")

    worker_args = [
        (os.path.join(pdb_dir, fname), ref_pdb, args.tmalign_bin)
        for fname in first_designs
    ]

    tm_scores = {}
    with Pool(num_workers) as pool:
        results = pool.map(compute_tmscore_worker, worker_args)
        for fname, tm, error in results:
            tm_scores[fname] = tm
            if error:
                print(f"  WARNING: {fname}: {error}")

    successful = sum(1 for v in tm_scores.values() if v is not None)
    print(f"  Computed TM-scores for {successful}/{len(first_designs)} structures")

    # Step 2: Identify structures needing DSSP (TM < 0.5)
    below_threshold = [fname for fname, tm in tm_scores.items() if tm is not None and tm < 0.5]
    print(f"\n--- Step 2: Computing beta strands for {len(below_threshold)} structures (TM < 0.5) ---")

    strand_counts = {}
    if below_threshold:
        worker_args = [
            (os.path.join(pdb_dir, fname), args.mkdssp_bin)
            for fname in below_threshold
        ]

        with Pool(num_workers) as pool:
            results = pool.map(count_beta_strands_worker, worker_args)
            for fname, n_strands, error in results:
                strand_counts[fname] = n_strands
                if error:
                    print(f"  WARNING: {fname}: {error}")

        successful = sum(1 for v in strand_counts.values() if v is not None)
        print(f"  Computed beta strands for {successful}/{len(below_threshold)} structures")

    # Step 3: Determine partition labels
    print("\n--- Step 3: Assigning partition labels ---")
    partition_labels = {}
    for fname in first_designs:
        tm = tm_scores.get(fname)
        if tm is None:
            partition_labels[fname] = "failed"
        elif tm >= 0.5:
            partition_labels[fname] = "tm_above_0.5"
        else:
            n_strands = strand_counts.get(fname)
            if n_strands == 11:
                partition_labels[fname] = "tm_below_0.5_11_stranded"
            else:
                partition_labels[fname] = "tm_below_0.5_not_11_stranded"

    # Count partitions
    partition_counts = {}
    for label in partition_labels.values():
        partition_counts[label] = partition_counts.get(label, 0) + 1

    print("  Partition counts:")
    for label, count in sorted(partition_counts.items()):
        print(f"    {label}: {count}")

    # Step 4: Create subdirectories and move/copy files
    print("\n--- Step 4: Organizing files into subdirectories ---")

    # Create all partition directories
    for label in set(partition_labels.values()):
        os.makedirs(os.path.join(pdb_dir, label), exist_ok=True)

    move_fn = shutil.copy2 if args.copy_instead_of_move else shutil.move
    action = "Copying" if args.copy_instead_of_move else "Moving"

    moved_count = 0
    for fname in first_designs:
        prefix = get_design_prefix(fname)
        if prefix is None:
            print(f"  WARNING: Could not parse prefix from {fname}")
            continue

        label = partition_labels[fname]
        related_files = get_related_designs(prefix, all_pdb_set, args.designs_per_structure)

        for related_fname in related_files:
            src = os.path.join(pdb_dir, related_fname)
            dst = os.path.join(pdb_dir, label, related_fname)
            if os.path.exists(src):
                move_fn(src, dst)
                moved_count += 1

    print(f"  {action} {moved_count} files")

    # Save CSV summary
    csv_path = os.path.join(pdb_dir, "partition_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["design_prefix", "first_design", "tm_score", "n_beta_strands", "partition"])
        for fname in sorted(first_designs):
            prefix = get_design_prefix(fname)
            tm = tm_scores.get(fname)
            n_strands = strand_counts.get(fname)
            label = partition_labels.get(fname, "unknown")
            writer.writerow([
                prefix,
                fname,
                f"{tm:.4f}" if tm is not None else "",
                n_strands if n_strands is not None else "",
                label,
            ])
    print(f"\nSummary saved to {csv_path}")

    # Print final summary
    print("\n=== Final Summary ===")
    for label in sorted(partition_counts.keys()):
        count = partition_counts[label]
        total_files = count * args.designs_per_structure
        print(f"  {label}: {count} structures ({total_files} PDB files)")


if __name__ == "__main__":
    main()
