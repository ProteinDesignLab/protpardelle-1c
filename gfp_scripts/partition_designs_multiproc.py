#!/usr/bin/env python3
"""Partition PDB structures by TM-score and beta barrel classification with multiprocessing.

This script:
1. Filters based on the first design (_0.pdb) per structure group
2. Moves all 8 designs (_0.pdb through _7.pdb) to appropriate subfolders
3. Does NOT discard anything - just partitions into subfolders
4. Uses multiprocessing for TM-score and DSSP computations
"""

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from Bio.PDB import PDBParser


def parse_args():
    parser = argparse.ArgumentParser(
        description="Partition PDB structures by TM-score and beta barrel classification."
    )
    parser.add_argument("--pdb_dir", required=True, help="Directory containing PDB structures")
    parser.add_argument("--ref_pdb", required=True, help="Path to reference PDB (e.g. 2AWJ)")
    parser.add_argument("--tmalign_bin", default="TMalign", help="Path to TMalign binary")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--tm_threshold", type=float, default=0.5, help="TM-score threshold")
    parser.add_argument("--strand_target", type=int, default=11, help="Target number of beta strands")
    return parser.parse_args()


def compute_tmscore(query_pdb, ref_pdb, tmalign_bin):
    """Run TMalign and return TM-score normalized by reference length."""
    try:
        result = subprocess.run(
            [tmalign_bin, query_pdb, ref_pdb],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return None

        tm_scores = re.findall(r"TM-score=\s*([\d.]+)", result.stdout)
        if len(tm_scores) < 2:
            return None
        return float(tm_scores[1])
    except Exception:
        return None


def calc_dihedral(p0, p1, p2, p3):
    """Calculate dihedral angle between four points."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.degrees(math.atan2(y, x))


def count_beta_strands_geometric(pdb_path):
    """Count distinct beta strands using phi/psi angle analysis.

    Beta sheet regions have phi ~ -120 to -140 and psi ~ 110 to 135 degrees.
    We use a relaxed definition: phi in [-180, -60] and psi in [90, 180] or [-180, -150]
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]

        # Extract backbone atoms
        residues = []
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  # Standard residue
                    try:
                        n = residue['N'].get_coord()
                        ca = residue['CA'].get_coord()
                        c = residue['C'].get_coord()
                        residues.append((n, ca, c))
                    except KeyError:
                        residues.append(None)

        if len(residues) < 4:
            return None

        # Compute phi/psi angles
        is_beta = []
        for i in range(len(residues)):
            if residues[i] is None:
                is_beta.append(False)
                continue

            # Phi: C(i-1) - N(i) - CA(i) - C(i)
            # Psi: N(i) - CA(i) - C(i) - N(i+1)
            phi = None
            psi = None

            if i > 0 and residues[i-1] is not None:
                c_prev = residues[i-1][2]
                n_curr, ca_curr, c_curr = residues[i]
                phi = calc_dihedral(c_prev, n_curr, ca_curr, c_curr)

            if i < len(residues) - 1 and residues[i+1] is not None:
                n_curr, ca_curr, c_curr = residues[i]
                n_next = residues[i+1][0]
                psi = calc_dihedral(n_curr, ca_curr, c_curr, n_next)

            # Check if in beta sheet region
            # Extended beta: phi ~ -120, psi ~ 120
            in_beta = False
            if phi is not None and psi is not None:
                # Relaxed beta region
                if -180 <= phi <= -60:
                    if (90 <= psi <= 180) or (-180 <= psi <= -150):
                        in_beta = True
            is_beta.append(in_beta)

        # Count contiguous beta strand regions (minimum 3 residues)
        n_strands = 0
        in_strand = False
        strand_len = 0

        for beta in is_beta:
            if beta:
                if not in_strand:
                    in_strand = True
                    strand_len = 1
                else:
                    strand_len += 1
            else:
                if in_strand and strand_len >= 3:
                    n_strands += 1
                in_strand = False
                strand_len = 0

        # Check last strand
        if in_strand and strand_len >= 3:
            n_strands += 1

        return n_strands
    except Exception as e:
        return None


def process_tmscore_worker(args):
    """Worker function for TM-score computation."""
    pdb_path, ref_pdb, tmalign_bin = args
    tm = compute_tmscore(pdb_path, ref_pdb, tmalign_bin)
    return pdb_path, tm


def process_dssp_worker(args):
    """Worker function for beta strand computation."""
    pdb_path = args[0]
    n = count_beta_strands_geometric(pdb_path)
    return pdb_path, n


def get_base_name(pdb_file):
    """Extract base name without the design index suffix (e.g., d0_0.pdb -> d0)."""
    match = re.match(r'^(.+)_[0-7]\.pdb$', pdb_file)
    if match:
        return match.group(1)
    return None


def main():
    args = parse_args()

    pdb_dir = os.path.abspath(args.pdb_dir)
    ref_pdb = os.path.abspath(args.ref_pdb)

    # Find all _0.pdb files (first design per structure group)
    all_files = os.listdir(pdb_dir)
    first_designs = sorted(f for f in all_files if f.endswith("_0.pdb"))

    if not first_designs:
        print(f"No *_0.pdb files found in {pdb_dir}")
        sys.exit(1)

    print(f"Found {len(first_designs)} structure groups in {pdb_dir}")
    print(f"Using {args.num_workers} workers")

    # Step 1: Compute TM-scores in parallel
    print("\n--- Step 1: Computing TM-scores (parallel) ---")
    tm_scores = {}
    tasks = [(os.path.join(pdb_dir, f), ref_pdb, args.tmalign_bin) for f in first_designs]

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_tmscore_worker, t): t[0] for t in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            pdb_path, tm = future.result()
            fname = os.path.basename(pdb_path)
            tm_scores[fname] = tm
            if completed % 100 == 0 or completed == len(first_designs):
                print(f"  TM-score progress: {completed}/{len(first_designs)}")

    # Partition by TM-score
    print(f"\n--- Step 2: Partitioning by TM-score (threshold={args.tm_threshold}) ---")
    above_thresh = []
    below_thresh = []
    failed = []

    for fname, tm in tm_scores.items():
        if tm is None:
            failed.append(fname)
        elif tm >= args.tm_threshold:
            above_thresh.append(fname)
        else:
            below_thresh.append(fname)

    print(f"  TM >= {args.tm_threshold}: {len(above_thresh)} structures")
    print(f"  TM <  {args.tm_threshold}: {len(below_thresh)} structures")
    if failed:
        print(f"  Failed TM computation: {len(failed)} structures")

    # Step 3: Compute beta strands for below-threshold structures
    print("\n--- Step 3: Computing beta strands (parallel) ---")
    strand_counts = {}
    tasks = [(os.path.join(pdb_dir, f),) for f in below_thresh]

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_dssp_worker, t): t[0] for t in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            pdb_path, n = future.result()
            fname = os.path.basename(pdb_path)
            strand_counts[fname] = n
            if completed % 100 == 0 or completed == len(below_thresh):
                print(f"  DSSP progress: {completed}/{len(below_thresh)}")

    # Partition by strand count
    target_strands = []
    other_strands = []
    for fname, n in strand_counts.items():
        if n == args.strand_target:
            target_strands.append(fname)
        else:
            other_strands.append(fname)

    print(f"\n  {args.strand_target}-stranded: {len(target_strands)} structures")
    print(f"  Not {args.strand_target}-stranded: {len(other_strands)} structures")

    # Create output directories
    print("\n--- Step 4: Creating partition directories and moving files ---")
    above_dir = os.path.join(pdb_dir, f"tm_above_{args.tm_threshold}")
    below_target_dir = os.path.join(pdb_dir, f"tm_below_{args.tm_threshold}", f"{args.strand_target}_stranded")
    below_other_dir = os.path.join(pdb_dir, f"tm_below_{args.tm_threshold}", f"not_{args.strand_target}_stranded")
    failed_dir = os.path.join(pdb_dir, "failed")

    os.makedirs(above_dir, exist_ok=True)
    os.makedirs(below_target_dir, exist_ok=True)
    os.makedirs(below_other_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    def move_all_designs(first_design_fname, dest_dir):
        """Move all 8 designs for a structure group to destination."""
        base = get_base_name(first_design_fname)
        if base is None:
            return 0
        moved = 0
        for i in range(8):
            fname = f"{base}_{i}.pdb"
            src = os.path.join(pdb_dir, fname)
            if os.path.exists(src):
                shutil.move(src, os.path.join(dest_dir, fname))
                moved += 1
        return moved

    # Move files
    moved_above = 0
    for fname in above_thresh:
        moved_above += move_all_designs(fname, above_dir)
    print(f"  Moved {moved_above} files to {above_dir}")

    moved_target = 0
    for fname in target_strands:
        moved_target += move_all_designs(fname, below_target_dir)
    print(f"  Moved {moved_target} files to {below_target_dir}")

    moved_other = 0
    for fname in other_strands:
        moved_other += move_all_designs(fname, below_other_dir)
    print(f"  Moved {moved_other} files to {below_other_dir}")

    moved_failed = 0
    for fname in failed:
        moved_failed += move_all_designs(fname, failed_dir)
    print(f"  Moved {moved_failed} files to {failed_dir}")

    # Save summary CSV
    csv_path = os.path.join(pdb_dir, "partition_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["first_design", "tm_score", "n_beta_strands", "partition"])

        for fname in sorted(tm_scores.keys()):
            tm = tm_scores[fname]
            n = strand_counts.get(fname)

            if tm is None:
                partition = "failed"
            elif tm >= args.tm_threshold:
                partition = f"tm_above_{args.tm_threshold}"
            elif n == args.strand_target:
                partition = f"tm_below_{args.tm_threshold}/{args.strand_target}_stranded"
            else:
                partition = f"tm_below_{args.tm_threshold}/not_{args.strand_target}_stranded"

            writer.writerow([
                fname,
                f"{tm:.4f}" if tm is not None else "",
                n if n is not None else "",
                partition,
            ])

    print(f"\nSummary saved to {csv_path}")
    print("\n--- Partition Complete ---")
    print(f"Total structure groups: {len(first_designs)}")
    print(f"  tm_above_{args.tm_threshold}: {len(above_thresh)} groups ({moved_above} files)")
    print(f"  tm_below_{args.tm_threshold}/{args.strand_target}_stranded: {len(target_strands)} groups ({moved_target} files)")
    print(f"  tm_below_{args.tm_threshold}/not_{args.strand_target}_stranded: {len(other_strands)} groups ({moved_other} files)")
    print(f"  failed: {len(failed)} groups ({moved_failed} files)")


if __name__ == "__main__":
    main()
