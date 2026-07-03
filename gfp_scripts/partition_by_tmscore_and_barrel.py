#!/usr/bin/env python3
"""Partition PDB structures by TM-score against a reference and beta barrel classification."""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys

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
    return parser.parse_args()


def compute_tmscore(query_pdb, ref_pdb, tmalign_bin):
    """Run TMalign and return TM-score normalized by reference length (second TM-score line)."""
    result = subprocess.run(
        [tmalign_bin, query_pdb, ref_pdb],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"WARNING: TMalign failed for {query_pdb}: {result.stderr.strip()}")
        return None

    # Parse stdout for TM-score lines. TMalign outputs two:
    #   TM-score= 0.XXXX (if normalized by length of Chain_1)
    #   TM-score= 0.XXXX (if normalized by length of Chain_2)
    # We want the second one (normalized by reference = Chain_2).
    tm_scores = re.findall(r"TM-score=\s*([\d.]+)", result.stdout)
    if len(tm_scores) < 2:
        print(f"WARNING: Could not parse TM-scores from TMalign output for {query_pdb}")
        return None
    return float(tm_scores[1])


def count_beta_strands(pdb_path, mkdssp_bin):
    """Count distinct beta strands (consecutive runs of 'E') using DSSP."""
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
    return n_strands


def main():
    args = parse_args()

    pdb_dir = os.path.abspath(args.pdb_dir)
    ref_pdb = os.path.abspath(args.ref_pdb)

    # Gather PDB files
    pdb_files = sorted(f for f in os.listdir(pdb_dir) if f.endswith(".pdb"))
    if not pdb_files:
        print(f"No .pdb files found in {pdb_dir}")
        sys.exit(1)
    print(f"Found {len(pdb_files)} PDB files in {pdb_dir}")

    # Step 1: Compute TM-scores
    print("\n--- Step 1: Computing TM-scores ---")
    tm_scores = {}
    for i, fname in enumerate(pdb_files, 1):
        query_path = os.path.join(pdb_dir, fname)
        tm = compute_tmscore(query_path, ref_pdb, args.tmalign_bin)
        tm_scores[fname] = tm
        print(f"  [{i}/{len(pdb_files)}] {fname}: TM-score = {tm}")

    # Step 2: Partition by TM-score
    print("\n--- Step 2: Partitioning by TM-score (threshold=0.5) ---")
    above_dir = os.path.join(pdb_dir, "tm_above_0.5")
    below_dir = os.path.join(pdb_dir, "tm_below_0.5")
    os.makedirs(above_dir, exist_ok=True)
    os.makedirs(below_dir, exist_ok=True)

    above_files = []
    below_files = []
    failed_files = []
    for fname, tm in tm_scores.items():
        src = os.path.join(pdb_dir, fname)
        if tm is None:
            failed_files.append(fname)
            continue
        if tm >= 0.5:
            shutil.move(src, os.path.join(above_dir, fname))
            above_files.append(fname)
        else:
            shutil.move(src, os.path.join(below_dir, fname))
            below_files.append(fname)

    print(f"  TM >= 0.5: {len(above_files)} structures -> {above_dir}")
    print(f"  TM <  0.5: {len(below_files)} structures -> {below_dir}")
    if failed_files:
        print(f"  Failed (not moved): {len(failed_files)} structures")

    # Step 3: Beta barrel classification on tm_below_0.5
    print("\n--- Step 3: Counting beta strands (tm_below_0.5) ---")
    strand_counts = {}
    for i, fname in enumerate(below_files, 1):
        pdb_path = os.path.join(below_dir, fname)
        try:
            n = count_beta_strands(pdb_path, args.mkdssp_bin)
        except Exception as e:
            print(f"  WARNING: DSSP failed for {fname}: {e}")
            n = None
        strand_counts[fname] = n
        print(f"  [{i}/{len(below_files)}] {fname}: {n} beta strands")

    # Step 4: Further partition tm_below_0.5
    print("\n--- Step 4: Partitioning by 11-stranded classification ---")
    stranded_dir = os.path.join(below_dir, "11_stranded")
    not_stranded_dir = os.path.join(below_dir, "not_11_stranded")
    os.makedirs(stranded_dir, exist_ok=True)
    os.makedirs(not_stranded_dir, exist_ok=True)

    eleven_files = []
    other_files = []
    for fname, n in strand_counts.items():
        src = os.path.join(below_dir, fname)
        if n == 11:
            shutil.move(src, os.path.join(stranded_dir, fname))
            eleven_files.append(fname)
        else:
            shutil.move(src, os.path.join(not_stranded_dir, fname))
            other_files.append(fname)

    print(f"  11-stranded:     {len(eleven_files)} structures -> {stranded_dir}")
    print(f"  Not 11-stranded: {len(other_files)} structures -> {not_stranded_dir}")

    # Save CSV summary
    csv_path = os.path.join(pdb_dir, "partition_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "tm_score", "partition", "n_beta_strands", "is_11_stranded"])
        for fname in sorted(tm_scores.keys()):
            tm = tm_scores[fname]
            n = strand_counts.get(fname)
            if tm is None:
                partition = "failed"
                is_11 = ""
            elif tm >= 0.5:
                partition = "tm_above_0.5"
                is_11 = ""
            elif n == 11:
                partition = "tm_below_0.5/11_stranded"
                is_11 = "True"
            else:
                partition = "tm_below_0.5/not_11_stranded"
                is_11 = "False"
            writer.writerow([
                fname,
                f"{tm:.4f}" if tm is not None else "",
                partition,
                n if n is not None else "",
                is_11,
            ])
    print(f"\nSummary saved to {csv_path}")


if __name__ == "__main__":
    main()
