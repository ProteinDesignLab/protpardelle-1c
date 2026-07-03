"""Run LigandMPNN over a chunk of the manifest.

For each row, invokes LigandMPNN's run.py via subprocess (preserves full LMPNN
arg surface). Outputs are written under <out_dir>/<model>/<motif>/<sample_idx>/
with FASTA + sequence stats files.

Resulting structure:
  <out_dir>/<model>/<motif>/<sample_idx>/
    seqs/<sample_basename>.fa     # 8 designed sequences (LigandMPNN default)
    backbones/                    # if requested
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-seqs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--array-id", type=int, default=0)
    ap.add_argument("--num-arrays", type=int, default=1)
    ap.add_argument("--lmpnn-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.1,
                    help="LigandMPNN sampling temperature")
    ap.add_argument("--model-type", default="ligand_mpnn",
                    help="LigandMPNN model_type (e.g. ligand_mpnn, soluble_mpnn)")
    ap.add_argument("--ligand-mpnn-ckpt",
                    default="/scratch/users/tianyulu/ligandmpnn_model_params/ligandmpnn_v_32_010_25.pt",
                    help="LigandMPNN checkpoint path")
    args = ap.parse_args()

    rows = []
    with open(args.manifest) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rows.append(row)

    chunk_size = math.ceil(len(rows) / args.num_arrays)
    start = args.array_id * chunk_size
    end = min(start + chunk_size, len(rows))
    rows = rows[start:end]
    print(f"[lmpnn] array_id={args.array_id}/{args.num_arrays}: rows {start}..{end} (n={len(rows)})")

    out_root = Path(args.out_dir)

    for row in rows:
        model = row["model_name"]
        motif = row["motif_name"]
        sidx = int(row["sample_idx"])
        sample_path = row["sample_path"]
        fixed = row["scaffold_motif_resnums"]  # comma-separated 1-indexed
        # LigandMPNN expects "A13 A26 ..." — chain A by construction in our pipeline
        fixed_residues = " ".join(f"A{r}" for r in fixed.split(",") if r.strip())

        out_dir = out_root / model / motif / f"{sidx:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        sample_basename = Path(sample_path).stem
        if (out_dir / "seqs" / f"{sample_basename}.fa").exists():
            print(f"  skip (exists): {model}/{motif}/{sidx:02d}")
            continue

        cmd = [
            "python", "run.py",
            "--seed", str(args.seed),
            "--pdb_path", sample_path,
            "--out_folder", str(out_dir),
            "--fixed_residues", fixed_residues,
            "--number_of_batches", str(max(1, args.num_seqs // args.batch_size)),
            "--batch_size", str(min(args.num_seqs, args.batch_size)),
            "--temperature", str(args.temperature),
            "--save_stats", "1",
            "--model_type", args.model_type,
            "--checkpoint_ligand_mpnn", args.ligand_mpnn_ckpt,
        ]
        print(f"  running {model}/{motif}/{sidx:02d}: {' '.join(cmd[:6])} ...")
        try:
            subprocess.run(cmd, cwd=args.lmpnn_dir, check=True,
                           env={**os.environ, "PYTHONUNBUFFERED": "1"})
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: {model}/{motif}/{sidx:02d} failed (rc={e.returncode})")
            continue


if __name__ == "__main__":
    main()
