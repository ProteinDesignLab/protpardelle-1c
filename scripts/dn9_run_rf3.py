"""Run RF3 structure prediction on filter-passing dn9 candidates.

Reads a filtered CSV of sample PDB paths, writes FASTA sequences (single chain),
and submits/runs RF3 fold for each. Stores per-sample mean pLDDT alongside the
filter CSV.

This is intentionally simple — sequential RF3 calls. For larger sets, batch
multiple jobs via the rf3 CLI's batch input mode.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


# Map 1-letter back to 3-letter for fasta header (FASTA stays as 1-letter, but
# we keep this in case we want CIF outputs).


def write_fasta(seq: str, name: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(f">{name}\n{seq}\n")


def run_rf3_fold(input_fasta: Path, out_dir: Path, rf3_python: str, env_extras: dict) -> dict:
    """Invoke rf3 fold CLI. Returns dict with mean_plddt and any errors."""
    cmd = [
        rf3_python, "fold",
        f"inputs={input_fasta}",
        f"out_dir={out_dir}",
        "annotate_b_factor_with_plddt=true",
    ]
    env = os.environ.copy()
    env.update(env_extras)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if result.returncode != 0:
        return {"error": f"rc={result.returncode}\n{result.stderr[-1000:]}", "mean_plddt": None}

    # rf3 places output in out_dir/<example_id>/. example_id = input file stem.
    example_id = input_fasta.stem
    summary_path = out_dir / example_id / f"{example_id}_summary_confidences.json"
    if not summary_path.exists():
        # Try other layouts
        candidates = list((out_dir / example_id).glob("*_summary_confidences.json"))
        if candidates:
            summary_path = candidates[0]
    if not summary_path.exists():
        return {"error": f"no summary at {summary_path}", "mean_plddt": None}

    with summary_path.open() as f:
        data = json.load(f)
    return {
        "mean_plddt": data.get("overall_plddt"),
        "ptm": data.get("ptm"),
        "summary_path": str(summary_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="RF3 output dir (one subdir per sample)")
    ap.add_argument("--rf3-python", default="/scratch/users/tianyulu/envs/rf3/bin/rf3")
    ap.add_argument("--limit", type=int, default=None, help="Limit to N top candidates")
    ap.add_argument("--require-passed", action="store_true", default=True)
    args = ap.parse_args()

    df = pd.read_csv(args.filter_csv)
    if args.require_passed:
        df = df[df["passed"] == True]
    if args.limit is not None:
        df = df.head(args.limit)
    print(f"Running RF3 on {len(df)} candidates")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir = args.out_dir / "_fasta"
    fasta_dir.mkdir(exist_ok=True)

    rows = []
    for i, row in df.iterrows():
        name = Path(row["full_path"]).stem
        fasta = fasta_dir / f"{name}.fasta"
        seq = row["sequence"].replace("X", "G")  # Replace UNK with GLY for prediction
        write_fasta(seq, name, fasta)

        rf3_out = args.out_dir / "predictions"
        rf3_out.mkdir(exist_ok=True)
        if (rf3_out / name / f"{name}_summary_confidences.json").exists():
            with (rf3_out / name / f"{name}_summary_confidences.json").open() as f:
                data = json.load(f)
            r = {"mean_plddt": data.get("overall_plddt"), "ptm": data.get("ptm")}
        else:
            r = run_rf3_fold(fasta, rf3_out, args.rf3_python, {})
        r["pdb"] = row["pdb"]
        r["full_path"] = row["full_path"]
        rows.append(r)
        if (i + 1) % 10 == 0:
            pass

    out_df = pd.DataFrame(rows)
    out_csv = args.out_dir / "rf3_predictions.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    valid = out_df.dropna(subset=["mean_plddt"])
    if len(valid) > 0:
        print(f"Mean pLDDT distribution: min={valid['mean_plddt'].min():.1f}, "
              f"median={valid['mean_plddt'].median():.1f}, "
              f"max={valid['mean_plddt'].max():.1f}")
        print(f">50: {(valid['mean_plddt'] > 50).sum()}/{len(valid)}")


if __name__ == "__main__":
    main()
