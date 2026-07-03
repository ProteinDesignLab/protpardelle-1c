"""Prepare FASTA inputs for RF3 from dn9 filter-passers."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.filter_csv)
    df = df[df["passed"] == True].copy()
    if args.limit is not None:
        df = df.head(args.limit)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        seq = row["sequence"].replace("X", "G")
        name = Path(row["full_path"]).stem
        fasta = args.out_dir / f"{name}.fasta"
        with fasta.open("w") as f:
            f.write(f">{name}\n{seq}\n")
    print(f"Wrote {len(df)} FASTAs to {args.out_dir}")


if __name__ == "__main__":
    main()
