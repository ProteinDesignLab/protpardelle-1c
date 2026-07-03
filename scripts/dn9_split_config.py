"""Split a dn9 sampling config into N chunks for parallel execution."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--num-chunks", type=int, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.input.read_text())
    n = len(cfg["motifs"])
    chunk_size = math.ceil(n / args.num_chunks)

    list_fields = ["motifs", "motif_contigs", "total_lengths", "hotspots", "ssadj",
                   "motif_sse", "binder_sse_prop"]
    scalar_fields = ["motif_sse_flank"]

    for chunk_id in range(args.num_chunks):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, n)
        if start >= n:
            break
        sub = {"search_space": cfg["search_space"]}
        for fld in list_fields:
            sub[fld] = cfg[fld][start:end]
        for fld in scalar_fields:
            if fld in cfg:
                sub[fld] = cfg[fld]
        out = args.out_prefix.parent / f"{args.out_prefix.name}_chunk{chunk_id:02d}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            yaml.safe_dump(sub, f, sort_keys=False)
        print(f"Wrote {out} with {end-start} motif rows")


if __name__ == "__main__":
    main()
