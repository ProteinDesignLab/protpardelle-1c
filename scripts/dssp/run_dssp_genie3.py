"""Sharded P-SEA secondary-structure assignment for genie3 AFDB + Pinder.

Mirrors run_dssp_teddymer.py but keys each row by the bare ID (no underscore
split), because:

- AFDB monomer names (e.g. AF-A0A4P6L4I8-F1-model_v4) contain no underscore,
  so they're already the lookup key.
- Pinder dimer ids (e.g. 8phr__X4_UNDEFINED--8phr__W4_UNDEFINED) contain
  internal underscores, so splitting on '_' would corrupt them. We write
  the full id as pdb_code; the dataset loader's multi-strategy lookup
  matches the full pdb_key first.

Inputs are taken as the list of bare IDs from the genie3 info.csv (built
upstream into a names.list with one ID per line) plus a directory of
structures. The script handles both .pdb and .pdb.gz.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calc_dssp import annotate_sse  # noqa: E402

SSE_LETTERS = np.array(["H", "E", "L", "X"])


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def parse_ca_coords(pdb_path: Path) -> np.ndarray:
    coords = []
    with _open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
    return np.asarray(coords, dtype=np.float64)


def sse_string(ca: np.ndarray) -> str:
    onehot = annotate_sse(ca, percentage_mask=0).numpy()
    idx = onehot.argmax(axis=-1)
    return "".join(SSE_LETTERS[idx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-list", required=True, help="One ID per line (no extension).")
    ap.add_argument("--pdb-dir", required=True, help="Directory containing the structure files.")
    ap.add_argument(
        "--filename-template",
        default="{id}.pdb",
        help="How to construct the filename from each ID. e.g. '{id}.pdb.gz' for AFDB, '{id}.pdb' for Pinder.",
    )
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    with Path(args.ids_list).open() as f:
        all_ids = [line.strip() for line in f if line.strip()]

    if args.num_shards > 1:
        ids = all_ids[args.shard_id :: args.num_shards]
    else:
        ids = all_ids
    if args.limit:
        ids = ids[: args.limit]

    out_path = Path(f"{args.output_prefix}_shard_{args.shard_id:04d}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"shard {args.shard_id}/{args.num_shards}: {len(ids)} ids "
        f"(total {len(all_ids)}) -> {out_path}",
        flush=True,
    )

    t0 = time.time()
    n_ok = n_fail = 0
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb_code", "source_pdb", "length", "sse"])
        for i, pid in enumerate(ids, start=1):
            filename = args.filename_template.format(id=pid)
            pdb = pdb_dir / filename
            try:
                if not pdb.exists():
                    raise FileNotFoundError(filename)
                ca = parse_ca_coords(pdb)
                if len(ca) < 5:
                    raise ValueError(f"only {len(ca)} CA atoms")
                sse = sse_string(ca)
                w.writerow([pid, filename, len(ca), sse])
                n_ok += 1
            except Exception as e:
                w.writerow([pid, filename, 0, f"ERROR: {e}"])
                n_fail += 1
            if i % 2000 == 0 or i == len(ids):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(ids) - i) / rate if rate > 0 else float("inf")
                print(
                    f"  shard {args.shard_id}: {i}/{len(ids)} ok={n_ok} fail={n_fail} "
                    f"elapsed={elapsed:.1f}s rate={rate:.1f}/s eta={eta:.1f}s",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(
        f"shard {args.shard_id} done in {elapsed:.1f}s. ok={n_ok} fail={n_fail}. -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
