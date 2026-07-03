"""Sharded P-SEA secondary-structure assignment for the Teddymer datasets.

Designed to be invoked from an sbatch array. Each shard reads
`shard_id`-th slice of the keys list (mod `num_shards`), writes its rows to
`{output_prefix}_shard_{shard_id:04d}.csv`, and a final pass (see
run_dssp_teddymer.sbatch) concatenates all shard outputs into a single CSV
that matches the schema the dataset loader expects:
    pdb_code, source_pdb, length, sse

Teddymer filenames look like `100014568DI_AF-A0A384A0U5-F1-model_v4.pdb`.
Unlike the CATH set there are no ProteinMPNN redesign suffixes, but the
dataset still looks up rows by `Path(pdb_key).stem.split("_", 1)[0]` — the
dimer-ID prefix before the first underscore (e.g. `100014568DI`). We therefore
write that prefix as `pdb_code` so the lookup hits.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calc_dssp import annotate_sse  # noqa: E402

SSE_LETTERS = np.array(["H", "E", "L", "X"])


def parse_ca_coords(pdb_path: Path) -> np.ndarray:
    coords = []
    with pdb_path.open() as f:
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
    ap.add_argument("--keys-list", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument(
        "--output-prefix",
        required=True,
        help="Shard CSV path is {output_prefix}_shard_{shard_id:04d}.csv",
    )
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N keys *in this shard* (0 = all).",
    )
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    with Path(args.keys_list).open() as f:
        all_keys = [line.strip() for line in f if line.strip()]

    if args.num_shards > 1:
        keys = all_keys[args.shard_id :: args.num_shards]
    else:
        keys = all_keys
    if args.limit:
        keys = keys[: args.limit]

    out_path = Path(f"{args.output_prefix}_shard_{args.shard_id:04d}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"shard {args.shard_id}/{args.num_shards}: {len(keys)} keys "
        f"(total {len(all_keys)}) -> {out_path}",
        flush=True,
    )

    t0 = time.time()
    n_ok = n_fail = 0
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb_code", "source_pdb", "length", "sse"])
        for i, name in enumerate(keys, start=1):
            # Match dataset.py: lookup key is the prefix before the first '_'.
            code = Path(name).stem.split("_", 1)[0]
            pdb = pdb_dir / name
            try:
                if not pdb.exists():
                    raise FileNotFoundError(name)
                ca = parse_ca_coords(pdb)
                if len(ca) < 5:
                    raise ValueError(f"only {len(ca)} CA atoms")
                sse = sse_string(ca)
                w.writerow([code, name, len(ca), sse])
                n_ok += 1
            except Exception as e:
                w.writerow([code, name, 0, f"ERROR: {e}"])
                n_fail += 1
            if i % 2000 == 0 or i == len(keys):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(keys) - i) / rate if rate > 0 else float("inf")
                print(
                    f"  shard {args.shard_id}: {i}/{len(keys)} ok={n_ok} fail={n_fail} "
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
