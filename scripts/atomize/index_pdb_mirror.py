"""Index the local PDB mmCIF mirror: which chemical components each entry holds.

Scans ``_atom_site`` records only (no full mmCIF parse), so it runs at ~30 ms per
gzipped entry. Output is one TSV per shard:

    pdb_id  n_atoms  n_poly_res_approx  comps
    7zoe    5551     645                ALA:38,ARG:12,...,ACV:1,FE:1

The downstream selector (``build_train_index.py``) turns these into a training
list, and in particular into the *oversampled* list of entries that contain
modified amino acids or interesting ligands.
"""

from __future__ import annotations

import argparse
import gzip
import os
from collections import Counter
from pathlib import Path


def scan_one(path: Path) -> tuple[str, int, int, Counter] | None:
    comps: Counter = Counter()
    n_atoms = 0
    # label_comp_id is field 6 and label_seq_id field 9 of _atom_site in the
    # wwPDB-produced mmCIFs; both are unquoted tokens.
    try:
        with gzip.open(path, "rt", errors="replace") as fh:
            seen_res: set[tuple[str, str, str]] = set()
            for line in fh:
                if not (line.startswith("ATOM ") or line.startswith("HETATM ")):
                    continue
                f = line.split()
                if len(f) < 10:
                    continue
                n_atoms += 1
                comp = f[5]
                comps[comp] += 1
                seen_res.add((comp, f[6], f[8]))
    except (OSError, EOFError):
        return None
    if n_atoms == 0:
        return None
    return path.name.split(".")[0], n_atoms, len(seen_res), comps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", default="/scratch/users/tianyulu/datasets/pdb_mirror/mmCIF")
    ap.add_argument("--out-dir", default="/scratch/users/tianyulu/datasets/pdb_atomized/index")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()

    mirror = Path(args.mirror)
    subdirs = sorted(d for d in os.listdir(mirror) if (mirror / d).is_dir())
    mine = [d for i, d in enumerate(subdirs) if i % args.n_shards == args.shard]

    files: list[Path] = []
    for d in mine:
        files.extend(sorted((mirror / d).glob("*.cif.gz")))
    print(f"shard {args.shard}: {len(mine)} dirs, {len(files)} files", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"comps_{args.shard:03d}.tsv"

    from multiprocessing import Pool

    n_ok = 0
    with Pool(args.procs) as pool, open(out_path, "w") as out:
        for res in pool.imap_unordered(scan_one, files, chunksize=64):
            if res is None:
                continue
            pdb_id, n_atoms, n_res, comps = res
            comp_str = ",".join(f"{k}:{v}" for k, v in comps.most_common())
            out.write(f"{pdb_id}\t{n_atoms}\t{n_res}\t{comp_str}\n")
            n_ok += 1
            if n_ok % 5000 == 0:
                print(f"  {n_ok}/{len(files)}", flush=True)
    print(f"shard {args.shard}: wrote {n_ok} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
