"""Run P-SEA secondary-structure assignment on the CATH training set.

Each line in the keys list looks like `1jvbA02_K.pdb` where K indexes a different
ProteinMPNN redesign of the same backbone. We only need one structure per unique
PDB code (everything before the first underscore), so we dedupe and run on the
representative `_0.pdb` file when available.

The actual SSE call comes from RosettaCommons/protein_generator's calc_dssp.py
(P-SEA algorithm via biotite), which only needs CA coordinates.
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

SSE_LETTERS = np.array(["H", "E", "L", "X"])  # a=H helix, b=E strand, c=L loop, mask=X


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


def pick_representative(keys_path: Path) -> dict[str, str]:
    """Map unique PDB code -> chosen filename. Prefer the `_0.pdb` redesign."""
    chosen: dict[str, str] = {}
    with keys_path.open() as f:
        for raw in f:
            name = raw.strip()
            if not name:
                continue
            code = name.split("_", 1)[0]
            if code in chosen:
                if name.endswith("_0.pdb"):
                    chosen[code] = name
            else:
                chosen[code] = name
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keys-list",
        default="/scratch/users/tianyulu/augmented_ingraham_cath_bugfree/train_designable_pdb_keys.list",
    )
    ap.add_argument(
        "--pdb-dir",
        action="append",
        default=None,
        help=(
            "Directory containing the PDB files. Can be passed multiple times; "
            "the first directory containing a given filename wins."
        ),
    )
    ap.add_argument(
        "--output",
        default="/scratch/users/tianyulu/augmented_ingraham_cath_bugfree/train_dssp.csv",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N unique PDB codes (0 = all).",
    )
    args = ap.parse_args()

    keys_path = Path(args.keys_list)
    pdb_dirs = [Path(p) for p in (args.pdb_dir or [
        "/scratch/users/tianyulu/augmented_ingraham_cath_bugfree/mpnn_esmfold",
        "/scratch/users/tianyulu/augmented_ingraham_cath_bugfree/dne_mpnn",
    ])]
    out_path = Path(args.output)

    representatives = pick_representative(keys_path)
    codes = sorted(representatives)
    if args.limit:
        codes = codes[: args.limit]
    print(f"Unique PDB codes: {len(representatives)} (processing {len(codes)})")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_ok = n_fail = 0
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb_code", "source_pdb", "length", "sse"])
        for i, code in enumerate(codes, start=1):
            name = representatives[code]
            pdb = next((d / name for d in pdb_dirs if (d / name).exists()), None)
            try:
                if pdb is None:
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
            if i % 500 == 0 or i == len(codes):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(codes) - i) / rate if rate > 0 else float("inf")
                print(
                    f"  {i}/{len(codes)} ok={n_ok} fail={n_fail} "
                    f"elapsed={elapsed:.1f}s rate={rate:.1f}/s eta={eta:.1f}s",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. ok={n_ok} fail={n_fail}. Output: {out_path}")


if __name__ == "__main__":
    main()
