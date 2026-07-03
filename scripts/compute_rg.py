"""Compute per-chain radius of gyration for the pulchra_teddymer dataset and
append the values to filtered_metadata.tsv as new columns Rg_A and Rg_B."""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

PDB_DIR_DEFAULT = "/scratch/users/tianyulu/datasets/teddymer/pulchra_teddymer"
META_DEFAULT = "/scratch/users/tianyulu/datasets/teddymer/filtered_metadata.tsv"
OUT_DEFAULT = "/scratch/users/tianyulu/datasets/teddymer/filtered_metadata_rg.tsv"


def parse_pdb_chain_coords(path):
    """Return dict chain -> (Nx3 float32 array of ATOM coordinates, residue count)."""
    coords = {"A": [], "B": []}
    residues = {"A": set(), "B": set()}
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            if chain not in coords:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords[chain].append((x, y, z))
            residues[chain].add((line[22:27]))  # resSeq + iCode
    return {
        c: (np.asarray(coords[c], dtype=np.float32), len(residues[c]))
        for c in coords
    }


def radius_of_gyration(xyz):
    if xyz.size == 0:
        return np.nan
    com = xyz.mean(axis=0)
    return float(np.sqrt(((xyz - com) ** 2).sum(axis=1).mean()))


def process_one(args):
    dimer_index, path = args
    try:
        chains = parse_pdb_chain_coords(path)
        xyz_a, n_a = chains["A"]
        xyz_b, n_b = chains["B"]
        rg_a = radius_of_gyration(xyz_a)
        rg_b = radius_of_gyration(xyz_b)
        norm_a = rg_a / (n_a ** (1.0 / 3.0)) if n_a > 0 else np.nan
        norm_b = rg_b / (n_b ** (1.0 / 3.0)) if n_b > 0 else np.nan
        return dimer_index, rg_a, rg_b, n_a, n_b, norm_a, norm_b
    except Exception:  # noqa: BLE001
        return dimer_index, np.nan, np.nan, 0, 0, np.nan, np.nan


def build_index(pdb_dir):
    """Map DimerIndex (int) -> pdb path."""
    paths = glob.glob(os.path.join(pdb_dir, "*DI_*.pdb"))
    idx = {}
    for p in paths:
        name = os.path.basename(p)
        try:
            di = int(name.split("DI_", 1)[0])
        except ValueError:
            continue
        idx[di] = p
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-dir", default=PDB_DIR_DEFAULT)
    ap.add_argument("--meta", default=META_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N (for testing).")
    args = ap.parse_args()

    print(f"[info] reading metadata: {args.meta}", flush=True)
    meta = pd.read_csv(args.meta, sep="\t")
    print(f"[info] rows: {len(meta):,}", flush=True)

    print(f"[info] indexing PDBs in {args.pdb_dir}", flush=True)
    idx = build_index(args.pdb_dir)
    print(f"[info] found {len(idx):,} pdb files", flush=True)

    tasks = []
    missing = 0
    for di in meta["DimerIndex"].tolist():
        p = idx.get(int(di))
        if p is None:
            missing += 1
            continue
        tasks.append((int(di), p))
    print(f"[info] tasks: {len(tasks):,}  missing: {missing:,}", flush=True)

    if args.limit > 0:
        tasks = tasks[: args.limit]
        print(f"[info] limiting to first {len(tasks):,} tasks for testing", flush=True)

    results = {}
    nan_tuple = (np.nan, np.nan, 0, 0, np.nan, np.nan)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="rg", unit="pdb"):
            di, rga, rgb, na, nb, norma, normb = fut.result()
            results[di] = (rga, rgb, na, nb, norma, normb)

    def col(i):
        return meta["DimerIndex"].map(lambda d: results.get(int(d), nan_tuple)[i])

    meta["Rg_A"] = col(0)
    meta["Rg_B"] = col(1)
    meta["N_A"] = col(2).astype("Int64")
    meta["N_B"] = col(3).astype("Int64")
    meta["RgNorm_A"] = col(4)
    meta["RgNorm_B"] = col(5)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta.to_csv(args.out, sep="\t", index=False, float_format="%.4f")
    print(f"[info] wrote {args.out}", flush=True)
    print(meta[["DimerIndex", "Rg_A", "Rg_B", "N_A", "N_B", "RgNorm_A", "RgNorm_B"]].head(), flush=True)
    print(f"[info] Rg_A nan: {meta['Rg_A'].isna().sum():,}  Rg_B nan: {meta['Rg_B'].isna().sum():,}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
