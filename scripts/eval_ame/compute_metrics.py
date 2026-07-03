"""Compute success metrics for each (sample, designed_seq) pair after RF3 prediction.

Metrics:
  scRMSD              — CA RMSD between protpardelle sample and RF3 prediction
                         (after Kabsch alignment). Threshold: <= 2 Å.
  motif_scRMSD        — RMSD of catalytic functional-group atoms (per ame.csv)
                         between the protpardelle sample and the RF3 prediction
                         at the matched scaffold positions (both in scaffold frame).
                         Threshold: <= 1.5 Å.
  ligand_clashes      — count of binder atoms within 1.5 Å of any ligand atom
                         in the RF3 prediction. Threshold: == 0.
  success             — all three thresholds satisfied.

Inputs:
  --mapping       rf3_mapping.csv (from prepare_rf3_inputs.py)
  --rf3-out-dir   directory containing RF3 predicted CIFs/PDBs
                   (assumed naming: <rf3_name>.pdb or .cif under any subdir)
  --out           output metrics CSV
  --workers       number of parallel processes (default: cpu_count())

The RF3 output directory is searched recursively at startup to build an index
{rf3_name -> path}. Per-row work then runs in parallel.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# ---------- PDB / CIF parsing ----------

def _read_atoms(pdb_or_cif: Path):
    atoms = []
    if pdb_or_cif.suffix.lower() == ".cif":
        return _read_cif_atoms(pdb_or_cif)
    with open(pdb_or_cif) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                atom = line[12:16].strip()
                alt = line[16]
                if alt not in (" ", "A"):
                    continue
                resname = line[17:20].strip()
                chain = line[21]
                rn = int(line[22:26])
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except Exception:
                continue
            atoms.append({
                "chain": chain, "resnum": rn, "resname": resname,
                "atom": atom, "xyz": xyz, "hetatm": line.startswith("HETATM"),
            })
    return atoms


def _read_cif_atoms(cif: Path):
    atoms = []
    fields = []
    in_loop = False
    with open(cif) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("loop_"):
                fields = []
                in_loop = True
                continue
            if in_loop and line.startswith("_atom_site."):
                fields.append(line[len("_atom_site."):])
                continue
            if in_loop and (line.startswith("#") or line.strip() == ""):
                in_loop = False
                fields = []
                continue
            if in_loop and fields and not line.startswith("_") and not line.startswith("loop_"):
                parts = line.split()
                if len(parts) < len(fields):
                    continue
                rec = dict(zip(fields, parts))
                try:
                    chain = rec.get("auth_asym_id", rec.get("label_asym_id", "A"))
                    rn = int(rec.get("auth_seq_id", rec.get("label_seq_id")))
                    atom = rec.get("auth_atom_id", rec.get("label_atom_id")).strip('"')
                    resname = rec.get("auth_comp_id", rec.get("label_comp_id"))
                    xyz = np.array([float(rec["Cartn_x"]), float(rec["Cartn_y"]), float(rec["Cartn_z"])])
                except Exception:
                    continue
                atoms.append({
                    "chain": chain, "resnum": rn, "resname": resname,
                    "atom": atom, "xyz": xyz,
                    "hetatm": rec.get("group_PDB", "ATOM") == "HETATM",
                })
    return atoms


# ---------- RMSD utilities ----------

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    P = P - P.mean(0); Q = Q - Q.mean(0)
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    S = np.eye(3); S[2, 2] = d
    R = Vt.T @ S @ U.T
    P_aligned = P @ R.T
    return float(np.sqrt(((P_aligned - Q) ** 2).sum(-1).mean()))


# ---------- Per-row metric computation ----------

def get_ca_array(atoms, chain: str = "A") -> np.ndarray:
    cas = [(a["resnum"], a["xyz"]) for a in atoms
           if a["atom"] == "CA" and a["chain"] == chain and not a["hetatm"]]
    cas.sort()
    return np.array([c for _, c in cas]) if cas else np.zeros((0, 3))


def get_atom(atoms, chain: str, resnum: int, atom_name: str):
    for a in atoms:
        if a["chain"] == chain and a["resnum"] == resnum and a["atom"] == atom_name and not a["hetatm"]:
            return a["xyz"]
    return None


def compute_motif_rmsd(sample_atoms, rf3_atoms, mapping, conditioning_atoms_by_resid):
    """Motif scRMSD: how well RF3 reproduces the protpardelle sample's motif atoms.

    Both sample and RF3 are in the scaffold frame (chain A, 1-indexed), so atoms
    are looked up at the same scaffold_resnum in both.
    """
    pairs = []
    n_expected = 0
    for orig_resnum, scaffold_resnum in mapping:
        found_atoms = None
        for chain_key in ("A", "B", "C"):
            key = f"{chain_key}{orig_resnum}"
            if key in conditioning_atoms_by_resid:
                found_atoms = conditioning_atoms_by_resid[key]
                break
        if not found_atoms:
            continue
        n_expected += len(found_atoms)
        for atom_name in found_atoms:
            sample_xyz = get_atom(sample_atoms, "A", scaffold_resnum, atom_name)
            rf3_xyz    = get_atom(rf3_atoms,    "A", scaffold_resnum, atom_name)
            if sample_xyz is None or rf3_xyz is None:
                continue
            pairs.append((sample_xyz, rf3_xyz))
    if len(pairs) < 3:
        return float("nan"), 0, n_expected
    src = np.array([p[0] for p in pairs])
    ref = np.array([p[1] for p in pairs])
    return kabsch_rmsd(src, ref), len(pairs), n_expected


def build_rf3_index(rf3_dir: Path) -> dict[str, Path]:
    """One-pass scan: map rf3_name (file stem prefix) -> first matching path.

    RF3 emits files named like `<rf3_name>.pdb` or `<rf3_name>_model_0.pdb` etc.
    We index by the longest stem first, so any file beginning with a registered
    rf3_name will resolve via prefix match.
    """
    index: dict[str, list[Path]] = {}
    for p in rf3_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".pdb", ".cif"):
            continue
        stem = p.stem
        index.setdefault(stem, []).append(p)
    # collapse to single best path per stem (prefer shortest filename)
    out = {}
    for stem, paths in index.items():
        out[stem] = sorted(paths, key=lambda x: (len(x.name), str(x)))[0]
    return out


def lookup_rf3_path(rf3_index: dict[str, Path], rf3_name: str) -> Path | None:
    """Find the RF3 output path for a given rf3_name. Tries exact stem match,
    then any stem starting with `rf3_name`."""
    if rf3_name in rf3_index:
        return rf3_index[rf3_name]
    matches = [p for stem, p in rf3_index.items() if stem.startswith(rf3_name)]
    return sorted(matches, key=lambda x: (len(x.name), str(x)))[0] if matches else None


# ---------- Worker function (runs in subprocess) ----------

def _process_row(args_tuple):
    row, rf3_path_str, scrmsd_threshold, motif_scrmsd_threshold, clash_threshold = args_tuple
    rf3_path = Path(rf3_path_str)

    sample_path = Path(row["sample_path"])
    try:
        sample_atoms = _read_atoms(sample_path)
        rf3_atoms = _read_atoms(rf3_path)
    except Exception as e:
        return {**row, "rf3_path": str(rf3_path), "error": str(e)}

    sample_ca = get_ca_array(sample_atoms, "A")
    rf3_ca = get_ca_array(rf3_atoms, "A")
    L = min(len(sample_ca), len(rf3_ca))
    scrmsd = kabsch_rmsd(sample_ca[:L], rf3_ca[:L]) if L >= 3 else float("nan")

    scaffold_resnums = [int(x) for x in row["scaffold_motif_resnums"].split(",") if x.strip()]
    contig_order = [tok for tok in row["contig_order"].split(",") if tok]
    orig_resnums = []
    for tok in contig_order:
        i = 0
        while i < len(tok) and tok[i].isalpha():
            i += 1
        if i < len(tok):
            orig_resnums.append(int(tok[i:]))
    mapping_pairs = list(zip(orig_resnums[:len(scaffold_resnums)], scaffold_resnums))
    cond_atoms = json.loads(row["conditioning_atoms_json"]) if row.get("conditioning_atoms_json") else {}
    motif_rmsd, n_matched, n_expected = compute_motif_rmsd(
        sample_atoms, rf3_atoms, mapping_pairs, cond_atoms)

    binder = np.array([a["xyz"] for a in rf3_atoms if not a["hetatm"]])
    ligand = np.array([a["xyz"] for a in rf3_atoms if a["hetatm"]])
    if len(binder) > 0 and len(ligand) > 0:
        D = np.linalg.norm(binder[:, None, :] - ligand[None, :, :], axis=-1)
        clashes = int((D < clash_threshold).any(axis=1).sum())
    else:
        clashes = 0

    success = (
        np.isfinite(scrmsd) and scrmsd <= scrmsd_threshold
        and np.isfinite(motif_rmsd) and motif_rmsd <= motif_scrmsd_threshold
        and clashes == 0
    )

    return {
        **row,
        "rf3_path": str(rf3_path),
        "scrmsd": f"{scrmsd:.3f}",
        "motif_scrmsd": f"{motif_rmsd:.3f}",
        "n_motif_atoms_matched": n_matched,
        "n_motif_atoms_expected": n_expected,
        "ligand_clashes": clashes,
        "n_binder_atoms": len(binder),
        "n_ligand_atoms": len(ligand),
        "success": int(bool(success)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--rf3-out-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scrmsd-threshold", type=float, default=2.0)
    ap.add_argument("--motif-scrmsd-threshold", type=float, default=1.5)
    ap.add_argument("--clash-threshold", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel workers (default: cpu_count())")
    args = ap.parse_args()

    rf3_dir = Path(args.rf3_out_dir)
    print(f"[index] scanning {rf3_dir} ...")
    rf3_index = build_rf3_index(rf3_dir)
    print(f"[index] {len(rf3_index)} rf3 output files")

    rows = []
    skipped_no_rf3 = 0
    with open(args.mapping) as f:
        for row in csv.DictReader(f):
            rf3_path = lookup_rf3_path(rf3_index, row["rf3_name"])
            if rf3_path is None:
                skipped_no_rf3 += 1
                continue
            rows.append((row, str(rf3_path),
                         args.scrmsd_threshold,
                         args.motif_scrmsd_threshold,
                         args.clash_threshold))
    print(f"[plan] {len(rows)} rows to process; {skipped_no_rf3} skipped (no rf3 output)")

    if not rows:
        print("No rows had RF3 outputs.")
        return

    workers = args.workers or os.cpu_count() or 4
    out_rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_row, t) for t in rows]
        for i, fut in enumerate(as_completed(futures)):
            out_rows.append(fut.result())
            if (i + 1) % 500 == 0:
                print(f"  processed {i + 1}/{len(rows)}")

    fields = list(out_rows[0].keys())
    for r in out_rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    n_succ = sum(1 for r in out_rows if r.get("success") == 1)
    print(f"wrote {args.out} ({len(out_rows)} rows; skipped {skipped_no_rf3} without RF3 output)")
    print(f"  successes: {n_succ} / {len(out_rows)} ({100 * n_succ / max(1, len(out_rows)):.1f}%)")


if __name__ == "__main__":
    main()
