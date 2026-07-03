"""Compute motif scRMSD for chroma_08 caliby RF3 predictions vs designed backbones.

Usage:
    python 20b_chroma08_caliby_rf3_motif_scrmsd.py [--workers N]
"""

import argparse
import json
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, Superimposer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RF3_BASE = Path("/scratch/users/tianyulu/gfp/denovo/chroma_08/caliby")
FIXPOS_CSV = Path("/scratch/users/tianyulu/gfp/denovo/chroma_08/top1000_rmsd_fixpos.csv")
DESIGN_SAMPLES_DIR = Path(
    "/scratch/users/tianyulu/allatom-design-sampling-results/"
    "chroma_08_top1000_rmsd/chroma_08_top1000_rmsd/samples"
)
SEQDES_CSV = Path(
    "/scratch/users/tianyulu/allatom-design-sampling-results/"
    "chroma_08_top1000_rmsd/chroma_08_top1000_rmsd/seq_des_outputs.csv"
)
OUT_DIR = RF3_BASE

BB_ATOMS = ("N", "CA", "C", "O")
CRO_BB_MAP = {"N1": "N", "CA1": "CA", "C3": "C", "O3": "O"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_fixpos_seq(fixpos_str):
    return [int(x.strip().lstrip("A")) for x in fixpos_str.split(",")]


def parse_rf3_dirname(dirname):
    """Parse chroma_08 RF3 dir name like 'p0_d1043_0_seed-0_sample-4_sample0_cro'.

    Returns (pdb_key, caliby_sample, fold_type) or None.
    """
    # Strip fold_type suffix
    if dirname.endswith("_no_cro"):
        fold_type = "no_cro"
        rest = dirname[: -len("_no_cro")]
    elif dirname.endswith("_cro"):
        fold_type = "cro"
        rest = dirname[: -len("_cro")]
    else:
        return None

    # Last _sampleN is caliby sample
    m = re.match(r"^(.+)_(sample\d+)$", rest)
    if not m:
        return None
    pdb_key = m.group(1)
    caliby_sample = m.group(2)
    return pdb_key, caliby_sample, fold_type


def get_bb_atoms_from_residue(residue):
    resname = residue.get_resname().strip()
    atoms = {a.name: a for a in residue.get_atoms()}
    if resname == "CRO":
        return {CRO_BB_MAP[k]: atoms[k] for k in CRO_BB_MAP if k in atoms}
    return {name: atoms[name] for name in BB_ATOMS if name in atoms}


def build_residue_map(structure, chain_id="A"):
    model = structure[0]
    chain = model[chain_id]
    res_map = {}
    for r in chain.get_residues():
        resname = r.get_resname().strip()
        if r.id[0] == " " or resname == "CRO":
            res_map[r.id[1]] = r
    return res_map


def compute_scrmsd(pred_res_map, design_res_map, pred_resids, design_resids,
                   motif_design_resids=None):
    all_fixed, all_moving = [], []
    motif_fixed_coords, motif_moving_coords = [], []
    motif_set = set(motif_design_resids) if motif_design_resids else set()

    for pred_rid, des_rid in zip(pred_resids, design_resids):
        if pred_rid not in pred_res_map or des_rid not in design_res_map:
            continue
        pred_bb = get_bb_atoms_from_residue(pred_res_map[pred_rid])
        des_bb = get_bb_atoms_from_residue(design_res_map[des_rid])
        for aname in BB_ATOMS:
            if aname in pred_bb and aname in des_bb:
                all_fixed.append(des_bb[aname])
                all_moving.append(pred_bb[aname])
                if des_rid in motif_set:
                    motif_fixed_coords.append(des_bb[aname].get_vector().get_array())
                    motif_moving_coords.append(pred_bb[aname].get_vector().get_array())

    if len(all_fixed) < 4:
        return {"global_bb_scrmsd": None, "motif_bb_scrmsd": None}

    sup = Superimposer()
    sup.set_atoms(all_fixed, all_moving)
    result = {"global_bb_scrmsd": round(sup.rms, 4)}

    if motif_fixed_coords:
        rot, tran = sup.rotran
        transformed = np.dot(np.array(motif_moving_coords), rot) + tran
        diff = np.array(motif_fixed_coords) - transformed
        result["motif_bb_scrmsd"] = round(
            float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))), 4
        )
    else:
        result["motif_bb_scrmsd"] = None
    return result


def build_correspondence_nocro(fix_pos, n_design_res):
    r = list(range(1, n_design_res + 1))
    return r, r


def build_correspondence_cro(fix_pos, n_design_res):
    tyg_start = fix_pos[7]
    pred_resids, design_resids = [], []
    for i in range(1, tyg_start):
        pred_resids.append(i)
        design_resids.append(i)
    pred_resids.append(tyg_start)
    design_resids.append(tyg_start)
    for i in range(tyg_start + 3, n_design_res + 1):
        pred_resids.append(i - 2)
        design_resids.append(i)
    return pred_resids, design_resids


def find_rf3_cifs(design_dir):
    cifs = []
    for sample_dir in sorted(design_dir.glob("seed-*_sample-*")):
        for f in sample_dir.glob("*_model.cif"):
            cifs.append((sample_dir.name, f))
    return cifs


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def process_one(args):
    rf3_dir_str, fold_type, pdb_key, caliby_sample, fix_pos, design_cif_path = args
    rf3_dir = Path(rf3_dir_str)
    parser = MMCIFParser(QUIET=True)

    try:
        design_struct = parser.get_structure("design", str(design_cif_path))
        design_res_map = build_residue_map(design_struct)
    except Exception as e:
        return [{"pdb_key": pdb_key, "caliby_sample": caliby_sample,
                 "fold_type": fold_type, "error": f"design_load: {e}"}]

    n_design_res = max(design_res_map.keys()) if design_res_map else 0

    if fold_type == "no_cro":
        pred_resids, design_resids = build_correspondence_nocro(fix_pos, n_design_res)
    else:
        pred_resids, design_resids = build_correspondence_cro(fix_pos, n_design_res)

    rf3_cifs = find_rf3_cifs(rf3_dir)
    if not rf3_cifs:
        return [{"pdb_key": pdb_key, "caliby_sample": caliby_sample,
                 "fold_type": fold_type, "error": "no_rf3_cifs"}]

    rows = []
    for rf3_sample_name, cif_path in rf3_cifs:
        row = {
            "pdb_key": pdb_key,
            "caliby_sample": caliby_sample,
            "fold_type": fold_type,
            "rf3_sample": rf3_sample_name,
            "rf3_cif": str(cif_path),
        }

        # pLDDT
        json_path = cif_path.parent / cif_path.name.replace(
            "_model.cif", "_summary_confidences.json"
        )
        if json_path.exists():
            with open(json_path) as f:
                row["plddt"] = json.load(f).get("overall_plddt")

        try:
            pred_struct = parser.get_structure("pred", str(cif_path))
            pred_res_map = build_residue_map(pred_struct)
        except Exception as e:
            row["error"] = f"pred_load: {e}"
            rows.append(row)
            continue

        try:
            row.update(compute_scrmsd(
                pred_res_map, design_res_map,
                pred_resids, design_resids,
                motif_design_resids=fix_pos,
            ))
        except Exception as e:
            row["error"] = f"rmsd: {e}"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(32, cpu_count()))
    args = ap.parse_args()

    # Load fixpos
    fp_df = pd.read_csv(FIXPOS_CSV)
    fixpos_map = {}
    for _, row in fp_df.iterrows():
        fixpos_map[row["pdb_key"]] = parse_fixpos_seq(row["fixed_pos_seq"])
    print(f"Loaded {len(fixpos_map)} fix_pos entries")

    # Load sequences
    seq_df = pd.read_csv(SEQDES_CSV)
    seq_map = {}
    for _, row in seq_df.iterrows():
        eid = row["example_id"]
        sample_name = Path(row["out_pdb"]).stem.replace(f"{eid}_", "")
        seq_map[(eid, sample_name)] = row["seq"]
    print(f"Loaded {len(seq_map)} sequences")

    # Discover RF3 directories
    work_items = []
    for fold_type in ("cro", "no_cro"):
        rf3_type_dir = RF3_BASE / fold_type
        if not rf3_type_dir.exists():
            continue
        for chunk_dir in sorted(rf3_type_dir.iterdir()):
            if not chunk_dir.is_dir() or not chunk_dir.name.isdigit():
                continue
            for design_dir in chunk_dir.iterdir():
                if not design_dir.is_dir():
                    continue
                parsed = parse_rf3_dirname(design_dir.name)
                if parsed is None:
                    continue
                pdb_key, caliby_sample, ft = parsed
                if ft != fold_type:
                    continue
                if pdb_key not in fixpos_map:
                    continue
                fix_pos = fixpos_map[pdb_key]
                design_cif = DESIGN_SAMPLES_DIR / f"{pdb_key}_{caliby_sample}.cif"
                if not design_cif.exists():
                    continue
                work_items.append((
                    str(design_dir), fold_type, pdb_key, caliby_sample,
                    fix_pos, str(design_cif),
                ))

    print(f"\nTotal RF3 directories to process: {len(work_items)}")
    print(f"Using {args.workers} workers")

    all_rows = []
    with Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_one, work_items, chunksize=16),
            total=len(work_items),
            desc="Computing scRMSD",
        ))
    for result in results:
        all_rows.extend(result)

    if not all_rows:
        print("ERROR: No predictions processed!")
        return

    df = pd.DataFrame(all_rows)
    print(f"\nTotal predictions: {len(df)}")
    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        if n_err > 0:
            print(f"  Errors: {n_err}")

    # Save full results
    out_path = OUT_DIR / "chroma08_caliby_rf3_motif_scrmsd.csv"
    df.to_csv(out_path, index=False)
    print(f"Full results saved to {out_path}")

    # Summary
    valid = df.dropna(subset=["global_bb_scrmsd", "motif_bb_scrmsd"])
    print(f"Valid predictions: {len(valid)}")
    for fold_type in ("cro", "no_cro"):
        sub = valid[valid["fold_type"] == fold_type]
        if len(sub) == 0:
            continue
        print(f"\n  {fold_type} ({len(sub)} predictions):")
        for col in ("global_bb_scrmsd", "motif_bb_scrmsd", "plddt"):
            v = sub[col].dropna()
            if len(v) > 0:
                print(f"    {col:25s}  mean={v.mean():.3f}  median={v.median():.3f}  "
                      f"[{v.min():.3f} - {v.max():.3f}]")

    # Best per design, top 1000
    best = valid.loc[
        valid.groupby(["pdb_key", "caliby_sample", "fold_type"])["motif_bb_scrmsd"].idxmin()
    ]
    best = best.sort_values("motif_bb_scrmsd")

    # Add sequences
    seqs = []
    for _, row in best.iterrows():
        key = (row["pdb_key"], row["caliby_sample"])
        seqs.append(seq_map.get(key))
    best = best.copy()
    best["seq"] = seqs

    top1000 = best.head(1000)
    top_path = OUT_DIR / "chroma08_caliby_rf3_motif_scrmsd_top1000.csv"
    top1000.to_csv(top_path, index=False)
    print(f"\nTop 1000 saved to {top_path}")
    print(f"  motif_bb_scrmsd range: {top1000['motif_bb_scrmsd'].min():.4f} - "
          f"{top1000['motif_bb_scrmsd'].max():.4f}")
    print(f"  Sequences found: {top1000['seq'].notna().sum()}/{len(top1000)}")

    # Threshold analysis
    print(f"\n  Threshold analysis (best of RF3 samples):")
    for fold_type in ("cro", "no_cro"):
        sub = best[best["fold_type"] == fold_type]
        n_total = len(sub)
        print(f"    {fold_type} (n={n_total}):")
        for thresh in [1.0, 1.5, 2.0, 3.0, 5.0]:
            n = (sub["motif_bb_scrmsd"] < thresh).sum()
            pct = 100 * n / n_total if n_total > 0 else 0
            print(f"      motif_scrmsd < {thresh:.1f}: {n}/{n_total} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
