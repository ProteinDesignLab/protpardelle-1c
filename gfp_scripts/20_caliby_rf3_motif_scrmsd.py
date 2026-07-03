"""Compute motif scRMSD for caliby RF3 predictions vs designed backbones.

For each RF3 prediction (consensus + 5 samples), compares to the original
designed backbone on motif residues (from fix_pos).

Metrics:
  - global_bb_scrmsd: backbone RMSD over all residues (after global BB alignment)
  - motif_bb_scrmsd: backbone RMSD over motif residues only (after global BB alignment)
  - plddt: overall pLDDT from RF3 confidence

Handles both CRO and no_CRO predictions. For CRO, maps CRO backbone atoms
(N1->N, CA1->CA, C3->C, O3->O) and shifts post-CRO residues by -2.

Usage:
    python 20_caliby_rf3_motif_scrmsd.py [--workers N]
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
RF3_BASE = Path("/scratch/users/tianyulu/gfp/rf3_outputs/caliby")
DESIGN_BASE = Path("/scratch/users/tianyulu/allatom-design-sampling-results/denovo_gfp_design_ensemble")
FIXPOS_DIR = Path("/scratch/users/tianyulu/gfp/denovo/chroma_05")
OUT_DIR = RF3_BASE

# Prefix -> (design subdir, fixpos CSV)
GROUPS = {
    "gt05": (
        DESIGN_BASE / "tm_gt_05/denovo_gfp_design_ensemble/samples",
        FIXPOS_DIR / "design_input_sasa_lt_50_caliby_fixpos_tm_above_0.5.csv",
    ),
    "lt05n11": (
        DESIGN_BASE / "tm_lt_05_not_11_stranded/denovo_gfp_design_ensemble/samples",
        FIXPOS_DIR / "design_input_sasa_lt_50_caliby_fixpos_tm_below_0.5_not_11_stranded.csv",
    ),
    "lt05_11s": (
        DESIGN_BASE / "tm_lt_05_11_stranded/denovo_gfp_design_ensemble/samples",
        FIXPOS_DIR / "design_input_sasa_lt_50_caliby_fixpos_tm_below_0.5_11_stranded.csv",
    ),
}

BB_ATOMS = ("N", "CA", "C", "O")
CRO_BB_MAP = {"N1": "N", "CA1": "CA", "C3": "C", "O3": "O"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_fixpos_seq(fixpos_str):
    """Parse 'A59,A60,...' -> [59, 60, ...]."""
    return [int(x.strip().lstrip("A")) for x in fixpos_str.split(",")]


def parse_rf3_dirname(dirname):
    """Parse RF3 directory name like 'gt05_d180_3_sample0_cro'.

    Returns (prefix, example_id, sample_name, fold_type) or None.
    """
    # Try CRO suffix first
    m = re.match(r"^(\w+?)_(d\d+_\d+)_(sample\d+)_(cro|no_cro)$", dirname)
    if m:
        return m.group(1), m.group(2), m.group(3), m.group(4)
    return None


def get_bb_atoms_from_residue(residue):
    """Get backbone atom coordinates from a residue, handling CRO."""
    resname = residue.get_resname().strip()
    atoms = {a.name: a for a in residue.get_atoms()}

    if resname == "CRO":
        result = {}
        for cro_name, std_name in CRO_BB_MAP.items():
            if cro_name in atoms:
                result[std_name] = atoms[cro_name]
        return result
    else:
        return {name: atoms[name] for name in BB_ATOMS if name in atoms}


def build_residue_map(structure, chain_id="A"):
    """Build {resid: residue} map from structure, including CRO."""
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
    """Compute global and motif backbone scRMSD.

    Aligns on all paired backbone atoms, then computes RMSD globally
    and on motif residues.

    Returns dict with global_bb_scrmsd and motif_bb_scrmsd.
    """
    # Collect all paired backbone atoms for alignment
    all_fixed, all_moving = [], []
    # Track which atoms belong to motif
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

    # Global alignment
    sup = Superimposer()
    sup.set_atoms(all_fixed, all_moving)
    global_rmsd = sup.rms

    result = {"global_bb_scrmsd": round(global_rmsd, 4)}

    # Motif RMSD after global alignment
    if motif_fixed_coords:
        rot, tran = sup.rotran
        transformed = np.dot(np.array(motif_moving_coords), rot) + tran
        diff = np.array(motif_fixed_coords) - transformed
        motif_rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        result["motif_bb_scrmsd"] = round(motif_rmsd, 4)
    else:
        result["motif_bb_scrmsd"] = None

    return result


def build_correspondence_nocro(fix_pos, n_design_res):
    """Build (pred_resids, design_resids) for no_CRO prediction.

    Both have standard amino acids, same numbering.
    """
    pred_resids = list(range(1, n_design_res + 1))
    design_resids = list(range(1, n_design_res + 1))
    return pred_resids, design_resids


def build_correspondence_cro(fix_pos, n_design_res):
    """Build (pred_resids, design_resids) for CRO prediction.

    CRO replaces 3 residues (TYG at fix_pos[7:10]) with 1 CRO.
    Residues before CRO: same numbering.
    CRO itself: pred has 1 residue at fix_pos[7], design has 3 at fix_pos[7:10].
    Residues after CRO: pred resids shifted by -2.
    """
    tyg_start = fix_pos[7]  # First TYG residue in design

    pred_resids = []
    design_resids = []

    # Before CRO: 1 to tyg_start-1
    for i in range(1, tyg_start):
        pred_resids.append(i)
        design_resids.append(i)

    # CRO: pred has 1 residue at tyg_start, design has residue at tyg_start
    # We include CRO for backbone comparison (using CRO_BB_MAP)
    # But CRO spans 3 design residues - only compare backbone of first (N) and last (C,O)
    # For simplicity, map CRO to design tyg_start (first of TYG)
    pred_resids.append(tyg_start)
    design_resids.append(tyg_start)

    # After CRO: design has tyg_start+3 onwards, pred has tyg_start+1 onwards
    for i in range(tyg_start + 3, n_design_res + 1):
        pred_resids.append(i - 2)
        design_resids.append(i)

    return pred_resids, design_resids


def find_rf3_cifs(design_dir):
    """Find all RF3 CIF model files (consensus + samples)."""
    cifs = []
    # Consensus model
    for f in design_dir.glob("*_model.cif"):
        if "sample" not in f.name and "seed" not in f.name:
            cifs.append(("consensus", f))
    # Sample models
    for sample_dir in sorted(design_dir.glob("seed-*_sample-*")):
        for f in sample_dir.glob("*_model.cif"):
            cifs.append((sample_dir.name, f))
    return cifs


def get_plddt(design_dir, rf3_name):
    """Get pLDDT from summary confidences JSON."""
    # Consensus
    json_path = design_dir / f"{rf3_name}_summary_confidences.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        return data.get("overall_plddt")
    return None


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def process_one(args):
    """Process a single RF3 prediction directory."""
    rf3_dir_str, fold_type, prefix, example_id, sample_name, \
        fix_pos, design_cif_path = args

    rf3_dir = Path(rf3_dir_str)
    rf3_name = rf3_dir.name  # e.g. gt05_d180_3_sample0_cro

    parser = MMCIFParser(QUIET=True)

    # Load designed structure
    try:
        design_struct = parser.get_structure("design", str(design_cif_path))
        design_res_map = build_residue_map(design_struct)
    except Exception as e:
        return [{"design_id": example_id, "prefix": prefix, "sample": sample_name,
                 "fold_type": fold_type, "error": f"design_load: {e}"}]

    n_design_res = max(design_res_map.keys()) if design_res_map else 0

    # Build motif residue set (in design numbering)
    motif_design_resids = fix_pos

    # Build residue correspondence
    if fold_type == "no_cro":
        pred_resids, design_resids = build_correspondence_nocro(fix_pos, n_design_res)
    else:
        pred_resids, design_resids = build_correspondence_cro(fix_pos, n_design_res)

    # Find all RF3 CIF files
    rf3_cifs = find_rf3_cifs(rf3_dir)
    if not rf3_cifs:
        return [{"design_id": example_id, "prefix": prefix, "sample": sample_name,
                 "fold_type": fold_type, "error": "no_rf3_cifs"}]

    rows = []
    for rf3_sample_name, cif_path in rf3_cifs:
        row = {
            "design_id": example_id,
            "prefix": prefix,
            "sample": sample_name,
            "fold_type": fold_type,
            "rf3_sample": rf3_sample_name,
            "rf3_cif": str(cif_path),
        }

        # Get pLDDT
        if rf3_sample_name == "consensus":
            plddt = get_plddt(rf3_dir, rf3_name)
        else:
            plddt_path = cif_path.parent / cif_path.name.replace(
                "_model.cif", "_summary_confidences.json"
            )
            if plddt_path.exists():
                with open(plddt_path) as f:
                    data = json.load(f)
                plddt = data.get("overall_plddt")
            else:
                plddt = None
        row["plddt"] = plddt

        try:
            pred_struct = parser.get_structure("pred", str(cif_path))
            pred_res_map = build_residue_map(pred_struct)
        except Exception as e:
            row["error"] = f"pred_load: {e}"
            rows.append(row)
            continue

        try:
            rmsd_results = compute_scrmsd(
                pred_res_map, design_res_map,
                pred_resids, design_resids,
                motif_design_resids=motif_design_resids,
            )
            row.update(rmsd_results)
        except Exception as e:
            row["error"] = f"rmsd: {e}"

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(32, cpu_count()),
                    help="Number of parallel workers")
    args = ap.parse_args()

    # Load all fixpos CSVs
    fixpos_maps = {}  # prefix -> {example_id: fix_pos_list}
    for prefix, (_, fixpos_csv) in GROUPS.items():
        if not fixpos_csv.exists():
            print(f"WARNING: fixpos CSV not found: {fixpos_csv}")
            continue
        df = pd.read_csv(fixpos_csv)
        fmap = {}
        for _, row in df.iterrows():
            pdb_key = row["pdb_key"]
            fp = parse_fixpos_seq(row["fixed_pos_seq"])
            fmap[pdb_key] = fp
        fixpos_maps[prefix] = fmap
        print(f"  {prefix}: {len(fmap)} designs with fix_pos")

    # Discover all RF3 output directories
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

                prefix, example_id, sample_name, ft = parsed
                if ft != fold_type:
                    continue

                if prefix not in fixpos_maps:
                    continue
                if example_id not in fixpos_maps[prefix]:
                    continue

                fix_pos = fixpos_maps[prefix][example_id]

                # Find designed structure CIF
                design_samples_dir = GROUPS[prefix][0]
                design_cif = design_samples_dir / f"{example_id}_{sample_name}.cif"
                if not design_cif.exists():
                    continue

                work_items.append((
                    str(design_dir), fold_type, prefix, example_id,
                    sample_name, fix_pos, str(design_cif),
                ))

    print(f"\nTotal RF3 directories to process: {len(work_items)}")
    print(f"Using {args.workers} workers")

    # Process with multiprocessing
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
            print(df[df["error"].notna()]["error"].value_counts().head(5))

    # Save full results
    out_path = OUT_DIR / "caliby_rf3_motif_scrmsd.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    # Summary statistics
    valid = df.dropna(subset=["global_bb_scrmsd", "motif_bb_scrmsd"])
    print(f"\nValid predictions: {len(valid)}")

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

        # Per-prefix breakdown
        for prefix in sorted(sub["prefix"].unique()):
            psub = sub[sub["prefix"] == prefix]
            print(f"    {prefix}: n={len(psub)}  "
                  f"motif_scrmsd={psub['motif_bb_scrmsd'].mean():.3f}±{psub['motif_bb_scrmsd'].std():.3f}  "
                  f"global_scrmsd={psub['global_bb_scrmsd'].mean():.3f}±{psub['global_bb_scrmsd'].std():.3f}")

    # Consensus-only summary
    consensus = valid[valid["rf3_sample"] == "consensus"]
    if len(consensus) > 0:
        print(f"\n  Consensus models only ({len(consensus)}):")
        for fold_type in ("cro", "no_cro"):
            sub = consensus[consensus["fold_type"] == fold_type]
            if len(sub) == 0:
                continue
            print(f"    {fold_type}: motif_scrmsd={sub['motif_bb_scrmsd'].mean():.3f}±{sub['motif_bb_scrmsd'].std():.3f}  "
                  f"global_scrmsd={sub['global_bb_scrmsd'].mean():.3f}±{sub['global_bb_scrmsd'].std():.3f}  "
                  f"plddt={sub['plddt'].mean():.3f}±{sub['plddt'].std():.3f}")

    # Threshold analysis
    print(f"\n  Designs with motif scRMSD below threshold (consensus, no_cro):")
    cons_nocro = consensus[consensus["fold_type"] == "no_cro"]
    for thresh in [0.5, 1.0, 1.5, 2.0, 3.0]:
        n = (cons_nocro["motif_bb_scrmsd"] < thresh).sum()
        pct = 100 * n / len(cons_nocro) if len(cons_nocro) > 0 else 0
        print(f"    < {thresh:.1f} Å: {n}/{len(cons_nocro)} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
