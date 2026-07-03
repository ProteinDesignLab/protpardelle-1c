"""Dual RMSD filter for chroma_08 MiniCaliby designs.

Adapted from 19b_chroma08_dual_rmsd_filter.py for caliby-designed sequences.

Metric 1: no_cro vs CRO RMSD (pre-cyclic vs post-cyclic consistency)
Metric 2: CRO vs crystal (2AWK) motif RMSD (chromophore geometry fidelity)

All RMSD metrics are all-atom (sidechains included), with alignment on backbone.
For motif residues with mismatched types (CYS at crystal 70, MET->ARG at 96),
only backbone atoms contribute to the RMSD.

Tier assignment:
  T1: helix_allatom_rmsd < 1A AND motif_allatom_rmsd < 1A  (best)
  T2: helix_allatom_rmsd >= 1A AND motif_allatom_rmsd < 1A  (crystal-like CRO)
  T3: helix_allatom_rmsd < 1A AND motif_allatom_rmsd >= 1A  (consistent but wrong CRO)
  T4: both fail

Top 1000 selected by tier priority (T1 > T2 > T3 > T4), sorted by
motif_allatom_rmsd within each tier. No duplicate sequences.

Usage:
    python 21_caliby_dual_rmsd_filter.py [--workers N]
"""

import argparse
import json
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, Superimposer
from tqdm import tqdm

PLDDT_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RF3_BASE = Path("/scratch/users/tianyulu/gfp/denovo/chroma_08/caliby")
FIXPOS_CSV = Path("/scratch/users/tianyulu/gfp/denovo/chroma_08/top1000_rmsd_fixpos.csv")
SEQDES_CSV = Path(
    "/scratch/users/tianyulu/allatom-design-sampling-results/"
    "chroma_08_top1000_rmsd/chroma_08_top1000_rmsd/seq_des_outputs.csv"
)
SCRMSD_CSV = RF3_BASE / "chroma08_caliby_rf3_motif_scrmsd.csv"
CRYSTAL_CIF = "/scratch/users/tianyulu/gfp/rf3/2AWK.cif"
OUT_DIR = RF3_BASE

T = 1.0  # RMSD threshold in Angstroms
MOTIF_SCRMSD_THRESHOLD = 1.5  # Max motif backbone scRMSD to include in top 1000

# CRO atom -> (offset from first TYG residue, standard atom name)
CRO_TO_TYG_MAP = {
    # THR (offset 0)
    "N1": (0, "N"), "CA1": (0, "CA"), "CB1": (0, "CB"),
    "CG1": (0, "CG2"), "OG1": (0, "OG1"), "C1": (0, "C"),
    # TYR (offset 1)
    "N2": (1, "N"), "CA2": (1, "CA"), "CB2": (1, "CB"),
    "CG2": (1, "CG"), "CD1": (1, "CD1"), "CD2": (1, "CD2"),
    "CE1": (1, "CE1"), "CE2": (1, "CE2"), "CZ": (1, "CZ"),
    "OH": (1, "OH"), "C2": (1, "C"), "O2": (1, "O"),
    # GLY (offset 2)
    "N3": (2, "N"), "CA3": (2, "CA"), "C3": (2, "C"), "O3": (2, "O"),
}

CRYSTAL_CRO_RESID = 66
CRYSTAL_R96 = 96
BACKBONE_ONLY_CRYSTAL = {70, 96}
BB_ATOMS = {"N", "CA", "C", "O"}


# ---------------------------------------------------------------------------
# Structure parsing
# ---------------------------------------------------------------------------
def get_all_atoms(structure, chain_id="A"):
    """Extract all heavy atoms per residue -> {resid: {atom_name: atom_obj}}."""
    model = structure[0]
    chain = model[chain_id]
    residues = {}
    for r in chain.get_residues():
        if r.id[0] != " " and r.get_resname().strip() != "CRO":
            continue
        resid = r.id[1]
        atoms = {}
        for atom in r.get_atoms():
            name = atom.name
            if name[0] in ("H", "1", "2", "3"):
                continue
            atoms[name] = atom
        if atoms:
            residues[resid] = atoms
    return residues


def get_bb_atoms(all_atoms, resid):
    if resid not in all_atoms:
        return {}
    atoms = all_atoms[resid]
    if "N1" in atoms:  # CRO
        mapping = {"N1": "N", "CA1": "CA", "C3": "C", "O3": "O"}
        return {mapping[k]: v for k, v in atoms.items() if k in mapping}
    return {k: v for k, v in atoms.items() if k in BB_ATOMS}


# ---------------------------------------------------------------------------
# Superimposition and RMSD
# ---------------------------------------------------------------------------
def superimpose_on_backbone(fixed_all, moving_all, fixed_resids, moving_resids):
    fixed_atoms, moving_atoms = [], []
    for fr, mr in zip(fixed_resids, moving_resids):
        fbb = get_bb_atoms(fixed_all, fr)
        mbb = get_bb_atoms(moving_all, mr)
        for aname in ("N", "CA", "C", "O"):
            a1, a2 = fbb.get(aname), mbb.get(aname)
            if a1 is not None and a2 is not None:
                fixed_atoms.append(a1)
                moving_atoms.append(a2)
    if len(fixed_atoms) < 4:
        return None, False
    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)
    return sup, True


def rmsd_after_sup(sup, fixed_coords, moving_coords):
    if len(fixed_coords) == 0:
        return None
    rot, tran = sup.rotran
    transformed = np.dot(np.array(moving_coords), rot) + tran
    diff = np.array(fixed_coords) - transformed
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def collect_bb_coord_pairs(fixed_all, moving_all, fixed_resids, moving_resids):
    fc, mc = [], []
    for fr, mr in zip(fixed_resids, moving_resids):
        fbb = get_bb_atoms(fixed_all, fr)
        mbb = get_bb_atoms(moving_all, mr)
        for aname in ("N", "CA", "C", "O"):
            a1, a2 = fbb.get(aname), mbb.get(aname)
            if a1 is not None and a2 is not None:
                fc.append(a1.get_vector().get_array())
                mc.append(a2.get_vector().get_array())
    return fc, mc


def collect_allatom_pairs(fixed_all, moving_all, fixed_resids, moving_resids,
                          bb_only_fixed_resids=None):
    if bb_only_fixed_resids is None:
        bb_only_fixed_resids = set()
    fc, mc = [], []
    for fr, mr in zip(fixed_resids, moving_resids):
        if fr not in fixed_all or mr not in moving_all:
            continue
        fatoms = fixed_all[fr]
        matoms = moving_all[mr]
        use_bb_only = fr in bb_only_fixed_resids
        for aname, atom in fatoms.items():
            if use_bb_only and aname not in BB_ATOMS:
                continue
            if aname in matoms:
                fc.append(atom.get_vector().get_array())
                mc.append(matoms[aname].get_vector().get_array())
    return fc, mc


def collect_cro_to_tyg_pairs(cro_all, nocro_all, cro_resid, tyg_start):
    if cro_resid not in cro_all:
        return [], []
    cro_atoms = cro_all[cro_resid]
    fc, mc = [], []
    for cro_aname, (offset, std_aname) in CRO_TO_TYG_MAP.items():
        nocro_resid = tyg_start + offset
        ca = cro_atoms.get(cro_aname)
        na = nocro_all.get(nocro_resid, {}).get(std_aname)
        if ca is not None and na is not None:
            fc.append(ca.get_vector().get_array())
            mc.append(na.get_vector().get_array())
    return fc, mc


def collect_cro_to_cro_pairs(design_all, crystal_all, design_cro_resid,
                             crystal_cro_resid=CRYSTAL_CRO_RESID):
    if design_cro_resid not in design_all or crystal_cro_resid not in crystal_all:
        return [], []
    d_atoms = design_all[design_cro_resid]
    c_atoms = crystal_all[crystal_cro_resid]
    fc, mc = [], []
    for aname, atom in c_atoms.items():
        if aname in d_atoms:
            fc.append(atom.get_vector().get_array())
            mc.append(d_atoms[aname].get_vector().get_array())
    return fc, mc


# ---------------------------------------------------------------------------
# Residue correspondences from fix_pos
# ---------------------------------------------------------------------------
def parse_fixpos(fixpos_str):
    return [int(x.strip().lstrip("A")) for x in fixpos_str.split(",")]


def build_nocro_cro_correspondence(fix_pos, n_nocro):
    fp7 = fix_pos[7]
    fp9 = fix_pos[9]
    before_nocro = list(range(1, fp7))
    before_cro = list(range(1, fp7))
    after_nocro = list(range(fp9 + 1, n_nocro + 1))
    after_cro = list(range(fp7 + 1, fp7 + 1 + len(after_nocro)))
    return before_nocro + after_nocro, before_cro + after_cro


def build_helix_correspondence(fix_pos):
    helix_nocro, helix_cro = [], []
    for i in range(2, 7):
        helix_nocro.append(fix_pos[i])
        helix_cro.append(fix_pos[i])
    for i in range(10, 14):
        helix_nocro.append(fix_pos[i])
        helix_cro.append(fix_pos[i] - 2)
    return helix_nocro, helix_cro


def build_crystal_motif_correspondence(fix_pos):
    fp7 = fix_pos[7]
    design_resids, crystal_resids = [], []
    for i in range(7):
        design_resids.append(fix_pos[i])
        crystal_resids.append(58 + i)
    for i, gfp_res in zip(range(10, 14), [68, 69, 70, 71]):
        design_resids.append(fix_pos[i] - 2)
        crystal_resids.append(gfp_res)
    design_resids.append(fix_pos[14] - 2)
    crystal_resids.append(CRYSTAL_R96)
    return design_resids, crystal_resids, fp7, CRYSTAL_CRO_RESID


# ---------------------------------------------------------------------------
# RF3 directory parsing
# ---------------------------------------------------------------------------
def parse_rf3_dirname(dirname):
    """Parse caliby RF3 dir name -> (pdb_key, caliby_sample, fold_type) or None.

    Examples:
      p0_d1043_0_seed-0_sample-4_sample0_cro -> (p0_d1043_0_seed-0_sample-4, sample0, cro)
      p0_d3559_0_consensus_sample3_no_cro    -> (p0_d3559_0_consensus, sample3, no_cro)
    """
    if dirname.endswith("_no_cro"):
        fold_type = "no_cro"
        rest = dirname[: -len("_no_cro")]
    elif dirname.endswith("_cro"):
        fold_type = "cro"
        rest = dirname[: -len("_cro")]
    else:
        return None
    m = re.match(r"^(.+)_(sample\d+)$", rest)
    if not m:
        return None
    return m.group(1), m.group(2), fold_type


def read_plddt(cif_path):
    summary_path = Path(str(cif_path).replace("_model.cif", "_summary_confidences.json"))
    if not summary_path.exists():
        return None
    try:
        with open(summary_path) as f:
            return json.load(f).get("overall_plddt")
    except Exception:
        return None


def find_rf3_samples(design_dir):
    """Find all RF3 CIF files (consensus + diffusion samples)."""
    cifs = []
    # Consensus model
    for f in design_dir.glob("*_model.cif"):
        if "sample" not in f.name:
            cifs.append(("consensus", f))
    # Diffusion samples
    for sample_dir in sorted(design_dir.glob("seed-*_sample-*")):
        for f in sample_dir.glob("*_model.cif"):
            cifs.append((sample_dir.name, f))
    return cifs


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------
def process_one_design(args):
    """Process one (pdb_key, caliby_sample) pair across CRO and no_CRO."""
    cro_dir_str, nocro_dir_str, pdb_key, caliby_sample, fix_pos_str, crystal_cif = args
    cro_dir = Path(cro_dir_str)
    nocro_dir = Path(nocro_dir_str)
    fix_pos = parse_fixpos(fix_pos_str)

    parser = MMCIFParser(QUIET=True)
    crystal_struct = parser.get_structure("crystal", crystal_cif)
    crystal_all = get_all_atoms(crystal_struct)

    cro_samples = find_rf3_samples(cro_dir)
    nocro_samples = find_rf3_samples(nocro_dir)
    if not cro_samples or not nocro_samples:
        return []

    nocro_by_sample = {name: path for name, path in nocro_samples}

    rows = []
    for sample_name, cro_path in cro_samples:
        nocro_path = nocro_by_sample.get(sample_name)
        if nocro_path is None:
            continue

        cro_plddt = read_plddt(cro_path)
        nocro_plddt = read_plddt(nocro_path)
        if cro_plddt is not None and cro_plddt < PLDDT_THRESHOLD:
            continue
        if nocro_plddt is not None and nocro_plddt < PLDDT_THRESHOLD:
            continue

        row = {
            "pdb_key": pdb_key,
            "caliby_sample": caliby_sample,
            "rf3_sample": sample_name,
            "cro_plddt": cro_plddt,
            "nocro_plddt": nocro_plddt,
        }

        try:
            cro_struct = parser.get_structure("cro", str(cro_path))
            cro_all = get_all_atoms(cro_struct)
            nocro_struct = parser.get_structure("nocro", str(nocro_path))
            nocro_all = get_all_atoms(nocro_struct)
        except Exception as e:
            row["error"] = str(e)
            rows.append(row)
            continue

        n_nocro = max(nocro_all.keys()) if nocro_all else 0
        fp7 = fix_pos[7]

        # ---- Metric 1: no_cro vs CRO ----
        try:
            nocro_nonchrom, cro_nonchrom = build_nocro_cro_correspondence(
                fix_pos, n_nocro
            )
            helix_nocro, helix_cro = build_helix_correspondence(fix_pos)

            sup1, ok1 = superimpose_on_backbone(
                cro_all, nocro_all, cro_nonchrom, nocro_nonchrom
            )
            if ok1:
                gc, gm = collect_allatom_pairs(
                    cro_all, nocro_all, cro_nonchrom, nocro_nonchrom
                )
                row["global_allatom_rmsd"] = rmsd_after_sup(sup1, gc, gm)

                gbc, gbm = collect_bb_coord_pairs(
                    cro_all, nocro_all, cro_nonchrom, nocro_nonchrom
                )
                row["global_bb_rmsd"] = rmsd_after_sup(sup1, gbc, gbm)

                hc, hm = collect_allatom_pairs(
                    cro_all, nocro_all, helix_cro, helix_nocro
                )
                cc, cm = collect_cro_to_tyg_pairs(cro_all, nocro_all, fp7, fp7)
                row["chrom_allatom_rmsd"] = rmsd_after_sup(sup1, cc, cm)

                helix_full_c = hc + cc
                helix_full_m = hm + cm
                row["helix_allatom_rmsd"] = rmsd_after_sup(
                    sup1, helix_full_c, helix_full_m
                )
                hbc, hbm = collect_bb_coord_pairs(
                    cro_all, nocro_all, helix_cro, helix_nocro
                )
                row["helix_bb_rmsd"] = rmsd_after_sup(sup1, hbc, hbm)
            else:
                row["error_m1"] = "superimpose_failed"
        except Exception as e:
            row["error_m1"] = str(e)

        # ---- Metric 2: CRO vs Crystal ----
        try:
            design_motif, crystal_motif, d_cro_resid, c_cro_resid = \
                build_crystal_motif_correspondence(fix_pos)

            sup2, ok2 = superimpose_on_backbone(
                crystal_all, cro_all, crystal_motif, design_motif
            )
            if ok2:
                ma_c, ma_m = collect_allatom_pairs(
                    crystal_all, cro_all, crystal_motif, design_motif,
                    bb_only_fixed_resids=BACKBONE_ONLY_CRYSTAL,
                )
                row["motif_allatom_rmsd"] = rmsd_after_sup(sup2, ma_c, ma_m)

                mb_c, mb_m = collect_bb_coord_pairs(
                    crystal_all, cro_all, crystal_motif, design_motif
                )
                row["motif_bb_rmsd"] = rmsd_after_sup(sup2, mb_c, mb_m)

                cro_c, cro_m = collect_cro_to_cro_pairs(
                    cro_all, crystal_all, d_cro_resid, c_cro_resid
                )
                row["cro_allatom_rmsd"] = rmsd_after_sup(sup2, cro_c, cro_m)
            else:
                row["error_m2"] = "superimpose_failed"
        except Exception as e:
            row["error_m2"] = str(e)

        for col in ("global_allatom_rmsd", "global_bb_rmsd",
                     "helix_allatom_rmsd", "helix_bb_rmsd",
                     "chrom_allatom_rmsd",
                     "motif_allatom_rmsd", "motif_bb_rmsd",
                     "cro_allatom_rmsd"):
            if col in row and row[col] is not None:
                row[col] = round(row[col], 4)

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------
def assign_tiers(df):
    df = df.copy()
    df["pass_m1"] = df["helix_allatom_rmsd"] < T
    df["pass_m2"] = df["motif_allatom_rmsd"] < T

    def _tier(row):
        m1 = row.get("pass_m1", False)
        m2 = row.get("pass_m2", False)
        if pd.isna(m1) or pd.isna(m2):
            return 4
        if m1 and m2:
            return 1
        elif not m1 and m2:
            return 2
        elif m1 and not m2:
            return 3
        else:
            return 4

    df["pred_tier"] = df.apply(_tier, axis=1)
    return df


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_pairs():
    """Find all matching (cro_dir, nocro_dir, pdb_key, caliby_sample) across partitions."""
    # Index all cro dirs by (pdb_key, caliby_sample)
    cro_index = {}
    for fold_type in ("cro", "no_cro"):
        type_dir = RF3_BASE / fold_type
        if not type_dir.exists():
            continue
        for chunk_dir in sorted(type_dir.iterdir()):
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
                key = (pdb_key, caliby_sample)
                if fold_type == "cro":
                    cro_index[key] = design_dir
                else:
                    # Store nocro dirs separately
                    if key not in cro_index:
                        cro_index[key] = None
                    # Use a second dict for nocro
                    pass

    # Re-do more cleanly: separate dicts
    cro_dirs = {}
    nocro_dirs = {}
    for fold_type, target_dict in [("cro", cro_dirs), ("no_cro", nocro_dirs)]:
        type_dir = RF3_BASE / fold_type
        if not type_dir.exists():
            continue
        for chunk_dir in sorted(type_dir.iterdir()):
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
                target_dict[(pdb_key, caliby_sample)] = design_dir

    # Find pairs
    pairs = []
    for key in sorted(cro_dirs.keys()):
        if key in nocro_dirs:
            pairs.append((cro_dirs[key], nocro_dirs[key], key[0], key[1]))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Dual RMSD filter for chroma_08 caliby designs."
    )
    ap.add_argument("--workers", type=int, default=min(32, cpu_count()))
    args = ap.parse_args()

    # Load fix_pos
    fp_df = pd.read_csv(FIXPOS_CSV)
    fixpos_map = {}
    for _, row in fp_df.iterrows():
        fixpos_map[row["pdb_key"]] = row["fixed_pos_seq"]
    print(f"Loaded {len(fixpos_map)} fix_pos entries")

    # Load sequences
    seq_df = pd.read_csv(SEQDES_CSV)
    seq_map = {}
    for _, row in seq_df.iterrows():
        eid = row["example_id"]
        sample_name = Path(row["out_pdb"]).stem.replace(f"{eid}_", "")
        seq_map[(eid, sample_name)] = row["seq"]
    print(f"Loaded {len(seq_map)} sequences")

    # Discover paired CRO/no_CRO directories
    print("Discovering design pairs...")
    pairs = discover_pairs()
    print(f"  {len(pairs)} paired (pdb_key, caliby_sample) entries")

    # Build work items
    work_items = []
    skipped = 0
    for cro_dir, nocro_dir, pdb_key, caliby_sample in pairs:
        if pdb_key in fixpos_map:
            work_items.append((
                str(cro_dir), str(nocro_dir), pdb_key, caliby_sample,
                fixpos_map[pdb_key], CRYSTAL_CIF,
            ))
        else:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} (no fix_pos)")
    print(f"  {len(work_items)} designs to process")
    print(f"Using {args.workers} workers")

    if not work_items:
        print("No work items, exiting.")
        return

    all_rows = []
    with Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_one_design, work_items, chunksize=16),
            total=len(work_items),
            desc="Dual RMSD",
        ))
    for result in results:
        all_rows.extend(result)

    if not all_rows:
        print("No predictions processed!")
        return

    df = pd.DataFrame(all_rows)
    print(f"\nTotal predictions: {len(df)}")
    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        if n_err > 0:
            print(f"  Errors: {n_err}")

    # Assign tiers
    valid = df.dropna(subset=["helix_allatom_rmsd", "motif_allatom_rmsd"])
    print(f"Valid predictions (both metrics): {len(valid)}")
    valid = assign_tiers(valid)

    # Save full results with tiers
    out_all = OUT_DIR / "caliby_dual_rmsd_all.csv"
    valid.to_csv(out_all, index=False)
    print(f"Full results saved to {out_all}")

    # Tier summary
    for tier in sorted(valid["pred_tier"].unique()):
        sub = valid[valid["pred_tier"] == tier]
        n_pred = len(sub)
        n_designs = sub.groupby(["pdb_key", "caliby_sample"]).ngroups
        print(f"  Tier {tier}: {n_pred} predictions, {n_designs} unique (pdb_key, caliby_sample)")

    # Best prediction per (pdb_key, caliby_sample): best tier, then best motif_allatom_rmsd
    valid_sorted = valid.sort_values(["pred_tier", "motif_allatom_rmsd"])
    best = valid_sorted.drop_duplicates(subset=["pdb_key", "caliby_sample"], keep="first")
    best = best.copy()

    # Assign best_tier per design
    best_tier = best.groupby("pdb_key")["pred_tier"].min().reset_index()
    best_tier.columns = ["pdb_key", "best_tier"]
    best = best.merge(best_tier, on="pdb_key")

    # Add sequences
    seqs = []
    for _, row in best.iterrows():
        key = (row["pdb_key"], row["caliby_sample"])
        seqs.append(seq_map.get(key))
    best["seq"] = seqs

    print(f"\nBest-per-design summary ({len(best)} designs):")
    for tier in sorted(best["pred_tier"].unique()):
        sub = best[best["pred_tier"] == tier]
        print(f"  Tier {tier}: {len(sub)} designs")
        for col in ("helix_allatom_rmsd", "motif_allatom_rmsd"):
            v = sub[col].dropna()
            if len(v) > 0:
                print(f"    {col:25s}  mean={v.mean():.3f}  median={v.median():.3f}  "
                      f"[{v.min():.3f} - {v.max():.3f}]")

    # Load motif scRMSD data and compute best per (pdb_key, caliby_sample)
    if SCRMSD_CSV.exists():
        scrmsd_df = pd.read_csv(SCRMSD_CSV)
        best_scrmsd = scrmsd_df.groupby(["pdb_key", "caliby_sample"]).agg(
            best_motif_scrmsd=("motif_bb_scrmsd", "min"),
            best_global_scrmsd=("global_bb_scrmsd", "min"),
        ).reset_index()
        best = best.merge(best_scrmsd, on=["pdb_key", "caliby_sample"], how="left")
        n_with = best["best_motif_scrmsd"].notna().sum()
        print(f"\nMotif scRMSD data matched: {n_with}/{len(best)}")
        print(f"  best_motif_scrmsd: mean={best['best_motif_scrmsd'].mean():.3f}  "
              f"median={best['best_motif_scrmsd'].median():.3f}")
    else:
        print(f"\nWARNING: {SCRMSD_CSV} not found, skipping motif scRMSD filter")
        best["best_motif_scrmsd"] = None
        best["best_global_scrmsd"] = None

    # Select top 1000: tier priority, then motif_allatom_rmsd, no duplicate sequences
    # Filter: require motif scRMSD < threshold
    best_sorted = best.sort_values(["pred_tier", "motif_allatom_rmsd"])
    top1000 = []
    seen_seqs = set()
    n_filtered_scrmsd = 0
    for _, row in best_sorted.iterrows():
        # Skip designs with poor motif scRMSD
        scrmsd_val = row.get("best_motif_scrmsd")
        if pd.notna(scrmsd_val) and scrmsd_val >= MOTIF_SCRMSD_THRESHOLD:
            n_filtered_scrmsd += 1
            continue
        seq = row.get("seq")
        if seq is not None and seq in seen_seqs:
            continue
        top1000.append(row)
        if seq is not None:
            seen_seqs.add(seq)
        if len(top1000) >= 1000:
            break

    top_df = pd.DataFrame(top1000)
    top_path = OUT_DIR / "caliby_dual_rmsd_top1000.csv"
    top_df.to_csv(top_path, index=False)

    print(f"\nTop 1000 saved to {top_path}")
    print(f"  Filtered by motif_scrmsd >= {MOTIF_SCRMSD_THRESHOLD}A: {n_filtered_scrmsd}")
    print(f"  Final count: {len(top_df)}")
    print(f"  Unique sequences: {top_df['seq'].nunique()}")
    for tier in sorted(top_df["pred_tier"].unique()):
        n = (top_df["pred_tier"] == tier).sum()
        print(f"  Tier {tier}: {n}")
    print(f"  motif_allatom_rmsd range: "
          f"{top_df['motif_allatom_rmsd'].min():.4f} - "
          f"{top_df['motif_allatom_rmsd'].max():.4f}")
    print(f"  helix_allatom_rmsd range: "
          f"{top_df['helix_allatom_rmsd'].min():.4f} - "
          f"{top_df['helix_allatom_rmsd'].max():.4f}")
    if "best_motif_scrmsd" in top_df.columns:
        v = top_df["best_motif_scrmsd"].dropna()
        print(f"  best_motif_scrmsd range: {v.min():.4f} - {v.max():.4f}")
    print(f"  Sequences found: {top_df['seq'].notna().sum()}/{len(top_df)}")


if __name__ == "__main__":
    main()
