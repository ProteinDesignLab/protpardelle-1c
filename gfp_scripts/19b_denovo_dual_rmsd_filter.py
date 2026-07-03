"""Dual RMSD filter for de novo GFP scaffold designs + downselection.

Metric 1: no_cro vs CRO RMSD (pre-cyclic vs post-cyclic consistency)
Metric 2: CRO vs crystal (2AWK) motif RMSD (chromophore geometry fidelity)

Design-specific chromophore positions from fix_pos column.
RF3 predictions: consensus model + 5 samples per design.

All RMSD metrics are all-atom (sidechains included), with alignment on backbone.
For motif residues with mismatched types (CYS at crystal 70, MET->ARG at 96),
only backbone atoms contribute to the RMSD.

Tier assignment:
  Tier 1: helix_allatom_rmsd < 1Å AND motif_allatom_rmsd < 1Å  (best)
  Tier 2: helix_allatom_rmsd >= 1Å AND motif_allatom_rmsd < 1Å  (crystal-like CRO)
  Tier 3: helix_allatom_rmsd < 1Å AND motif_allatom_rmsd >= 1Å  (consistent but wrong CRO)
  Tier 4: both fail

Outputs one CSV per group + combined selected_sequences CSV.

Usage:
    conda run -n farfalle python 19b_denovo_dual_rmsd_filter.py [--workers N]
"""

import argparse
import re
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, Superimposer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DESIGN_CSV = "/home/pdl/results/gfp/round2/denovo/design_input_sasa_lt_50_design_output.csv"
RF3_BASE = Path("/home/pdl/results/gfp/round2/denovo/rf3_outputs")
CRYSTAL_CIF = "/home/pdl/results/gfp/round2/inputs/2AWK.cif"
WATER_HBOND_DIR = Path(
    "/home/pdl/results/gfp/round2/denovo/design_input_sasa_lt_50_chroma_repack_lowest_energy"
)
OUT_DIR = Path("/home/pdl/results/gfp/round2/denovo/rf3_outputs")

# Groups: label -> (cro_subdir, no_cro_subdir, water_hbond_csv)
GROUPS = {
    "tm_above_0.5": (
        "chroma_05_tm_above_0.5_cro",
        "chroma_05_tm_above_0.5_no_cro",
        WATER_HBOND_DIR / "water_hbond_analysis_tm_above_0.5.csv",
    ),
    "tm_below_0.5_11_stranded": (
        "chroma_05_tm_below_0.5_11_stranded_cro",
        "chroma_05_tm_below_0.5_11_stranded_no_cro",
        WATER_HBOND_DIR / "water_hbond_analysis_tm_below_0.5_11_stranded.csv",
    ),
    "tm_below_0.5_not_11_stranded": (
        "chroma_05_tm_below_0.5_not_11_stranded_cro",
        "chroma_05_tm_below_0.5_not_11_stranded_no_cro",
        WATER_HBOND_DIR / "water_hbond_analysis_tm_below_0.5_not_11_stranded.csv",
    ),
}

T = 1.0  # RMSD threshold in Angstroms

# CRO atom -> (offset from first TYG residue, standard atom name)
# offset 0 = THR, offset 1 = TYR, offset 2 = GLY
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

# Crystal motif: GFP residue 70 is CYS (missing in designs),
# GFP residue 96 is MET (replaced by ARG in designs).
# These positions get backbone-only matching in all-atom RMSD.
CRYSTAL_CRO_RESID = 66
CRYSTAL_R96 = 96
# fix_pos index -> crystal GFP residue for the mismatched positions
BACKBONE_ONLY_CRYSTAL = {70, 96}

BB_ATOMS = {"N", "CA", "C", "O"}


# ---------------------------------------------------------------------------
# Structure parsing
# ---------------------------------------------------------------------------
def get_all_atoms(structure, chain_id="A"):
    """Extract all heavy atoms per residue -> {resid: {atom_name: atom_obj}}.

    Excludes hydrogen atoms (names starting with H or digit).
    """
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
    """Get backbone atom dict from all_atoms, handling CRO."""
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
    """Superimpose moving onto fixed using backbone of paired residues.

    Returns (Superimposer, success).
    """
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
    """RMSD on coordinate arrays after applying superimposition."""
    if len(fixed_coords) == 0:
        return None
    rot, tran = sup.rotran
    transformed = np.dot(np.array(moving_coords), rot) + tran
    diff = np.array(fixed_coords) - transformed
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def collect_bb_coord_pairs(fixed_all, moving_all, fixed_resids, moving_resids):
    """Collect paired backbone coords."""
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
    """All-atom coord pairs matched by atom name.

    For residues in bb_only_fixed_resids, only backbone atoms are used
    (for mismatched residue types like CYS->PHE, MET->ARG).
    """
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
    """CRO vs TYG atom pairs using the known cyclization mapping."""
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
    """CRO-to-CRO all-atom pairs (direct atom name matching)."""
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
# Per-design residue mapping from fix_pos
# ---------------------------------------------------------------------------
def parse_fix_pos(fix_pos_str):
    """Parse comma-separated fix_pos string -> list of ints."""
    return [int(x.strip()) for x in fix_pos_str.split(",")]


def build_nocro_cro_correspondence(fix_pos, n_nocro):
    """Build non-chromophore residue mapping between no_cro and CRO structures.

    Returns (nocro_nonchrom, cro_nonchrom) lists of paired residue IDs.
    """
    fp7 = fix_pos[7]
    fp9 = fix_pos[9]
    before_nocro = list(range(1, fp7))
    before_cro = list(range(1, fp7))
    after_nocro = list(range(fp9 + 1, n_nocro + 1))
    after_cro = list(range(fp7 + 1, fp7 + 1 + len(after_nocro)))
    return before_nocro + after_nocro, before_cro + after_cro


def build_helix_correspondence(fix_pos):
    """Build helix non-chromophore residue pairs for no_cro vs CRO.

    Helix = fix_pos[2..6] (before CRO) + fix_pos[10..13] (after CRO, shifted).
    """
    helix_nocro, helix_cro = [], []
    for i in range(2, 7):
        helix_nocro.append(fix_pos[i])
        helix_cro.append(fix_pos[i])
    for i in range(10, 14):
        helix_nocro.append(fix_pos[i])
        helix_cro.append(fix_pos[i] - 2)
    return helix_nocro, helix_cro


def build_crystal_motif_correspondence(fix_pos):
    """Build motif residue pairs for CRO design vs crystal.

    Returns (design_resids, crystal_resids, design_cro_resid, crystal_cro_resid).
    Crystal residues 70 (CYS) and 96 (MET) have mismatched types -> bb only.
    """
    fp7 = fix_pos[7]
    design_resids, crystal_resids = [], []

    # Before CRO: fix_pos[0..6] -> crystal 58..64
    for i in range(7):
        design_resids.append(fix_pos[i])
        crystal_resids.append(58 + i)

    # After CRO: fix_pos[10..13] shifted by -2 -> crystal 68..71
    for i, gfp_res in zip(range(10, 14), [68, 69, 70, 71]):
        design_resids.append(fix_pos[i] - 2)
        crystal_resids.append(gfp_res)

    # R96 equivalent: fix_pos[14] - 2 -> crystal 96
    design_resids.append(fix_pos[14] - 2)
    crystal_resids.append(CRYSTAL_R96)

    return design_resids, crystal_resids, fp7, CRYSTAL_CRO_RESID


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------
def find_rf3_samples(design_dir):
    """Find all RF3 CIF files in a design directory (consensus + samples)."""
    cifs = []
    for f in design_dir.glob("*_model.cif"):
        if "sample" not in f.name:
            cifs.append(("consensus", f))
    for sample_dir in sorted(design_dir.glob("seed-*_sample-*")):
        for f in sample_dir.glob("*_model.cif"):
            cifs.append((sample_dir.name, f))
    return cifs


def extract_design_id(dirname):
    """Extract design ID from directory name like 'd101_cro' -> 'd101'."""
    m = re.match(r"(d\d+)_(cro|no_cro)", dirname)
    return m.group(1) if m else None


def process_one_design(args):
    """Process a single design (all samples). Called by multiprocessing pool.

    args: (cro_dir_str, nocro_dir_str, group_name, fix_pos_str, crystal_cif)
    """
    cro_dir_str, nocro_dir_str, group_name, fix_pos_str, crystal_cif = args
    cro_dir = Path(cro_dir_str)
    nocro_dir = Path(nocro_dir_str)

    design_id = extract_design_id(cro_dir.name)
    if design_id is None:
        return []

    fix_pos = parse_fix_pos(fix_pos_str)

    parser = MMCIFParser(QUIET=True)

    # Load crystal (each worker loads its own copy)
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

        row = {
            "design_id": design_id,
            "group": group_name,
            "sample": sample_name,
            "cro_path": str(cro_path),
            "nocro_path": str(nocro_path),
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

            # Superimpose no_cro onto CRO using non-chromophore backbone
            sup1, ok1 = superimpose_on_backbone(
                cro_all, nocro_all, cro_nonchrom, nocro_nonchrom
            )
            if ok1:
                # Global all-atom RMSD (non-chromophore)
                gc, gm = collect_allatom_pairs(
                    cro_all, nocro_all, cro_nonchrom, nocro_nonchrom
                )
                row["global_allatom_rmsd"] = rmsd_after_sup(sup1, gc, gm)

                # Global backbone RMSD
                gbc, gbm = collect_bb_coord_pairs(
                    cro_all, nocro_all, cro_nonchrom, nocro_nonchrom
                )
                row["global_bb_rmsd"] = rmsd_after_sup(sup1, gbc, gbm)

                # Helix all-atom RMSD (non-chromophore helix sidechains)
                hc, hm = collect_allatom_pairs(
                    cro_all, nocro_all, helix_cro, helix_nocro
                )
                # Add chromophore all-atom (CRO vs TYG mapped)
                cc, cm = collect_cro_to_tyg_pairs(cro_all, nocro_all, fp7, fp7)
                row["chrom_allatom_rmsd"] = rmsd_after_sup(sup1, cc, cm)

                helix_full_c = hc + cc
                helix_full_m = hm + cm
                row["helix_allatom_rmsd"] = rmsd_after_sup(
                    sup1, helix_full_c, helix_full_m
                )

                # Helix backbone RMSD (for reference)
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

            # Superimpose design CRO onto crystal using non-CRO motif backbone
            sup2, ok2 = superimpose_on_backbone(
                crystal_all, cro_all, crystal_motif, design_motif
            )
            if ok2:
                # Motif all-atom RMSD (bb-only for CYS/MET mismatched positions)
                ma_c, ma_m = collect_allatom_pairs(
                    crystal_all, cro_all, crystal_motif, design_motif,
                    bb_only_fixed_resids=BACKBONE_ONLY_CRYSTAL,
                )
                row["motif_allatom_rmsd"] = rmsd_after_sup(sup2, ma_c, ma_m)

                # Motif backbone RMSD (for reference)
                mb_c, mb_m = collect_bb_coord_pairs(
                    crystal_all, cro_all, crystal_motif, design_motif
                )
                row["motif_bb_rmsd"] = rmsd_after_sup(sup2, mb_c, mb_m)

                # CRO-to-CRO all-atom RMSD
                cro_c, cro_m = collect_cro_to_cro_pairs(
                    cro_all, crystal_all, d_cro_resid, c_cro_resid
                )
                row["cro_allatom_rmsd"] = rmsd_after_sup(sup2, cro_c, cro_m)
            else:
                row["error_m2"] = "superimpose_failed"
        except Exception as e:
            row["error_m2"] = str(e)

        # Round RMSD values
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
# Tier assignment and selection
# ---------------------------------------------------------------------------
def assign_tiers(df):
    """Assign tier per prediction and best tier per design."""
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
    best = df.groupby("design_id")["pred_tier"].min().reset_index()
    best.columns = ["design_id", "best_tier"]
    df = df.merge(best, on="design_id")
    return df


def pick_representatives(df):
    """Pick one representative per design: best tier, lowest motif_allatom_rmsd."""
    rows = []
    for design_id, grp in df.groupby("design_id"):
        tier = grp["best_tier"].iloc[0]
        tier_preds = grp[grp["pred_tier"] == tier]
        valid = tier_preds.dropna(subset=["motif_allatom_rmsd"])
        if len(valid) > 0:
            best_idx = valid["motif_allatom_rmsd"].idxmin()
        else:
            best_idx = tier_preds.index[0]
        rows.append(best_idx)
    return df.loc[rows].copy()


def merge_water_hbond(selected, water_csv_path):
    """Merge water hbond data with selected designs."""
    if not water_csv_path.exists():
        print(f"  WARNING: water hbond CSV not found: {water_csv_path}")
        return selected

    water = pd.read_csv(water_csv_path)
    water["design_id"] = water["pdb_file"].str.extract(r"(d\d+)_")

    water_cols = [
        "design_id", "pdb_file",
        "n_water_hbonds_verified", "n_still_unsat_total",
        "native_water_filled_by_sc", "native_sc_filled",
        "n_unsatisfied_total_dry",
    ]
    available = [c for c in water_cols if c in water.columns]
    water_sub = water[available].copy()
    water_sub = water_sub.rename(columns={"pdb_file": "design_pdb_water"})

    merged = selected.merge(water_sub, on="design_id", how="left")
    return merged


def select_sequences(all_selected, max_total=1000):
    """Select up to max_total sequences across all groups."""
    tier_names = {1: "T1_both_pass", 2: "T2_m2_only", 3: "T3_m1_only", 4: "T4_both_fail"}
    all_selected["tier"] = all_selected["best_tier"].map(tier_names)
    all_selected = all_selected.sort_values(["best_tier", "motif_allatom_rmsd"])

    tier1 = all_selected[all_selected["best_tier"] == 1]
    tier2 = all_selected[all_selected["best_tier"] == 2]
    tier3 = all_selected[all_selected["best_tier"] == 3]
    tier4 = all_selected[all_selected["best_tier"] == 4]

    core_n = len(tier1) + len(tier2)
    remaining = max_total - core_n

    n_tier3 = min(len(tier3), max(remaining, 0))
    remaining -= n_tier3
    n_tier4 = min(len(tier4), max(remaining, 0))

    if n_tier4 < len(tier4) and n_tier4 > 0:
        tier4_sorted = tier4.sort_values("motif_allatom_rmsd")
        indices = np.linspace(0, len(tier4_sorted) - 1, n_tier4, dtype=int)
        tier4_sel = tier4_sorted.iloc[indices]
    else:
        tier4_sel = tier4.head(n_tier4)

    final = pd.concat([tier1, tier2, tier3.head(n_tier3), tier4_sel])
    return final.sort_values(["best_tier", "motif_allatom_rmsd"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(16, cpu_count()),
                    help="Number of parallel workers")
    args = ap.parse_args()

    print("Loading design CSV...")
    design_df = pd.read_csv(DESIGN_CSV)
    print(f"  {len(design_df)} design rows")

    # Pre-compute design_id -> fix_pos mapping
    design_df["_design_id"] = design_df["design_pdb"].str.extract(r"(d\d+)_")
    fix_pos_map = design_df.groupby("_design_id")["fix_pos"].first().to_dict()

    # Build work items: (cro_dir, nocro_dir, group_name, fix_pos_str, crystal_cif)
    work_items = []
    for group_name, (cro_sub, nocro_sub, _) in GROUPS.items():
        cro_base = RF3_BASE / cro_sub
        nocro_base = RF3_BASE / nocro_sub
        if not cro_base.exists() or not nocro_base.exists():
            print(f"WARNING: missing dirs for {group_name}")
            continue

        cro_dirs = sorted([d for d in cro_base.iterdir() if d.is_dir()])
        print(f"  {group_name}: {len(cro_dirs)} CRO design directories")

        for cro_dir in cro_dirs:
            design_id = extract_design_id(cro_dir.name)
            if design_id is None or design_id not in fix_pos_map:
                continue
            nocro_dir = nocro_base / f"{design_id}_no_cro"
            if not nocro_dir.exists():
                continue
            work_items.append((
                str(cro_dir), str(nocro_dir), group_name,
                fix_pos_map[design_id], CRYSTAL_CIF,
            ))

    print(f"\nTotal designs to process: {len(work_items)}")
    print(f"Using {args.workers} workers")

    # Process with multiprocessing
    all_rows = []
    with Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_one_design, work_items),
            total=len(work_items),
            desc="Processing designs",
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

    # Filter to valid predictions (have both metrics)
    valid = df.dropna(subset=["helix_allatom_rmsd", "motif_allatom_rmsd"])
    print(f"Valid predictions (both metrics): {len(valid)}")

    # Assign tiers
    valid = assign_tiers(valid)

    # Pick representatives
    selected = pick_representatives(valid)
    print(f"Unique designs with representatives: {len(selected)}")

    # Save per-group CSVs
    for group_name, (_, _, water_csv) in GROUPS.items():
        group_all = valid[valid["group"] == group_name]
        out_all = OUT_DIR / f"denovo_rmsd_all_{group_name}.csv"
        group_all.to_csv(out_all, index=False)
        print(f"\n  {group_name}: {len(group_all)} predictions -> {out_all}")

        group_sel = selected[selected["group"] == group_name].copy()
        group_sel = merge_water_hbond(group_sel, water_csv)
        out_sel = OUT_DIR / f"denovo_rmsd_selected_{group_name}.csv"
        group_sel.to_csv(out_sel, index=False)
        print(f"  {group_name}: {len(group_sel)} selected -> {out_sel}")

    # Merge all groups for final selection
    all_selected = []
    for group_name, (_, _, water_csv) in GROUPS.items():
        group_sel = selected[selected["group"] == group_name].copy()
        group_sel = merge_water_hbond(group_sel, water_csv)
        all_selected.append(group_sel)
    all_selected = pd.concat(all_selected, ignore_index=True)

    # Final selection (~1000)
    final = select_sequences(all_selected, max_total=1000)

    # Add sequence and energy from design CSV
    seq_map = design_df.groupby("_design_id")[["seq", "energy"]].first().reset_index()
    seq_map = seq_map.rename(columns={"_design_id": "design_id"})
    final = final.merge(seq_map, on="design_id", how="left")

    out_cols = [
        "tier", "design_id", "group", "sample", "seq", "energy",
        "helix_allatom_rmsd", "motif_allatom_rmsd",
        "chrom_allatom_rmsd", "cro_allatom_rmsd",
        "global_allatom_rmsd", "helix_bb_rmsd", "motif_bb_rmsd", "global_bb_rmsd",
        "cro_path", "nocro_path",
    ]
    water_cols = [
        "n_water_hbonds_verified", "n_still_unsat_total",
        "native_water_filled_by_sc", "native_sc_filled",
        "n_unsatisfied_total_dry",
    ]
    for c in water_cols:
        if c in final.columns:
            out_cols.append(c)

    # Keep only columns that exist
    out_cols = [c for c in out_cols if c in final.columns]

    out_path = OUT_DIR / "denovo_selected_sequences_for_experiment.csv"
    final[out_cols].to_csv(out_path, index=False)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"FINAL SELECTION: {len(final)} unique designs")
    print(f"{'=' * 70}")
    tier_order = {"T1_both_pass": 1, "T2_m2_only": 2, "T3_m1_only": 3, "T4_both_fail": 4}
    for t in sorted(final["tier"].unique(), key=lambda x: tier_order.get(x, 99)):
        sub = final[final["tier"] == t]
        print(f"\n  {t} (n={len(sub)}):")
        for col in ("helix_allatom_rmsd", "motif_allatom_rmsd",
                     "chrom_allatom_rmsd", "cro_allatom_rmsd"):
            v = sub[col].dropna()
            if len(v) > 0:
                print(f"    {col:25s}  mean={v.mean():.3f}  "
                      f"[{v.min():.3f} - {v.max():.3f}]")
        gc = sub["group"].value_counts()
        print(f"    Groups: {', '.join(f'{g}:{c}' for g, c in gc.items())}")

    print(f"\nSaved to {out_path}")

    # Sensitivity analysis
    print(f"\n{'=' * 70}")
    print("Sensitivity: designs passing metric 2 at various thresholds")
    print(f"{'=' * 70}")
    for thresh in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        n = valid[valid["motif_allatom_rmsd"] < thresh]["design_id"].nunique()
        print(f"  motif_allatom_rmsd < {thresh:.2f} Å:  {n} unique designs")


if __name__ == "__main__":
    main()
