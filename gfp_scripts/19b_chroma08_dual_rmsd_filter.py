"""Dual RMSD filter for chroma_08 de novo GFP scaffold designs.

Adapted from 19b_denovo_dual_rmsd_filter.py for the chroma_08 dataset.

Metric 1: no_cro vs CRO RMSD (pre-cyclic vs post-cyclic consistency)
Metric 2: CRO vs crystal (2AWK) motif RMSD (chromophore geometry fidelity)

Design-specific chromophore positions from fix_pos column in design_output CSVs.
RF3 predictions: consensus model + 5 samples per design.

All RMSD metrics are all-atom (sidechains included), with alignment on backbone.
For motif residues with mismatched types (CYS at crystal 70, MET->ARG at 96),
only backbone atoms contribute to the RMSD.

Tier assignment:
  Tier 1: helix_allatom_rmsd < 1A AND motif_allatom_rmsd < 1A  (best)
  Tier 2: helix_allatom_rmsd >= 1A AND motif_allatom_rmsd < 1A  (crystal-like CRO)
  Tier 3: helix_allatom_rmsd < 1A AND motif_allatom_rmsd >= 1A  (consistent but wrong CRO)
  Tier 4: both fail

Parallelized via SLURM array job: each task processes a subset of RF3 batch
directories.

Usage:
    python 19b_chroma08_dual_rmsd_filter.py --chunk-id 0 --num-chunks 10 [--workers N]
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

PLDDT_THRESHOLD = 0.70  # Skip predictions below this pLDDT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHROMA_DIR = Path("/scratch/users/tianyulu/gfp/denovo/chroma_08/chroma")
RF3_BASE = Path("/scratch/users/tianyulu/gfp/rf3_outputs/all_out")
CRYSTAL_CIF = "/scratch/users/tianyulu/gfp/rf3/2AWK.cif"
OUT_DIR = Path("/scratch/users/tianyulu/gfp/denovo/chroma_08")

CRO_SUBDIR = "chroma_08_cro"
NOCRO_SUBDIR = "chroma_08_no_cro"

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
    """Superimpose moving onto fixed using backbone of paired residues."""
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
    """All-atom coord pairs matched by atom name."""
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
    """Build non-chromophore residue mapping between no_cro and CRO structures."""
    fp7 = fix_pos[7]
    fp9 = fix_pos[9]
    before_nocro = list(range(1, fp7))
    before_cro = list(range(1, fp7))
    after_nocro = list(range(fp9 + 1, n_nocro + 1))
    after_cro = list(range(fp7 + 1, fp7 + 1 + len(after_nocro)))
    return before_nocro + after_nocro, before_cro + after_cro


def build_helix_correspondence(fix_pos):
    """Build helix non-chromophore residue pairs for no_cro vs CRO."""
    helix_nocro, helix_cro = [], []
    for i in range(2, 7):
        helix_nocro.append(fix_pos[i])
        helix_cro.append(fix_pos[i])
    for i in range(10, 14):
        helix_nocro.append(fix_pos[i])
        helix_cro.append(fix_pos[i] - 2)
    return helix_nocro, helix_cro


def build_crystal_motif_correspondence(fix_pos):
    """Build motif residue pairs for CRO design vs crystal."""
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
def read_plddt(cif_path):
    """Read overall_plddt from the summary_confidences.json next to a model CIF."""
    summary_path = Path(str(cif_path).replace("_model.cif", "_summary_confidences.json"))
    if not summary_path.exists():
        return None
    try:
        with open(summary_path) as f:
            return json.load(f).get("overall_plddt")
    except Exception:
        return None


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
    """Extract design base from directory name like 'p0_d2009_0_cro' -> 'p0_d2009_0'."""
    m = re.match(r"(.+?)_(cro|no_cro)$", dirname)
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

        # pLDDT filter: skip if either prediction is below threshold
        cro_plddt = read_plddt(cro_path)
        nocro_plddt = read_plddt(nocro_path)
        if cro_plddt is not None and cro_plddt < PLDDT_THRESHOLD:
            continue
        if nocro_plddt is not None and nocro_plddt < PLDDT_THRESHOLD:
            continue

        row = {
            "design_id": design_id,
            "group": group_name,
            "sample": sample_name,
            "cro_plddt": cro_plddt,
            "nocro_plddt": nocro_plddt,
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


# ---------------------------------------------------------------------------
# Build fix_pos mapping and discover design pairs
# ---------------------------------------------------------------------------
def build_fix_pos_map():
    """Build mapping from design base name (e.g. 'p0_d2009_0') to fix_pos string."""
    fix_pos_map = {}
    for part in range(6):
        csv_path = CHROMA_DIR / f"design_input_part{part}_design_output.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, usecols=["design_pdb", "fix_pos"])
        for _, row in df.iterrows():
            design_name = row["design_pdb"].replace(".pdb", "")
            dir_base = f"p{part}_{design_name}"
            fix_pos_map[dir_base] = row["fix_pos"]
    return fix_pos_map


def discover_design_pairs(batch_dirs):
    """Find all matching (cro_dir, nocro_dir, design_base) pairs in given batches."""
    cro_base = RF3_BASE / CRO_SUBDIR
    nocro_base = RF3_BASE / NOCRO_SUBDIR

    pairs = []
    for batch_name in batch_dirs:
        cro_batch = cro_base / batch_name
        nocro_batch = nocro_base / batch_name
        if not cro_batch.exists() or not nocro_batch.exists():
            continue

        for cro_dir in sorted(cro_batch.iterdir()):
            if not cro_dir.is_dir():
                continue
            m = re.match(r"(.+)_cro$", cro_dir.name)
            if not m:
                continue
            design_base = m.group(1)
            nocro_dir = nocro_batch / f"{design_base}_no_cro"
            if nocro_dir.exists():
                pairs.append((str(cro_dir), str(nocro_dir), design_base))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Dual RMSD filter for chroma_08 designs (array job chunk)."
    )
    ap.add_argument("--chunk-id", type=int, required=True,
                    help="Array task chunk index (0-based)")
    ap.add_argument("--num-chunks", type=int, default=10,
                    help="Total number of chunks")
    ap.add_argument("--workers", type=int, default=min(16, cpu_count()),
                    help="Number of parallel workers per chunk")
    args = ap.parse_args()

    print(f"Chunk {args.chunk_id}/{args.num_chunks}")

    # Discover all batch directories
    cro_base = RF3_BASE / CRO_SUBDIR
    all_batches = sorted([d.name for d in cro_base.iterdir() if d.is_dir()])
    print(f"Total batch directories: {len(all_batches)}")

    # Split batch directories across chunks
    chunk_size = len(all_batches) // args.num_chunks
    remainder = len(all_batches) % args.num_chunks
    # Distribute remainder: first 'remainder' chunks get one extra batch
    if args.chunk_id < remainder:
        start = args.chunk_id * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = remainder * (chunk_size + 1) + (args.chunk_id - remainder) * chunk_size
        end = start + chunk_size
    my_batches = all_batches[start:end]
    print(f"Processing batches: {my_batches[0]}..{my_batches[-1]} ({len(my_batches)} batches)")

    # Load fix_pos mapping
    print("Loading fix_pos mapping from design CSVs...")
    fix_pos_map = build_fix_pos_map()
    print(f"  {len(fix_pos_map)} designs in fix_pos map")

    # Discover design pairs in our batches
    print("Discovering design pairs...")
    pairs = discover_design_pairs(my_batches)
    print(f"  {len(pairs)} design pairs found")

    # Build work items
    work_items = []
    skipped = 0
    for cro_dir, nocro_dir, design_base in pairs:
        if design_base in fix_pos_map:
            work_items.append((
                cro_dir, nocro_dir, "chroma_08",
                fix_pos_map[design_base], CRYSTAL_CIF,
            ))
        else:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} designs (no fix_pos)")
    print(f"  {len(work_items)} designs to process")
    print(f"Using {args.workers} workers")

    if not work_items:
        print("No work items, exiting.")
        return

    # Process with multiprocessing
    all_rows = []
    with Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_one_design, work_items),
            total=len(work_items),
            desc=f"Chunk {args.chunk_id}",
        ))
    for result in results:
        all_rows.extend(result)

    if not all_rows:
        print("No predictions processed (all filtered by pLDDT < {:.0f}?)!".format(
            PLDDT_THRESHOLD * 100))
        return

    df = pd.DataFrame(all_rows)
    print(f"\nTotal predictions (pLDDT >= {PLDDT_THRESHOLD * 100:.0f}): {len(df)}")
    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        if n_err > 0:
            print(f"  Errors: {n_err}")

    # Save chunk CSV (all predictions)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_path = OUT_DIR / f"rmsd_chunk_{args.chunk_id:02d}.csv"
    df.to_csv(chunk_path, index=False)
    print(f"Saved chunk to {chunk_path}")

    # Quick summary
    valid = df.dropna(subset=["helix_allatom_rmsd", "motif_allatom_rmsd"])
    print(f"Valid predictions (both metrics): {len(valid)}")
    if len(valid) > 0:
        valid = assign_tiers(valid)
        for tier in sorted(valid["pred_tier"].unique()):
            n = (valid["pred_tier"] == tier).sum()
            n_designs = valid[valid["pred_tier"] == tier]["design_id"].nunique()
            print(f"  Tier {tier}: {n} predictions, {n_designs} unique designs")


if __name__ == "__main__":
    main()
