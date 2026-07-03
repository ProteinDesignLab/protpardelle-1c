"""Walk a protpardelle-1c sampling output directory tree and produce a manifest CSV.

For each sample PDB, records:
- sample_path:           absolute path to sample_*.pdb
- motif_name:            motif id (e.g. M0024_1nzy_specific_atoms)
- model_name:            model used (e.g. cc138)
- sample_idx:            0-based index within the (model, motif) batch
- scaffold_length:       number of residues in the sample
- motif_pdb_path:        absolute path to the source motif PDB (loaded by sample.py)
- motif_residues_pdb:    comma-separated original residue numbers in the motif PDB
                          (e.g. "33,60,62,65,123,215" for M0050 singlechain)
- scaffold_motif_resnums: comma-separated 1-indexed PDB residue numbers in the
                          sample where motif residues were placed
                          (e.g. "13,26,36,65,74,101")
- contig_order:          comma-separated motif residue numbers in contig order
                          (matches scaffold_motif_resnums positionally)
- ligands:               comma-separated CCD codes from ame.csv (e.g. "ADP,MG,3PG")
- conditioning_atoms:    serialized per-residue atom selections from ame.csv

This manifest is the input to all downstream eval steps. Run once after sampling
finishes; subsequent steps read from it. Idempotent.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict


def parse_atoms_field(s: str) -> dict:
    """Parse 'A64: [O, C]; A86: [CB, CA, N, C]; ...' -> dict[(chain, resnum)] -> list[atom]"""
    out = {}
    for piece in s.split(";"):
        piece = piece.strip()
        if not piece:
            continue
        m = re.match(r"([A-Z])(\d+):\s*\[([^\]]*)\]", piece)
        if not m:
            continue
        chain, resnum = m.group(1), int(m.group(2))
        atoms = [a.strip() for a in m.group(3).split(",") if a.strip()]
        out[(chain, resnum)] = atoms
    return out


def parse_motif_placement(placement_str: str, motif_residue_count: int) -> list[int]:
    """Convert a placement string like '12/A/12/A/9/A/16/A/27' into a list of
    0-indexed scaffold positions where the motif residues were placed.
    """
    if not placement_str or placement_str == "unindexed":
        return []
    parts = placement_str.split("/")
    positions = []
    idx = 0
    for p in parts:
        if p.isalpha():
            positions.append(idx)
            idx += 1
        else:
            try:
                idx += int(p)
            except ValueError:
                continue
    return positions


def parse_matched_motif_idx(s: str) -> list[int]:
    """For unindexed runs: 'matched_motif_idx' is comma-separated 1-indexed PDB residue numbers."""
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def get_motif_residues_in_pdb_order(motif_pdb: Path) -> list[tuple[str, int]]:
    """Return list of (chain, resnum) for unique residues in the motif PDB,
    in the order they appear (matches model.sample's load order)."""
    seen = []
    seen_set = set()
    with open(motif_pdb) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            chain = line[21]
            try:
                rn = int(line[22:26])
            except ValueError:
                continue
            key = (chain, rn)
            if key not in seen_set:
                seen_set.add(key)
                seen.append(key)
    return seen


def parse_contig(contig: str) -> list[tuple[str, int]]:
    """From '5-30;A86-86;5-30;A88-88;...' return [(A, 86), (A, 88), ...]
    in contig order. For multi-chain contigs (';/;'-separated), flattens."""
    out = []
    for chain_seg in contig.split(";/;"):
        for piece in chain_seg.split(";"):
            piece = piece.strip()
            m = re.match(r"^([A-Z])(\d+)-(\d+)$", piece)
            if m:
                chain = m.group(1)
                start = int(m.group(2))
                end = int(m.group(3))
                for rn in range(start, end + 1):
                    out.append((chain, rn))
    return out


def load_csv_metadata(csv_path: Path) -> dict:
    """Load ame.csv -> dict keyed by motif_name (e.g. 'M0024_1nzy_specific_atoms')."""
    meta = {}
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            m_id = row["M-CSA ID"]
            pdb_id = row["PDB ID"]
            atoms_field = row["Selected motif residue atoms"]
            ligands = row["Ligands"]
            for variant in (
                f"{m_id}_{pdb_id}_specific_atoms",
                f"{m_id}_{pdb_id}_specific_atoms_singlechain",
            ):
                meta[variant] = {
                    "m_id": m_id,
                    "pdb_id": pdb_id,
                    "pdb_id_lower": pdb_id.lower(),
                    "ligands": [x.strip() for x in ligands.split(",") if x.strip()],
                    "conditioning_atoms": parse_atoms_field(atoms_field),
                }
    return meta


def parse_pdb_residue_index(pdb_path: Path) -> list[int]:
    """Return list of unique residue numbers in the PDB (in order of first appearance)."""
    seen, seen_set = [], set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                rn = int(line[22:26])
            except ValueError:
                continue
            if rn not in seen_set:
                seen_set.add(rn)
                seen.append(rn)
    return seen


def count_ligand_stoichiometry(source_pdb: Path,
                                motif_residue_keys: list[tuple[str, int]],
                                ligand_ccds: list[str],
                                cutoff_ang: float = 8.0) -> dict[str, int]:
    """Count how many residues of each CCD code in `ligand_ccds` have any atom
    within `cutoff_ang` Å of any motif residue's atoms in the source PDB.

    Returns dict[ccd -> count]. CCD codes not found get count=0.
    """
    import numpy as np
    motif_set = set(motif_residue_keys)  # set of (chain, resnum)
    motif_atoms = []      # list of np.array([x,y,z])
    hetatm_by_ccd = {}    # ccd -> dict[(chain, resnum) -> list of xyz]

    ligand_ccds_norm = [c.strip().upper() for c in ligand_ccds]
    if not ligand_ccds_norm:
        return {}

    with open(source_pdb) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                resname = line[17:20].strip().upper()
                chain = line[21]
                rn = int(line[22:26])
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except Exception:
                continue
            if line.startswith("ATOM"):
                # Motif residues — collect all atom coords
                if (chain, rn) in motif_set:
                    motif_atoms.append(xyz)
            elif resname in ligand_ccds_norm:
                hetatm_by_ccd.setdefault(resname, {}).setdefault((chain, rn), []).append(xyz)

    if not motif_atoms:
        # No motif anchor — return 1 each as fallback
        return {c: 1 for c in ligand_ccds_norm}

    motif_atoms_arr = np.array(motif_atoms)
    counts = {}
    for ccd in ligand_ccds_norm:
        n_in_active_site = 0
        for (chain, rn), atom_xyzs in hetatm_by_ccd.get(ccd, {}).items():
            # Check if any atom of this ligand instance is within cutoff of any motif atom
            atom_arr = np.array(atom_xyzs)
            d = np.linalg.norm(
                atom_arr[:, None, :] - motif_atoms_arr[None, :, :], axis=-1
            ).min()
            if d <= cutoff_ang:
                n_in_active_site += 1
        counts[ccd] = n_in_active_site if n_in_active_site > 0 else 1
    return counts


def load_contigs_from_yaml(yaml_path: Path) -> dict:
    """Return dict[motif_name -> contig_string] from a sampling YAML."""
    import yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    motifs = cfg.get("motifs", [])
    contigs = cfg.get("motif_contigs", [])
    return {m: c for m, c in zip(motifs, contigs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True,
                    help="Top-level protpardelle output dir (e.g. ame_cc138/)")
    ap.add_argument("--motif-dir", required=True,
                    help="Directory containing the motif PDBs used by the sampling run")
    ap.add_argument("--csv", required=True,
                    help="ame.csv path")
    ap.add_argument("--yaml", default=None,
                    help="Sampling YAML — used to recover contig order per motif. "
                         "If not given, falls back to PDB order (may misalign for "
                         "indexed motifs whose contig reorders residues).")
    ap.add_argument("--source-pdb-dir",
                    default="/home/users/tianyulu/code/protpardelle-1c/examples/motifs/ame/pdb_files",
                    help="Directory of source PDBs (used to count ligand stoichiometry).")
    ap.add_argument("--ligand-cutoff-ang", type=float, default=8.0,
                    help="Distance to any motif atom for a ligand instance to count "
                         "as active-site occupant.")
    ap.add_argument("--out", required=True, help="Output manifest CSV")
    args = ap.parse_args()

    yaml_contigs = load_contigs_from_yaml(Path(args.yaml)) if args.yaml else {}
    source_pdb_dir = Path(args.source_pdb_dir)

    # Cache ligand stoichiometry per motif (computed once)
    stoich_cache: dict[str, dict[str, int]] = {}

    sample_dir = Path(args.sample_dir)
    motif_dir = Path(args.motif_dir)
    meta = load_csv_metadata(Path(args.csv))

    rows = []
    # sample_dir typically contains one or more "<model>-epoch<N>-..." subdirs,
    # each containing per-motif subdirs.
    for model_root in sorted(sample_dir.iterdir()):
        if not model_root.is_dir():
            continue
        # Parse model name + epoch from dir name: e.g. cc138-epoch248-...
        m = re.match(r"^([^-]+(?:-[^-]+)*?)-epoch(\d+)-", model_root.name)
        model_name = m.group(1) if m else model_root.name
        for motif_dir_path in sorted(model_root.iterdir()):
            if not motif_dir_path.is_dir():
                continue
            motif_name = motif_dir_path.name
            scaffold_csv = motif_dir_path / "scaffold_info.csv"
            if not scaffold_csv.exists():
                continue

            motif_pdb = motif_dir / f"{motif_name}.pdb"
            if not motif_pdb.exists():
                continue

            with open(scaffold_csv) as f:
                rdr = csv.DictReader(f)
                placements = list(rdr)

            for row in placements:
                idx = int(row["sample_num"])
                placement = row.get("motif_placements", "")

                sample_pdb = motif_dir_path / f"sample_{motif_name}_{idx:02d}.pdb"
                if not sample_pdb.exists():
                    continue

                # Determine motif residue placement in scaffold
                csv_meta = meta.get(motif_name, {})
                cond_atoms = csv_meta.get("conditioning_atoms", {})

                # contig_order = (chain, resnum) tuples in the order the contig
                # listed them. We need this to map placements back to motif residues.
                # For unindexed runs, we use matched_motif_idx (1-indexed PDB resnums).
                contig_order = []
                scaffold_motif_resnums = []

                if placement == "unindexed":
                    matched = parse_matched_motif_idx(row.get("matched_motif_idx", ""))
                    scaffold_motif_resnums = matched
                    motif_residues_in_order = get_motif_residues_in_pdb_order(motif_pdb)
                    # for unindexed, the linear_sum_assignment output ordering
                    # is the order matched_motif_idx is written. But which
                    # motif residue corresponds to which entry isn't recorded;
                    # use PDB order as a best-effort label.
                    contig_order = motif_residues_in_order
                else:
                    n_motif = sum(1 for p in placement.split("/") if p.isalpha())
                    pos_0idx = parse_motif_placement(placement, n_motif)
                    scaffold_motif_resnums = [p + 1 for p in pos_0idx]
                    # Recover contig order from the YAML if provided; otherwise
                    # fall back to PDB order (incorrect for reordered contigs).
                    contig = yaml_contigs.get(motif_name)
                    if contig and contig != "unindexed":
                        contig_order = parse_contig(contig)
                    else:
                        contig_order = get_motif_residues_in_pdb_order(motif_pdb)

                # Compute ligand stoichiometry from source PDB (cached per motif).
                # cond_atoms is dict[(chain, resnum) -> list[atom]] from parse_atoms_field.
                if motif_name not in stoich_cache:
                    pdb_id_lower = csv_meta.get("pdb_id_lower")
                    src_pdb = (source_pdb_dir / f"{pdb_id_lower}.pdb") if pdb_id_lower else None
                    if src_pdb and src_pdb.exists() and cond_atoms:
                        motif_residue_keys = list(cond_atoms.keys())
                        stoich_cache[motif_name] = count_ligand_stoichiometry(
                            src_pdb, motif_residue_keys,
                            csv_meta.get("ligands", []),
                            cutoff_ang=args.ligand_cutoff_ang,
                        )
                    else:
                        stoich_cache[motif_name] = {c.upper(): 1 for c in csv_meta.get("ligands", [])}
                stoich = stoich_cache[motif_name]

                rows.append({
                    "model_name": model_name,
                    "motif_name": motif_name,
                    "sample_idx": idx,
                    "sample_path": str(sample_pdb.resolve()),
                    "scaffold_length": len(parse_pdb_residue_index(sample_pdb)),
                    "motif_pdb_path": str(motif_pdb.resolve()),
                    "scaffold_motif_resnums": ",".join(str(x) for x in scaffold_motif_resnums),
                    "contig_order": ",".join(f"{c}{r}" for c, r in contig_order),
                    "ligands": ",".join(csv_meta.get("ligands", [])),
                    "ligand_stoichiometry_json": json.dumps(stoich),
                    "conditioning_atoms_json": json.dumps(
                        {f"{c}{r}": atoms for (c, r), atoms in cond_atoms.items()}
                    ),
                })

    if not rows:
        print(f"WARN: no samples found under {sample_dir}")
        return

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
