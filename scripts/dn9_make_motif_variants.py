"""Generate bb-O diversification variants for dn9 architector motifs.

For each base architector PDB (ASP + HOH atoms), produce variant PDBs that
re-route which oxygen atoms participate as backbone-O motif anchors versus
sidechain (acidic-tip) anchors. Diversification logic:

  - base   : keep PDB as-is (ASP -> acidic tip, HOH -> bb-O)
  - bbo1   : convert 1 most solvent-exposed ASP into HOH (its OD1 becomes bb-O)
  - bbo2   : convert 2 most solvent-exposed ASPs into HOH (drop 1 atom each)
  - drop1  : delete the most solvent-exposed coordinating atom entirely
             (the protein leaves that slot for a structural water)

The "solvent-exposed" judgment uses distance of CB/OD/O atoms from the
geometric centroid of the metal cluster's heavy atoms — outermost gets
selected first. Coarse but functional.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np


def parse_pdb(path: Path):
    lines = path.read_text().splitlines()
    records = []
    for ln in lines:
        if ln.startswith(("ATOM", "HETATM")):
            records.append({
                "raw": ln,
                "kind": ln[:6].strip(),
                "atom_num": int(ln[6:11]),
                "atom_name": ln[12:16].strip(),
                "resname": ln[17:20].strip(),
                "chain": ln[21],
                "resseq": int(ln[22:26]),
                "x": float(ln[30:38]),
                "y": float(ln[38:46]),
                "z": float(ln[46:54]),
                "element": ln[76:78].strip(),
            })
    return records


def group_by_residue(records):
    groups = {}
    for r in records:
        key = (r["chain"], r["resseq"], r["resname"])
        groups.setdefault(key, []).append(r)
    return groups


def _fmt_atom_name(atom_name: str, element: str) -> str:
    """Format atom name into 4-char PDB column 13-16 per convention."""
    if len(atom_name) >= 4:
        return atom_name[:4]
    # 1-3 char atom name; element 1-char gets leading space
    if len(element) == 1:
        return f" {atom_name:<3s}"
    return f"{atom_name:<4s}"


def write_pdb(path: Path, records, header=None):
    with path.open("w") as f:
        if header:
            f.write(header + "\n")
        for i, r in enumerate(records, start=1):
            kind = r["kind"]
            name = _fmt_atom_name(r["atom_name"], r["element"])
            line = (
                f"{kind:<6s}{i:>5d} {name}"  # cols 1-16
                f" {r['resname']:<3s} {r['chain']:1s}{r['resseq']:>4d}    "
                f"{r['x']:>8.3f}{r['y']:>8.3f}{r['z']:>8.3f}  1.00  0.00          "
                f"{r['element']:>2s}"
            )
            f.write(line + "\n")
        f.write("END\n")


def find_metal_center(records):
    metals = [r for r in records if r["kind"] == "HETATM" and r["resname"] != "HOH"]
    if not metals:
        return None
    return np.array([metals[0]["x"], metals[0]["y"], metals[0]["z"]])


def coordinating_atoms_of_residue(rg):
    """Return the heavy-atom coords that interact with the metal for a residue."""
    if rg[0]["resname"] == "ASP":
        atoms = [r for r in rg if r["atom_name"] in ("OD1", "OD2")]
    elif rg[0]["resname"] == "GLU":
        atoms = [r for r in rg if r["atom_name"] in ("OE1", "OE2")]
    elif rg[0]["resname"] == "HOH":
        atoms = [r for r in rg if r["atom_name"] == "O"]
    else:
        atoms = []
    return atoms


def furthest_residue_from(records, metal_xyz, residue_keys):
    """Among given residue keys, return the one whose coordinating atoms are
    closest to (i.e. *bigger* magnitude of "solvent exposed" in the architector
    snapshot — outermost from metal coordination radius)."""
    groups = group_by_residue(records)
    scores = []
    for k in residue_keys:
        atoms = coordinating_atoms_of_residue(groups[k])
        if not atoms:
            scores.append((-1, k))
            continue
        coords = np.array([[a["x"], a["y"], a["z"]] for a in atoms])
        dists = np.linalg.norm(coords - metal_xyz, axis=1)
        scores.append((float(dists.max()), k))
    scores.sort(reverse=True)
    return [k for _, k in scores]


def convert_asp_to_hoh(records, key_to_convert):
    """Return a new list of records where the ASP residue at key_to_convert
    has been replaced with a single HOH residue at the location of its OD1."""
    groups = group_by_residue(records)
    rg = groups[key_to_convert]
    od1 = next((r for r in rg if r["atom_name"] == "OD1"), None)
    if od1 is None:
        return records
    new_records = []
    for r in records:
        k = (r["chain"], r["resseq"], r["resname"])
        if k == key_to_convert:
            continue
        new_records.append(r)
    # Append HOH
    new_records.append({
        "raw": "",
        "kind": "HETATM",
        "atom_num": 0,
        "atom_name": "O",
        "resname": "HOH",
        "chain": od1["chain"],
        "resseq": od1["resseq"],
        "x": od1["x"], "y": od1["y"], "z": od1["z"],
        "element": "O",
    })
    return new_records


def drop_residue(records, key_to_drop):
    """Remove a residue entirely (treat as solvent)."""
    return [r for r in records if (r["chain"], r["resseq"], r["resname"]) != key_to_drop]


def generate_variants(input_pdb: Path, output_dir: Path):
    records = parse_pdb(input_pdb)
    # Keep only heavy atoms (no H)
    records = [r for r in records if r["element"] != "H"]
    metal_xyz = find_metal_center(records)
    if metal_xyz is None:
        print(f"[skip] {input_pdb.name}: no metal found")
        return []

    groups = group_by_residue(records)
    asp_keys = [k for k in groups if k[2] == "ASP"]
    hoh_keys = [k for k in groups if k[2] == "HOH"]

    # Rank ASPs by furthest-coordinating-atom from metal (proxy: bigger distance = more solvent-exposed-edge)
    ranked_asp = furthest_residue_from(records, metal_xyz, asp_keys)

    base_name = input_pdb.stem
    variants = []

    # base — unmodified (HOH retained as bb-O)
    write_pdb(output_dir / f"{base_name}__base.pdb", records, header=f"# variant: base of {base_name}")
    variants.append(f"{base_name}__base")

    # bbo1 — convert most-outward ASP to HOH (one more bb-O anchor)
    if len(ranked_asp) >= 2:
        rec_v = convert_asp_to_hoh(records, ranked_asp[0])
        write_pdb(output_dir / f"{base_name}__bbo1.pdb", rec_v, header=f"# variant: bbo1 (asp {ranked_asp[0]} -> HOH) of {base_name}")
        variants.append(f"{base_name}__bbo1")

    # bbo2 — convert 2 most-outward ASPs to HOH
    if len(ranked_asp) >= 3:
        rec_v = convert_asp_to_hoh(records, ranked_asp[0])
        rec_v = convert_asp_to_hoh(rec_v, ranked_asp[1])
        write_pdb(output_dir / f"{base_name}__bbo2.pdb", rec_v, header=f"# variant: bbo2 of {base_name}")
        variants.append(f"{base_name}__bbo2")

    # drop1 — delete one of the HOHs (or most outer ASP if no HOH) — fully solvent
    if hoh_keys:
        # Drop the HOH furthest from metal — leave site for solvent
        ranked_hoh = furthest_residue_from(records, metal_xyz, hoh_keys)
        rec_v = drop_residue(records, ranked_hoh[0])
        write_pdb(output_dir / f"{base_name}__drop1.pdb", rec_v, header=f"# variant: drop1 (drop {ranked_hoh[0]}) of {base_name}")
        variants.append(f"{base_name}__drop1")

    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_variants = []
    for pdb in sorted(args.motif_dir.glob("*.pdb")):
        if "__" in pdb.stem:  # skip already-generated variants
            continue
        vs = generate_variants(pdb, args.output_dir)
        all_variants.extend(vs)
    print(f"\nGenerated {len(all_variants)} variant motifs in {args.output_dir}")
    for v in all_variants:
        print(f"  {v}")


if __name__ == "__main__":
    main()
