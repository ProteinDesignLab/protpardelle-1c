"""Prepare 8FNS for protpardelle cc89 overfitting.

Creates:
1. A PDB file with the 8FNS protein + 4 Nd metals as GLY pseudo-residues on chain B
2. A data directory structure matching augmented_ingraham_cath_bugfree
3. The train_designable_pdb_keys.list file
"""

import argparse
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cif-path",
        type=str,
        default="/home/users/tianyulu/code/rhaven/data/metal_diffusion/8FNS.cif",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/users/tianyulu/code/protpardelle-1c/data/8fns_overfit_augmented_ingraham_cath_bugfree",
    )
    args = parser.parse_args()

    cif_path = args.cif_path
    out_dir = Path(args.out_dir)

    # Parse CIF
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("8fns", cif_path)
    model = next(structure.get_models())

    # Extract protein residues and metal atoms
    protein_residues = []
    metal_coords = []

    for chain in model:
        for res in chain:
            if res.id[0] == " ":  # standard residue
                protein_residues.append(res)
            elif res.id[0].startswith("H_"):  # HETATM
                resname = res.get_resname().strip()
                if resname == "ND":  # Nd³⁺
                    for atom in res:
                        metal_coords.append(atom.get_vector().get_array())

    print(f"Protein residues: {len(protein_residues)}")
    print(f"Metal atoms (Nd): {len(metal_coords)}")

    # Write PDB with protein chain A + metals as GLY pseudo-residues on chain B
    pdb_lines = []
    atom_num = 1
    res_num = 1

    # Backbone atom names for GLY: N, CA, C, O
    bb_atoms = ["N", "CA", "C", "O"]

    def fmt_atom_line(atom_num, atom_name, resname, chain_id, res_num, coord, element):
        """Format a PDB ATOM line with proper column alignment."""
        # Columns: 1-6 record, 7-11 serial, 13-16 name, 17 altloc, 18-20 resname, 22 chain, 23-26 resseq
        if len(atom_name) < 4:
            name_field = f" {atom_name:<3s}"
        else:
            name_field = f"{atom_name:<4s}"
        return (
            f"ATOM  {atom_num:5d} {name_field} {resname:>3s} {chain_id}{res_num:4d}    "
            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
            f"{1.0:6.2f}{0.0:6.2f}          {element:>2s}  \n"
        )

    for res in protein_residues:
        for atom in res:
            name = atom.get_name()
            coord = atom.get_vector().get_array()
            resname = res.get_resname()
            line = fmt_atom_line(atom_num, name, resname, "A", res_num, coord, name[0])
            pdb_lines.append(line)
            atom_num += 1
        res_num += 1

    pdb_lines.append("TER\n")

    # Add metals as GLY pseudo-residues on chain B
    # Each metal gets its coord placed at the CA position; N, C, O are set to the same coord
    for mi, mcoord in enumerate(metal_coords):
        metal_res_num = mi + 1
        for aname in bb_atoms:
            line = fmt_atom_line(atom_num, aname, "GLY", "B", metal_res_num, mcoord, aname[0])
            pdb_lines.append(line)
            atom_num += 1

    pdb_lines.append("TER\n")
    pdb_lines.append("END\n")

    pdb_content = "".join(pdb_lines)

    # Create directory structure matching augmented_ingraham_cath_bugfree
    mpnn_dir = out_dir / "mpnn_esmfold"
    mpnn_dir.mkdir(parents=True, exist_ok=True)

    pdb_filename = "8fns_with_metals.pdb"
    pdb_path = mpnn_dir / pdb_filename
    with open(pdb_path, "w") as f:
        f.write(pdb_content)

    # Also save as the motif PDB for sampling (partial diffusion input)
    motif_dir = out_dir / "motifs"
    motif_dir.mkdir(parents=True, exist_ok=True)
    motif_pdb_path = motif_dir / "8fns_with_metals.pdb"
    with open(motif_pdb_path, "w") as f:
        f.write(pdb_content)

    # Create train_designable_pdb_keys.list (overfit = repeat key many times)
    n_repeats = 1000  # large enough for the dataset to have many "examples"
    keys_path = out_dir / "train_designable_pdb_keys.list"
    with open(keys_path, "w") as f:
        for _ in range(n_repeats):
            f.write(f"{pdb_filename}\n")

    print(f"\nCreated data directory: {out_dir}")
    print(f"  PDB: {pdb_path}")
    print(f"  Motif PDB: {motif_pdb_path}")
    print(f"  Keys: {keys_path} ({n_repeats} repeats)")

    # Verify by loading with protpardelle's reader
    from protpardelle.data.pdb_io import load_feats_from_pdb
    feats, hetero = load_feats_from_pdb(str(pdb_path), chain_residx_gap=200)
    print(f"\nVerification:")
    print(f"  Residues: {feats['aatype'].shape[0]}")
    print(f"  Chains: {feats['chain_index'].unique().tolist()}")
    print(f"  Residue index range: {feats['residue_index'].min().item()} - {feats['residue_index'].max().item()}")
    print(f"  Atom mask sum: {feats['atom_mask'].sum().item()}")


if __name__ == "__main__":
    main()
