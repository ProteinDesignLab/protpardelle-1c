"""PDB file input/output.

Authors: Alex Chu, Jinho Kim, Richard Shuai, Tianyu Lu, Zhaoyang Li
"""

import numpy as np
import torch
from Bio.PDB import PDBParser
from einops import rearrange
from torchtyping import TensorType

from protpardelle.common import residue_constants
from protpardelle.data import protein
from protpardelle.data.atom import (
    atom37_coords_to_atom14,
    atom37_mask_from_aatype,
    atom37_to_atom73,
)


def add_chain_gap(
    residue_index: TensorType["n"],
    chain_index: TensorType["n"],
    chain_residx_gap: int = 200,
) -> TensorType["n"]:
    """
    Add a residue index gap between chains.
    e.g. if chain A has 5 residues and chain B has 3 residues,
    with chain_residx_gap=200, the residue indices for chain B will be 206, 207, 208.

    Iteratively shift residue_index per chain, starting from last residx from previous chain
    """
    if torch.sum(chain_index) == 0:
        return residue_index

    for ci in range(1, int(torch.max(chain_index).item()) + 1):
        curr_chain_pos = torch.nonzero(chain_index == ci).flatten()
        prev_chain_residue_idx_end = residue_index[curr_chain_pos[0] - 1]
        curr_chain_residue_idx_start = residue_index[curr_chain_pos[0]]
        target_start = prev_chain_residue_idx_end + chain_residx_gap
        residx_delta = target_start - curr_chain_residue_idx_start
        # special case when chain_residx_gap is zero: keep original indices for each chain, reset to 1
        if chain_residx_gap == 0:
            residue_index[curr_chain_pos] = (
                residue_index[curr_chain_pos] - residue_index[curr_chain_pos[0]] + 1
            )
        else:
            residue_index[curr_chain_pos] = residue_index[curr_chain_pos] + residx_delta

    return residue_index


def load_feats_from_pdb(
    pdb,
    bb_atoms=["N", "CA", "C", "O"],
    load_atom73=False,
    chain_residx_gap: int = 200,
    chain_id: str | None = None,
    atom14: bool = False,
    include_pos_feats: bool = False,
):
    """
    Load model input features from a PDB file or mmcif file.
    - bb_atoms: list of backbone atom names to load
    - load_atom73: if True, also load atom73 features
    - chain_residx_gap: residue index gap for chain breaks for PDBs with multiple chains
    - include_pos_feats: if True, include chain_id_mapping and residue_index_orig in feats for specifying specific positions
    """
    feats = {}
    protein_obj, hetero_obj, chain_id_mapping = protein.read_pdb(pdb, chain_id=chain_id)
    bb_idxs = [residue_constants.atom_order[a] for a in bb_atoms]
    bb_coords = torch.from_numpy(protein_obj.atom_positions[:, bb_idxs])
    feats["bb_coords"] = bb_coords.float()
    for k, v in vars(protein_obj).items():
        feats[k] = torch.Tensor(v)
    feats["aatype"] = feats["aatype"].long()
    if load_atom73:
        feats["atom73_coords"], feats["atom73_mask"] = atom37_to_atom73(
            feats["atom_positions"], feats["aatype"], return_mask=True
        )
    if atom14:
        feats["atom_positions"], feats["b_factors"], feats["atom_mask"] = (
            atom37_coords_to_atom14(
                feats["atom_positions"],
                feats["b_factors"],
                feats["aatype"],
            )
        )

    # For users to specify conditioning: keep track of original residx and mapping of chain ID to chain index
    if include_pos_feats:
        feats["residue_index_orig"] = feats["residue_index"].clone()
        feats["chain_id_mapping"] = chain_id_mapping

    # Always start first residx from 1 (shouldn't matter with relative pos encoding though)
    feats["residue_index"] = (
        feats["residue_index"] - torch.min(feats["residue_index"]) + 1
    )

    # Handle residue index for PDBs with multiple chains
    feats["residue_index"] = add_chain_gap(
        feats["residue_index"], feats["chain_index"], chain_residx_gap=chain_residx_gap
    )

    return feats, hetero_obj


def feats_to_pdb_str(
    atom_positions,
    aatype=None,
    atom_mask=None,
    residue_index=None,
    chain_index=None,
    b_factors=None,
    atom_lines_only=True,
):
    # Expects unbatched, cropped inputs. needs at least one of atom_mask, aatype
    # Uses all-GLY aatype if aatype not given: does not infer from atom_mask
    assert aatype is not None or atom_mask is not None
    if atom_mask is None:
        aatype = aatype.cpu()
        atom_mask = atom37_mask_from_aatype(aatype, torch.ones_like(aatype))
    if aatype is None:
        seq_mask = atom_mask[:, residue_constants.atom_order["CA"]].cpu()
        aatype = seq_mask * residue_constants.restype_order["G"]
    if residue_index is None:
        residue_index = torch.arange(aatype.shape[-1]) + 1  # start residue index from 1
    if chain_index is None:
        chain_index = torch.ones_like(aatype)
    if b_factors is None:
        b_factors = torch.ones_like(atom_mask)

    cast = lambda x: np.array(x.detach().cpu()) if isinstance(x, torch.Tensor) else x
    prot = protein.Protein(
        atom_positions=cast(atom_positions),
        atom_mask=cast(atom_mask),
        aatype=cast(aatype),
        residue_index=cast(residue_index),
        chain_index=cast(chain_index),
        b_factors=cast(b_factors),
    )
    pdb_str = protein.to_pdb(prot)

    if atom_lines_only:
        pdb_lines = pdb_str.split("\n")
        atom_lines = [
            line
            for line in pdb_lines
            if len(line.split()) > 1 and line.split()[0] == "ATOM"
        ]
        pdb_str = "\n".join(atom_lines) + "\n"

    return pdb_str


def bb_coords_to_pdb_str(
    coords,
    atoms=["N", "CA", "C", "O"],
    chain_index: TensorType["n"] | None = None,
    aatype: TensorType["n"] | None = None,
):
    """
    Save backbone coords to pdb string.
    - chain_index: 0-indexed chain index for each residue, starting from A
    - aatype: aatype for each residue (if not specified, default to GLY)
    """

    def _bb_pdb_line(atom, atomnum, resnum, chain_idx, coords, elem, res="GLY"):
        atm = "ATOM".ljust(6)
        atomnum = str(atomnum).rjust(5)
        atomname = atom.center(4)
        resname = res.ljust(3)
        chain = chr(ord("A") + chain_idx).rjust(1)
        resnum = str(resnum).rjust(4)
        x = str("%8.3f" % (float(coords[0]))).rjust(8)
        y = str("%8.3f" % (float(coords[1]))).rjust(8)
        z = str("%8.3f" % (float(coords[2]))).rjust(8)
        occ = str("%6.2f" % (float(1))).rjust(6)
        temp = str("%6.2f" % (float(20))).ljust(6)
        elname = elem.rjust(12)
        return "%s%s %s %s %s%s    %s%s%s%s%s%s\n" % (
            atm,
            atomnum,
            atomname,
            resname,
            chain,
            resnum,
            x,
            y,
            z,
            occ,
            temp,
            elname,
        )

    n = coords.shape[0]
    na = len(atoms)
    pdb_str = ""
    res_counter = {}
    for j in range(0, n, na):
        for idx, atom in enumerate(atoms):
            residx = j // na  # 0-indexed residue index

            # Handle aatype
            if aatype is not None:
                # save with aatype if specified
                restype = residue_constants.restypes[aatype[residx].item()]
                resname = residue_constants.restype_1to3[restype]
            else:
                # otherwise, default to GLY
                resname = "GLY"

            # Handle chain index
            if chain_index is not None:
                chain_idx = chain_index[residx].long().item()
            else:
                chain_idx = 0

            if chain_idx not in res_counter:
                res_counter[chain_idx] = 1
            resnum = res_counter[chain_idx]

            pdb_str += _bb_pdb_line(
                atom,
                j + idx + 1,
                resnum,
                chain_idx,
                coords[j + idx],
                atom[0],
                resname,
            )
        res_counter[chain_idx] += 1
    return pdb_str


def write_pdb_str(pdb_str, filename, append=False, write_to_frames=False):
    write_mode = "a" if append else "w"
    with open(filename, write_mode) as f:
        if write_to_frames:
            f.write("MODEL\n")
        f.write(pdb_str)
        if write_to_frames:
            f.write("ENDMDL\n")


def load_coords_from_pdb(
    pdb,
    atoms=["N", "CA", "C", "O"],
    method="raw",
    also_bfactors=False,
    normalize_bfactors=True,
):
    """Returns array of shape (1, n_res, len(atoms), 3)"""
    coords = []
    bfactors = []
    if method == "raw":  # Raw numpy implementation, faster than biopdb
        # Indexing into PDB format, allowing XXXX.XXX
        coords_in_pdb = [slice(30, 38), slice(38, 46), slice(46, 54)]
        # Indexing into PDB format, allowing XXX.XX
        bfactor_in_pdb = slice(60, 66)

        with open(pdb, "r") as f:
            resi_prev = 1
            counter = 0
            for l in f:
                l_split = l.rstrip("\n").split()
                if len(l_split) > 0 and l_split[0] == "ATOM" and l_split[2] in atoms:
                    resi = l_split[5]
                    if resi == resi_prev:
                        counter += 1
                    else:
                        counter = 0
                    if counter < len(atoms):
                        xyz = [
                            np.array(l[s].strip()).astype(float) for s in coords_in_pdb
                        ]
                        coords.append(xyz)
                        if also_bfactors:
                            bfactor = np.array(l[bfactor_in_pdb].strip()).astype(float)
                            bfactors.append(bfactor)
                    resi_prev = resi
            coords = torch.Tensor(np.array(coords)).view(1, -1, len(atoms), 3)
            if also_bfactors:
                bfactors = torch.Tensor(np.array(bfactors)).view(1, -1, len(atoms))
    elif method == "biopdb":
        structure = PDBParser(QUIET=True).get_structure(pdb[:-3], pdb)
        for model in structure:
            for chain in model:
                for res in chain:
                    for atom in atoms:
                        try:
                            coords.append(np.asarray(res[atom].get_coord()))
                            if also_bfactors:
                                bfactors.append(np.asarray(res[atom].get_bfactor()))
                        except:
                            continue
    else:
        raise NotImplementedError(f"Invalid method for reading coords: {method}")
    if also_bfactors:
        if normalize_bfactors:  # Normalize over Calphas
            mean_b = bfactors[..., 1].mean()
            std_b = bfactors[..., 1].var().sqrt()
            bfactors = (bfactors - mean_b) / (std_b + 1e-6)
        return coords, bfactors
    return coords


def write_coords_to_pdb(
    coords_in,
    filename,
    batched=True,
    write_to_frames=False,
    **all_atom_feats,
):
    if not (batched or write_to_frames):
        coords_in = [coords_in]
        filename = [filename]
        all_atom_feats = {k: [v] for k, v in all_atom_feats.items()}

    n_atoms_in = coords_in[0].shape[-2]
    is_bb_or_ca_pdb = n_atoms_in <= 4
    for i, c in enumerate(coords_in):
        n_res = c.shape[0]
        if isinstance(filename, list):
            fname = filename[i]
        elif write_to_frames or len(coords_in) == 1:
            fname = filename
        else:
            fname = f"{filename[:-4]}_{i}.pdb"

        if is_bb_or_ca_pdb:
            c_flat = rearrange(c, "n a c -> (n a) c")
            if n_atoms_in == 1:
                atoms = ["CA"]
            if n_atoms_in == 3:
                atoms = ["N", "CA", "C"]
            if n_atoms_in == 4:
                atoms = ["N", "CA", "C", "O"]
            feats_i = {k: v[i][:n_res] for k, v in all_atom_feats.items()}
            pdb_str = bb_coords_to_pdb_str(
                c_flat,
                atoms,
                aatype=feats_i.get("aatype", None),
                chain_index=feats_i.get("chain_index", None),
            )
        else:
            feats_i = {k: v[i][:n_res] for k, v in all_atom_feats.items()}
            pdb_str = feats_to_pdb_str(c, **feats_i)
        write_pdb_str(pdb_str, fname, append=write_to_frames and i > 0)
