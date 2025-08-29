"""Atom representation handling.

Authors: Alex Chu, Richard Shuai, Zhaoyang Li
"""

import torch
from torchtyping import TensorType

from protpardelle.common import residue_constants


def atom37_mask_from_aatype(
    aatype: torch.Tensor, seq_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Generate a mask for atom37 representation from amino acid type.

    Args:
        aatype (torch.Tensor): Amino acid type tensor of shape (B, L).
        seq_mask (torch.Tensor | None, optional): Sequence mask tensor of shape (B, L). Defaults to None.

    Returns:
        torch.Tensor: Atom37 mask tensor of shape (B, L, 37).
    """

    # source_mask is (21, 37) originally
    source_mask = torch.tensor(residue_constants.restype_atom37_mask).to(aatype)
    bb_atoms = source_mask[residue_constants.restype_order["G"]].unsqueeze(0)
    # Use only the first 20 plus bb atoms for X, mask
    source_mask = torch.cat([source_mask[:-1], bb_atoms, bb_atoms], 0)
    atom_mask = source_mask[aatype]
    if seq_mask is not None:
        atom_mask = atom_mask * seq_mask.unsqueeze(-1)

    return atom_mask


def atom14_mask_from_aatype(
    aatype: torch.Tensor, seq_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Generate a mask for atom14 representation from amino acid type.

    Args:
        aatype (torch.Tensor): Amino acid type tensor of shape (B, L).
        seq_mask (torch.Tensor | None, optional): Sequence mask tensor of shape (B, L). Defaults to None.

    Returns:
        torch.Tensor: Atom14 mask tensor of shape (B, L, 14).
    """

    # source_mask is (21, 14) originally
    source_mask = torch.tensor(residue_constants.restype_atom14_mask).to(aatype.device)
    bb_atoms = source_mask[residue_constants.restype_order["G"]].unsqueeze(0)
    # Use only the first 20 plus bb atoms for X, mask
    source_mask = torch.cat([source_mask[:-1], bb_atoms, bb_atoms], 0)
    atom_mask = source_mask[aatype]
    if seq_mask is not None:
        atom_mask = atom_mask * seq_mask.unsqueeze(-1)
    return atom_mask


def atom37_coords_from_atom14(
    atom14_coords: torch.Tensor, aatype: torch.Tensor
) -> torch.Tensor:
    """Convert atom14 coordinates to atom37 coordinates.

    Args:
        atom14_coords (torch.Tensor): Atom 14 coordinates. (B, L, 14, 3)
        aatype (torch.Tensor): Amino acid type. (B, L)

    Returns:
        torch.Tensor: Atom 37 coordinates. (B, L, 37, 3)
    """

    device = atom14_coords.device
    B, L = atom14_coords.shape[:2]
    atom37_coords = torch.zeros((B, L, 37, 3), device=device)
    for i in range(B):  # per batch
        for j in range(L):  # per residue
            aa = aatype[i, j]
            if aa.item() < residue_constants.restype_num:
                aa_3name = residue_constants.restype_1to3[
                    residue_constants.restypes[aa]
                ]
            else:
                aa_3name = "UNK"

            atom14_atoms = residue_constants.restype_name_to_atom14_names[aa_3name]
            for k in range(14):
                atom_name = atom14_atoms[k]
                if atom_name != "":
                    atom37_idx = residue_constants.atom_order[atom_name]
                    atom37_coords[i, j, atom37_idx, :] = atom14_coords[i, j, k, :]

    return atom37_coords


def atom37_coords_to_atom14(
    atom37_coords: TensorType["b 37 3"],
    b_factors_37: TensorType["b 37"],
    aatype: TensorType["b"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Unbatched
    device = atom37_coords.device
    atom14_coords = torch.zeros((atom37_coords.shape[0], 14, 3)).to(device)
    b_factors_14 = torch.zeros((atom37_coords.shape[0], 14)).to(device)
    for i in range(atom37_coords.shape[0]):  # per residue
        aa = aatype[i]
        aa_3name = residue_constants.restype_1to3[residue_constants.restypes[aa]]
        atom14_atoms = residue_constants.restype_name_to_atom14_names[aa_3name]
        for j in range(14):
            atom_name = atom14_atoms[j]
            if atom_name != "":
                atom37_idx = residue_constants.atom_order[atom_name]
                atom14_coords[i, j, :] = atom37_coords[i, atom37_idx, :]
                b_factors_14[i, j] = b_factors_37[i, atom37_idx]

    atom14_mask = atom14_mask_from_aatype(aatype)

    return atom14_coords, b_factors_14, atom14_mask


def atom37_coords_from_bb(
    bb_coords: TensorType["n 4 3"], return_mask=False
) -> TensorType["n 37 3"]:
    """
    Unbatched. Takes in coords with N, CA, C, O backbone atoms and returns in atom37 format.
    """
    atom37_coords = torch.zeros((bb_coords.shape[0], 37, 3)).to(bb_coords)
    bb_idxs = [
        residue_constants.atom_order[atom_name] for atom_name in ["N", "CA", "C", "O"]
    ]
    atom37_coords[:, bb_idxs] = bb_coords

    if return_mask:
        atom37_mask = torch.zeros((bb_coords.shape[0], 37)).to(bb_coords)
        atom37_mask[:, bb_idxs] = 1
        return atom37_coords, atom37_mask
    return atom37_coords


def atom73_mask_from_aatype(aatype, seq_mask=None):
    source_mask = torch.tensor(residue_constants.restype_atom73_mask).to(aatype.device)
    atom_mask = source_mask[aatype]
    if seq_mask is not None:
        atom_mask = atom_mask * seq_mask.unsqueeze(-1)
    return atom_mask


def atom37_to_atom73(atom37, aatype, return_mask=False):
    # Unbatched
    atom73 = torch.zeros((atom37.shape[0], 73, 3)).to(atom37)
    for i in range(atom37.shape[0]):
        aa = aatype[i]
        aa1 = residue_constants.restypes[aa]
        for j, atom37_name in enumerate(residue_constants.atom_types):
            atom73_name = atom37_name
            if atom37_name not in ["N", "CA", "C", "O", "CB"]:
                atom73_name = aa1 + atom73_name
            if atom73_name in residue_constants.atom73_names_to_idx:
                atom73_idx = residue_constants.atom73_names_to_idx[atom73_name]
                atom73[i, atom73_idx, :] = atom37[i, j, :]

    if return_mask:
        atom73_mask = atom73_mask_from_aatype(aatype)
        return atom73, atom73_mask
    return atom73


def atom73_to_atom37(atom73, aatype, return_mask=False):
    # Unbatched
    atom37_coords = torch.zeros((atom73.shape[0], 37, 3)).to(atom73)
    for i in range(atom73.shape[0]):  # per residue
        aa = aatype[i]
        aa1 = residue_constants.restypes[aa]
        for j, atom_type in enumerate(residue_constants.atom_types):
            atom73_name = atom_type
            if atom73_name not in ["N", "CA", "C", "O", "CB"]:
                atom73_name = aa1 + atom73_name
            if atom73_name in residue_constants.atom73_names_to_idx:
                atom73_idx = residue_constants.atom73_names_to_idx[atom73_name]
                atom37_coords[i, j, :] = atom73[i, atom73_idx, :]

    if return_mask:
        atom37_mask = atom37_mask_from_aatype(aatype)
        return atom37_coords, atom37_mask
    return atom37_coords


def fill_in_cbeta_for_atom37(coords):
    b = coords[..., 1, :] - coords[..., 0, :]
    c = coords[..., 2, :] - coords[..., 1, :]
    a = torch.cross(b, c, dim=-1)
    cbeta = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + coords[..., 1, :]
    new_coords = torch.clone(coords)
    new_coords[..., 3, :] = cbeta
    return new_coords
