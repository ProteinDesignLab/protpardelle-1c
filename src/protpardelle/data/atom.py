"""Atom representation handling.

Authors: Alex Chu, Richard Shuai, Zhaoyang Li
"""

import torch

from protpardelle.common import residue_constants


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
    device = aatype.device
    source_mask = torch.tensor(residue_constants.restype_atom14_mask, device=device)
    bb_atoms = source_mask[residue_constants.restype_order["G"]].unsqueeze(0)
    # Use only the first 20 plus bb atoms for X, mask
    source_mask = torch.cat([source_mask[:-1], bb_atoms, bb_atoms], 0)
    atom_mask = source_mask[aatype]

    if seq_mask is not None:
        atom_mask = atom_mask * seq_mask.unsqueeze(-1)
    return atom_mask


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
    device = aatype.device
    source_mask = torch.tensor(residue_constants.restype_atom37_mask, device=device)
    bb_atoms = source_mask[residue_constants.restype_order["G"]].unsqueeze(0)
    # Use only the first 20 plus bb atoms for X, mask
    source_mask = torch.cat([source_mask[:-1], bb_atoms, bb_atoms], 0)
    atom_mask = source_mask[aatype]

    if seq_mask is not None:
        atom_mask = atom_mask * seq_mask.unsqueeze(-1)
    return atom_mask


def atom73_mask_from_aatype(
    aatype: torch.Tensor, seq_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Generate a mask for atom73 representation from amino acid type.

    Args:
        aatype (torch.Tensor): Amino acid type tensor of shape (B, L).
        seq_mask (torch.Tensor | None, optional): Sequence mask tensor of shape (B, L). Defaults to None.

    Returns:
        torch.Tensor: Atom73 mask tensor of shape (B, L, 73).
    """

    device = aatype.device
    source_mask = torch.tensor(residue_constants.restype_atom73_mask, device=device)
    atom_mask = source_mask[aatype]

    if seq_mask is not None:
        atom_mask = atom_mask * seq_mask.unsqueeze(-1)
    return atom_mask


def atom14_coords_to_atom37_coords(
    atom14_coords: torch.Tensor, aatype: torch.Tensor
) -> torch.Tensor:
    """Convert atom14 coordinates to atom37 coordinates.

    Not batched.

    Args:
        atom14_coords (torch.Tensor): Atom 14 coordinates. (L, 14, 3)
        aatype (torch.Tensor): Amino acid type. (L,)

    Returns:
        torch.Tensor: Atom 37 coordinates. (L, 37, 3)
    """

    device = atom14_coords.device
    atom37_coords = torch.zeros((atom14_coords.shape[0], 37, 3), device=device)
    for i in range(atom14_coords.shape[0]):  # per residue
        aa = aatype[i]
        if aa.item() < residue_constants.restype_num:
            aa_3name = residue_constants.restype_1to3[residue_constants.restypes[aa]]
        else:
            aa_3name = "UNK"

        atom14_atoms = residue_constants.restype_name_to_atom14_names[aa_3name]
        for k in range(14):
            atom_name = atom14_atoms[k]
            if atom_name != "":
                atom37_idx = residue_constants.atom_order[atom_name]
                atom37_coords[i, atom37_idx, :] = atom14_coords[i, k, :]

    return atom37_coords


def atom14_coords_to_atom37_coords_batched(
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


def atom37_coords_to_atom14_coords(
    atom37_coords: torch.Tensor,
    aatype: torch.Tensor,
) -> torch.Tensor:
    """Convert atom37 coordinates to atom14 coordinates.

    Not batched.

    Args:
        atom37_coords (torch.Tensor): Atom37 coordinates. (L, 37, 3)
        aatype (torch.Tensor): Amino acid types. (L,)

    Returns:
        torch.Tensor: Atom14 coordinates. (L, 14, 3)
    """

    device = atom37_coords.device
    atom14_coords = torch.zeros((atom37_coords.shape[0], 14, 3), device=device)
    for i in range(atom37_coords.shape[0]):  # per residue
        aa = aatype[i]
        aa_3name = residue_constants.restype_1to3[residue_constants.restypes[aa]]
        atom14_atoms = residue_constants.restype_name_to_atom14_names[aa_3name]
        for j in range(14):
            if atom_name := atom14_atoms[j]:
                atom37_idx = residue_constants.atom_order[atom_name]
                atom14_coords[i, j, :] = atom37_coords[i, atom37_idx, :]

    return atom14_coords


def b_factors_37_to_b_factors_14(
    b_factors_37: torch.Tensor,
    aatype: torch.Tensor,
) -> torch.Tensor:
    """Convert B-factors from atom37 representation to atom14 representation.

    Not batched.

    Args:
        b_factors_37 (torch.Tensor): B-factors in atom37 representation. (L, 37)
        aatype (torch.Tensor): Amino acid types. (L,)

    Returns:
        torch.Tensor: B-factors in atom14 representation. (L, 14)
    """

    device = b_factors_37.device
    b_factors_14 = torch.zeros((b_factors_37.shape[0], 14), device=device)
    for i in range(b_factors_37.shape[0]):  # per residue
        aa = aatype[i]
        aa_3name = residue_constants.restype_1to3[residue_constants.restypes[aa]]
        atom14_atoms = residue_constants.restype_name_to_atom14_names[aa_3name]
        for j in range(14):
            if atom_name := atom14_atoms[j]:
                atom37_idx = residue_constants.atom_order[atom_name]
                b_factors_14[i, j] = b_factors_37[i, atom37_idx]

    return b_factors_14


def bb_coords_to_atom37_coords(bb_coords: torch.Tensor) -> torch.Tensor:
    """Convert backbone coordinates to atom37 coordinates.

    Not batched.

    Args:
        bb_coords (torch.Tensor): Backbone coordinates. (L, 4, 3)

    Returns:
        torch.Tensor: Atom37 coordinates. (L, 37, 3)
    """

    device = bb_coords.device
    atom37_coords = torch.zeros((bb_coords.shape[0], 37, 3), device=device)
    bb_idxs = [
        residue_constants.atom_order[atom_name] for atom_name in ["N", "CA", "C", "O"]
    ]
    atom37_coords[:, bb_idxs] = bb_coords

    return atom37_coords


def atom37_coords_to_atom73_coords(
    atom37_coords: torch.Tensor, aatype: torch.Tensor
) -> torch.Tensor:
    """Convert atom37 coordinates to atom73 coordinates.

    Not batched.

    Args:
        atom37_coords (torch.Tensor): Atom37 coordinates. (L, 37, 3)
        aatype (torch.Tensor): Amino acid types. (L,)

    Returns:
        torch.Tensor: Atom73 coordinates. (L, 73, 3)
    """

    device = atom37_coords.device
    atom73_coords = torch.zeros((atom37_coords.shape[0], 73, 3), device=device)
    for i in range(atom37_coords.shape[0]):
        aa = aatype[i]
        aa1 = residue_constants.restypes[aa]
        for j, atom37_name in enumerate(residue_constants.atom_types):
            atom73_name = atom37_name
            if atom73_name not in ["N", "CA", "C", "O", "CB"]:
                atom73_name = aa1 + atom73_name
            if atom73_name in residue_constants.atom73_names_to_idx:
                atom73_idx = residue_constants.atom73_names_to_idx[atom73_name]
                atom73_coords[i, atom73_idx, :] = atom37_coords[i, j, :]

    return atom73_coords


def atom73_coords_to_atom37_coords(
    atom73_coords: torch.Tensor, aatype: torch.Tensor
) -> torch.Tensor:
    """Convert atom73 coordinates to atom37 coordinates.

    Not batched.

    Args:
        atom73_coords (torch.Tensor): Atom73 coordinates. (L, 73, 3)
        aatype (torch.Tensor): Amino acid types. (L,)

    Returns:
        torch.Tensor: Atom37 coordinates. (L, 37, 3)
    """

    device = atom73_coords.device
    atom37_coords = torch.zeros((atom73_coords.shape[0], 37, 3), device=device)
    for i in range(atom73_coords.shape[0]):  # per residue
        aa = aatype[i]
        aa1 = residue_constants.restypes[aa]
        for j, atom_type in enumerate(residue_constants.atom_types):
            atom73_name = atom_type
            if atom73_name not in ["N", "CA", "C", "O", "CB"]:
                atom73_name = aa1 + atom73_name
            if atom73_name in residue_constants.atom73_names_to_idx:
                atom73_idx = residue_constants.atom73_names_to_idx[atom73_name]
                atom37_coords[i, j, :] = atom73_coords[i, atom73_idx, :]

    return atom37_coords


def fill_in_cbeta_for_atom37_coords(atom37_coords: torch.Tensor) -> torch.Tensor:
    """Fill in the CBeta atom coordinates for a given set of atom37 coordinates.

    Args:
        atom37_coords (torch.Tensor): Atom37 coordinates. (L, 37, 3)

    Returns:
        torch.Tensor: Updated Atom37 coordinates with CBeta filled in. (L, 37, 3)
    """

    b = atom37_coords[..., 1, :] - atom37_coords[..., 0, :]
    c = atom37_coords[..., 2, :] - atom37_coords[..., 1, :]
    a = torch.cross(b, c, dim=-1)
    cbeta = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + atom37_coords[..., 1, :]
    updated_atom37_coords = torch.clone(atom37_coords)
    updated_atom37_coords[..., 3, :] = cbeta

    return updated_atom37_coords
