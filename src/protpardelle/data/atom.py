"""Atom representation handling.

Authors: Alex Chu, Richard Shuai, Zhaoyang Li
"""

from typing import Literal

import numpy as np
import torch
from jaxtyping import Float, Int

from protpardelle.common import residue_constants


def atom14_mask_from_aatype(
    aatype: Int[torch.Tensor, "B L"], seq_mask: Float[torch.Tensor, "B L"] | None = None
) -> Float[torch.Tensor, "B L 14"]:
    """Generate a mask for atom14 representation from amino acid type.

    Args:
        aatype (torch.Tensor): Amino acid type tensor.
        seq_mask (torch.Tensor | None, optional): Sequence mask tensor. Defaults to None.

    Returns:
        torch.Tensor: Atom14 mask tensor.
    """

    device = aatype.device
    source_mask = torch.from_numpy(
        residue_constants.restype_atom14_mask.astype(np.float32)
    ).to(device)
    bb_atoms = source_mask[residue_constants.restype_order["G"]].unsqueeze(0)
    # Use only the first 20 plus bb atoms for X, mask
    source_mask = torch.cat([source_mask[:-1], bb_atoms, bb_atoms], 0)
    atom_mask = source_mask[aatype]

    if seq_mask is not None:
        atom_mask = atom_mask * seq_mask.unsqueeze(-1)
    return atom_mask


def atom37_mask_from_aatype(
    aatype: Int[torch.Tensor, "B L"], seq_mask: Float[torch.Tensor, "B L"] | None = None
) -> Float[torch.Tensor, "B L 37"]:
    """Generate a mask for atom37 representation from amino acid type.

    Args:
        aatype (torch.Tensor): Amino acid type tensor.
        seq_mask (torch.Tensor | None, optional): Sequence mask tensor. Defaults to None.

    Returns:
        torch.Tensor: Atom37 mask tensor.
    """

    device = aatype.device
    source_mask = torch.from_numpy(
        residue_constants.restype_atom37_mask.astype(np.float32)
    ).to(device)
    bb_atoms = source_mask[residue_constants.restype_order["G"]].unsqueeze(0)
    # Use only the first 20 plus bb atoms for X, mask
    source_mask = torch.cat([source_mask[:-1], bb_atoms, bb_atoms], 0)
    atom_mask = source_mask[aatype]

    if seq_mask is not None:
        atom_mask = atom_mask * seq_mask.unsqueeze(-1)
    return atom_mask


def atom73_mask_from_aatype(
    aatype: Int[torch.Tensor, "B L"], seq_mask: Float[torch.Tensor, "B L"] | None = None
) -> Float[torch.Tensor, "B L 73"]:
    """Generate a mask for atom73 representation from amino acid type.

    Args:
        aatype (torch.Tensor): Amino acid type tensor.
        seq_mask (torch.Tensor | None, optional): Sequence mask tensor. Defaults to None.

    Returns:
        torch.Tensor: Atom73 mask tensor.
    """

    device = aatype.device
    source_mask = torch.from_numpy(
        residue_constants.restype_atom73_mask.astype(np.float32)
    ).to(device)
    atom_mask = source_mask[aatype]

    if seq_mask is not None:
        atom_mask = atom_mask * seq_mask.unsqueeze(-1)
    return atom_mask


def atom14_coords_to_atom37_coords(
    atom14_coords: Float[torch.Tensor, "L 14 3"], aatype: Int[torch.Tensor, "L"]
) -> Float[torch.Tensor, "L 37 3"]:
    """Convert atom14 coordinates to atom37 coordinates.

    Not batched.

    Args:
        atom14_coords (torch.Tensor): Atom14 coordinates.
        aatype (torch.Tensor): Amino acid type.

    Returns:
        torch.Tensor: Atom37 coordinates.
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
    atom14_coords: Float[torch.Tensor, "B L 14 3"], aatype: Int[torch.Tensor, "B L"]
) -> Float[torch.Tensor, "B L 37 3"]:
    """Convert atom14 coordinates to atom37 coordinates.

    Args:
        atom14_coords (torch.Tensor): Atom14 coordinates.
        aatype (torch.Tensor): Amino acid type.

    Returns:
        torch.Tensor: Atom37 coordinates.
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
    atom37_coords: Float[torch.Tensor, "L 37 3"],
    aatype: Int[torch.Tensor, "L"],
) -> Float[torch.Tensor, "L 14 3"]:
    """Convert atom37 coordinates to atom14 coordinates.

    Not batched.

    Args:
        atom37_coords (torch.Tensor): Atom37 coordinates.
        aatype (torch.Tensor): Amino acid types.

    Returns:
        torch.Tensor: Atom14 coordinates.
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
    b_factors_37: Float[torch.Tensor, "L 37"],
    aatype: Int[torch.Tensor, "L"],
) -> Float[torch.Tensor, "L 14"]:
    """Convert B-factors from atom37 representation to atom14 representation.

    Not batched.

    Args:
        b_factors_37 (torch.Tensor): B-factors in atom37 representation.
        aatype (torch.Tensor): Amino acid types.

    Returns:
        torch.Tensor: B-factors in atom14 representation.
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


def bb_coords_to_atom37_coords(
    bb_coords: Float[torch.Tensor, "L 4 3"],
) -> Float[torch.Tensor, "L 37 3"]:
    """Convert backbone coordinates to atom37 coordinates.

    Not batched.

    Args:
        bb_coords (torch.Tensor): Backbone coordinates.

    Returns:
        torch.Tensor: Atom37 coordinates.
    """

    device = bb_coords.device
    atom37_coords = torch.zeros((bb_coords.shape[0], 37, 3), device=device)
    bb_idxs = [
        residue_constants.atom_order[atom_name] for atom_name in ["N", "CA", "C", "O"]
    ]
    atom37_coords[:, bb_idxs] = bb_coords

    return atom37_coords


def atom37_coords_to_atom73_coords(
    atom37_coords: Float[torch.Tensor, "L 37 3"], aatype: Int[torch.Tensor, "L"]
) -> Float[torch.Tensor, "L 73 3"]:
    """Convert atom37 coordinates to atom73 coordinates.

    Not batched.

    Args:
        atom37_coords (torch.Tensor): Atom37 coordinates.
        aatype (torch.Tensor): Amino acid types.

    Returns:
        torch.Tensor: Atom73 coordinates.
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
    atom73_coords: Float[torch.Tensor, "L 73 3"], aatype: Int[torch.Tensor, "L"]
) -> Float[torch.Tensor, "L 37 3"]:
    """Convert atom73 coordinates to atom37 coordinates.

    Not batched.

    Args:
        atom73_coords (torch.Tensor): Atom73 coordinates.
        aatype (torch.Tensor): Amino acid types.

    Returns:
        torch.Tensor: Atom37 coordinates.
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


def fill_in_cbeta_for_atom37_coords(
    atom37_coords: Float[torch.Tensor, "... 37 3"],
) -> Float[torch.Tensor, "... 37 3"]:
    """Fill in the CB atom coordinates for a given set of atom37 coordinates.

    Args:
        atom37_coords (torch.Tensor): Atom37 coordinates.

    Returns:
        torch.Tensor: Updated Atom37 coordinates with CB filled in.
    """

    b = atom37_coords[..., 1, :] - atom37_coords[..., 0, :]
    c = atom37_coords[..., 2, :] - atom37_coords[..., 1, :]
    a = torch.linalg.cross(b, c)  # pylint: disable=not-callable

    cbeta = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + atom37_coords[..., 1, :]
    updated_atom37_coords = torch.clone(atom37_coords)
    updated_atom37_coords[..., 3, :] = cbeta

    return updated_atom37_coords


def dummy_fill(
    atom37_coords: Float[torch.Tensor, "... 37 3"],
    atom37_mask: Float[torch.Tensor, "... 37"],
    mode: Literal["CA", "zero"] = "zero",
) -> Float[torch.Tensor, "... 37 3"]:
    """Fill in ghost side chain atoms with either the CA or CB atom value
    for each residue, depending on the mode.

    Args:
        atom37_coords (torch.Tensor): Input coordinates.
        atom37_mask (torch.Tensor): Atom mask.
        mode (Literal["CA", "zero"], optional): Mode for filling in ghost atoms.
            Defaults to "zero".

    Returns:
        torch.Tensor: Filled coordinates.
    """

    dummy_fill_mask: Float[torch.Tensor, "... 37"] = 1.0 - atom37_mask

    if mode == "CA":
        dummy_fill_value = atom37_coords[..., 1:2, :]  # CA
    elif mode == "CB":  # deprecated
        dummy_fill_value = fill_in_cbeta_for_atom37_coords(atom37_coords)[
            ..., 3:4, :
        ]  # idealized CB
    elif mode == "zero":
        dummy_fill_value = torch.zeros_like(atom37_coords)
    else:
        raise ValueError(f"Unknown dummy fill mode: {mode}")

    atom37_coords = atom37_coords * atom37_mask.unsqueeze(
        -1
    ) + dummy_fill_value * dummy_fill_mask.unsqueeze(-1)

    return atom37_coords
