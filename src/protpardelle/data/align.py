"""Alignment functions.

Authors: Alex Chu, Zhaoyang Li
"""

from typing import Literal

import torch
from jaxtyping import Float


def kabsch_align(p: Float[torch.Tensor, "N 3"], q: Float[torch.Tensor, "N 3"]) -> tuple[
    Float[torch.Tensor, "N 3"],
    tuple[Float[torch.Tensor, "3 3"], Float[torch.Tensor, "3"]],
]:
    """Aligns two sets of points using the Kabsch algorithm.

    Args:
        p (torch.Tensor): The first set of points.
        q (torch.Tensor): The second set of points.

    Returns:
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: The aligned points and the transformation (rotation, translation).
    """

    assert p.shape == q.shape
    assert p.shape[-1] == 3
    assert p.ndim == 2

    p_ctr = p - p.mean(0)
    t = q.mean(0)
    q_ctr = q - t
    H = p_ctr.t() @ q_ctr
    U, _, V = torch.svd(H)
    R = V @ U.t()
    I = torch.eye(3).to(p)
    I[-1, -1] = R.det().sign()
    R = V @ I @ U.t()
    p_aligned = p_ctr @ R.t() + t

    return p_aligned, (R, t)


def tm_score(
    coords1: Float[torch.Tensor, "L A 3"],
    coords2: Float[torch.Tensor, "L A 3"],
    atom_mask: Float[torch.Tensor, "L A"] | None = None,
) -> torch.Tensor:
    """Compute the TM-score between two sets of coordinates.

    Args:
        coords1 (torch.Tensor): First set of coordinates.
        coords2 (torch.Tensor): Second set of coordinates.
        atom_mask (torch.Tensor | None, optional): Mask to apply. Defaults to None.

    Returns:
        torch.Tensor: The TM-score.
    """

    assert coords1.shape == coords2.shape

    length = coords2.shape[0]
    dists_square = (coords1 - coords2).square().sum(-1)
    d0 = 1.24 * ((length - 15) ** (1 / 3)) - 1.8
    term = 1 / (1 + dists_square / d0**2)

    if atom_mask is None:
        return term.mean()

    return torch.sum(term * atom_mask) / atom_mask.sum().clamp(min=1)


def lddt(
    all_atom_pred_pos: Float[torch.Tensor, "... N 3"],
    all_atom_positions: Float[torch.Tensor, "... N 3"],
    all_atom_mask: Float[torch.Tensor, "... N"],
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    """Compute the LDDT score.

    Adapted from https://github.com/aqlaboratory/openfold
    Changed all_atom_mask shape from (..., N, 1) to (..., N)

    Args:
        all_atom_pred_pos (torch.Tensor): Predicted atom positions.
        all_atom_positions (torch.Tensor): True atom positions.
        all_atom_mask (torch.Tensor): Mask for valid atoms.
        cutoff (float, optional): Distance cutoff for considering pairs. Defaults to 15.0.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-10.
        per_residue (bool, optional): Whether to compute the score per residue. Defaults to True.

    Returns:
        torch.Tensor: The computed LDDT score.
    """

    N = all_atom_mask.shape[-1]
    device = all_atom_mask.device

    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions.unsqueeze(-2) - all_atom_positions.unsqueeze(-3)
            ).square(),
            dim=-1,
        )
    )
    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos.unsqueeze(-2) - all_atom_pred_pos.unsqueeze(-3)
            ).square(),
            dim=-1,
        )
    )

    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask.unsqueeze(-1)
        * all_atom_mask.unsqueeze(-2)
        * (1.0 - torch.eye(N, device=device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


def compute_structure_metric(
    coords1: Float[torch.Tensor, "L A 3"],
    coords2: Float[torch.Tensor, "L A 3"],
    metric: Literal["ca_rmsd", "tm_score"] = "ca_rmsd",
) -> torch.Tensor:
    """Compute structure metric between two sets of coordinates.

    Args:
        coords1 (torch.Tensor): First set of coordinates.
        coords2 (torch.Tensor): Second set of coordinates.
        metric (Literal["ca_rmsd", "tm_score"], optional): Metric to compute. Defaults to "ca_rmsd".

    Raises:
        NotImplementedError: If the metric is not implemented.

    Returns:
        torch.Tensor: The computed structure metric.
    """

    aligned_coords1_ca, _ = kabsch_align(coords1[:, 1], coords2[:, 1])

    if metric == "ca_rmsd":
        ca_rmsd = (aligned_coords1_ca - coords2[:, 1]).square().sum(-1).mean().sqrt()
        return ca_rmsd
    if metric == "tm_score":
        return tm_score(aligned_coords1_ca, coords2[:, 1])

    raise NotImplementedError(f"Metric {metric} not implemented.")


def compute_allatom_structure_metric(
    coords1: Float[torch.Tensor, "L A 3"],
    coords2: Float[torch.Tensor, "L A 3"],
    atom_mask: Float[torch.Tensor, "L A"],
    metric: Literal["allatom_rmsd", "allatom_tm", "allatom_lddt"] = "allatom_rmsd",
) -> torch.Tensor:
    """Compute all-atom structure metric between two sets of coordinates.

    Args:
        coords1 (torch.Tensor): First set of coordinates.
        coords2 (torch.Tensor): Second set of coordinates.
        atom_mask (torch.Tensor): Mask to apply.
        metric (Literal["allatom_rmsd", "allatom_tm", "allatom_lddt"], optional):
            Metric to compute. Defaults to "allatom_rmsd".

    Raises:
        NotImplementedError: If the metric is not implemented.

    Returns:
        torch.Tensor: The computed structure metric.
    """

    if metric == "allatom_lddt":
        return lddt(
            coords1.reshape(-1, 3),
            coords2.reshape(-1, 3),
            atom_mask.flatten(),
            per_residue=False,
        )

    atom_mask_bool = atom_mask.bool()
    _, (R, t) = kabsch_align(coords1[atom_mask_bool], coords2[atom_mask_bool])
    aligned_coords1 = coords1 - coords1[atom_mask_bool].mean(dim=0)
    aligned_coords1 = aligned_coords1 @ R.t() + t

    if metric == "allatom_rmsd":
        aligned_coords1_masked = aligned_coords1 * atom_mask.unsqueeze(-1)
        coords2_masked = coords2 * atom_mask.unsqueeze(-1)
        allatom_rmsd = torch.sqrt(
            (aligned_coords1_masked - coords2_masked).square().sum(-1).sum()
            / atom_mask.sum()
        )
        return allatom_rmsd
    if metric == "allatom_tm":
        return tm_score(aligned_coords1, coords2, atom_mask=atom_mask)

    raise NotImplementedError(f"Metric {metric} not implemented.")
