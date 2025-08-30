"""Alignment functions.

Authors: Alex Chu, Zhaoyang Li
"""

import torch


def kabsch_align(
    p: torch.Tensor, q: torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Aligns two sets of points using the Kabsch algorithm.

    Args:
        p (torch.Tensor): The first set of points.
        q (torch.Tensor): The second set of points.

    Returns:
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: The aligned points and the transformation (rotation, translation).
    """

    if len(p.shape) > 2:
        p = p.reshape(-1, 3)
    if len(q.shape) > 2:
        q = q.reshape(-1, 3)
    p_ctr = p - p.mean(0, keepdim=True)
    t = q.mean(0, keepdim=True)
    q_ctr = q - t
    H = p_ctr.t() @ q_ctr
    U, S, V = torch.svd(H)
    R = V @ U.t()
    I_ = torch.eye(3).to(p)
    I_[-1, -1] = R.det().sign()
    R = V @ I_ @ U.t()
    p_aligned = p_ctr @ R.t() + t

    return p_aligned, (R, t)


def tm_score(
    coords1: torch.Tensor, coords2: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute the TM-score between two sets of coordinates.

    Args:
        coords1 (torch.Tensor): First set of coordinates. (L, A, 3)
        coords2 (torch.Tensor): Second set of coordinates. (L, A, 3)
        mask (torch.Tensor | None, optional): Mask to apply. Defaults to None.

    Returns:
        torch.Tensor: The TM-score.
    """

    assert coords1.shape == coords2.shape

    length = coords2.shape[0]
    dists_square = (coords1 - coords2).square().sum(-1)
    d0 = 1.24 * ((length - 15) ** (1 / 3)) - 1.8
    term = 1 / (1 + dists_square / d0**2)

    if mask is None:
        return term.mean()

    return torch.sum(term * mask) / mask.sum().clamp(min=1)
