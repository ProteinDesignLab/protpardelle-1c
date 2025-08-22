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
