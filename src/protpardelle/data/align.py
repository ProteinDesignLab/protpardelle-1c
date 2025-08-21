"""Alignment functions.

Author: Alex Chu
"""

import torch


def kabsch_align(p, q):
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
