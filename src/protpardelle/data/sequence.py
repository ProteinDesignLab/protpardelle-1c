"""Sequence-related utils.

Author: Alex Chu, Zhaoyang Li
"""

import torch
import torch.nn.functional as F

from protpardelle.common import residue_constants


def aatype_to_seq(aatype, seq_mask=None):
    if seq_mask is None:
        seq_mask = torch.ones_like(aatype)

    mapping = residue_constants.restypes_with_x
    mapping = mapping + ["<mask>"]

    unbatched = False
    if len(aatype.shape) == 1:
        unbatched = True
        aatype = [aatype]
        seq_mask = [seq_mask]

    seqs = []
    for i, ai in enumerate(aatype):
        seq = []
        for j, aa in enumerate(ai):
            if seq_mask[i][j] == 1:
                try:
                    seq.append(mapping[aa])
                except IndexError:
                    print(aatype[i])
                    raise Exception(f"Error in mapping {aa} at {i},{j}")
        seqs.append("".join(seq))

    if unbatched:
        seqs = seqs[0]
    return seqs


def seq_to_aatype(seq, num_tokens=21):
    if num_tokens == 20:
        mapping = residue_constants.restype_order
    if num_tokens == 21:
        mapping = residue_constants.restype_order_with_x
    if num_tokens == 22:
        mapping = residue_constants.restype_order_with_x
        mapping["<mask>"] = 21
    return torch.Tensor([mapping[aa] for aa in seq]).long()


def batched_seq_to_aatype_and_mask(seqs, max_len=None):
    if max_len is None:
        max_len = max([len(s) for s in seqs])
    aatypes = []
    seq_mask = []
    for s in seqs:
        pad_size = max_len - len(s)
        aatype = seq_to_aatype(s)
        aatypes.append(F.pad(aatype, (0, pad_size)))
        mask = torch.ones_like(aatype).float()
        seq_mask.append(F.pad(mask, (0, pad_size)))
    return torch.stack(aatypes), torch.stack(seq_mask)
