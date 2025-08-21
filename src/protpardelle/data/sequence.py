"""Sequence-related utils.

Author: Alex Chu, Zhaoyang Li
"""

from typing import Literal

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


def seq_to_aatype(seq: str, num_tokens: Literal[20, 21, 22] = 21) -> torch.Tensor:
    """Convert a protein sequence to its amino acid type representation.

    Args:
        seq (str): The protein sequence.
        num_tokens (Literal[20, 21, 22], optional): The number of tokens to use. Defaults to 21.

    Raises:
        ValueError: If the number of tokens is not supported.

    Returns:
        torch.Tensor: The amino acid type representation of the sequence.
    """

    if num_tokens == 20:
        mapping = residue_constants.restype_order
    elif num_tokens == 21:
        mapping = residue_constants.restype_order_with_x
    elif num_tokens == 22:
        mapping = residue_constants.restype_order_with_x
        mapping["<mask>"] = 21
    else:
        raise ValueError(f"num_tokens {num_tokens} not supported")

    return torch.tensor([mapping[aa] for aa in seq], dtype=torch.long)


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
