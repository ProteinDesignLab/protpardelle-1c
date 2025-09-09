"""Sequence-related utils.

Author: Alex Chu, Zhaoyang Li
"""

from typing import Literal

import torch
import torch.nn.functional as F

from protpardelle.common import residue_constants


def seq_to_aatype(seq: str, num_tokens: Literal[20, 21, 22] = 21) -> torch.Tensor:
    """Convert a protein sequence to its amino acid type representation.

    Args:
        seq (str): The protein sequence.
        num_tokens (Literal[20, 21, 22], optional): The number of tokens to use. Defaults to 21.

    Raises:
        ValueError: If the number of tokens is not supported.

    Returns:
        torch.Tensor: The amino acid type representation of the sequence. (L,)
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


def seq_to_aatype_batched(seqs: list[str], max_len: int | None = None) -> torch.Tensor:
    """Convert a batch of protein sequences to their amino acid type representations.

    Args:
        seqs (list[str]): The protein sequences.
        max_len (int | None, optional): The maximum length of the sequences. Defaults to None.

    Returns:
        torch.Tensor: The amino acid type representations of the sequences. (B, L)
    """

    if max_len is None:
        max_len = max(len(s) for s in seqs)
    aatypes = []
    for s in seqs:
        pad_size = max_len - len(s)
        aatype = seq_to_aatype(s)
        aatypes.append(F.pad(aatype, (0, pad_size)))

    return torch.stack(aatypes)
