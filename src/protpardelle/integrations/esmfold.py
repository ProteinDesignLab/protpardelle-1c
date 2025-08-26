"""ESMFold wrapper functions.

Author: Zhaoyang Li
"""

from collections import defaultdict

import torch
import torch.nn as nn
from torch.types import Device
from tqdm.auto import tqdm
from transformers import EsmForProteinFolding

from protpardelle.common import residue_constants
from protpardelle.data import atom
from protpardelle.data.sequence import seq_to_aatype
from protpardelle.env import ESMFOLD_PATH
from protpardelle.utils import clean_gpu_cache, get_default_device


def collate_dense_tensors(
    samples: list[torch.Tensor], pad_v: float = 0.0
) -> torch.Tensor:
    """Adapted from https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/misc.py"""
    if not samples:
        return torch.tensor([])
    if len(set(x.dim() for x in samples)) != 1:
        raise RuntimeError(
            f"Samples has varying dimensions: {[x.dim() for x in samples]}"
        )
    (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    result = torch.empty(
        len(samples), *max_shape, dtype=samples[0].dtype, device=device
    )
    result.fill_(pad_v)
    for i in range(len(samples)):
        result_i = result[i]
        t = samples[i]
        result_i[tuple(slice(0, k) for k in t.shape)] = t

    return result


def encode_sequence(
    seq: str,
    residue_index_offset: int | None = 512,
    chain_linker: str | None = "G" * 25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/misc.py"""
    if chain_linker is None:
        chain_linker = ""
    if residue_index_offset is None:
        residue_index_offset = 0

    chains = seq.split(":")
    seq = chain_linker.join(chains)

    unk_idx = residue_constants.restype_order_with_x["X"]
    encoded = torch.tensor(
        [residue_constants.restype_order_with_x.get(aa, unk_idx) for aa in seq]
    )
    residx = torch.arange(len(encoded))

    if residue_index_offset > 0:
        start = 0
        for i, chain in enumerate(chains):
            residx[start : start + len(chain) + len(chain_linker)] += (
                i * residue_index_offset
            )
            start += len(chain) + len(chain_linker)

    linker_mask = torch.ones_like(encoded, dtype=torch.float)
    chain_index = []
    offset = 0
    for i, chain in enumerate(chains):
        if i > 0:
            chain_index.extend([i - 1] * len(chain_linker))
        chain_index.extend([i] * len(chain))
        offset += len(chain)
        linker_mask[offset : offset + len(chain_linker)] = 0
        offset += len(chain_linker)

    chain_index_tensor = torch.tensor(chain_index, dtype=torch.long)

    return encoded, residx, linker_mask, chain_index_tensor


def batch_encode_sequences(
    sequences: list[str],
    residue_index_offset: int | None = 512,
    chain_linker: str | None = "G" * 25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/misc.py"""
    aatype_list = []
    residx_list = []
    linker_mask_list = []
    chain_index_list = []
    for seq in sequences:
        aatype_seq, residx_seq, linker_mask_seq, chain_index_seq = encode_sequence(
            seq,
            residue_index_offset=residue_index_offset,
            chain_linker=chain_linker,
        )
        aatype_list.append(aatype_seq)
        residx_list.append(residx_seq)
        linker_mask_list.append(linker_mask_seq)
        chain_index_list.append(chain_index_seq)

    aatype = collate_dense_tensors(aatype_list)
    mask = collate_dense_tensors(
        [aatype.new_ones(len(aatype_seq)) for aatype_seq in aatype_list]
    )
    residx = collate_dense_tensors(residx_list)
    linker_mask = collate_dense_tensors(linker_mask_list)
    chain_index_tensor = collate_dense_tensors(chain_index_list, -1)

    return aatype, mask, residx, linker_mask, chain_index_tensor


class EsmForProteinFoldingNew(EsmForProteinFolding):
    """HuggingFace ESMFold model with the original infer method."""

    @torch.no_grad()
    def infer_(
        self,
        seqs_list: str | list[str],
        residx: torch.Tensor | None = None,
        masking_pattern: torch.Tensor | None = None,
        num_recycles: int | None = None,
        residue_index_offset: int | None = 512,
        chain_linker: str | None = "G" * 25,
    ) -> dict[str, torch.Tensor]:
        """Rewrite the infer method as the original ESMFold model."""
        if isinstance(seqs_list, str):
            seqs_list = [seqs_list]

        aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            seqs_list, residue_index_offset, chain_linker
        )

        if residx is None:
            residx = _residx
        elif not isinstance(residx, torch.Tensor):
            residx = collate_dense_tensors(residx)

        device = next(self.parameters()).device
        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(device), (aatype, mask, residx, linker_mask)
        )

        output = self(
            aatype,
            attention_mask=mask,
            position_ids=residx,
            masking_pattern=masking_pattern,
            num_recycles=num_recycles,
        )

        # Apply linker mask to remove poly-G
        output["atom37_atom_exists"] = output[
            "atom37_atom_exists"
        ] * linker_mask.unsqueeze(2)

        return output


class ESMFold(nn.Module):
    """ESMFold model for predicting the 3D structure of a protein from its sequence."""

    def __init__(self, device: Device = None, chunk_size: int | None = None) -> None:
        """Initialize the ESMFold model."""
        super().__init__()

        self.model_name = ESMFOLD_PATH
        if device is None:
            device = get_default_device()
        self._device = device
        self.chunk_size = chunk_size

        self.model = self._load_model()

    def _load_model(self) -> EsmForProteinFoldingNew:
        """Load the ESMFold model from the transformers library."""
        model = EsmForProteinFoldingNew.from_pretrained(self.model_name)
        model = model.to(self._device)  # type: ignore
        model = model.eval()
        model.trunk.set_chunk_size(self.chunk_size)

        return model

    @property
    def device(self) -> torch.device:
        """Return the device on which the model is loaded."""
        return next(self.model.parameters()).device

    @clean_gpu_cache
    @torch.no_grad()
    def batch_predict(
        self,
        seqs_list: list[str],
        num_recycles: int = 4,
        residue_index_offset: int = 512,
        chain_linker: str = "G" * 25,
    ) -> dict[str, torch.Tensor]:
        """Predict the 3D structure of proteins from a list of sequences.

        Args:
            seqs_list (list[str]): List of protein sequences.
            num_recycles (int, optional): Number of recycling steps. Defaults to 4.
            residue_index_offset (int, optional): Offset for the residue index. Defaults to 512.
            chain_linker (str, optional): A string of Gs representing the linker between the two chains. Defaults to "G"*25.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the predicted 3D structure of the protein.
        """

        # Collect single output
        output_list_dict: dict[str, list[torch.Tensor]] = defaultdict(list)
        for seqs in tqdm(seqs_list, desc="Predicting structures"):
            output = self.model.infer_(
                seqs,
                num_recycles=num_recycles,
                residue_index_offset=residue_index_offset,
                chain_linker=chain_linker,
            )

            output_list_dict["positions"].append(
                output["positions"][-1]
            )  # (B, L', 14, 3)
            output_list_dict["atom37_atom_exists"].append(
                output["atom37_atom_exists"]
            )  # (B, L', 37)
            output_list_dict["plddt"].append(output["plddt"])  # (B, L', 37)
            output_list_dict["predicted_aligned_error"].append(
                output["predicted_aligned_error"]
            )  # (B, L', L)

        # Concatenate outputs
        outputs_dict: dict[str, torch.Tensor] = {
            key: torch.concat(value) for key, value in output_list_dict.items()
        }

        return outputs_dict


def predict_structures(
    seqs_list: str | list[str], device: Device = None
) -> dict[str, torch.Tensor]:

    if isinstance(seqs_list, str):
        seqs_list = [seqs_list]

    # Predict structures using ESMFold
    if device is None:
        device = get_default_device()
    esmfold_model = ESMFold(device=device)
    outputs_dict = esmfold_model.batch_predict(seqs_list)

    # Recreate linker mask for poly-G
    atom37_atom_exists = outputs_dict["atom37_atom_exists"]  # (B, L', 37)
    linker_mask_ = torch.any(atom37_atom_exists, dim=-1)  # (B, L')
    assert (linker_mask_ == linker_mask_[0]).all()
    linker_mask = linker_mask_[0]  # (L',)

    # Apply linker mask to outputs
    positions = outputs_dict["positions"][:, linker_mask]  # (B, L, 14, 3)
    atom37_atom_exists = outputs_dict["atom37_atom_exists"][
        :, linker_mask
    ]  # (B, L, 37)
    plddt = outputs_dict["plddt"][:, linker_mask]  # (B, L, 37)
    predicted_aligned_error = outputs_dict["predicted_aligned_error"][
        :, linker_mask, :
    ][
        :, :, linker_mask
    ]  # (B, L, L)

    # Transform to final outputs
    aatype = torch.stack(
        [seq_to_aatype(seqs.replace(":", "")).to(device) for seqs in seqs_list]
    )  # (B, L)
    atom37_coords = atom.atom37_coords_from_atom14(positions, aatype)  # (B, L, 37, 3)
    all_atom_plddt = plddt.clone()  # (B, L, 37)
    # Extract C-alpha and take the average
    plddt = plddt[:, :, 1].mean(-1)  # (B,)
    pae = predicted_aligned_error.mean((-2, -1))  # (B,)

    return {
        "atom37_coords": atom37_coords,
        "plddt": plddt,
        "all_atom_plddt": all_atom_plddt,
        "pae": pae,
    }
