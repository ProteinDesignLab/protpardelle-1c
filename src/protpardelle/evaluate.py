"""Entrypoint for Protpardelle-1c self-consistency evals.

Authors: Alex Chu, Jinho Kim, Richard Shuai, Tianyu Lu, Zhaoyang Li
"""

from collections import defaultdict

import numpy as np
import torch
from jaxtyping import Float

from protpardelle.common import residue_constants
from protpardelle.data.align import (
    compute_allatom_structure_metric,
    compute_structure_metric,
)
from protpardelle.integrations.esmfold import predict_structures
from protpardelle.integrations.protein_mpnn import design_sequence


def _insert_chain_gaps(
    seq: str,
    chain_mask: Float[torch.Tensor, "L"],
) -> str:
    """Insert gaps in the sequence at chain boundaries.

    Args:
        seq (str): The input sequence.
        chain_mask (torch.Tensor): A mask indicating chain boundaries.

    Returns:
        str: The modified sequence with gaps inserted.
    """

    aa_list = [seq[0]]  # start with the first residue
    for i in range(1, len(seq)):
        if chain_mask[i] != chain_mask[i - 1]:  # detect chain boundary
            aa_list.append(":")  # insert gap
        aa_list.append(seq[i])  # append residue
    new_seq = "".join(aa_list)

    return new_seq


def compute_self_consistency(
    comparison_structures: list[torch.Tensor],  # can be sampled or ground truth
    trimmed_chain_index: torch.Tensor | None = None,
    sampled_sequences: list[str] | None = None,
    num_seqs: int = 1,
    motif_idx: list[list[int]] | None = None,
    motif_coords: torch.Tensor | None = None,
    motif_aatypes: torch.Tensor | None = None,
    allatom: bool = False,
    atom_mask: torch.Tensor | None = None,
    motif_atom_mask: torch.Tensor | None = None,
):

    aux = defaultdict(list)

    for i, coords in enumerate(comparison_structures):
        if sampled_sequences is None:
            input_aatype = (
                torch.ones(coords.shape[0], dtype=torch.long)
                * residue_constants.restype_order["G"]
            )
            fixed_pos_mask = torch.zeros_like(
                input_aatype
            )  #  0 for positions to redesign, 1 for positions to keep fixed
            if motif_aatypes is not None:
                # fix motif aatypes during design
                input_aatype[motif_idx[i]] = motif_aatypes[i]
                fixed_pos_mask[motif_idx[i]] = 1

            seqs_to_predict = design_sequence(
                coords,
                num_seqs=num_seqs,
                disallow_aas=["X"],
                chain_index=trimmed_chain_index[i].cpu(),
                input_aatype=input_aatype,
                fixed_pos_mask=fixed_pos_mask,
            )
        else:
            seqs_to_predict = [sampled_sequences[i][: coords.shape[0]]]

        if trimmed_chain_index is not None:
            multichain_sequences = []
            for seq in seqs_to_predict:
                modified_sequence = _insert_chain_gaps(seq, trimmed_chain_index[i])
                multichain_sequences.append(modified_sequence)
            seqs_to_predict = multichain_sequences

        esmfold_predictions = predict_structures(seqs_to_predict)

        pred_coords = esmfold_predictions["atom37_coords"]
        all_atom_plddts = esmfold_predictions["all_atom_plddt"]
        plddts = esmfold_predictions["plddt"]
        paes = esmfold_predictions["pae"]

        # compute full protein rmsd
        ca_scaffold_scrmsds = []
        allatom_scaffold_scrmsds = []
        for pred in pred_coords:
            if not allatom:
                ca_rmsd = compute_structure_metric(
                    coords.to(pred),
                    pred[:, :3],
                    metric="ca_rmsd",
                )
                ca_scaffold_scrmsds.append(ca_rmsd.item())
                allatom_scaffold_scrmsds.append(999)
            else:
                ca_rmsd = compute_structure_metric(
                    coords.to(pred),
                    pred,
                    metric="ca_rmsd",
                )
                allatom_rmsd = compute_allatom_structure_metric(
                    coords.to(pred),
                    pred,
                    metric="allatom_rmsd",
                    atom_mask=atom_mask[i],
                )
                ca_scaffold_scrmsds.append(ca_rmsd.item())
                allatom_scaffold_scrmsds.append(allatom_rmsd.item())

        # compute motif rmsd with sampled structure
        ca_motif_sample_rmsds = []
        allatom_motif_sample_rmsds = []
        if motif_idx is not None and motif_coords is not None:
            for pred in pred_coords:
                if not allatom:
                    ca_rmsd = compute_structure_metric(
                        motif_coords[i].to(pred),
                        coords[motif_idx[i], :3].to(pred),
                        metric="ca_rmsd",
                    )
                    ca_motif_sample_rmsds.append(ca_rmsd.item())
                    allatom_motif_sample_rmsds.append(999)
                else:
                    ca_rmsd = compute_structure_metric(
                        motif_coords[i].to(pred),
                        coords[motif_idx[i]].to(pred),
                        metric="ca_rmsd",
                    )
                    allatom_rmsd = compute_allatom_structure_metric(
                        motif_coords[i].to(pred),
                        coords[motif_idx[i]].to(pred),
                        metric="allatom_rmsd",
                        atom_mask=motif_atom_mask[i],
                    )
                    ca_motif_sample_rmsds.append(ca_rmsd.item())
                    allatom_motif_sample_rmsds.append(allatom_rmsd.item())
        else:
            ca_motif_sample_rmsds = [999 for _ in range(len(seqs_to_predict))]
            allatom_motif_sample_rmsds = [999 for _ in range(len(seqs_to_predict))]

        # compute motif rmsd with predicted structure, can do this with both backbone-only and allatom models
        ca_motif_pred_rmsds = []
        allatom_motif_pred_rmsds = []
        if motif_idx is not None and motif_coords is not None:
            for pred in pred_coords:
                ca_rmsd = compute_structure_metric(
                    motif_coords[i].to(pred),
                    pred[motif_idx[i]],
                    metric="ca_rmsd",
                )
                allatom_rmsd = compute_allatom_structure_metric(
                    motif_coords[i].to(pred),
                    pred[motif_idx[i]],
                    metric="allatom_rmsd",
                    atom_mask=motif_atom_mask[i],
                )
                ca_motif_pred_rmsds.append(ca_rmsd.item())
                allatom_motif_pred_rmsds.append(allatom_rmsd.item())
        else:
            ca_motif_pred_rmsds = [999 for _ in range(len(seqs_to_predict))]
            allatom_motif_pred_rmsds = [999.0 for _ in range(len(seqs_to_predict))]

        aux["pred"].extend(pred_coords)
        seqs_to_predict_arr = seqs_to_predict
        if isinstance(seqs_to_predict_arr, str):
            seqs_to_predict_arr = [seqs_to_predict_arr]

        aux["structure_index"].extend([i] * len(seqs_to_predict))
        aux["seqs"].extend(seqs_to_predict_arr)
        aux["all_atom_plddt"].extend(all_atom_plddts)
        aux["plddt"].extend(plddts)
        aux["pae"].extend(paes)
        aux["ca_scaffold_scrmsd"].extend(ca_scaffold_scrmsds)
        aux["allatom_scaffold_scrmsd"].extend(allatom_scaffold_scrmsds)
        aux["ca_motif_sample_rmsd"].extend(ca_motif_sample_rmsds)
        aux["allatom_motif_sample_rmsd"].extend(allatom_motif_sample_rmsds)
        aux["ca_motif_pred_rmsd"].extend(ca_motif_pred_rmsds)
        aux["allatom_motif_pred_rmsd"].extend(allatom_motif_pred_rmsds)

    for k, v in aux.items():
        if isinstance(v[0], float):
            aux[k] = np.array(v)

    return aux
