"""Utils for computing evaluation metrics.

Authors: Alex Chu, Jinho Kim, Richard Shuai, Tianyu Lu, Zhaoyang Li
"""

import json
import shutil
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchtyping import TensorType

import protpardelle.core.modules as modules
import protpardelle.data.pdb_io
from protpardelle.common import residue_constants
from protpardelle.data import align
from protpardelle.env import PROTEINMPNN_WEIGHTS
from protpardelle.integrations import protein_mpnn as mpnn
from protpardelle.integrations.esmfold import predict_structures


def design_sequence(
    coords,
    model=None,
    num_seqs=1,
    disallow_aas=["C"],
    tmp_prefix: str = "",
    chain_index: TensorType["n"] | None = None,
    input_aatype: TensorType["n"] | None = None,
    fixed_pos_mask: TensorType["n"] | None = None,
):
    # Returns list of strs; seqs like 'MKRLLDS', not aatypes
    if model is None:
        model = mpnn.get_mpnn_model(PROTEINMPNN_WEIGHTS)
    if isinstance(coords, str):
        using_tmp_dir = False
        pdb_fn = coords
        fixed_pos_jsonl = None
    else:
        # Create a temporary directory for storing pdb file + fixed positions jsonl for ProteinMPNN design
        unique_id = uuid.uuid4().hex  # unique ID for temp processing dir
        tmp_dir = f"tmp-{unique_id}"
        using_tmp_dir = True
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)

        pdb_fn = f"{tmp_dir}/{tmp_prefix}_tmp.pdb"

        if input_aatype is None:
            # default to all glycine sequence
            gly_idx = residue_constants.restype_order["G"]
            input_aatype = (torch.ones(coords.shape[0]) * gly_idx).long()

        if fixed_pos_mask is None:
            # design on all positions
            fixed_pos_mask = torch.zeros_like(input_aatype)

        protpardelle.data.pdb_io.write_coords_to_pdb(
            coords, pdb_fn, batched=False, aatype=input_aatype, chain_index=chain_index
        )

        # make fixed pos jsonl
        fixed_pos_jsonl = make_fixed_pos_jsonl(chain_index, fixed_pos_mask, pdb_fn)

    with torch.no_grad():
        designed_seqs = mpnn.run_proteinmpnn(
            model=model,
            pdb_path=pdb_fn,
            num_seq_per_target=num_seqs,
            omit_AAs=disallow_aas,
            fixed_positions_jsonl=fixed_pos_jsonl,
        )

    if using_tmp_dir:
        shutil.rmtree(tmp_dir)
    return designed_seqs


def make_fixed_pos_jsonl(
    chain_index: TensorType["n"], fixed_pos_mask: TensorType["n"], pdb_fn: str
) -> str:
    """
    Create a temporary jsonl file for fixed positions.
    Maps pdb filename to fixed positions (assuming chain index starts from chain A).
    ProteinMPNN expects 1-indexed indices into the sequence (not PDB residue indices).

    e.g.
        {"5TTA": {"A": [1, 2, 3, 7, 8, 9, 22, 25, 33], "B": []}, "3LIS": {"A": [], "B": []}}

    Args:
    - chain_index: (n,) tensor of 0-indexed chain indices for each residue
    - fixed_pos_mask: (n,) tensor of 0/1 for positions to fix, 1 for positions to fix, 0 for positions to redesign
    - pdb_fn: input pdb file
    """
    if fixed_pos_mask.sum() == 0:
        # skip if no fixed positions
        return ""

    fixed_pos_dict_i = {}
    for i in chain_index.long().unique():
        chain_mask = chain_index == i
        chain_fixed_pos_indices = (
            torch.nonzero(fixed_pos_mask[chain_mask], as_tuple=True)[0] + 1
        )  # 1-indexed indices
        chain_letter = chr(ord("A") + i.item())
        fixed_pos_dict_i[chain_letter] = chain_fixed_pos_indices.tolist()

    pdb_name = Path(pdb_fn).stem
    fixed_pos_dict = {pdb_name: fixed_pos_dict_i}
    fixed_pos_jsonl = pdb_fn.replace(".pdb", "-fixed_pos.jsonl")
    with open(fixed_pos_jsonl, "w") as f:
        json.dump(fixed_pos_dict, f)

    return fixed_pos_jsonl


def recursive_to(obj, **kwargs):
    # from omegafold.torch_utils
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(**kwargs)
        except RuntimeError:
            kwargs.pop("non_blocking")
            return obj.to(**kwargs)
    elif isinstance(obj, list):
        return [recursive_to(o, **kwargs) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, **kwargs) for o in obj)
    elif isinstance(obj, set):
        return set(recursive_to(o, **kwargs) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, **kwargs) for k, v in obj.items()}
    elif hasattr(obj, "to"):
        # this takes care of classes that implements the ~to method
        return obj.to(**kwargs)
    else:
        return obj


def compute_structure_metric(coords1, coords2, metric="ca_rmsd", atom_mask=None):
    # coords1 tensor[l][a][3]
    def _tmscore(a, b, mask=None):
        length = len(b)
        dists = (a - b).pow(2).sum(-1)
        d0 = 1.24 * ((length - 15) ** (1 / 3)) - 1.8
        term = 1 / (1 + ((dists) / (d0**2)))
        if mask is None:
            return term.mean()
        else:
            term = term * mask
            return term.sum() / mask.sum().clamp(min=1)

    aligned_coords1_ca, (R, t) = align.kabsch_align(coords1[:, 1], coords2[:, 1])
    aligned_coords1_based_on_ca = coords1 - coords1[:, 1:2].mean(0, keepdim=True)
    aligned_coords1_based_on_ca = aligned_coords1_based_on_ca @ R.t() + t

    if "allatom" in metric:
        atom_mask = atom_mask[: coords1.shape[0]].bool()
        _, (R, t) = align.kabsch_align(coords1[atom_mask], coords2[atom_mask])
        aligned_coords1 = coords1 - coords1[atom_mask].mean(0, keepdim=True)
        aligned_coords1 = aligned_coords1 @ R.t() + t
    else:
        aligned_coords1 = aligned_coords1_based_on_ca

    ca_rmsd = (aligned_coords1_ca - coords2[:, 1]).pow(2).sum(-1).mean().sqrt()

    if "allatom" in metric:
        aligned_coords1_masked = aligned_coords1 * atom_mask[..., None]
        coords2_masked = coords2 * atom_mask[..., None]
        allatom_rmsd = torch.sqrt(
            (aligned_coords1_masked - coords2_masked).pow(2).sum(-1).sum()
            / atom_mask.sum()
        )

    if metric == "ca_and_allatom_rmsd":
        return ca_rmsd, allatom_rmsd
    elif metric == "allatom_rmsd":
        return allatom_rmsd
    elif metric == "ca_rmsd":
        return ca_rmsd
    elif metric == "tm_score":
        tm = _tmscore(aligned_coords1_ca, coords2[:, 1])
        return tm
    elif metric == "allatom_tm":
        # Align on Ca, compute allatom TM
        assert atom_mask is not None
        return _tmscore(aligned_coords1, coords2, mask=atom_mask)
    elif metric == "allatom_lddt":
        assert atom_mask is not None
        lddt = modules.lddt(
            coords1.reshape(-1, 3),
            coords2.reshape(-1, 3),
            atom_mask.reshape(-1, 1),
            per_residue=False,
        )
        return lddt
    else:
        raise NotImplementedError


def compute_self_consistency(
    comparison_structures,  # can be sampled or ground truth
    trimmed_chain_index=None,
    sampled_sequences=None,
    mpnn_model=None,
    num_seqs: int = 1,
    motif_idx: list[list[int]] | None = None,
    motif_coords=None,
    motif_aatypes=None,
    tmp_prefix: str = "",
    allatom: bool = False,
    atom_mask=None,
    motif_atom_mask=None,
):
    aux = defaultdict(list)

    def insert_chain_gaps(sequence, chain_mask):
        modified_sequence = sequence[0]  # Start with the first residue
        for i in range(1, len(sequence)):
            if chain_mask[i] != chain_mask[i - 1]:  # Detect chain boundary
                modified_sequence += ":"  # Insert gap
            modified_sequence += sequence[i]  # Append residue
        return modified_sequence

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
                model=mpnn_model,
                num_seqs=num_seqs,
                tmp_prefix=tmp_prefix,
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
                modified_sequence = insert_chain_gaps(seq, trimmed_chain_index[i])
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
                    atom_mask=atom_mask[i],
                )
                ca_scaffold_scrmsds.append(ca_rmsd.item())
                allatom_scaffold_scrmsds.append(999)
            else:
                ca_rmsd, allatom_rmsd = compute_structure_metric(
                    coords.to(pred),
                    pred,
                    metric="ca_and_allatom_rmsd",
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
                        atom_mask=motif_atom_mask[i],
                    )
                    ca_motif_sample_rmsds.append(ca_rmsd.item())
                    allatom_motif_sample_rmsds.append(999)
                else:
                    ca_rmsd, allatom_rmsd = compute_structure_metric(
                        motif_coords[i].to(pred),
                        coords[motif_idx[i]].to(pred),
                        metric="ca_and_allatom_rmsd",
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
                ca_rmsd, allatom_rmsd = compute_structure_metric(
                    motif_coords[i].to(pred),
                    pred[motif_idx[i]],
                    metric="ca_and_allatom_rmsd",
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
