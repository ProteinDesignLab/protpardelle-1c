"""ESMFold wrapper functions.

Authors: Alex Chu, Tianyu Lu, Zhaoyang Li
"""

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import protpardelle.data.sequence
from protpardelle.data import atom
from protpardelle.env import ESMFOLD_PATH
from protpardelle.utils import get_default_device


def get_esmfold_model(model_path, device=None):
    if model_path is None:
        raise ValueError("Environment variable ESMFOLD_PATH not set")
    if device is None:
        device = get_default_device()
    model = torch.load(model_path).to(device)
    model.esm = model.esm.half()
    return model


def inference_esmfold(
    sequence_list,
    model,
    tokenizer,
    return_all_atom_plddt=False,
    huggingface=False,
    multichain=False,
):
    # * code adapted from ESMFold colab notebook
    # * https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ESMFold.ipynb

    if huggingface:
        inputs = tokenizer(
            sequence_list,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(model.device)

        outputs = model(**inputs)

        # positions is shape (l, b, n, a, c)
        pred_coords = outputs.positions[-1].contiguous()
        all_atom_plddt = outputs.plddt
        plddt = outputs.plddt[:, :, 1].mean(-1)
        pae = outputs.predicted_aligned_error.mean((-2, -1))

    else:
        outputs = {}
        print("Predicting structures...")
        for seq in tqdm(sequence_list):
            single_output = model.infer([seq], residue_index_offset=512)
            # Initialize output structure if first sequence
            if not outputs:
                outputs = {key: [] for key in single_output.keys()}
            # Append results
            for key, tensor in single_output.items():
                outputs[key].append(tensor)
        outputs["positions"] = torch.concat(outputs["positions"], dim=1)
        outputs["atom37_atom_exists"] = torch.concat(
            outputs["atom37_atom_exists"], dim=0
        )
        outputs["plddt"] = torch.concat(outputs["plddt"], dim=0)
        outputs["predicted_aligned_error"] = torch.concat(
            outputs["predicted_aligned_error"], dim=0
        )

        pred_coords = outputs["positions"][-1].contiguous()

        if multichain:
            mask = outputs["atom37_atom_exists"][:, :, 1] == 1

            B, N, C, D = pred_coords.shape

            # Get indices where mask is True
            indices = mask.nonzero(as_tuple=True)  # Returns (batch_idx, seq_idx)

            # Gather indices separately for each batch to avoid flattening
            batch_idx, seq_idx = indices  # Unpack batch and sequence indices

            # Use indexing to retain batch dimension
            selected_coords = pred_coords[
                batch_idx, seq_idx
            ]  # This results in a 1D array
            num_selected = seq_idx.view(B, -1).shape[
                1
            ]  # Number of selected items per batch
            pred_coords = selected_coords.view(
                B, num_selected, C, D
            )  # Restore batch dim

            B, N, C = outputs["plddt"].shape
            all_atom_plddt = outputs["plddt"][batch_idx, seq_idx].view(
                B, num_selected, C
            )
            plddt = all_atom_plddt[..., 1].mean(-1)

            seq_idx_per_batch = [seq_idx[batch_idx == b] for b in range(B)]
            pae = torch.stack(
                [
                    outputs["predicted_aligned_error"][b, seq_idx_per_batch[b]][
                        :, seq_idx_per_batch[b]
                    ]
                    for b in range(B)
                ]
            )
            pae = pae.mean((-2, -1))
        else:
            all_atom_plddt = outputs["plddt"]
            plddt = all_atom_plddt[..., 1].mean(-1)
            pae = outputs["predicted_aligned_error"].mean((-2, -1))

    if return_all_atom_plddt:
        return pred_coords, (all_atom_plddt, plddt, pae)
    else:
        return pred_coords, (plddt, pae)  # * conf metrics is a tuple of lists


def predict_structures(
    sequences, model="esmfold", tokenizer=None, return_all_atom_plddt=False
):
    # Expects seqs like 'MKRLLDS', not aatypes
    # model can be a model, or a string describing which pred model to load
    if isinstance(sequences, str):
        sequences = [sequences]
    if model == "esmfold":
        model = get_esmfold_model(ESMFOLD_PATH)
    device = model.device
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    aatype = [
        protpardelle.data.sequence.seq_to_aatype(seq.replace(":", "")).to(device)
        for seq in sequences
    ]

    with torch.no_grad():
        pred_coords, conf_metrics = inference_esmfold(
            sequences, model, tokenizer, return_all_atom_plddt=return_all_atom_plddt
        )

    seq_lens = []
    trimmed_coords = []
    trimmed_plddts = []
    for i, s in enumerate(sequences):
        if ":" in s:
            per_chain_lens = [len(chain_seq) for chain_seq in s.split(":")]
            seq_lens.append(per_chain_lens)
            trimmed_coords.append(
                torch.cat(
                    [
                        pred_coords[i, : per_chain_lens[0]],
                        pred_coords[
                            i,
                            per_chain_lens[0]
                            + 25 : per_chain_lens[0]
                            + 25
                            + per_chain_lens[1],
                        ],
                    ],
                    dim=-3,
                )
            )
            trimmed_plddts.append(
                torch.cat(
                    [
                        conf_metrics[0][i, : per_chain_lens[0]],
                        conf_metrics[0][
                            i,
                            per_chain_lens[0]
                            + 25 : per_chain_lens[0]
                            + 25
                            + per_chain_lens[1],
                        ],
                    ],
                    dim=-2,
                )
            )
        else:
            seq_lens.append(len(s))
            trimmed_coords.append(pred_coords[i, : len(s)])
            trimmed_plddts.append(conf_metrics[0][i])

    trimmed_coords = torch.stack(trimmed_coords)
    trimmed_plddts = torch.stack(trimmed_plddts)
    conf_metrics = (trimmed_plddts, conf_metrics[1], conf_metrics[2])

    trimmed_coords_atom37 = [
        atom.atom37_coords_from_atom14(c, aatype[i])
        for i, c in enumerate(trimmed_coords)
    ]
    return trimmed_coords_atom37, conf_metrics
