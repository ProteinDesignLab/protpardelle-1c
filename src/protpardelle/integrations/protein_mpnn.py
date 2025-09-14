# MIT License

# Copyright (c) 2022 Justas Dauparas

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""ProteinMPNN wrapper functions.

Adapted from https://github.com/dauparas/ProteinMPNN

Authors: Alex Chu, Zhaoyang Li
"""


from __future__ import annotations

import copy
import itertools
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.types import Device
from torchtyping import TensorType

from protpardelle.common import residue_constants
from protpardelle.common.protein import PDB_CHAIN_IDS, PDB_MAX_CHAINS
from protpardelle.data.pdb_io import write_coords_to_pdb
from protpardelle.env import PROJECT_ROOT_DIR, PROTEINMPNN_WEIGHTS
from protpardelle.utils import StrPath, get_default_device, seed_everything


def get_mpnn_model(
    model_name: Literal["v_48_002", "v_48_010", "v_48_020", "v_48_030"],
    device: Device = None,
) -> ProteinMPNN:
    if device is None:
        device = get_default_device()

    checkpoint_path = PROTEINMPNN_WEIGHTS / f"{model_name}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)  # type: ignore
    model = ProteinMPNN(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    )

    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


@torch.no_grad()
def run_protein_mpnn(
    model: ProteinMPNN,
    pdb_path="",
    pdb_path_chains="",
    model_name="v_48_020",
    seed: int | None = None,
    ca_only=False,
    out_folder="",
    num_seq_per_target=1,
    batch_size=1,
    sampling_temps=[0.1],
    max_length=200000,
    omit_AAs=[],
    print_all=False,
    chain_id_jsonl="",
    fixed_positions_jsonl="",
    pssm_jsonl="",
    omit_AA_jsonl="",
    bias_AA_jsonl="",
    tied_positions_jsonl="",
    bias_by_res_jsonl="",
    jsonl_path="",
    pssm_threshold=0.0,
    pssm_multi=0.0,
    pssm_log_odds_flag=False,
    pssm_bias_flag=False,
    write_output_files=False,
):

    if seed is not None:
        seed_everything(seed)

    NUM_BATCHES = num_seq_per_target // batch_size
    BATCH_COPIES = batch_size
    temperatures = sampling_temps
    omit_AAs_list = omit_AAs
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    device = get_default_device()
    if os.path.isfile(chain_id_jsonl):
        with open(chain_id_jsonl, "r") as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            chain_id_dict = json.loads(json_str)
    else:
        chain_id_dict = None
        if print_all:
            print(40 * "-")
            print("chain_id_jsonl is NOT loaded")

    if os.path.isfile(fixed_positions_jsonl):
        with open(fixed_positions_jsonl, "r") as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            fixed_positions_dict = json.loads(json_str)
    else:
        if print_all:
            print(40 * "-")
            print("fixed_positions_jsonl is NOT loaded")
        fixed_positions_dict = None

    if os.path.isfile(pssm_jsonl):
        with open(pssm_jsonl, "r") as json_file:
            json_list = list(json_file)
        pssm_dict = {}
        for json_str in json_list:
            pssm_dict.update(json.loads(json_str))
    else:
        if print_all:
            print(40 * "-")
            print("pssm_jsonl is NOT loaded")
        pssm_dict = None

    if os.path.isfile(omit_AA_jsonl):
        with open(omit_AA_jsonl, "r") as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            omit_AA_dict = json.loads(json_str)
    else:
        if print_all:
            print(40 * "-")
            print("omit_AA_jsonl is NOT loaded")
        omit_AA_dict = None

    if os.path.isfile(bias_AA_jsonl):
        with open(bias_AA_jsonl, "r") as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            bias_AA_dict = json.loads(json_str)
    else:
        if print_all:
            print(40 * "-")
            print("bias_AA_jsonl is NOT loaded")
        bias_AA_dict = None

    if os.path.isfile(tied_positions_jsonl):
        with open(tied_positions_jsonl, "r") as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            tied_positions_dict = json.loads(json_str)
    else:
        if print_all:
            print(40 * "-")
            print("tied_positions_jsonl is NOT loaded")
        tied_positions_dict = None

    if os.path.isfile(bias_by_res_jsonl):
        with open(bias_by_res_jsonl, "r") as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            bias_by_res_dict = json.loads(json_str)
        if print_all:
            print("bias by residue dictionary is loaded")
    else:
        if print_all:
            print(40 * "-")
            print("bias by residue dictionary is not loaded, or not provided")
        bias_by_res_dict = None

    if print_all:
        print(40 * "-")
    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
        for n, AA in enumerate(alphabet):
            if AA in list(bias_AA_dict.keys()):
                bias_AAs_np[n] = bias_AA_dict[AA]

    if pdb_path:
        pdb_dict_list = parse_PDB(pdb_path, ca_only=ca_only)
        dataset_valid = StructureDatasetPDB(
            pdb_dict_list, truncate=None, max_length=max_length
        )
        all_chain_list = [
            item[-1:] for item in list(pdb_dict_list[0]) if item[:9] == "seq_chain"
        ]  # ['A','B', 'C',...]
        if pdb_path_chains:
            designed_chain_list = [str(item) for item in pdb_path_chains.split()]
        else:
            designed_chain_list = all_chain_list
        fixed_chain_list = [
            letter for letter in all_chain_list if letter not in designed_chain_list
        ]
        chain_id_dict = {}
        chain_id_dict[pdb_dict_list[0]["name"]] = (
            designed_chain_list,
            fixed_chain_list,
        )
    else:
        dataset_valid = StructureDataset(
            jsonl_path, truncate=None, max_length=max_length, verbose=print_all
        )

    # Build paths for experiment
    if write_output_files:
        folder_for_outputs = out_folder
        base_folder = folder_for_outputs
        if base_folder[-1] != "/":
            base_folder = base_folder + "/"
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        if not os.path.exists(base_folder + "seqs"):
            os.makedirs(base_folder + "seqs")

    # Validation epoch
    new_mpnn_seqs = []

    for protein in dataset_valid:
        score_list = []
        global_score_list = []
        all_probs_list = []
        all_log_probs_list = []
        S_sample_list = []
        batch_clones = [copy.deepcopy(protein) for _ in range(BATCH_COPIES)]
        (
            X,
            S,
            mask,
            lengths,
            chain_M,
            chain_encoding_all,
            chain_list_list,
            visible_list_list,
            masked_list_list,
            masked_chain_length_list_list,
            chain_M_pos,
            omit_AA_mask,
            residue_idx,
            dihedral_mask,
            tied_pos_list_of_lists_list,
            pssm_coef,
            pssm_bias,
            pssm_log_odds_all,
            bias_by_res_all,
            tied_beta,
        ) = tied_featurize(
            batch_clones,
            device,
            chain_id_dict,
            fixed_positions_dict,
            omit_AA_dict,
            tied_positions_dict,
            pssm_dict,
            bias_by_res_dict,
            ca_only=ca_only,
        )
        pssm_log_odds_mask = (
            pssm_log_odds_all > pssm_threshold
        ).float()  # 1.0 for true, 0.0 for false
        name_ = batch_clones[0]["name"]
        randn_1 = torch.randn(chain_M.shape, device=X.device)
        log_probs = model(
            X,
            S,
            mask,
            chain_M * chain_M_pos,
            residue_idx,
            chain_encoding_all,
            randn_1,
        )
        mask_for_loss = mask * chain_M * chain_M_pos
        scores = _scores(
            S, log_probs, mask_for_loss
        )  # score only the redesigned part
        native_score = scores.cpu().data.numpy()
        global_scores = _scores(
            S, log_probs, mask
        )  # score the whole structure-sequence
        global_native_score = global_scores.cpu().data.numpy()
        # Generate some sequences
        if write_output_files:
            ali_file = base_folder + "/seqs/" + batch_clones[0]["name"] + ".fa"
            f = open(ali_file, "w")
        if print_all:
            print(f"Generating sequences for: {name_}")
        t0 = time.time()
        for temp in temperatures:
            for j in range(NUM_BATCHES):
                randn_2 = torch.randn(chain_M.shape, device=X.device)
                if tied_positions_dict is None:
                    sample_dict = model.sample(
                        X,
                        randn_2,
                        S,
                        chain_M,
                        chain_encoding_all,
                        residue_idx,
                        mask=mask,
                        temperature=temp,
                        omit_AAs_np=omit_AAs_np,
                        bias_AAs_np=bias_AAs_np,
                        chain_M_pos=chain_M_pos,
                        omit_AA_mask=omit_AA_mask,
                        pssm_coef=pssm_coef,
                        pssm_bias=pssm_bias,
                        pssm_multi=pssm_multi,
                        pssm_log_odds_flag=bool(pssm_log_odds_flag),
                        pssm_log_odds_mask=pssm_log_odds_mask,
                        pssm_bias_flag=bool(pssm_bias_flag),
                        bias_by_res=bias_by_res_all,
                    )
                else:
                    sample_dict = model.tied_sample(
                        X,
                        randn_2,
                        S,
                        chain_M,
                        chain_encoding_all,
                        residue_idx,
                        mask=mask,
                        temperature=temp,
                        omit_AAs_np=omit_AAs_np,
                        bias_AAs_np=bias_AAs_np,
                        chain_M_pos=chain_M_pos,
                        omit_AA_mask=omit_AA_mask,
                        pssm_coef=pssm_coef,
                        pssm_bias=pssm_bias,
                        pssm_multi=pssm_multi,
                        pssm_log_odds_flag=bool(pssm_log_odds_flag),
                        pssm_log_odds_mask=pssm_log_odds_mask,
                        pssm_bias_flag=bool(pssm_bias_flag),
                        tied_pos=tied_pos_list_of_lists_list[0],
                        tied_beta=tied_beta,
                        bias_by_res=bias_by_res_all,
                    )
                S_sample = sample_dict["S"]
                log_probs = model(
                    X,
                    S_sample,
                    mask,
                    chain_M * chain_M_pos,
                    residue_idx,
                    chain_encoding_all,
                    randn_2,
                    use_input_decoding_order=True,
                    decoding_order=sample_dict["decoding_order"],
                )
                mask_for_loss = mask * chain_M * chain_M_pos
                scores = _scores(S_sample, log_probs, mask_for_loss)
                scores = scores.cpu().data.numpy()

                global_scores = _scores(
                    S_sample, log_probs, mask
                )  # score the whole structure-sequence
                global_scores = global_scores.cpu().data.numpy()

                all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                all_log_probs_list.append(log_probs.cpu().data.numpy())
                S_sample_list.append(S_sample.cpu().data.numpy())
                for b_ix in range(BATCH_COPIES):
                    masked_chain_length_list = masked_chain_length_list_list[b_ix]
                    masked_list = masked_list_list[b_ix]
                    seq_recovery_rate = torch.sum(
                        torch.sum(
                            F.one_hot(S[b_ix], 21) * F.one_hot(S_sample[b_ix], 21),
                            axis=-1,
                        )
                        * mask_for_loss[b_ix]
                    ) / torch.sum(mask_for_loss[b_ix])
                    seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                    new_mpnn_seqs.append(seq)
                    score = scores[b_ix]
                    score_list.append(score)
                    global_score = global_scores[b_ix]
                    global_score_list.append(global_score)
                    native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                    if b_ix == 0 and j == 0 and temp == temperatures[0]:
                        start = 0
                        end = 0
                        list_of_AAs = []
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(native_seq[start:end])
                            start = end
                        native_seq = "".join(
                            list(np.array(list_of_AAs)[np.argsort(masked_list)])
                        )
                        l0 = 0
                        for mc_length in list(
                            np.array(masked_chain_length_list)[
                                np.argsort(masked_list)
                            ]
                        )[:-1]:
                            l0 += mc_length
                            native_seq = native_seq[:l0] + "/" + native_seq[l0:]
                            l0 += 1
                        sorted_masked_chain_letters = np.argsort(
                            masked_list_list[0]
                        )
                        print_masked_chains = [
                            masked_list_list[0][i]
                            for i in sorted_masked_chain_letters
                        ]
                        sorted_visible_chain_letters = np.argsort(
                            visible_list_list[0]
                        )
                        print_visible_chains = [
                            visible_list_list[0][i]
                            for i in sorted_visible_chain_letters
                        ]
                        native_score_print = np.format_float_positional(
                            np.float32(native_score.mean()),
                            unique=False,
                            precision=4,
                        )
                        global_native_score_print = np.format_float_positional(
                            np.float32(global_native_score.mean()),
                            unique=False,
                            precision=4,
                        )
                        try:
                            commit_str = subprocess.check_output(
                                [
                                    "git",
                                    "-C",
                                    str(PROJECT_ROOT_DIR),
                                    "rev-parse",
                                    "HEAD",
                                ],
                                stderr=subprocess.DEVNULL,
                                text=True,
                            ).strip()
                        except subprocess.CalledProcessError:
                            commit_str = "unknown"
                        print_model_name = (
                            "CA_model_name" if ca_only else "model_name"
                        )
                        if write_output_files:
                            f.write(
                                f">{name_}, score={native_score_print}, global_score={global_native_score_print}, fixed_chains={print_visible_chains}, designed_chains={print_masked_chains}, {print_model_name}={model_name}, git_hash={commit_str}, seed={seed}\n{native_seq}\n"
                            )
                    start = 0
                    end = 0
                    list_of_AAs = []
                    for mask_l in masked_chain_length_list:
                        end += mask_l
                        list_of_AAs.append(seq[start:end])
                        start = end

                    seq = "".join(
                        list(np.array(list_of_AAs)[np.argsort(masked_list)])
                    )
                    l0 = 0
                    for mc_length in list(
                        np.array(masked_chain_length_list)[np.argsort(masked_list)]
                    )[:-1]:
                        l0 += mc_length
                        seq = seq[:l0] + "/" + seq[l0:]
                        l0 += 1
                    score_print = np.format_float_positional(
                        np.float32(score), unique=False, precision=4
                    )
                    global_score_print = np.format_float_positional(
                        np.float32(global_score), unique=False, precision=4
                    )
                    seq_rec_print = np.format_float_positional(
                        np.float32(seq_recovery_rate.detach().cpu().numpy()),
                        unique=False,
                        precision=4,
                    )
                    sample_number = j * BATCH_COPIES + b_ix + 1
                    if write_output_files:
                        f.write(
                            f">T={temp}, sample={sample_number}, score={score_print}, global_score={global_score_print}, seq_recovery={seq_rec_print}\n{seq}\n"
                        )

        t1 = time.time()
        dt = round(float(t1 - t0), 4)
        num_seqs = len(temperatures) * NUM_BATCHES * BATCH_COPIES
        total_length = X.shape[1]
        if print_all:
            print(
                f"{num_seqs} sequences of length {total_length} generated in {dt} seconds"
            )
        if write_output_files:
            f.close()

    return new_mpnn_seqs


def make_fixed_pos_jsonl(
    chain_index: TensorType["n"], fixed_pos_mask: TensorType["n"], pdb_fn: str
) -> str:
    """Create a temporary jsonl file for fixed positions.

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


def design_sequence(
    coords: torch.Tensor,
    model: ProteinMPNN,
    num_seqs=1,
    disallow_aas=["C"],
    tmp_prefix: str = "",
    chain_index: TensorType["n"] | None = None,
    input_aatype: TensorType["n"] | None = None,
    fixed_pos_mask: TensorType["n"] | None = None,
):
    # Returns list of strs; seqs like 'MKRLLDS', not aatypes

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

        write_coords_to_pdb(
            coords, pdb_fn, aatype=input_aatype, chain_index=chain_index
        )

        # make fixed pos jsonl
        fixed_pos_jsonl = make_fixed_pos_jsonl(chain_index, fixed_pos_mask, pdb_fn)

    with torch.no_grad():
        designed_seqs = run_protein_mpnn(
            model=model,
            pdb_path=pdb_fn,
            num_seq_per_target=num_seqs,
            omit_AAs=disallow_aas,
            fixed_positions_jsonl=fixed_pos_jsonl,
        )

    if using_tmp_dir:
        shutil.rmtree(tmp_dir)
    return designed_seqs


########################################
# Adapted from https://github.com/dauparas/ProteinMPNN/blob/main/protein_mpnn_utils.py


def _scores(
    S: torch.Tensor,
    log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    criterion = nn.NLLLoss(reduction="none")
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores


def _S_to_seq(S: torch.Tensor, mask: torch.Tensor) -> str:
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    seq = "".join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])
    return seq


def parse_PDB_biounits(x, atoms=["N", "CA", "C"], chain=None):
    """
    input:  x = PDB filename
            atoms = atoms to extract (optional)
    output: (length, atoms, coords=(x,y,z)), sequence
    """

    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    alpha_3 = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "GAP",
    ]

    aa_3_N = {a: n for n, a in enumerate(alpha_3)}
    aa_N_1 = dict(enumerate(alpha_1))

    def N_to_AA(x):
        # [[0,1,2,3]] -> ["ARND"]
        x = np.array(x)
        if x.ndim == 1:
            x = x[None]
        return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    for line in open(x, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()

        if line[:6] == "HETATM" and line[17 : 17 + 3] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12 : 12 + 4].strip()
                resi = line[17 : 17 + 3]
                resn = line[22 : 22 + 5].strip()
                x, y, z = [float(line[i : (i + 8)]) for i in [30, 38, 46]]

                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1]) - 1
                else:
                    resa, resn = "", int(resn) - 1
                if resn < min_resn:
                    min_resn = resn
                if resn > max_resn:
                    max_resn = resn
                if resn not in xyz:
                    xyz[resn] = {}
                if resa not in xyz[resn]:
                    xyz[resn][resa] = {}
                if resn not in seq:
                    seq[resn] = {}
                if resa not in seq[resn]:
                    seq[resn][resa] = resi

                if atom not in xyz[resn][resa]:
                    xyz[resn][resa][atom] = np.array([x, y, z])

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    try:
        for resn in range(min_resn, max_resn + 1):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(aa_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(np.full(3, np.nan))
            else:
                for atom in atoms:
                    xyz_.append(np.full(3, np.nan))
        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_))
    except TypeError:
        return "no_chain", "no_chain"


def parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False):
    c = 0
    pdb_dict_list = []
    init_alphabet = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    if input_chain_list:
        chain_alphabet = input_chain_list

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ""
        for letter in chain_alphabet:
            sidechain_atoms = ["CA"] if ca_only else ["N", "CA", "C", "O"]
            xyz, seq = parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict["seq_chain_" + letter] = seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain["CA_chain_" + letter] = xyz.tolist()
                else:
                    coords_dict_chain["N_chain_" + letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain["CA_chain_" + letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain["C_chain_" + letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain["O_chain_" + letter] = xyz[:, 3, :].tolist()
                my_dict["coords_chain_" + letter] = coords_dict_chain
                s += 1
        fi = biounit.rfind("/")
        my_dict["name"] = biounit[(fi + 1) : -4]
        my_dict["num_of_chains"] = s
        my_dict["seq"] = concat_seq
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c += 1
    return pdb_dict_list


def tied_featurize(
    batch,
    device,
    chain_dict,
    fixed_position_dict=None,
    omit_AA_dict=None,
    tied_positions_dict=None,
    pssm_dict=None,
    bias_by_res_dict=None,
    ca_only=False,
):
    """Pack and pad batch into torch tensors"""
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    B = len(batch)
    lengths = np.array(
        [len(b["seq"]) for b in batch], dtype=np.int32
    )  # sum of chain seq lengths
    L_max = max([len(b["seq"]) for b in batch])
    if ca_only:
        X = np.zeros([B, L_max, 1, 3])
    else:
        X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted
    pssm_coef_all = np.zeros(
        [B, L_max], dtype=np.float32
    )  # 1.0 for the bits that need to be predicted
    pssm_bias_all = np.zeros(
        [B, L_max, 21], dtype=np.float32
    )  # 1.0 for the bits that need to be predicted
    pssm_log_odds_all = 10000.0 * np.ones(
        [B, L_max, 21], dtype=np.float32
    )  # 1.0 for the bits that need to be predicted
    chain_M_pos = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted
    bias_by_res_all = np.zeros([B, L_max, 21], dtype=np.float32)
    chain_encoding_all = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted
    S = np.zeros([B, L_max], dtype=np.int32)
    omit_AA_mask = np.zeros([B, L_max, len(alphabet)], dtype=np.int32)
    # Build the batch
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []
    for i, b in enumerate(batch):
        if chain_dict != None:
            masked_chains, visible_chains = chain_dict[
                b["name"]
            ]  # masked_chains a list of chain letters to predict [A, D, F]
        else:
            masked_chains = [item[-1:] for item in list(b) if item[:10] == "seq_chain_"]
            visible_chains = []
        masked_chains.sort()  # sort masked_chains
        visible_chains.sort()  # sort visible_chains
        all_chains = masked_chains + visible_chains
    for i, b in enumerate(batch):
        a = 0
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        letter_list = []
        global_idx_start_list = [0]
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list = []
        pssm_coef_list = []
        pssm_bias_list = []
        pssm_log_odds_list = []
        bias_by_res_list = []
        l0 = 0
        l1 = 0
        for letter in all_chains:
            if letter in visible_chains:
                letter_list.append(letter)
                visible_list.append(letter)
                chain_seq = b[f"seq_chain_{letter}"]
                chain_seq = "".join([a if a != "-" else "X" for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                chain_coords = b[f"coords_chain_{letter}"]  # this is a dictionary
                chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
                if ca_only:
                    x_chain = np.array(
                        chain_coords[f"CA_chain_{letter}"]
                    )  # [chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:, None, :]
                else:
                    x_chain = np.stack(
                        [
                            chain_coords[c]
                            for c in [
                                f"N_chain_{letter}",
                                f"CA_chain_{letter}",
                                f"C_chain_{letter}",
                                f"O_chain_{letter}",
                            ]
                        ],
                        1,
                    )  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                bias_by_res_list.append(np.zeros([chain_length, 21]))
            if letter in masked_chains:
                masked_list.append(letter)
                letter_list.append(letter)
                chain_seq = b[f"seq_chain_{letter}"]
                chain_seq = "".join([a if a != "-" else "X" for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                masked_chain_length_list.append(chain_length)
                chain_coords = b[f"coords_chain_{letter}"]  # this is a dictionary
                chain_mask = np.ones(chain_length)  # 1.0 for masked
                if ca_only:
                    x_chain = np.array(
                        chain_coords[f"CA_chain_{letter}"]
                    )  # [chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:, None, :]
                else:
                    x_chain = np.stack(
                        [
                            chain_coords[c]
                            for c in [
                                f"N_chain_{letter}",
                                f"CA_chain_{letter}",
                                f"C_chain_{letter}",
                                f"O_chain_{letter}",
                            ]
                        ],
                        1,
                    )  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                if fixed_position_dict != None:
                    fixed_pos_list = fixed_position_dict[b["name"]][letter]
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list) - 1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                if omit_AA_dict != None:
                    for item in omit_AA_dict[b["name"]][letter]:
                        idx_AA = np.array(item[0]) - 1
                        AA_idx = np.array(
                            [
                                np.argwhere(np.array(list(alphabet)) == AA)[0][0]
                                for AA in item[1]
                            ]
                        ).repeat(idx_AA.shape[0])
                        idx_ = np.array([[a, b] for a in idx_AA for b in AA_idx])
                        omit_AA_mask_temp[idx_[:, 0], idx_[:, 1]] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                if pssm_dict and pssm_dict[b["name"]][letter]:
                    pssm_coef = pssm_dict[b["name"]][letter]["pssm_coef"]
                    pssm_bias = pssm_dict[b["name"]][letter]["pssm_bias"]
                    pssm_log_odds = pssm_dict[b["name"]][letter]["pssm_log_odds"]
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                if bias_by_res_dict:
                    bias_by_res_list.append(bias_by_res_dict[b["name"]][letter])
                else:
                    bias_by_res_list.append(np.zeros([chain_length, 21]))

        letter_list_np = np.array(letter_list)
        tied_pos_list_of_lists = []
        tied_beta = np.ones(L_max)
        if tied_positions_dict != None:
            tied_pos_list = tied_positions_dict[b["name"]]
            if tied_pos_list:
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[
                            np.argwhere(letter_list_np == k)[0][0]
                        ]
                        if isinstance(v[0], list):
                            for v_count in range(len(v[0])):
                                one_list.append(
                                    start_idx + v[0][v_count] - 1
                                )  # make 0 to be the first
                                tied_beta[start_idx + v[0][v_count] - 1] = v[1][v_count]
                        else:
                            for v_ in v:
                                one_list.append(
                                    start_idx + v_ - 1
                                )  # make 0 to be the first
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)

        x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(
            chain_mask_list, 0
        )  # [L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)
        m_pos = np.concatenate(
            fixed_position_mask_list, 0
        )  # [L,], 1.0 for places that need to be predicted

        pssm_coef_ = np.concatenate(
            pssm_coef_list, 0
        )  # [L,], 1.0 for places that need to be predicted
        pssm_bias_ = np.concatenate(
            pssm_bias_list, 0
        )  # [L,], 1.0 for places that need to be predicted
        pssm_log_odds_ = np.concatenate(
            pssm_log_odds_list, 0
        )  # [L,], 1.0 for places that need to be predicted

        bias_by_res_ = np.concatenate(
            bias_by_res_list, 0
        )  # [L,21], 0.0 for places where AA frequencies don't need to be tweaked

        l = len(all_sequence)
        x_pad = np.pad(
            x, [[0, L_max - l], [0, 0], [0, 0]], "constant", constant_values=(np.nan,)
        )
        X[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, L_max - l]], "constant", constant_values=(0.0,))
        m_pos_pad = np.pad(m_pos, [[0, L_max - l]], "constant", constant_values=(0.0,))
        omit_AA_mask_pad = np.pad(
            np.concatenate(omit_AA_mask_list, 0),
            [[0, L_max - l]],
            "constant",
            constant_values=(0.0,),
        )
        chain_M[i, :] = m_pad
        chain_M_pos[i, :] = m_pos_pad
        omit_AA_mask[i,] = omit_AA_mask_pad

        chain_encoding_pad = np.pad(
            chain_encoding, [[0, L_max - l]], "constant", constant_values=(0.0,)
        )
        chain_encoding_all[i, :] = chain_encoding_pad

        pssm_coef_pad = np.pad(
            pssm_coef_, [[0, L_max - l]], "constant", constant_values=(0.0,)
        )
        pssm_bias_pad = np.pad(
            pssm_bias_, [[0, L_max - l], [0, 0]], "constant", constant_values=(0.0,)
        )
        pssm_log_odds_pad = np.pad(
            pssm_log_odds_, [[0, L_max - l], [0, 0]], "constant", constant_values=(0.0,)
        )

        pssm_coef_all[i, :] = pssm_coef_pad
        pssm_bias_all[i, :] = pssm_bias_pad
        pssm_log_odds_all[i, :] = pssm_log_odds_pad

        bias_by_res_pad = np.pad(
            bias_by_res_, [[0, L_max - l], [0, 0]], "constant", constant_values=(0.0,)
        )
        bias_by_res_all[i, :] = bias_by_res_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0

    # Conversion
    pssm_coef_all = torch.from_numpy(pssm_coef_all).to(dtype=torch.float, device=device)
    pssm_bias_all = torch.from_numpy(pssm_bias_all).to(dtype=torch.float, device=device)
    pssm_log_odds_all = torch.from_numpy(pssm_log_odds_all).to(
        dtype=torch.float, device=device
    )

    tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float, device=device)

    jumps = ((residue_idx[:, 1:] - residue_idx[:, :-1]) == 1).astype(np.float32)
    bias_by_res_all = torch.from_numpy(bias_by_res_all).to(
        dtype=torch.float, device=device
    )
    phi_mask = np.pad(jumps, [[0, 0], [1, 0]])
    psi_mask = np.pad(jumps, [[0, 0], [0, 1]])
    omega_mask = np.pad(jumps, [[0, 0], [0, 1]])
    dihedral_mask = np.concatenate(
        [phi_mask[:, :, None], psi_mask[:, :, None], omega_mask[:, :, None]], -1
    )  # [B,L,3]
    dihedral_mask = torch.from_numpy(dihedral_mask).to(dtype=torch.float, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float, device=device)
    chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float, device=device)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(
        dtype=torch.long, device=device
    )
    if ca_only:
        X_out = X[:, :, 0]
    else:
        X_out = X
    return (
        X_out,
        S,
        mask,
        lengths,
        chain_M,
        chain_encoding_all,
        letter_list_list,
        visible_list_list,
        masked_list_list,
        masked_chain_length_list_list,
        chain_M_pos,
        omit_AA_mask,
        residue_idx,
        dihedral_mask,
        tied_pos_list_of_lists_list,
        pssm_coef_all,
        pssm_bias_all,
        pssm_log_odds_all,
        bias_by_res_all,
        tied_beta,
    )


class StructureDataset:
    def __init__(
        self,
        jsonl_file,
        verbose=True,
        truncate=None,
        max_length=100,
        alphabet="ACDEFGHIKLMNPQRSTVWYX-",
    ):
        alphabet_set = set([a for a in alphabet])
        discard_count = {"bad_chars": 0, "too_long": 0, "bad_seq_length": 0}

        with open(jsonl_file) as f:
            self.data = []

            lines = f.readlines()
            start = time.time()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry["seq"]
                name = entry["name"]

                # Check if in alphabet
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry["seq"]) <= max_length:
                        if True:
                            self.data.append(entry)
                        else:
                            discard_count["bad_seq_length"] += 1
                    else:
                        discard_count["too_long"] += 1
                else:
                    if verbose:
                        print(name, bad_chars, entry["seq"])
                    discard_count["bad_chars"] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print(
                        "{} entries ({} loaded) in {:.1f} s".format(
                            len(self.data), i + 1, elapsed
                        )
                    )
            if verbose:
                print("discarded", discard_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureDatasetPDB:
    def __init__(
        self,
        pdb_dict_list,
        truncate=None,
        max_length=100,
        alphabet="ACDEFGHIKLMNPQRSTVWYX-",
    ):
        alphabet_set = set([a for a in alphabet])
        discard_count = {"bad_chars": 0, "too_long": 0, "bad_seq_length": 0}

        self.data = []

        for entry in pdb_dict_list:
            seq = entry["seq"]

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry["seq"]) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count["too_long"] += 1
            else:
                discard_count["bad_chars"] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        scale=30,
        time_cond_dim=None,
    ):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        if time_cond_dim is not None:
            self.time_block1 = nn.Sequential(
                Rearrange("b 1 d -> b 1 1 d"),
                nn.SiLU(),
                nn.Linear(time_cond_dim, num_hidden * 2),
            )
            self.time_block2 = nn.Sequential(
                Rearrange("b 1 d -> b 1 1 d"),
                nn.SiLU(),
                nn.Linear(time_cond_dim, num_hidden * 2),
            )

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W12 = nn.Linear(num_hidden, num_hidden)
        self.W13 = nn.Linear(num_hidden, num_hidden)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None, time_cond=None):
        """Parallel computation of full transformer layer"""

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)

        h_message = self.act(self.W2(self.act(self.W1(h_EV))))
        if time_cond is not None:
            scale, shift = self.time_block1(time_cond).chunk(2, dim=-1)
            h_message = h_message * (scale + 1) + shift
        h_message = self.W3(h_message)

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)

        h_message = self.act(self.W12(self.act(self.W11(h_EV))))
        if time_cond is not None:
            scale, shift = self.time_block2(time_cond).chunk(2, dim=-1)
            h_message = h_message * (scale + 1) + shift
        h_message = self.W13(h_message)

        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        scale=30,
        time_cond_dim=None,
    ):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        if time_cond_dim is not None:
            self.time_block = nn.Sequential(
                Rearrange("b 1 d -> b 1 1 d"),
                nn.SiLU(),
                nn.Linear(time_cond_dim, num_hidden * 2),
            )

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None, time_cond=None):
        """Parallel computation of full transformer layer"""

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.act(self.W2(self.act(self.W1(h_EV))))
        if time_cond is not None:
            scale, shift = self.time_block(time_cond).chunk(2, dim=-1)
            h_message = h_message * (scale + 1) + shift
        h_message = self.W3(h_message)

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff)
        self.W_out = nn.Linear(num_ff, num_hidden)
        self.act = nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = F.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class CAProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
    ):
        """Extract protein features"""
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # Normalization and embedding
        node_in, edge_in = 3, num_positional_embeddings + num_rbf * 9 + 7
        self.node_embedding = nn.Linear(node_in, node_features, bias=False)  # NOT USED
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _quaternions(self, R):
        """Convert a batch of 3D rotations [R] to quaternions [Q]
        R [...,3,3]
        Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(
            torch.abs(
                1
                + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1)
            )
        )
        _R = lambda i, j: R[:, :, :, i, j]
        signs = torch.sign(
            torch.stack(
                [_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1
            )
        )
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        return Q

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        dX = X[:, 1:, :] - X[:, :-1, :]
        dX_norm = torch.norm(dX, dim=-1)
        dX_mask = (dX_norm > 3.6) & (dX_norm < 4.0)
        dX = dX * dX_mask[:, :, None]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.linalg.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.linalg.cross(u_1, u_0), dim=-1)

        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1 + eps, 1 - eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack(
            (torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2
        )
        AD_features = F.pad(AD_features, (0, 0, 1, 2), "constant", 0)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.linalg.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0, 0, 1, 2), "constant", 0)
        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3, 3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU, Q), dim=-1)
        return AD_features, O_features

    def _dist(self, X, mask, eps=1e-6):
        """Pairwise euclidean distances"""
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, Ca, mask, residue_idx, chain_labels):
        """Featurize coordinates as an attributed graph"""
        if self.augment_eps > 0:
            Ca = Ca + self.augment_eps * torch.randn_like(Ca)

        D_neighbors, E_idx, mask_neighbors = self._dist(Ca, mask)

        Ca_0 = torch.zeros(Ca.shape, device=Ca.device)
        Ca_2 = torch.zeros(Ca.shape, device=Ca.device)
        Ca_0[:, 1:, :] = Ca[:, :-1, :]
        Ca_1 = Ca
        Ca_2[:, :-1, :] = Ca[:, 1:, :]

        V, O_features = self._orientations_coarse(Ca, E_idx)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca_1-Ca_1
        RBF_all.append(self._get_rbf(Ca_0, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_2, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_0, Ca_1, E_idx))
        RBF_all.append(self._get_rbf(Ca_0, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_1, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_1, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_2, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_2, Ca_1, E_idx))

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all, O_features), -1)

        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return E, E_idx


class ProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
    ):
        """Extract protein features"""
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        # X is ca coords (b, n, 3), mask is seq mask
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.linalg.cross(b, c)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class ProteinMPNN(nn.Module):
    def __init__(
        self,
        num_letters: int,
        node_features: int,
        edge_features: int,
        hidden_dim: int,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab: int = 21,
        k_neighbors: int = 64,
        augment_eps: float = 0.05,
        dropout: float = 0.1,
        ca_only: bool = False,
        time_cond_dim: int | None = None,
        input_S_is_embeddings: bool = False,
    ):
        super().__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        if ca_only:
            self.features = CAProteinFeatures(
                node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps
            )
            self.W_v = nn.Linear(node_features, hidden_dim)
        else:
            self.features = ProteinFeatures(
                node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps
            )

        self.W_e = nn.Linear(edge_features, hidden_dim)
        self.input_S_is_embeddings = input_S_is_embeddings
        if not self.input_S_is_embeddings:
            self.W_s = nn.Embedding(vocab, hidden_dim)

        if time_cond_dim is not None:
            self.time_block = nn.Sequential(
                nn.SiLU(), nn.Linear(time_cond_dim, hidden_dim)
            )

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                EncLayer(
                    hidden_dim,
                    hidden_dim * 2,
                    dropout=dropout,
                    time_cond_dim=time_cond_dim,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                DecLayer(
                    hidden_dim,
                    hidden_dim * 3,
                    dropout=dropout,
                    time_cond_dim=time_cond_dim,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.W_out = nn.Linear(hidden_dim, num_letters)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        randn: torch.Tensor,
        use_input_decoding_order=False,
        decoding_order=None,
        causal_mask=True,
        time_cond=None,
        return_node_embs=False,
    ):
        """Graph-conditioned sequence model"""
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        if time_cond is not None:
            time_cond_nodes = self.time_block(time_cond)
            h_V += time_cond_nodes  # time_cond is b, 1, c
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend, time_cond=time_cond)

        encoder_embs = h_V

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = S if self.input_S_is_embeddings else self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask  # update chain_M to include missing regions
        mask_size = E_idx.shape[1]
        if causal_mask:
            if not use_input_decoding_order:
                decoding_order = torch.argsort(
                    (chain_M + 0.0001) * (torch.abs(randn))
                )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            permutation_matrix_reverse = F.one_hot(
                decoding_order, num_classes=mask_size
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
        else:
            order_mask_backward = torch.ones(
                X.shape[0], mask_size, mask_size, device=device
            )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see.
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask, time_cond=time_cond)

        if return_node_embs:
            return h_V, encoder_embs
        logits = self.W_out(h_V)
        return F.log_softmax(logits, dim=-1)

    def sample(
        self,
        X: torch.Tensor,
        randn: torch.Tensor,
        S_true: torch.Tensor,
        chain_mask: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        residue_idx: torch.Tensor,
        mask: torch.Tensor,
        temperature=1.0,
        omit_AAs_np=None,
        bias_AAs_np=None,
        chain_M_pos=None,
        omit_AA_mask=None,
        pssm_coef=None,
        pssm_bias=None,
        pssm_multi=None,
        pssm_log_odds_flag=None,
        pssm_log_odds_mask=None,
        pssm_bias_flag=None,
        bias_by_res=None,
    ):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = (
            chain_mask * chain_M_pos * mask
        )  # update chain_M to include missing regions
        decoding_order = torch.argsort(
            (chain_mask + 0.0001) * (torch.abs(randn))
        )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = F.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        all_probs = torch.zeros(
            (N_batch, N_nodes, 21), device=device, dtype=torch.float
        )
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.long, device=device)
        h_V_stack = [h_V] + [
            torch.zeros_like(h_V, device=device)
            for _ in range(len(self.decoder_layers))
        ]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        # chain_mask_combined = chain_mask*chain_M_pos
        omit_AA_mask_flag = omit_AA_mask != None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_ in range(N_nodes):
            t = decoding_order[:, t_]  # [B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])  # [B]
            mask_gathered = torch.gather(mask, 1, t[:, None])  # [B]
            bias_by_res_gathered = torch.gather(
                bias_by_res, 1, t[:, None, None].repeat(1, 1, 21)
            )[
                :, 0, :
            ]  # [B, 21]
            if (mask_gathered == 0).all():  # for padded or missing regions only
                S_t = torch.gather(S_true, 1, t[:, None])
            else:
                # Hidden layers
                E_idx_t = torch.gather(
                    E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1])
                )
                h_E_t = torch.gather(
                    h_E,
                    1,
                    t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]),
                )
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(
                    h_EXV_encoder_fw,
                    1,
                    t[:, None, None, None].repeat(
                        1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]
                    ),
                )
                mask_t = torch.gather(mask, 1, t[:, None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(
                        h_V_stack[l],
                        1,
                        t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]),
                    )
                    h_ESV_t = (
                        torch.gather(
                            mask_bw,
                            1,
                            t[:, None, None, None].repeat(
                                1, 1, mask_bw.shape[-2], mask_bw.shape[-1]
                            ),
                        )
                        * h_ESV_decoder_t
                        + h_EXV_encoder_t
                    )
                    h_V_stack[l + 1].scatter_(
                        1,
                        t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                        layer(h_V_t, h_ESV_t, mask_V=mask_t),
                    )
                # Sampling step
                h_V_t = torch.gather(
                    h_V_stack[-1],
                    1,
                    t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]),
                )[:, 0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(
                    logits
                    - constant[None, :] * 1e8
                    + constant_bias[None, :] / temperature
                    + bias_by_res_gathered / temperature,
                    dim=-1,
                )
                if pssm_bias_flag:
                    pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:, None])[:, 0]
                    pssm_bias_gathered = torch.gather(
                        pssm_bias, 1, t[:, None, None].repeat(1, 1, pssm_bias.shape[-1])
                    )[:, 0]
                    probs = (
                        1 - pssm_multi * pssm_coef_gathered[:, None]
                    ) * probs + pssm_multi * pssm_coef_gathered[
                        :, None
                    ] * pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = torch.gather(
                        pssm_log_odds_mask,
                        1,
                        t[:, None, None].repeat(1, 1, pssm_log_odds_mask.shape[-1]),
                    )[
                        :, 0
                    ]  # [B, 21]
                    probs_masked = probs * pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked / torch.sum(
                        probs_masked, dim=-1, keepdim=True
                    )  # [B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(
                        omit_AA_mask,
                        1,
                        t[:, None, None].repeat(1, 1, omit_AA_mask.shape[-1]),
                    )[
                        :, 0
                    ]  # [B, 21]
                    probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                    probs = probs_masked / torch.sum(
                        probs_masked, dim=-1, keepdim=True
                    )  # [B, 21]
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(
                    1,
                    t[:, None, None].repeat(1, 1, 21),
                    (
                        chain_mask_gathered[
                            :,
                            :,
                            None,
                        ]
                        * probs[:, None, :]
                    ).float(),
                )
            S_true_gathered = torch.gather(S_true, 1, t[:, None])
            S_t = (
                S_t * chain_mask_gathered
                + S_true_gathered * (1.0 - chain_mask_gathered)
            ).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:, None, None].repeat(1, 1, temp1.shape[-1]), temp1)
            S.scatter_(1, t[:, None], S_t)
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict

    def tied_sample(
        self,
        X: torch.Tensor,
        randn: torch.Tensor,
        S_true: torch.Tensor,
        chain_mask: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        residue_idx: torch.Tensor,
        mask: torch.Tensor,
        temperature=1.0,
        omit_AAs_np=None,
        bias_AAs_np=None,
        chain_M_pos=None,
        omit_AA_mask=None,
        pssm_coef=None,
        pssm_bias=None,
        pssm_multi=None,
        pssm_log_odds_flag=None,
        pssm_log_odds_mask=None,
        pssm_bias_flag=None,
        tied_pos=None,
        tied_beta=None,
        bias_by_res=None,
    ):
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)
        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = (
            chain_mask * chain_M_pos * mask
        )  # update chain_M to include missing regions
        decoding_order = torch.argsort(
            (chain_mask + 0.0001) * (torch.abs(randn))
        )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]

        new_decoding_order = []
        for t_dec in list(decoding_order[0,].cpu().data.numpy()):
            if t_dec not in list(itertools.chain(*new_decoding_order)):
                if list_a := [item for item in tied_pos if t_dec in item]:
                    new_decoding_order.append(list_a[0])
                else:
                    new_decoding_order.append([t_dec])
        decoding_order = torch.tensor(
            list(itertools.chain(*new_decoding_order)), device=device
        )[
            None,
        ].repeat(
            X.shape[0], 1
        )

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = F.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        all_probs = torch.zeros(
            (N_batch, N_nodes, 21), device=device, dtype=torch.float
        )
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.long, device=device)
        h_V_stack = [h_V] + [
            torch.zeros_like(h_V, device=device)
            for _ in range(len(self.decoder_layers))
        ]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        omit_AA_mask_flag = omit_AA_mask != None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_list in new_decoding_order:
            logits = 0.0
            logit_list = []
            done_flag = False
            for t in t_list:
                if (mask[:, t] == 0).all():
                    S_t = S_true[:, t]
                    for t in t_list:
                        h_S[:, t, :] = self.W_s(S_t)
                        S[:, t] = S_t
                    done_flag = True
                    break
                else:
                    E_idx_t = E_idx[:, t : t + 1, :]
                    h_E_t = h_E[:, t : t + 1, :, :]
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = h_EXV_encoder_fw[:, t : t + 1, :, :]
                    mask_t = mask[:, t : t + 1]
                    for l, layer in enumerate(self.decoder_layers):
                        h_ESV_decoder_t = cat_neighbors_nodes(
                            h_V_stack[l], h_ES_t, E_idx_t
                        )
                        h_V_t = h_V_stack[l][:, t : t + 1, :]
                        h_ESV_t = (
                            mask_bw[:, t : t + 1, :, :] * h_ESV_decoder_t
                            + h_EXV_encoder_t
                        )
                        h_V_stack[l + 1][:, t, :] = layer(
                            h_V_t, h_ESV_t, mask_V=mask_t
                        ).squeeze(1)
                    h_V_t = h_V_stack[-1][:, t, :]
                    logit_list.append((self.W_out(h_V_t) / temperature) / len(t_list))
                    logits += (
                        tied_beta[t] * (self.W_out(h_V_t) / temperature) / len(t_list)
                    )
            if not done_flag:
                bias_by_res_gathered = bias_by_res[:, t, :]  # [B, 21]
                probs = F.softmax(
                    logits
                    - constant[None, :] * 1e8
                    + constant_bias[None, :] / temperature
                    + bias_by_res_gathered / temperature,
                    dim=-1,
                )
                if pssm_bias_flag:
                    pssm_coef_gathered = pssm_coef[:, t]
                    pssm_bias_gathered = pssm_bias[:, t]
                    probs = (
                        1 - pssm_multi * pssm_coef_gathered[:, None]
                    ) * probs + pssm_multi * pssm_coef_gathered[
                        :, None
                    ] * pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = pssm_log_odds_mask[:, t]
                    probs_masked = probs * pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked / torch.sum(
                        probs_masked, dim=-1, keepdim=True
                    )  # [B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = omit_AA_mask[:, t]
                    probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                    probs = probs_masked / torch.sum(
                        probs_masked, dim=-1, keepdim=True
                    )  # [B, 21]
                S_t_repeat = torch.multinomial(probs, 1).squeeze(-1)
                S_t_repeat = (
                    chain_mask[:, t] * S_t_repeat
                    + (1 - chain_mask[:, t]) * S_true[:, t]
                ).long()  # hard pick fixed positions
                for t in t_list:
                    h_S[:, t, :] = self.W_s(S_t_repeat)
                    S[:, t] = S_t_repeat
                    all_probs[:, t, :] = probs.float()
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return output_dict

    def conditional_probs(
        self,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        randn,
        backbone_only=False,
    ):
        """Graph-conditioned sequence model"""
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V_enc = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V_enc, h_E = layer(h_V_enc, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

        chain_M = chain_M * mask  # update chain_M to include missing regions

        chain_M_np = chain_M.cpu().numpy()
        idx_to_loop = np.argwhere(chain_M_np[0, :] == 1)[:, 0]
        log_conditional_probs = torch.zeros(
            [X.shape[0], chain_M.shape[1], 21], device=device
        ).float()

        for idx in idx_to_loop:
            h_V = torch.clone(h_V_enc)
            order_mask = torch.zeros(chain_M.shape[1], device=device).float()
            if backbone_only:
                order_mask = torch.ones(chain_M.shape[1], device=device).float()
                order_mask[idx] = 0.0
            else:
                order_mask = torch.zeros(chain_M.shape[1], device=device).float()
                order_mask[idx] = 1.0
            decoding_order = torch.argsort(
                (order_mask[None,] + 0.0001) * (torch.abs(randn))
            )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = F.one_hot(
                decoding_order, num_classes=mask_size
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see.
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)
            log_conditional_probs[:, idx, :] = log_probs[:, idx, :]
        return log_conditional_probs

    def unconditional_probs(self, X, mask, residue_idx, chain_encoding_all):
        """Graph-conditioned sequence model"""
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        order_mask_backward = torch.zeros(
            [X.shape[0], X.shape[1], X.shape[1]], device=device
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
