"""Top-level model definitions.

Typically these are initialized with config rather than arguments.

Authors: Alex Chu, Jinho Kim, Richard Shuai, Tianyu Lu, Zhaoyang Li
"""

import argparse
import copy
import logging
import os
import re
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from einops import rearrange, repeat
from omegaconf import DictConfig
from torch.types import Device
from torchtyping import TensorType
from tqdm.auto import tqdm

from protpardelle.common import residue_constants
from protpardelle.core import diffusion, modules
from protpardelle.data.atom import atom37_mask_from_aatype, atom73_mask_from_aatype
from protpardelle.data.atomize_io import N_ELEMENT_GROUPS
from protpardelle.data.dataset import make_fixed_size_1d, uniform_rand_rotation
from protpardelle.data.pdb_io import load_feats_from_pdb
from protpardelle.data.sequence import batched_seq_to_aatype_and_mask
from protpardelle.env import PROTEINMPNN_WEIGHTS
from protpardelle.evaluate import design_sequence
from protpardelle.integrations import protein_mpnn
from protpardelle.utils import (
    StrPath,
    apply_dotdict_recursively,
    get_default_device,
    load_config,
    norm_path,
    unsqueeze_trailing_dims,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fill_motif_seq(
    s_hat: TensorType["b n", int],
    motif_idx: list[list[int]],
    motif_aatype: list[list[int]],
) -> torch.Tensor:
    batch_size, seq_length = s_hat.shape
    for bi in range(batch_size):
        ii = 0
        for mi in range(seq_length):
            if mi in motif_idx[bi]:
                s_hat[bi, mi] = motif_aatype[bi][ii]
                ii += 1
    return s_hat


def apply_crop_cond_strategy(coords, motif_idx, motif_aatype, strategy: str):
    # Drop non-canonical aa3 from motif_aatype EXCEPT "HOH" — water placeholders
    # are routed through the backbone-O-only branch when strategy contains
    # "+water-bbO". Without the carve-out the per-residue iteration below would
    # shift indices and miss the water rows entirely.
    motif_aatype = [
        [ma for ma in mab if ma in residue_constants.restype_3to1 or ma == "HOH"]
        for mab in motif_aatype
    ]
    crop_cond_coords = coords.clone()
    if strategy is None:
        strategy = "backbone"
    water_bb_o = "water-bbO" in (strategy or "")
    bb_o_atom_idx = residue_constants.atom_order["O"]
    if "backbone" in strategy and "sidechain" not in strategy:
        crop_cond_coords[:, :, 3, :] = 0
        crop_cond_coords[:, :, 5:, :] = 0
    elif "acidic-tip" in strategy:
        # Base: only keep tip atoms for ASP/GLU, zero out everything for other residues.
        # Variants encoded via "+<aa3>-full" suffix(es): parse the strategy string
        # for tokens like "+tyr-full", "+his-full", etc. Listed residue types get
        # their FULL atom set preserved (backbone + sidechain). This matches the
        # training-time conditioning regime (model never saw a single atom alone).
        # The suffix is case-insensitive; the aa3 before "-full" should be the
        # standard 3-letter code (lowercased).
        full_aa3_set = set()
        # "+<aa3>-tip" gives a residue type its RFDIFFUSION tip atoms, and
        # "+<aa3>-bbo" gives it the backbone O alone. Both are needed to express
        # the natural lanmodulin loop pattern, where position 1 contributes an
        # Asn amide tip (OD1 to the metal, ND2 to a water) and position 7
        # contributes only its backbone carbonyl. Neither could be said before:
        # the base branch handles ASP/GLU tips, "-full" is all-or-nothing, and
        # backbone-O was reachable only through the HOH placeholder.
        # Both regimes are in cc151's training mix -- tip_only (p=0.35) applies
        # tips to every residue type including ASN/GLN, and acidic_tip_O_mix
        # (p=0.10) is per-residue "tip or backbone-O only".
        # "+<aa3>-otip" keeps ONLY the oxygens of a tip group, dropping the two
        # carbons. Standard tip conditioning pins CB/CG/OD1/OD2 (ASP) or
        # CG/CD/OE1/OE2 (GLU), so the metal-O-C angle is fully determined and comes
        # out native: measured 91.9 +- 6.2 deg over 382 bidentate contacts, 88%
        # inside the natural 85-101 window. Dropping the carbons frees that angle,
        # the same way the singly-conditioned backbone O is free (114 +- 34 deg).
        # "+otip@<k>" does it for the k-th motif residue only (1-based, in motif
        # order), so one carboxylate can be freed while the rest stay pinned and
        # the effect is attributable.
        #
        # The carbons are left at zero, and the condmask is derived per atom as
        # (coords.abs().sum(-1) > 0), so a dropped carbon reads as NOT CONDITIONED
        # rather than as an atom sitting at the origin.
        #
        # Note this is a mild extrapolation: cc151's tip_only / acidic_tip_only
        # modes always include the carbons, so it never saw a bare carboxylate
        # oxygen -- though backbone_O_tip (p=0.10) and acidic_tip_O_mix (p=0.10)
        # did show it isolated single oxygens.
        tip_aa3_set, bbo_aa3_set, otip_aa3_set = set(), set(), set()
        otip_indices: set[int] = set()
        for token in strategy.split("+"):
            token = token.strip().lower()
            if token.startswith("otip@"):
                try:
                    otip_indices.add(int(token[len("otip@"):]))
                except ValueError:
                    pass
                continue
            for suffix, target in (("-full", full_aa3_set), ("-otip", otip_aa3_set),
                                   ("-tip", tip_aa3_set), ("-bbo", bbo_aa3_set)):
                if token.endswith(suffix):
                    aa_code = token[: -len(suffix)].upper()
                    if aa_code in residue_constants.restype_3to1:
                        target.add(aa_code)
                    break
        coords_src = coords.clone()
        for bi, _ in enumerate(motif_idx):
            # Zero all motif atoms for this batch entry
            for mi in range(len(motif_idx[bi])):
                crop_cond_coords[bi, motif_idx[bi][mi], :, :] = 0
            # Restore kept atoms per aatype
            for mi, aa3 in enumerate(motif_aatype[bi]):
                pos = motif_idx[bi][mi]
                if aa3 in otip_aa3_set or (mi + 1) in otip_indices:
                    # oxygens of the tip group only; carbons stay zero/unconditioned
                    for at in residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS.get(aa3, []):
                        if not at.startswith("O"):
                            continue
                        ai = residue_constants.atom_order.get(at)
                        if ai is not None:
                            crop_cond_coords[bi, pos, ai, :] = coords_src[bi, pos, ai, :]
                elif aa3 in ("ASP", "GLU"):
                    atom_idx = [
                        residue_constants.atom_order.get(at)
                        for at in residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS[aa3]
                    ]
                    for ai in atom_idx:
                        crop_cond_coords[bi, pos, ai, :] = coords_src[bi, pos, ai, :]
                elif aa3 == "HOH" and water_bb_o:
                    # Water placeholder: restore only the backbone O atom.
                    crop_cond_coords[bi, pos, bb_o_atom_idx, :] = coords_src[bi, pos, bb_o_atom_idx, :]
                elif aa3 in full_aa3_set:
                    # Full residue (backbone + sidechain) — restore all original atoms
                    crop_cond_coords[bi, pos, :, :] = coords_src[bi, pos, :, :]
                elif aa3 in tip_aa3_set:
                    # This residue type's tip atoms (e.g. ASN -> CB/CG/ND2/OD1)
                    for at in residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS.get(aa3, []):
                        ai = residue_constants.atom_order.get(at)
                        if ai is not None:
                            crop_cond_coords[bi, pos, ai, :] = coords_src[bi, pos, ai, :]
                elif aa3 in bbo_aa3_set:
                    # Backbone carbonyl O only — the position-7 contribution in
                    # the natural loop, where the sidechain plays no part.
                    crop_cond_coords[bi, pos, bb_o_atom_idx, :] = coords_src[
                        bi, pos, bb_o_atom_idx, :]
                # else: leave zero
    elif "sidechain" in strategy:
        if "sidechain-tip" in strategy:
            for bi, _ in enumerate(motif_idx):
                for mi, aa3 in enumerate(motif_aatype[bi]):
                    if aa3 in residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS:
                        atom_idx = [
                            residue_constants.atom_order.get(at)
                            for at in residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS[
                                aa3
                            ]
                        ]
                        inv_atom_idx = np.delete(np.arange(37), atom_idx)
                        crop_cond_coords[bi, motif_idx[bi][mi], inv_atom_idx, :] = 0
        if "backbone" not in strategy:  # given sc, not given bb
            crop_cond_coords[:, :, (0, 1, 2, 4), :] = 0

    return crop_cond_coords

def place_motif_feats_on_contigs(
    motif_feats: dict,
    chain_id_mapping: dict,
    motif_placements_full: list[str],
) -> list[dict]:
    """Place a motif PDB's per-residue features onto each sample's contig layout.

    Returns one feature dict per entry of ``motif_placements_full`` (one per batch
    element), with every tensor concatenated over the motif segments of all chains.

    Factored out of Protpardelle.sample (verbatim) so the NEGATIVE motif used by
    negative-design guidance can be placed with the exact same padding/ordering
    logic as the positive motif.
    """
    all_motif_feats = []
    for mp_chains in motif_placements_full:
        chain_motif_feats = defaultdict(list)
        prev_motif_segments = 0
        for mp in mp_chains.split(";/;"):
            curr_motif_feats = copy.deepcopy(motif_feats)
            motif_chain_resids = [
                s for s in mp.split("/") if s[0].isalpha()
            ]
            if not motif_chain_resids:
                continue
            # Pad each motif segment to its NOMINAL contig length
            # (end - start + 1) rather than to the count of residues
            # actually present in the motif PDB. Residues unmodeled in
            # the PDB (gaps within the contig range) become zero-filled,
            # atom_mask=0 rows: they reserve a scaffold slot so the contig
            # placement (motif_idx) and per-chain total_lengths stay
            # consistent, but they are not conditioned on downstream
            # (crop_cond_mask is derived from non-zero coords). Without
            # this, a gapped motif yields fewer motif coords than the
            # contig reserves and the scatter into crop_cond_coords raises
            # a shape mismatch.
            for seg in motif_chain_resids:
                seg_match = re.match(
                    r"([A-Za-z])(\d+)(?:-(\d+))?$", seg
                )
                if not seg_match:
                    raise ValueError(
                        f"Invalid motif segment: {seg}"
                    )
                seg_chain = seg_match.group(1)
                seg_start = int(seg_match.group(2))
                seg_end = (
                    int(seg_match.group(3))
                    if seg_match.group(3)
                    else seg_start
                )
                nominal_len = seg_end - seg_start + 1
                seg_present_idx = parse_fixed_pos_str(
                    seg,
                    chain_id_mapping,
                    motif_feats["residue_index_orig"],
                    motif_feats["chain_index"],
                )
                present_resnums = (
                    motif_feats["residue_index_orig"][seg_present_idx]
                    .long()
                    .tolist()
                )
                offsets = [rn - seg_start for rn in present_resnums]
                seg_chain_int = int(chain_id_mapping[seg_chain])
                for k, v in curr_motif_feats.items():
                    pad_shape = (nominal_len,) + tuple(v.shape[1:])
                    if k in ("residue_index", "residue_index_orig"):
                        padded = torch.arange(
                            seg_start, seg_end + 1, dtype=v.dtype
                        )
                    elif k == "chain_index":
                        padded = torch.full(
                            pad_shape, seg_chain_int, dtype=v.dtype
                        )
                    else:
                        padded = torch.zeros(pad_shape, dtype=v.dtype)
                    if offsets:
                        padded[
                            torch.tensor(offsets, dtype=torch.long)
                        ] = v[
                            torch.tensor(
                                seg_present_idx, dtype=torch.long
                            )
                        ]
                    chain_motif_feats[k].append(padded)
                prev_motif_segments += nominal_len
        for k, v in chain_motif_feats.items():
            chain_motif_feats[k] = torch.cat(v, dim=0)
        all_motif_feats.append(chain_motif_feats)
    return all_motif_feats



def assert_identical_motif_atom_sets(
    pos_feats: dict,
    neg_feats: dict,
    pos_path: str,
    neg_path: str,
) -> None:
    """Hard guard for negative-design guidance: the positive and negative motifs
    must describe the SAME atoms, differing only in coordinates.

    x0_pos - x0_neg is only a meaningful direction if both conditioning tensors
    occupy the same (residue, atom) slots. If the atom sets differ, the difference
    silently mixes "this atom moved" with "this atom exists / does not exist", and
    the guidance is meaningless. So we require an exact match on residue count,
    residue numbering, chain, residue identity (aatype) and per-residue atom
    presence (atom_mask over the 37-atom layout), including ordering.

    Raises ValueError naming both files and the first mismatch found.
    """
    where = f"positive={pos_path!r}\nnegative={neg_path!r}"
    n_pos = int(pos_feats["aatype"].shape[0])
    n_neg = int(neg_feats["aatype"].shape[0])
    if n_pos != n_neg:
        raise ValueError(
            "Negative-design guidance requires identical atom sets, but the motifs "
            f"have different residue counts ({n_pos} vs {n_neg}).\n{where}"
        )

    def _res_label(feats, i):
        aa_idx = int(feats["aatype"][i])
        aa3 = (
            residue_constants.restype_1to3[residue_constants.restypes[aa_idx]]
            if aa_idx < len(residue_constants.restypes)
            else "UNK"
        )
        return f"{aa3}{int(feats['residue_index_orig'][i])} (chain idx {int(feats['chain_index'][i])})"

    for key in ("aatype", "residue_index_orig", "chain_index"):
        pv, nv = pos_feats[key], neg_feats[key]
        bad = torch.nonzero((pv != nv).reshape(pv.shape[0], -1).any(-1)).flatten()
        if bad.numel() > 0:
            i = int(bad[0])
            raise ValueError(
                "Negative-design guidance requires identical atom sets, but the motifs "
                f"differ in {key!r} at motif residue {i} "
                f"(positive {_res_label(pos_feats, i)} vs negative {_res_label(neg_feats, i)})."
                f"\n{where}"
            )

    pm, nm = pos_feats["atom_mask"], neg_feats["atom_mask"]
    bad = torch.nonzero(pm != nm)
    if bad.numel() > 0:
        i, a = int(bad[0, 0]), int(bad[0, 1])
        atom_name = (
            residue_constants.atom_types[a] if a < len(residue_constants.atom_types) else str(a)
        )
        present = "positive" if float(pm[i, a]) > 0 else "negative"
        raise ValueError(
            "Negative-design guidance requires identical atom sets, but atom "
            f"{atom_name!r} of motif residue {i} ({_res_label(pos_feats, i)}) is present "
            f"only in the {present} motif.\n{where}\n"
            "Fix: rebuild the two motif PDBs so they contain exactly the same "
            "residues and atom names in the same order."
        )

    is_w_pos = pos_feats.get("is_water_motif")
    is_w_neg = neg_feats.get("is_water_motif")
    if is_w_pos is not None and is_w_neg is not None and not torch.equal(is_w_pos, is_w_neg):
        raise ValueError(
            "Negative-design guidance requires identical atom sets, but the two motifs "
            f"flag different residues as HOH water placeholders.\n{where}"
        )



def get_time_dependent_scale(
    schedule: str, w: float, curr_step: int, n_steps: int, stage2: bool = False
) -> float:
    if schedule == "constant":
        scale = 1.0
    elif schedule == "cubic":
        scale = (curr_step / n_steps) ** 3
    elif schedule == "custom-stepscale":
        if curr_step < 250:
            scale = 0.67
        elif curr_step < 400:
            scale = 0.8
        elif curr_step < 450:
            scale = 0.9
        else:
            scale = 1.0
    elif schedule == "quadratic":
        scale = (curr_step / n_steps) ** 2
    elif stage2:
        if curr_step < 25:
            scale = 0.01
        elif curr_step < 40:
            scale = 0.1
        elif curr_step < 45:
            scale = 0.5
        else:
            scale = 1.0
    elif curr_step < 250:
        scale = 0.01
    elif curr_step < 400:
        scale = 0.1
    elif curr_step < 450:
        scale = 0.5
    else:
        scale = 1.0
    return w * scale


def motif_loss(x0_in, motif_idx, motif_coords, atom_mask) -> torch.Tensor:
    batch_size = x0_in.shape[0]
    losses = torch.zeros(
        batch_size,
    ).to(x0_in)
    for bi in range(batch_size):
        loss = (x0_in[bi, motif_idx[bi], :, :] - motif_coords[bi]).pow(2).sum(-1)
        losses[bi] = (loss * atom_mask[bi, motif_idx[bi]]).sum()
    return losses


def match_motif_to_scaffold_unindexed(
    x0_pred: torch.Tensor,
    motif_all_atom: torch.Tensor,
    motif_atom_mask: torch.Tensor,
    motif_condmask: torch.Tensor,
    motif_aatype_unindexed: list[torch.Tensor],
    seq_mask: torch.Tensor,
    motif_is_water_unindexed: list[torch.Tensor] | None = None,
) -> tuple[list[list[int]], list[torch.Tensor], list[torch.Tensor]]:
    """Match unindexed motif residues to scaffold positions via bipartite matching.

    For each motif residue, computes the centroid of its conditioned atoms, then
    uses linear_sum_assignment against scaffold CA positions to find a one-to-one
    assignment. Returns motif_idx (scaffold positions) and motif_aatype (aatypes)
    aligned so that motif_aatype[bi][k] belongs at motif_idx[bi][k].

    Args:
        x0_pred: [B, N, 37, 3] predicted scaffold coordinates
        motif_all_atom: [B, K, 37, 3] rotated motif coordinates
        motif_atom_mask: [B, K, 37] mask of valid atoms in motif
        motif_condmask: [B, K, 37] mask of conditioned atoms (tip atoms)
        motif_aatype_unindexed: list per batch of aatype tensors [K]
        seq_mask: [B, N] scaffold sequence mask
    """
    B = x0_pred.shape[0]
    # motif_condmask may be padded to max_motif_size; slice to actual K from motif_all_atom
    K = motif_all_atom.shape[1]
    motif_idx = []
    motif_aatype = []
    motif_is_water_match = []
    for bi in range(B):
        cm = motif_condmask[bi, :K]  # [K, 37]
        atom_counts = cm.sum(-1)  # [K]
        valid_motif = atom_counts > 0
        if valid_motif.sum().item() == 0:
            motif_idx.append([])
            motif_aatype.append(torch.tensor([], dtype=torch.long, device=x0_pred.device))
            motif_is_water_match.append(torch.tensor([], dtype=torch.bool))
            continue

        # Motif anchors: centroid of conditioned atoms
        mc = motif_all_atom[bi]  # [K, 37, 3]
        centroids = (mc * cm.unsqueeze(-1)).sum(-2) / atom_counts.clamp(min=1e-6).unsqueeze(-1)

        # Scaffold anchors: CA positions (atom index 1)
        scaffold_ca = x0_pred[bi, :, 1, :]  # [N, 3]
        sm = seq_mask[bi].bool()
        valid_scaffold = torch.nonzero(sm).squeeze(-1)
        if valid_scaffold.numel() == 0:
            motif_idx.append([])
            motif_aatype.append(torch.tensor([], dtype=torch.long, device=x0_pred.device))
            motif_is_water_match.append(torch.tensor([], dtype=torch.bool))
            continue

        valid_motif_positions = torch.nonzero(valid_motif).squeeze(-1)
        centroids_v = centroids[valid_motif_positions]  # [K_v, 3]
        scaffold_ca_v = scaffold_ca[valid_scaffold]  # [N_v, 3]

        cost = torch.cdist(centroids_v.unsqueeze(0), scaffold_ca_v.unsqueeze(0)).squeeze(0)
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

        # Map back to scaffold indices
        matched_positions = [int(valid_scaffold[col].item()) for col in col_ind]
        aatype_src = motif_aatype_unindexed[bi]
        # Keep on CPU to match load_feats_from_pdb convention used in indexed path
        matched_aatypes = torch.tensor(
            [int(aatype_src[int(valid_motif_positions[r].item())].item()) for r in row_ind],
            dtype=torch.long,
        )
        if motif_is_water_unindexed is not None:
            iw_src = motif_is_water_unindexed[bi]
            matched_is_water = torch.tensor(
                [bool(iw_src[int(valid_motif_positions[r].item())].item()) for r in row_ind],
                dtype=torch.bool,
            )
        else:
            matched_is_water = torch.zeros(len(matched_positions), dtype=torch.bool)

        motif_idx.append(matched_positions)
        motif_aatype.append(matched_aatypes)
        motif_is_water_match.append(matched_is_water)

    return motif_idx, motif_aatype, motif_is_water_match


def group_consecutive_idx(nums):

    nums = np.array(nums)
    # Find the indices where the difference between consecutive elements is greater than 1
    breaks = np.where(np.diff(nums) != 1)[0] + 1

    # Split the array at those indices
    result = np.split(nums, breaks)

    # Convert the subarrays to lists
    return [sublist.tolist() for sublist in result]


def contig_to_idx(contig: list[list[int]]) -> list[list[int]]:
    result = []
    start_idx = 0
    for c in contig:
        result.append(list(range(start_idx, start_idx + len(c))))
        start_idx = start_idx + len(c)

    return result


def _pad_seq_tail(x: torch.Tensor | None, m: int) -> torch.Tensor | None:
    """Zero-pad a per-residue tensor [B, N, ...] to [B, N+m, ...] along the seq dim."""

    if x is None or m <= 0:
        return x
    pad_shape = (x.shape[0], m) + tuple(x.shape[2:])
    pad = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


class MiniMPNN(nn.Module):
    """Wrapper for ProteinMPNN network to predict sequence from structure."""

    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.config = config
        self.model_config = cfg = config.model.mpnn_model
        self.n_tokens = config.data.n_aatype_tokens
        self.seq_emb_dim = cfg.n_channel
        time_cond_dim = cfg.n_channel * cfg.noise_cond_mult

        self.noise_block = modules.NoiseConditioningBlock(cfg.n_channel, time_cond_dim)
        self.token_embedding = nn.Linear(self.n_tokens, self.seq_emb_dim)
        self.mpnn_net = modules.NoiseConditionalProteinMPNN(
            n_channel=cfg.n_channel,
            n_layers=cfg.n_layers,
            n_neighbors=cfg.n_neighbors,
            time_cond_dim=time_cond_dim,
            vocab_size=config.data.n_aatype_tokens,
            input_S_is_embeddings=True,
        )
        self.proj_out = nn.Linear(cfg.n_channel, self.n_tokens)

    def forward(
        self,
        denoised_coords: TensorType["b n a x", float],
        coords_noise_level: TensorType["b", float],
        seq_mask: TensorType["b n", float],
        residue_index: TensorType["b n", int],
        seq_self_cond: TensorType["b n t", float] | None = None,  # logprobs
        seq_crop_cond: TensorType["b n", int] | None = None,  # motif aatypes
        return_embeddings: bool = False,
    ):
        coords_noise_level_scaled = 0.25 * torch.log(coords_noise_level)
        noise_cond = self.noise_block(coords_noise_level_scaled)

        b, n, _, _ = denoised_coords.shape
        if seq_self_cond is None or not self.model_config.use_self_conditioning:
            seq_emb_in = torch.zeros(b, n, self.seq_emb_dim).to(denoised_coords)
        else:
            seq_emb_in = self.token_embedding(seq_self_cond.exp())

        if seq_crop_cond is not None:
            seq_emb_in = seq_emb_in + self.token_embedding(seq_crop_cond.float())

        node_embs, encoder_embs = self.mpnn_net(
            denoised_coords, seq_emb_in, seq_mask, residue_index, noise_cond
        )

        logits = self.proj_out(node_embs)
        pred_logprobs = F.log_softmax(logits, -1)

        if return_embeddings:
            return pred_logprobs, node_embs, encoder_embs
        return pred_logprobs


class CoordinateDenoiser(nn.Module):
    """Wrapper for U-ViT/DiT module to denoise structure coordinates."""

    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.config = config

        # Configuration
        self.sigma_data = config.data.sigma_data
        m_cfg = config.model.struct_model
        nc = m_cfg.n_channel
        bb_atoms = ["N", "CA", "C", "O"]
        n_atoms = config.model.struct_model.n_atoms
        self.use_conv = len(m_cfg.uvit.n_filt_per_layer) > 0
        if self.use_conv and n_atoms == 37:
            n_atoms += 1  # make it an even number
        self.n_atoms = n_atoms
        self.bb_idxs = [residue_constants.atom_order.get(a) for a in bb_atoms]
        self.is_unindexed = (
            config.model.crop_conditional
            and "unindexed" in config.model.conditioning_style
        )
        self.is_flow_matching = getattr(config.model, "objective", "denoising") == "flow_matching"
        if self.is_unindexed:
            # Unindexed: noisy/3 + self_cond/3 + condmask/1 = 7
            n_xyz = 7
        else:
            n_xyz = (
                9
                if config.model.crop_conditional
                and "concat" in config.model.conditioning_style
                else 6
            )
            if (
                config.model.crop_conditional
                and "hotspot" in config.model.conditioning_style
            ):
                n_xyz += 1
            if config.model.crop_conditional and "ssadj" in config.model.conditioning_style:
                n_xyz += 3  # one-hot encoding [helix, strand, loop]
            if config.model.crop_conditional and "condmask" in config.model.conditioning_style:
                n_xyz += 1  # per-atom binary mask indicating which atoms have conditioning
        # Per-residue DSSP one-hot conditioning at motif positions (3 channels: H/E/L),
        # broadcast across the atom dim. Applies to both indexed and unindexed styles.
        self.has_dssp_cond = (
            config.model.crop_conditional
            and "dssp" in config.model.conditioning_style
        )
        if self.has_dssp_cond:
            n_xyz += 3
        # Binder-chain global SSE-proportion conditioning (3 channels: helix/strand/loop
        # fraction, quantized at train time). The same 3-float vector is broadcast to
        # every binder (non-motif) residue, zero at motif residues. CFG-friendly:
        # zeroed for 50% of training batches in train.py.
        self.has_binder_sse_prop_cond = (
            config.model.crop_conditional
            and "binderprop" in config.model.conditioning_style
        )
        if self.has_binder_sse_prop_cond:
            n_xyz += 3
        # Per-residue motif SASA burial-class conditioning (3 channels:
        # buried/partial/exposed one-hot), broadcast across the atom dim at motif
        # positions. Mirrors has_dssp_cond. Applies to indexed and unindexed styles.
        self.has_sasa_cond = (
            config.model.crop_conditional
            and "sasa" in config.model.conditioning_style
        )
        if self.has_sasa_cond:
            n_xyz += 3
        # Per-atom motif h-bond conditioning (1 channel): marks which of the 37
        # atoms at a motif residue participate in a side-chain h-bond. Per-atom
        # like condmask (NOT broadcast per-residue like sasa/dssp), so +1 channel.
        self.has_hbond_cond = (
            config.model.crop_conditional
            and "hbond" in config.model.conditioning_style
        )
        if self.has_hbond_cond:
            n_xyz += 1
        # Atomized conditioning (cc152): per-atom channels that say what each of
        # the 37 slots actually holds, so non-canonical residues, ligands and
        # metals can share the atom37 layout. Element one-hot + CCD reference
        # conformer xyz + a flag for slots used outside their canonical meaning.
        # For a purely canonical protein these are a deterministic function of
        # aatype, so zero-initialising their input weights reproduces cc89.
        self.has_atomized_cond = "atomized" in config.model.conditioning_style
        if self.has_atomized_cond:
            n_xyz += N_ELEMENT_GROUPS + 3 + 1
        # cc154: rotation-invariant bond-topology scalars per atom
        # (ideal distance to bonded parent, ideal distance to the token anchor,
        # bond degree). The reference-conformer channels give ideal *coordinates*,
        # from which a bond length needs |ref_a - ref_b| -- a pairwise operation
        # the architecture never performs within a token, which is why ablating
        # them changes almost nothing. These supply the distance directly.
        self.has_bondgeom_cond = "bondgeom" in config.model.conditioning_style
        if self.has_bondgeom_cond:
            n_xyz += 3
        nc_in = n_xyz * n_atoms  # xyz + selfcond xyz + maybe cropcond xyz

        # Neural networks
        n_noise_channel = nc * m_cfg.noise_cond_mult
        n_motif_channel = (
            nc * m_cfg.motif_cond_mult
            if (
                "motif_cond_mult" in m_cfg
                and "conditioning_style" in config.model
                and "separate_motif_track" in config.model.conditioning_style
            )
            else None
        )
        self.net = modules.TimeCondUViT(
            seq_len=config.data.fixed_size,
            patch_size=m_cfg.uvit.patch_size,
            dim=nc,
            depth=m_cfg.uvit.n_layers,
            n_filt_per_layer=m_cfg.uvit.n_filt_per_layer,
            heads=m_cfg.uvit.n_heads,
            dim_head=m_cfg.uvit.dim_head,
            conv_skip_connection=m_cfg.uvit.conv_skip_connection,
            n_atoms=n_atoms,
            channels_per_atom=n_xyz,
            time_cond_dim=n_noise_channel,
            motif_cond_dim=n_motif_channel,
            position_embedding_type=m_cfg.uvit.position_embedding_type,
            position_embedding_max=m_cfg.uvit.position_embedding_max,
            noise_residual=config.model.crop_conditional
            and "conditioning_style" in config.model
            and "noise_residual" in config.model.conditioning_style,
            ssadj_cond=config.model.crop_conditional
            and "conditioning_style" in config.model
            and "ssadj" in config.model.conditioning_style,
            attn_dropout=(
                m_cfg.uvit.attn_dropout if "attn_dropout" in m_cfg.uvit else 0.0
            ),
            out_dropout=m_cfg.uvit.out_dropout if "out_dropout" in m_cfg.uvit else 0.0,
            ff_dropout=m_cfg.uvit.ff_dropout if "ff_dropout" in m_cfg.uvit else 0.1,
            dit=m_cfg.arch == "dit",
        )
        self.noise_block = modules.NoiseConditioningBlock(nc, n_noise_channel)

    def forward(
        self,
        noisy_coords: TensorType["b n a x", float],
        noise_level: TensorType["b n", float],
        seq_mask: TensorType["b n", float],
        residue_index: TensorType["b n", int] | None = None,
        chain_index: TensorType["b n", int] | None = None,
        hotspot_mask: TensorType["b n", int] | None = None,
        struct_self_cond: TensorType["b n a x", float] | None = None,
        struct_crop_cond: TensorType["b n a x", float] | None = None,
        crop_cond_mask: TensorType["b n a", float] | None = None,
        sse_cond: TensorType["b n", int] | None = None,
        adj_cond: TensorType["b n n", int] | None = None,
        motif_sse_cond: TensorType["b n d", float] | None = None,
        binder_sse_prop_cond: TensorType["b n d", float] | None = None,
        motif_sasa_cond: TensorType["b n d", float] | None = None,
        motif_hbond_cond: TensorType["b n a", float] | None = None,
        atom_elem: TensorType["b n a", int] | None = None,
        atom_ref: TensorType["b n a x", float] | None = None,
        atom_noncanon: TensorType["b n a", float] | None = None,
        atom_bondfeat: TensorType["b n a d", float] | None = None,
        return_emb: bool = False,
    ):

        noise_level = noise_level.clamp(min=1e-6)

        # Prep inputs and time conditioning
        if self.is_flow_matching:
            # Flow matching: noise_level is t ∈ (0, 1)
            # Simple normalization by sigma_data
            emb = noisy_coords / self.sigma_data
            # Time conditioning via logit(t), scaled similarly to 0.25*log(σ)
            t = noise_level
            time_input = 0.25 * torch.log(t / (1 - t).clamp(min=1e-6))
            noise_cond = self.noise_block(time_input)
        else:
            # Denoising: Karras input scaling
            actual_var_data = self.sigma_data**2
            var_noisy_coords = noise_level**2 + actual_var_data
            emb = noisy_coords / unsqueeze_trailing_dims(
                var_noisy_coords.clamp(min=1e-6).sqrt(), noisy_coords
            )
            struct_noise_scaled = 0.25 * torch.log(noise_level)
            noise_cond = self.noise_block(struct_noise_scaled)

        # Prepare self- and crop-conditioning and concatenate along channels
        if struct_self_cond is None:
            struct_self_cond = torch.zeros_like(noisy_coords)
        if sse_cond is None:
            sse_cond = torch.zeros_like(residue_index).long()

        pos_bias_mask = None

        if self.config.model.crop_conditional:
            if self.is_unindexed:
                # Unindexed conditioning: motif coords are appended along sequence dim.
                # Channels: noisy(3) + self_cond(3) + condmask(1) = 7
                emb = torch.cat([emb, struct_self_cond], dim=-1)
                if crop_cond_mask is None:
                    condmask_channel = torch.zeros(
                        noisy_coords.shape[:-1] + (1,),
                        device=noisy_coords.device,
                        dtype=noisy_coords.dtype,
                    )
                else:
                    condmask_channel = crop_cond_mask.unsqueeze(-1).float()
                emb = torch.cat([emb, condmask_channel], dim=-1)

                # Zero out position bias for any pair involving a motif position
                if crop_cond_mask is not None:
                    is_motif = crop_cond_mask.any(-1).bool()  # [B, N+M]
                    is_protein = ~is_motif
                    pos_bias_mask = (
                        is_protein.unsqueeze(-1) & is_protein.unsqueeze(-2)
                    ).float().unsqueeze(1)  # [B, 1, N+M, N+M]

            elif (
                "conditioning_style" in self.config.model
                and "concat" in self.config.model.conditioning_style
            ):
                if struct_crop_cond is None:
                    struct_crop_cond = torch.zeros_like(noisy_coords)
                else:
                    struct_crop_cond = struct_crop_cond / self.sigma_data
                emb = torch.cat([emb, struct_self_cond, struct_crop_cond], dim=-1)

                if (
                    "condmask" in self.config.model.conditioning_style
                ):
                    if crop_cond_mask is None:
                        condmask_channel = torch.zeros(
                            noisy_coords.shape[:-1] + (1,),
                            device=noisy_coords.device,
                            dtype=noisy_coords.dtype,
                        )
                    else:
                        condmask_channel = crop_cond_mask.unsqueeze(-1).float()
                    emb = torch.cat([emb, condmask_channel], dim=-1)

                if (
                    "hotspot" in self.config.model.conditioning_style
                    and hotspot_mask is None
                ):
                    hotspot_mask = torch.zeros_like(seq_mask)

                if (
                    "hotspot" in self.config.model.conditioning_style
                    and hotspot_mask is not None
                ):
                    hotspot_mask_rep = torch.stack(
                        [hotspot_mask.clone() for _ in range(emb.shape[-2])], dim=-1
                    ).unsqueeze(-1)
                    hotspot_mask_rep[:, :, 3] = 0
                    hotspot_mask_rep[:, :, 5:] = 0

                    emb = torch.cat([emb, hotspot_mask_rep], dim=-1)

            else:
                emb = torch.cat([emb, struct_self_cond], dim=-1)

            if "conditioning_style" in self.config.model and (
                "noise_residual" in self.config.model.conditioning_style
                or "separate_motif_track" in self.config.model.conditioning_style
            ):
                if struct_crop_cond is None:
                    struct_crop_cond = torch.zeros_like(noisy_coords)
                struct_crop_cond = rearrange(struct_crop_cond, "b n a c -> b c n a")
                motif_cond = self.net.cond_to_patch_embedding(
                    struct_crop_cond
                )  # spacing info is leaked
                if "noise_residual" in self.config.model.conditioning_style:
                    noise_cond = noise_cond + motif_cond

        else:
            emb = torch.cat([emb, struct_self_cond], dim=-1)

        if (
            "conditioning_style" in self.config.model
            and "ssadj" in self.config.model.conditioning_style
        ):
            sse_cond = F.one_hot(sse_cond, num_classes=3)
            sse_cond = sse_cond.unsqueeze(-2).expand(
                -1, -1, emb.shape[-2], -1
            )  # expand along atom dimension to match emb shape
            emb = torch.cat([emb, sse_cond], dim=-1)

        if self.has_dssp_cond:
            if motif_sse_cond is None:
                motif_sse_cond = torch.zeros(
                    emb.shape[0], emb.shape[1], 3,
                    device=emb.device, dtype=emb.dtype,
                )
            motif_sse_per_atom = motif_sse_cond.unsqueeze(-2).expand(
                -1, -1, emb.shape[-2], -1
            )
            emb = torch.cat([emb, motif_sse_per_atom], dim=-1)

        if self.has_binder_sse_prop_cond:
            if binder_sse_prop_cond is None:
                binder_sse_prop_cond = torch.zeros(
                    emb.shape[0], emb.shape[1], 3,
                    device=emb.device, dtype=emb.dtype,
                )
            binder_sse_prop_per_atom = binder_sse_prop_cond.unsqueeze(-2).expand(
                -1, -1, emb.shape[-2], -1
            )
            emb = torch.cat([emb, binder_sse_prop_per_atom], dim=-1)

        if self.has_sasa_cond:
            if motif_sasa_cond is None:
                motif_sasa_cond = torch.zeros(
                    emb.shape[0], emb.shape[1], 3,
                    device=emb.device, dtype=emb.dtype,
                )
            motif_sasa_per_atom = motif_sasa_cond.unsqueeze(-2).expand(
                -1, -1, emb.shape[-2], -1
            )
            emb = torch.cat([emb, motif_sasa_per_atom], dim=-1)

        if self.has_hbond_cond:
            # Per-atom [B, N, 37] -> [B, N, 37, 1], concatenated directly (no
            # broadcast) exactly like the condmask channel.
            if motif_hbond_cond is None:
                hbond_channel = torch.zeros(
                    emb.shape[0], emb.shape[1], emb.shape[2], 1,
                    device=emb.device, dtype=emb.dtype,
                )
            else:
                hbond_channel = motif_hbond_cond.unsqueeze(-1).to(emb.dtype)
            emb = torch.cat([emb, hbond_channel], dim=-1)

        if self.has_atomized_cond:
            # All three are per-atom [B, N, 37, *] and concatenated directly (no
            # broadcast), like the condmask/hbond channels.
            b, n, a = emb.shape[0], emb.shape[1], emb.shape[2]
            if atom_elem is None:
                elem_oh = torch.zeros(
                    b, n, a, N_ELEMENT_GROUPS, device=emb.device, dtype=emb.dtype
                )
            else:
                elem_oh = F.one_hot(
                    atom_elem.long().clamp(0, N_ELEMENT_GROUPS - 1),
                    num_classes=N_ELEMENT_GROUPS,
                ).to(emb.dtype)
            if atom_ref is None:
                ref_ch = torch.zeros(b, n, a, 3, device=emb.device, dtype=emb.dtype)
            else:
                ref_ch = atom_ref.to(emb.dtype)
            if atom_noncanon is None:
                nc_ch = torch.zeros(b, n, a, 1, device=emb.device, dtype=emb.dtype)
            else:
                nc_ch = atom_noncanon.unsqueeze(-1).to(emb.dtype)
            emb = torch.cat([emb, elem_oh, ref_ch, nc_ch], dim=-1)

        if self.has_bondgeom_cond:
            if atom_bondfeat is None:
                bg = torch.zeros(
                    emb.shape[0], emb.shape[1], emb.shape[2], 3,
                    device=emb.device, dtype=emb.dtype,
                )
            else:
                bg = atom_bondfeat.to(emb.dtype)
            emb = torch.cat([emb, bg], dim=-1)

        emb, hidden = self.net(
            emb,
            noise_cond,
            motif_cond=None,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            return_emb=return_emb,
            pair_bias=adj_cond,
            pos_bias_mask=pos_bias_mask,
        )

        if self.is_flow_matching:
            # Flow matching: network output is velocity, scale back from normalized space
            velocity = emb * self.sigma_data
            velocity = velocity * unsqueeze_trailing_dims(seq_mask, velocity)
            return velocity, hidden
        else:
            # Preconditioning from Karras et al.
            out_scale = (
                noise_level
                * actual_var_data**0.5
                / torch.sqrt(var_noisy_coords.clamp(min=1e-6))
            )
            skip_scale = actual_var_data / var_noisy_coords.clamp(min=1e-6)
            emb = emb * unsqueeze_trailing_dims(out_scale, emb)
            skip_info = noisy_coords * unsqueeze_trailing_dims(skip_scale, noisy_coords)
            denoised_coords = emb + skip_info

            # Don't use atom mask; denoise all atoms
            denoised_coords = denoised_coords * unsqueeze_trailing_dims(
                seq_mask, denoised_coords
            )

            return denoised_coords, hidden


def parse_fixed_pos_str(
    fixed_pos_str: str,
    chain_id_mapping: dict[str, int],
    residue_index: TensorType["n", int],
    chain_index: TensorType["n", int],
) -> list[int]:
    """Parse a string of fixed positions in the format "A1, A10-25" and
    return the corresponding list of absolute indices.

    Args:
        fixed_pos_list (str): Comma-separated string representing fixed positions (e.g., "A1,A10-25").
        chain_id_mapping (dict[str, int]): Mapping of chain letter to chain index (e.g., {'A': 0, 'B': 1}).
        residue_index (torch.Tensor): Tensor of residue indices. (N,)
        chain_index (torch.Tensor): Tensor of chain indices. (N,)

    Returns:
        list[int]: The absolute indices of the fixed positions.
    """

    fixed_pos_str = fixed_pos_str.strip()
    if not fixed_pos_str:
        return []  # no positions specified

    fixed_indices = []

    fixed_pos_list = [pos for item in fixed_pos_str.split(",") if (pos := item.strip())]

    for pos in fixed_pos_list:
        # Match pattern like "A10" or "A10-25"
        match = re.match(r"([A-Za-z])(\d+)(?:-(\d+))?$", pos)
        if not match:
            raise ValueError(f"Invalid position format: {pos}")

        chain_letter = match.group(1)
        start_residue = int(match.group(2))
        end_residue = int(match.group(3)) if match.group(3) else start_residue

        if chain_letter not in chain_id_mapping:
            raise ValueError(f"Chain ID {chain_letter} not found in mapping.")

        # For the given chain, create a mask for all residues in the desired range
        chain_i = chain_id_mapping[chain_letter]
        range_mask = (
            (chain_index == chain_i)
            & (residue_index >= start_residue)
            & (residue_index <= end_residue)
        )
        matching_indices = torch.where(range_mask)[0]

        # Check that each residue in the requested range; warn if not found
        found_residues = residue_index[matching_indices].tolist()
        found_residues_set = set(found_residues)

        for r in range(start_residue, end_residue + 1):
            if r not in found_residues_set:
                logger.warning(
                    "Requested position %s%d not found in structure.", chain_letter, r
                )

        # Extend our fixed indices with whatever we did find
        fixed_indices.extend(matching_indices.tolist())

    return fixed_indices


class Protpardelle(nn.Module):
    """All-atom protein diffusion-based generative model.

    This class wraps a structure denoising network and a sequence prediction network
    to do structure/sequence co-design (for all-atom generation), or backbone generation.

    It can be trained for one of four main tasks. To produce the all-atom (co-design)
    Protpardelle model, we will typically pretrain an 'allatom' model, then use this
    to train a 'seqdes' model. A 'seqdes' model can be trained with either a backbone
    or allatom denoiser. The two can be combined to yield all-atom (co-design) Protpardelle
    without further training.
        'backbone': train only a backbone coords denoiser.
        'seqdes': train only a MiniMPNN, using a pretrained coords denoiser.
        'allatom': train only an allatom coords denoiser (cannot do all-atom generation
            by itself).
        'codesign': train both an allatom denoiser and MiniMPNN at once.
    """

    def __init__(self, config: argparse.Namespace, device: Device = None):
        super().__init__()

        self.config = config
        self.task = config.model.task
        self.n_tokens = config.data.n_aatype_tokens

        self.use_mpnn_model = self.task in ["seqdes", "codesign"]
        self.is_flow_matching = getattr(config.model, "objective", "denoising") == "flow_matching"

        # Modules
        self.bb_idxs = [0, 1, 2, 4]
        self.n_atoms = 37
        self.struct_model = CoordinateDenoiser(config)

        self.bb_idxs = self.struct_model.bb_idxs
        self.n_atoms = self.struct_model.n_atoms
        self.chain_residx_gap = config.data.chain_residx_gap

        if self.use_mpnn_model:
            self.mpnn_model = MiniMPNN(config)

        # Load any pretrained modules
        for module_name in self.config.model.pretrained_modules:
            self.load_pretrained_module(module_name)

        # Diffusion-related
        self.sigma_data = self.struct_model.sigma_data
        self.training_noise_schedule = partial(
            diffusion.noise_schedule,
            sigma_data=self.sigma_data,
            **vars(config.diffusion.training),
        )
        self.sampling_noise_schedule_default = self.make_sampling_noise_schedule()
        self.sampling_noise_schedule_bb = partial(
            diffusion.noise_schedule, function="backbone"
        )
        self.sampling_noise_schedule_sc = partial(
            diffusion.noise_schedule, function="sidechain"
        )

        if device is None:
            device = get_default_device()
        self.to(device)

    @property
    def device(self) -> torch.device:
        """Return the device on which the model is loaded."""
        return next(self.parameters()).device

    def load_pretrained_module(self, module_name: str, ckpt_path: str | None = None):
        """Load pretrained weights for a given module name."""
        assert module_name in ["struct_model", "mpnn_model"], module_name

        # Load pretrained checkpoint
        if ckpt_path is None:
            ckpt_path = getattr(self.config.model, f"{module_name}_checkpoint")
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=self.device)
        if isinstance(ckpt, dict):
            # training_state format: dict with "model_state_dict"
            model_state_dict = ckpt["model_state_dict"]
        else:
            # full model object saved via torch.save(model)
            model_state_dict = ckpt.state_dict()

        # Get only submodule state_dict
        submodule_state_dict = {
            sk[len(module_name) + 1 :]: sv
            for sk, sv in model_state_dict.items()
            if sk.startswith(module_name)
        }

        # Load into module
        module = dict(self.named_modules())[module_name]
        module.load_state_dict(submodule_state_dict)

        # Freeze unneeded modules
        if module_name == "struct_model":
            self.struct_model = module
            if self.task == "seqdes":
                for p in module.parameters():
                    p.requires_grad = False
        if module_name == "mpnn_model":
            self.mpnn_model = module
            if self.task not in ["codesign", "seqdes"]:
                for p in module.parameters():
                    p.requires_grad = False

        return module

    def load_minimpnn(self, mpnn_ckpt_path: str | None = None):
        """Convert an allatom model to a codesign model."""
        if mpnn_ckpt_path is None:
            mpnn_ckpt_path = "checkpoints/minimpnn_state_dict.pth"
        self.mpnn_model = MiniMPNN(self.config).to(self.device)
        self.load_pretrained_module("mpnn_model", ckpt_path=mpnn_ckpt_path)
        self.use_mpnn_model = True

    def remove_minimpnn(self):
        """Revert a codesign model to an allatom model."""
        self.use_mpnn_model = False
        self.mpnn_model = None

    def make_sampling_noise_schedule(self, **noise_kwargs):
        """Make the default sampling noise schedule function."""
        noise_schedule_kwargs = vars(self.config.diffusion.sampling)
        if len(noise_kwargs) > 0:
            noise_schedule_kwargs.update(noise_kwargs)
        return partial(diffusion.noise_schedule, **noise_schedule_kwargs)

    def forward(
        self,
        *,
        noisy_coords: TensorType["b n a x", float],
        noise_level: TensorType["b n", float],
        seq_mask: TensorType["b n", float],
        residue_index: TensorType["b n", int],
        chain_index: TensorType["b n", int] | None = None,
        hotspot_mask: TensorType["b n", int] | None = None,
        struct_self_cond: TensorType["b n a x", float] | None = None,
        struct_crop_cond: TensorType["b n a x", float] | None = None,
        crop_cond_mask: TensorType["b n a", float] | None = None,
        sse_cond: TensorType["b n", int] | None = None,
        adj_cond: TensorType["b n n", int] | None = None,
        motif_sse_cond: TensorType["b n d", float] | None = None,
        binder_sse_prop_cond: TensorType["b n d", float] | None = None,
        motif_sasa_cond: TensorType["b n d", float] | None = None,
        motif_hbond_cond: TensorType["b n a", float] | None = None,
        atom_elem: TensorType["b n a", int] | None = None,
        atom_ref: TensorType["b n a x", float] | None = None,
        atom_noncanon: TensorType["b n a", float] | None = None,
        atom_bondfeat: TensorType["b n a d", float] | None = None,
        seq_self_cond: TensorType["b n t", float] | None = None,  # logprobs
        seq_crop_cond: TensorType["b n", int] | None = None,  # motif aatypes
        motif_clean_coords: TensorType["b m a x", float] | None = None,
        motif_condmask: TensorType["b m a", float] | None = None,
        run_struct_model: bool = True,
        run_mpnn_model: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Main forward function for denoising/co-design.

        Arguments:
            noisy_coords: noisy array of xyz coordinates.
            noise_level: std of noise for each example in the batch
            seq_mask: mask indicating which indexes contain data.
            residue_index: residue ordering.
            struct_self_cond: denoised coordinates from the previous step, scaled
                down by sigma data.
            struct_crop_cond: unnoised coordinates. unscaled (scaled down by sigma
                data inside the denoiser)
            crop_cond_mask: per-atom binary mask indicating which atoms have conditioning.
            seq_self_cond: mpnn-predicted sequence logprobs from the previous step.
            motif_clean_coords: clean motif atom coordinates, padded to max_motif_size.
                Only used for unindexed conditioning — motif residues are appended
                along the sequence dim before the struct model, protein portion is
                extracted afterwards.
            motif_condmask: per-atom condmask for the motif residues.
            run_struct_model: flag to optionally not run structure denoiser.
            run_mpnn_model: flag to optionally not run MiniMPNN.
        """

        is_unindexed = getattr(self.struct_model, "is_unindexed", False)
        unindexed_active = (
            is_unindexed
            and motif_clean_coords is not None
            and motif_condmask is not None
        )
        protein_N = noisy_coords.shape[1]

        if unindexed_active:
            B = noisy_coords.shape[0]
            M = motif_clean_coords.shape[1]
            device = noisy_coords.device

            ext_noisy_coords = torch.cat([noisy_coords, motif_clean_coords], dim=1)

            motif_seq_mask = (motif_condmask.sum(-1) > 0).to(seq_mask.dtype)
            ext_seq_mask = torch.cat([seq_mask, motif_seq_mask], dim=1)

            ext_residue_index = torch.cat(
                [residue_index, torch.zeros(B, M, device=device, dtype=residue_index.dtype)],
                dim=1,
            )
            if chain_index is not None:
                ext_chain_index = torch.cat(
                    [chain_index, torch.zeros(B, M, device=device, dtype=chain_index.dtype)],
                    dim=1,
                )
            else:
                ext_chain_index = None

            ext_crop_cond_mask = torch.cat(
                [
                    torch.zeros(B, protein_N, motif_condmask.shape[-1], device=device, dtype=motif_condmask.dtype),
                    motif_condmask,
                ],
                dim=1,
            )

            if struct_self_cond is not None:
                ext_struct_self_cond = torch.cat(
                    [struct_self_cond, motif_clean_coords / self.sigma_data], dim=1
                )
            else:
                ext_struct_self_cond = None

            ext_hotspot_mask = None
            if hotspot_mask is not None:
                ext_hotspot_mask = torch.cat(
                    [hotspot_mask, torch.zeros(B, M, device=device, dtype=hotspot_mask.dtype)],
                    dim=1,
                )

            # Pad binder_sse_prop_cond along seq dim to [B, N+M, 3] (zeros for motif tail).
            ext_binder_sse_prop_cond = None
            if binder_sse_prop_cond is not None:
                ext_binder_sse_prop_cond = torch.cat(
                    [
                        binder_sse_prop_cond,
                        torch.zeros(
                            B, M, binder_sse_prop_cond.shape[-1],
                            device=device, dtype=binder_sse_prop_cond.dtype,
                        ),
                    ],
                    dim=1,
                )
            struct_inputs = dict(
                noisy_coords=ext_noisy_coords,
                noise_level=noise_level,
                seq_mask=ext_seq_mask,
                residue_index=ext_residue_index,
                chain_index=ext_chain_index,
                hotspot_mask=ext_hotspot_mask,
                struct_self_cond=ext_struct_self_cond,
                struct_crop_cond=None,  # unindexed does not use struct_crop_cond channels
                crop_cond_mask=ext_crop_cond_mask,
                sse_cond=sse_cond,
                adj_cond=adj_cond,
                motif_sse_cond=motif_sse_cond,
                binder_sse_prop_cond=ext_binder_sse_prop_cond,
                motif_sasa_cond=motif_sasa_cond,
                motif_hbond_cond=motif_hbond_cond,
                # Atomized channels are protein-indexed; pad the appended motif
                # tail with zeros (= "no atomized information").
                atom_elem=_pad_seq_tail(atom_elem, M),
                atom_ref=_pad_seq_tail(atom_ref, M),
                atom_noncanon=_pad_seq_tail(atom_noncanon, M),
                atom_bondfeat=_pad_seq_tail(atom_bondfeat, M),
            )
        else:
            struct_inputs = dict(
                noisy_coords=noisy_coords,
                noise_level=noise_level,
                seq_mask=seq_mask,
                residue_index=residue_index,
                chain_index=chain_index,
                hotspot_mask=hotspot_mask,
                struct_self_cond=struct_self_cond,
                struct_crop_cond=struct_crop_cond,
                crop_cond_mask=crop_cond_mask,
                sse_cond=sse_cond,
                adj_cond=adj_cond,
                motif_sse_cond=motif_sse_cond,
                binder_sse_prop_cond=binder_sse_prop_cond,
                motif_sasa_cond=motif_sasa_cond,
                motif_hbond_cond=motif_hbond_cond,
                atom_elem=atom_elem,
                atom_ref=atom_ref,
                atom_noncanon=atom_noncanon,
                atom_bondfeat=atom_bondfeat,
            )

        # Coordinate denoiser
        if run_struct_model:
            model_output, _ = self.struct_model(
                struct_inputs["noisy_coords"],
                struct_inputs["noise_level"],
                struct_inputs["seq_mask"],
                residue_index=struct_inputs["residue_index"],
                chain_index=struct_inputs["chain_index"],
                hotspot_mask=struct_inputs["hotspot_mask"],
                struct_self_cond=struct_inputs["struct_self_cond"],
                struct_crop_cond=struct_inputs["struct_crop_cond"],
                crop_cond_mask=struct_inputs["crop_cond_mask"],
                sse_cond=struct_inputs["sse_cond"],
                adj_cond=struct_inputs["adj_cond"],
                motif_sse_cond=struct_inputs["motif_sse_cond"],
                binder_sse_prop_cond=struct_inputs["binder_sse_prop_cond"],
                motif_sasa_cond=struct_inputs["motif_sasa_cond"],
                motif_hbond_cond=struct_inputs["motif_hbond_cond"],
                atom_elem=struct_inputs["atom_elem"],
                atom_ref=struct_inputs["atom_ref"],
                atom_noncanon=struct_inputs["atom_noncanon"],
                atom_bondfeat=struct_inputs["atom_bondfeat"],
                return_emb=True,
            )
            if unindexed_active:
                model_output = model_output[:, :protein_N]
        else:
            model_output = noisy_coords

        # For flow matching, derive x_0_hat from velocity for self-conditioning
        if self.is_flow_matching:
            velocity = model_output
            t = noise_level
            x_0_hat = noisy_coords + unsqueeze_trailing_dims(1 - t, velocity) * velocity
            struct_self_cond_out = x_0_hat.detach() / self.sigma_data
            denoised_for_mpnn = x_0_hat.detach()
        else:
            denoised_for_mpnn = model_output
            struct_self_cond_out = model_output.detach() / self.sigma_data

        # MiniMPNN
        if self.use_mpnn_model and run_mpnn_model:
            assert isinstance(self.mpnn_model, MiniMPNN)
            aatype_logprobs = self.mpnn_model(
                denoised_for_mpnn,
                noise_level,
                seq_mask,
                residue_index,
                seq_self_cond=seq_self_cond,
                seq_crop_cond=seq_crop_cond,
                return_embeddings=False,
            )
            aatype_logprobs = aatype_logprobs * seq_mask.unsqueeze(-1)
        else:
            aatype_logprobs = repeat(seq_mask, "b n -> b n t", t=self.n_tokens)
            aatype_logprobs = torch.ones_like(aatype_logprobs)
            aatype_logprobs = F.log_softmax(aatype_logprobs, -1)

        seq_self_cond_out = aatype_logprobs.detach()

        return model_output, aatype_logprobs, struct_self_cond_out, seq_self_cond_out

    def make_seq_mask_for_sampling(
        self,
        prot_lens_per_chain: TensorType["b c", int] | None = None,
        length_ranges_per_chain: TensorType["c 2", int] | None = None,
        num_samples: int | None = None,
        chain_residx_gap: int | None = None,
    ) -> tuple[
        TensorType["b n", float], TensorType["b n", float], TensorType["b n", int]
    ]:
        """Makes sequence mask, residue indices, and chain ids of varying protein lengths (only inputs required
        to begin sampling).

        Args:
        - prot_lens_per_chain: tensor of protein lengths for each chain (batch size, num_chains)
        - length_ranges_per_chain: tensor of min and max protein lengths for each chain (num_chains, 2) if prot_lens_per_chain is None
        - num_samples: number of samples to generate if providing length_ranges_per_chain
        - chain_residx_gap: gap between chains in residue indices (defaults to model config)

        Returns:
        - seq_mask: sequence mask (batch size, max_len)
        - residue_index: residue indices (batch size, max_len)
        - chain_index: chain ids (batch size, max_len)
        """
        # Ensure only one of prot_lens_per_chain or length_ranges_per_chain is provided
        assert (prot_lens_per_chain is None) != (
            length_ranges_per_chain is None
        ), f"Only one of prot_lens_per_chain or length_ranges_per_chain should be provided. Got prot_lens_per_chain={prot_lens_per_chain} and length_ranges_per_chain={length_ranges_per_chain}"

        # Make protein lengths by sampling from provided ranges
        if length_ranges_per_chain is not None:
            assert (
                num_samples is not None
            ), f"Must provide num_samples if providing length_ranges_per_chain"
            prot_lens_per_chain = torch.stack(
                [
                    torch.randint(low=start, high=end + 1, size=(num_samples,))
                    for start, end in length_ranges_per_chain
                ],
                dim=1,
            )

        num_samples = prot_lens_per_chain.shape[0]

        # Make sequence mask
        total_prot_lens = prot_lens_per_chain.sum(dim=1)  # total length across chains
        max_len = torch.max(total_prot_lens)
        residue_index = repeat(
            torch.arange(1, max_len + 1), "n -> b n", b=num_samples
        ).float()
        mask = (residue_index <= total_prot_lens.unsqueeze(-1)).float().to(self.device)

        # Get chain IDs
        cum_lens = torch.cumsum(prot_lens_per_chain, dim=1)
        chain_index = torch.zeros_like(residue_index).long()
        for ci in range(cum_lens.shape[1]):
            chain_index += (residue_index > cum_lens[:, ci].unsqueeze(-1)).long()

        # Add residue index gap between chains
        if chain_residx_gap is None:
            chain_residx_gap = self.chain_residx_gap
        if chain_residx_gap == 0 and torch.sum(chain_index) > 0:
            for bi in range(num_samples):
                for ci in range(1, int(torch.max(chain_index[bi]).item()) + 1):
                    curr_chain_pos = torch.nonzero(chain_index[bi] == ci).flatten()
                    residue_index[bi, curr_chain_pos] = (
                        residue_index[bi, curr_chain_pos]
                        - residue_index[bi, curr_chain_pos[0]]
                        + 1
                    )
        else:
            residue_index = residue_index + chain_residx_gap * chain_index

        # Apply sequence mask
        residue_index = residue_index.to(self.device) * mask
        chain_index = chain_index.to(self.device) * mask
        return mask, residue_index, chain_index

    def _forward_guided(
        self,
        guidance_model,
        autoguidance_weight,
        negative_crop_conds=None,
        negative_guidance_weight: float = 0.0,
        **kw,
    ):
        """forward() with optional AUTOGUIDANCE (Karras et al. 2024, "Guiding a
        Diffusion Model with a Bad Version of Itself"). D_bad = an undertrained
        early checkpoint fed the IDENTICAL noisy input + conditioning; we push the
        denoised prediction away from it:  x0 = x0_good + w*(x0_good - x0_bad).
        w = 0 (or no guidance_model) reduces to a plain forward pass.

        NEGATIVE MOTIF GUIDANCE (negative design). Same extrapolation, but the
        "bad" predictions come from the SAME model conditioned on DIFFERENT
        crop-conditional motif tensors (the negative motifs):

            x0 = x0_pos + w * sum_i[ w_i * (x0_pos - x0_neg_i) ]

        i.e. classifier-free-guidance-style repulsion away from every
        p(structure | negative motif i) and towards p(structure | positive motif).
        `negative_crop_conds` is a list of (coords, condmask, w_i) triples;
        `negative_guidance_weight` is the global scale w. Multiple negatives
        support band-pass selectivity (push away from both larger and smaller
        ions at once). Cost is (1 + n_negatives) denoiser forwards per step.

        w = 0 (or no negative tensors) is an exact no-op: no extra forward is run
        and the RNG stream is untouched, so results are bit-identical to
        positive-only sampling.
        """
        x0, s_logprobs, x_self_cond, s_self_cond = self.forward(**kw)
        if guidance_model is not None and autoguidance_weight != 0.0:
            with torch.no_grad():
                x0_bad = guidance_model.forward(**kw)[0]
            x0 = x0 + autoguidance_weight * (x0 - x0_bad)
        if negative_crop_conds and negative_guidance_weight != 0.0:
            delta = None
            for neg_coords, neg_mask, neg_w in negative_crop_conds:
                if neg_coords is None or neg_w == 0.0:
                    continue
                neg_kw = dict(kw)
                neg_kw["struct_crop_cond"] = neg_coords
                neg_kw["crop_cond_mask"] = neg_mask
                # We only need x0_neg; skip MiniMPNN (its logprobs/self-cond are
                # taken from the positive pass) to halve the extra cost.
                neg_kw["run_mpnn_model"] = False
                with torch.no_grad():
                    x0_neg = self.forward(**neg_kw)[0]
                term = neg_w * (x0 - x0_neg)
                delta = term if delta is None else delta + term
            if delta is not None:
                x0 = x0 + negative_guidance_weight * delta
        return x0, s_logprobs, x_self_cond, s_self_cond

    def sample(
        self,
        *,
        seq_mask: TensorType["b n", float],
        residue_index: TensorType["b n", int],
        chain_index: TensorType["b n", int] | None = None,
        hotspots: str | list[str] | None = None,
        sse_cond: TensorType["b n", int] | None = None,
        adj_cond: TensorType["b n n", int] | None = None,
        gt_aatype: TensorType["b n", int] | None = None,
        n_steps: int = 200,
        step_scale: float = 1.2,
        s_churn: float = 50.0,
        noise_scale: float = 1.0,
        s_t_min: float = 0.01,
        s_t_max: float = 50.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        disallow_aas: list[int] = [4, 20],  # CYS, UNK
        sidechain_mode: bool = False,
        skip_mpnn_proportion: float = 0.7,
        anneal_seq_resampling_rate: str | None = None,  # linear, cosine
        use_fullmpnn: bool = False,
        use_fullmpnn_for_final: bool = False,
        noise_schedule: Callable | None = None,
        tqdm_pbar: Callable | None = None,
        return_last: bool = True,
        return_aux: bool = False,
        jump_steps: bool = True,  # used to be called "use_superposition"
        uniform_steps: bool = False,  # alternative to superposition
        motif_file_path: str | None = None,
        multi_motif_list: list[tuple[str, float]] | None = None,
        dx: float | None = None,
        dy: float | None = None,
        dz: float | None = None,
        dummy_fill_mode: Literal["zero", "CA"] = "zero",
        xt_start: TensorType["b n a x", float] | None = None,
        xt_0_path: str | None = None,
        xt_1_path: str | None = None,
        partial_diffusion: DictConfig | None = None,
        conditional_cfg: DictConfig | None = None,
        motif_placements_full: list[str] | None = None,
        motif_all_atom_stage1: TensorType["b n a x", float] | None = None,
        motif_idx_stage1: list[list[int]] | None = None,
        stage2: bool = False,
        tip_atom_conditioning: bool = False,
        motif_sse_label: str | None = None,
        motif_sse_flank: int | None = None,
        binder_sse_prop: list[float] | tuple[float, float, float] | None = None,
        motif_sasa_label: str | None = None,
        motif_hbond_label: str | list[str] | None = None,
        guidance_model: "Protpardelle | None" = None,
        autoguidance_weight: float = 0.0,
        negative_motifs: str | list[tuple[str, float]] | None = None,
        negative_guidance_weight: float = 0.0,
        negative_motif_all_atom_stage1: list[TensorType["b n a x", float]] | None = None,
    ):
        """Sampling function for backbone or all-atom diffusion.

        seq_mask: mask defining the number and lengths of proteins to be sampled.
        residue_index: residue index of proteins to be sampled.
        chain_index: integers denoting different chains, e.g. 0, 0, 1, 1 denotes two chains of two residues each
        hotspots: {chain}{residx} list or comma-delimited string indicating hotspot residues, e.g. A33,A95,A98,A102
        sse_cond: integer denoting the secondary structure class, corresponds to ordering in scripts/make_secstruc_adj.py
        adj_cond: block-adjacency matrix indicating contact of secondary structure elements, obtained from scripts/make_secstruc_adj.py
        gt_coords: conditioning information for coords.
        gt_coords_traj: conditioning information for coords specified for each timestep
            (if gt_coords is not provided).
        gt_cond_atom_mask: mask identifying atoms to apply gt_coords.
        gt_aatype: conditioning information for sequence.
        n_steps: number of denoising steps (ODE discretizations).
        step_scale: scale to apply to the score.
        s_churn: gamma = s_churn / n_steps describes the additional noise to add
            relatively at each denoising step. Use 0.0 for deterministic sampling or
            0.2 * n_steps as a rough default for stochastic sampling.
        noise_scale: scale to apply to gamma.
        s_t_min: don't apply s_churn below this noise level.
        s_t_max: don't apply s_churn above this noise level.
        temperature: scale to apply to aatype logits.
        top_p: don't tokens which fall outside this proportion of the total probability.
        disallow_aas: don't sample these token indices.
        sidechain_mode: whether to do all-atom sampling (False for backbone-only).
        skip_mpnn_proportion: proportion of timesteps from the start to skip running
            MiniMPNN.
        anneal_seq_resampling_rate: whether and how to decay the probability of
            running MiniMPNN. None, 'linear', or 'cosine'
        use_fullmpnn: use "full" ProteinMPNN at each step.
        use_fullmpnn_for_final: use "full" ProteinMPNN at the final step.
        noise_schedule: specify the noise level timesteps for sampling.
        tqdm_pbar: progress bar in interactive contexts.
        return_last: return only the sampled structure and sequence.
        return_aux: return a dict of everything associated with the sampling run.
        jump_steps: use superposition scheme for sampling.
        uniform_steps: allatom denoising with same noise level changes at each step, not superposition scheme
        motif_file_path: path to .pdb structure containing motif info, possibly containing additional unused residues
        dx: x-axis translation applied to the motif coordinates
        dy: y-axis translation applied to the motif coordinates
        dz: z-axis translation applied to the motif coordinates
        dummy_fill_mode: for allatom sampling, how to fill in currently unobserved sidechain atom coordinates before adding fresh noise proportional to the current noise level.
        xt_start: instead of starting from random noise, start from this provided xt_start
        partial_diffusion: config for partial diffusion settings, see examples in protpardelle/configs/sampling/sampling_partial_diffusion.yaml
        conditional_cfg: config for conditional generation, see examples in protpardelle/configs/sampling/*.yaml
        motif_placements_full: for motif scaffolding, indicates the location of motif segments within the scaffold, e.g. 12/B1-7/40/A1-7/41 indicates 12 scaffold / B1-7 motif / 40 scaffold / A1-7 motif / 41 scaffold residues
        motif_all_atom_stage1: used in two-stage allatom sampling, cache the rotated and centered motif from stage1
        motif_idx_stage1: used in two-stage allatom sampling, cache the motif index used in stage1
        stage2: flag to indicate stage2
        tip_atom_conditioning: true to use reconstruction and replacement guidance for tip atom conditioning
        negative_motifs: motif(s) to steer AWAY from ("negative design"). Either a
            single path, or a list of (path, relative_weight) pairs — mirrors the
            multi_motif_list format. Every negative motif must contain exactly the
            same residues and atoms as motif_file_path (asserted); only the
            coordinates may differ. Each is placed on the same contig positions and
            rigid-transformed with the SAME centering/rotation/translation as the
            positive motif, so the only difference reaching the denoiser is motif
            geometry.
        negative_guidance_weight: global scale w in
            x0 = x0_pos + w * sum_i[w_i * (x0_pos - x0_neg_i)].
            0.0 disables negative guidance entirely (exact no-op, no extra forward).
        negative_motif_all_atom_stage1: like motif_all_atom_stage1 but a LIST of
            cached transformed negative-motif coords (one per negative motif), so
            stage2 / the Caliby re-denoise pass keep the negative conditioning
            active without having to recompute the (unavailable) centering/rotation.
        """

        # Autoguidance model can be attached to the model object (sample.py sets
        # model._guidance_model / _autoguidance_weight) instead of threaded through
        # the many sampling wrappers.
        if guidance_model is None:
            guidance_model = getattr(self, "_guidance_model", None)
        if autoguidance_weight == 0.0:
            autoguidance_weight = getattr(self, "_autoguidance_weight", 0.0)

        cc = apply_dotdict_recursively(conditional_cfg)  # shorthand
        pd = apply_dotdict_recursively(partial_diffusion)
        latent_interpolation = xt_0_path is not None and xt_1_path is not None

        if sse_cond is not None and adj_cond is not None:
            sse_cond = sse_cond.to(self.device)
            adj_cond = adj_cond.to(self.device)

        random_rots = None
        centering_delta = None

        if not cc.enabled:
            motif_idx = None
            motif_feats = None
            het_atom_pos = None
            motif_aatype = None
            motif_aa3 = None

        loss_weights = cc.reconstruction_guidance.loss_weights

        batch_size = seq_mask.shape[0]
        seq_length = seq_mask.shape[1]
        xt_rep = None

        all_motif_feats = []
        all_het_atom_pos = []

        motif_all_atom = None
        motif_atom_mask = None
        multi_motif_data = None
        if cc.enabled:
            if multi_motif_list is not None and len(multi_motif_list) > 0:
                # Multi-motif: load each motif and store with normalized weights
                multi_motif_data = []
                total_weight = sum(w for _, w in multi_motif_list)
                for motif_path, weight in multi_motif_list:
                    mf, _ = load_feats_from_pdb(motif_path, include_pos_feats=True)
                    normalized_weight = weight / total_weight
                    multi_motif_data.append({
                        'feats': mf,
                        'weight': normalized_weight,
                        'path': motif_path
                    })
                # Use first motif as template for downstream processing
                motif_feats = multi_motif_data[0]['feats']
                hetero_obj = None
                het_atom_pos = torch.zeros(0).to(seq_mask.device)
            else:
                # Single motif (existing logic)
                _cc_strategy = (
                    cc.crop_conditional_guidance.strategy
                    if cc.crop_conditional_guidance is not None
                    else ""
                ) or ""
                _inject_hoh = "water-bbO" in _cc_strategy
                motif_feats, hetero_obj = load_feats_from_pdb(
                    motif_file_path,
                    include_pos_feats=True,
                    inject_hoh_as_bb_o=_inject_hoh,
                )
                het_atom_pos = torch.from_numpy(
                    np.array(
                        [pos for res in hetero_obj.hetero_atom_positions for pos in res]
                    )
                ).to(seq_mask.device)
            all_motif_feats.append(motif_feats)
            all_het_atom_pos.append(het_atom_pos)

            motif_feats = all_motif_feats[0]

            # Parse hotspots on motif chains
            motif_hotspot_mask = torch.zeros_like(motif_feats["chain_index"])
            chain_id_mapping = motif_feats.pop("chain_id_mapping")
            if hotspots is not None:
                hotspot_indices = parse_fixed_pos_str(
                    hotspots,
                    chain_id_mapping,
                    motif_feats["residue_index_orig"],
                    motif_feats["chain_index"],
                )
                motif_hotspot_mask[hotspot_indices] = 1
            motif_feats["hotspot_mask"] = motif_hotspot_mask

            # record the contiguous stretches of a motif and only perform a single assignment for a contiguous stretch
            motif_residx = motif_feats["residue_index"].numpy().astype(int)
            motif_aatype = motif_feats["aatype"]
            # HOH placeholder residues (injected from waters) carry GLY's aatype
            # but should be labeled "HOH" in motif_aa3 so the condmask/MPNN logic
            # can route them through the backbone-O-tip branch and skip the
            # AA-pinning step.
            motif_is_water_vec = motif_feats.get(
                "is_water_motif", torch.zeros(len(motif_aatype), dtype=torch.bool)
            )
            motif_aa3 = [
                "HOH" if motif_is_water_vec[k] else
                residue_constants.restype_1to3[residue_constants.restypes[aa_idx]]
                for k, aa_idx in enumerate(motif_aatype)
            ]

            motif_all_atom = motif_feats["atom_positions"].to(
                self.device
            )  # [num_res, 37, 3]

        if cc.enabled:
            if motif_placements_full is not None:

                all_motif_feats = place_motif_feats_on_contigs(
                    motif_feats, chain_id_mapping, motif_placements_full
                )

                motif_residx = [
                    mf["residue_index"].numpy().astype(int) for mf in all_motif_feats
                ]
                motif_aatype = [mf["aatype"] for mf in all_motif_feats]
                motif_is_water_per_batch = [
                    mf.get("is_water_motif", torch.zeros(len(mf["aatype"]), dtype=torch.bool))
                    for mf in all_motif_feats
                ]
                motif_aa3 = []
                for ma, iw in zip(motif_aatype, motif_is_water_per_batch):
                    motif_aa3.append(
                        [
                            "HOH" if iw[k] else
                            residue_constants.restype_1to3[
                                residue_constants.restypes[aa_idx]
                            ]
                            for k, aa_idx in enumerate(ma)
                        ]
                    )
                motif_all_atom = torch.stack(
                    [mf["atom_positions"].to(self.device) for mf in all_motif_feats]
                )
                motif_atom_mask = torch.stack(
                    [mf["atom_mask"].to(self.device) for mf in all_motif_feats]
                )
                motif_hotspot_mask = torch.stack(
                    [mf["hotspot_mask"].to(self.device) for mf in all_motif_feats]
                )
            else:
                motif_all_atom = torch.tile(
                    motif_all_atom.unsqueeze(0), dims=(batch_size, 1, 1, 1)
                )
                # Also populate motif_atom_mask and motif_hotspot_mask for unindexed
                # or other paths that don't go through motif_placements_full.
                motif_atom_mask = torch.tile(
                    motif_feats["atom_mask"].to(self.device).unsqueeze(0),
                    dims=(batch_size, 1, 1),
                )
                motif_hotspot_mask = torch.tile(
                    motif_feats["hotspot_mask"].to(self.device).unsqueeze(0),
                    dims=(batch_size, 1),
                )

            # Extract relevant residues for multi-motif based on motif_placements_full
            if multi_motif_data is not None and motif_placements_full is not None:
                for mm_entry in multi_motif_data:
                    mm_feats = mm_entry['feats']
                    mm_chain_id_mapping = mm_feats.pop("chain_id_mapping") if "chain_id_mapping" in mm_feats else chain_id_mapping

                    mm_all_motif_feats = []
                    for mp_chains in motif_placements_full:
                        mm_chain_motif_feats = defaultdict(list)
                        for mp in mp_chains.split(";/;"):
                            curr_mm_feats = copy.deepcopy(mm_feats)
                            motif_chain_resids = [
                                s for s in mp.split("/") if s[0].isalpha()
                            ]
                            flat_raw_idx = parse_fixed_pos_str(
                                ",".join(motif_chain_resids),
                                mm_chain_id_mapping,
                                mm_feats["residue_index_orig"],
                                mm_feats["chain_index"],
                            )
                            if motif_chain_resids:
                                for k, v in curr_mm_feats.items():
                                    mm_chain_motif_feats[k].append(v[flat_raw_idx])
                        for k, v in mm_chain_motif_feats.items():
                            mm_chain_motif_feats[k] = torch.cat(v, dim=0)
                        mm_all_motif_feats.append(mm_chain_motif_feats)

                    # Stack extracted features for this multi-motif entry
                    mm_entry['extracted_all_atom'] = torch.stack(
                        [mf["atom_positions"].to(self.device) for mf in mm_all_motif_feats]
                    )
                    mm_entry['extracted_atom_mask'] = torch.stack(
                        [mf["atom_mask"].to(self.device) for mf in mm_all_motif_feats]
                    )

            if motif_all_atom_stage1 is not None:
                motif_all_atom = motif_all_atom_stage1.clone()
            else:
                # center the motif on CA coords
                if partial_diffusion is not None and pd.enabled:
                    pd_feats, _ = load_feats_from_pdb(pd.pdb_file_path, include_pos_feats=True)
                    centering_delta = torch.mean(pd_feats["atom_positions"][:, 1:2, :], dim=-3, keepdim=True).to(self.device)
                else:
                    centering_delta = torch.mean(
                        motif_all_atom[..., 1:2, :], dim=-3, keepdim=True
                    )
                motif_all_atom = motif_all_atom - centering_delta

                # randomly rotate the motif
                random_rots = torch.stack(
                    [
                        uniform_rand_rotation(1)[0].to(self.device)
                        for _ in range(batch_size)
                    ]
                )
                motif_all_atom = torch.einsum(
                    "bij,blnj->blni", random_rots, motif_all_atom
                )
                motif_all_atom = motif_all_atom * motif_atom_mask.unsqueeze(-1).to(
                    motif_all_atom
                )

            motif_size = motif_all_atom.shape[-3]
            print(
                f"Using motif from {motif_file_path} with {motif_size} motif residues."
            )

            # translate the motif
            if dx is not None and dx != "":
                motif_all_atom[..., 0] = motif_all_atom[..., 0] + dx
            if dy is not None and dy != "":
                motif_all_atom[..., 1] = motif_all_atom[..., 1] + dy
            if dz is not None and dz != "":
                motif_all_atom[..., 2] = motif_all_atom[..., 2] + dz

            # Transform multi-motif coordinates using same centering and rotation
            if multi_motif_data is not None:
                for mm_entry in multi_motif_data:
                    # Use extracted features if available, otherwise use full feats
                    if 'extracted_all_atom' in mm_entry:
                        mm_all_atom = mm_entry['extracted_all_atom']
                        mm_atom_mask = mm_entry['extracted_atom_mask']
                    else:
                        mm_all_atom = mm_entry['feats']["atom_positions"].to(self.device)
                        mm_atom_mask = mm_entry['feats']["atom_mask"].to(self.device)
                        # Expand to batch size
                        mm_all_atom = torch.tile(mm_all_atom.unsqueeze(0), dims=(batch_size, 1, 1, 1))
                        mm_atom_mask = torch.tile(mm_atom_mask.unsqueeze(0), dims=(batch_size, 1, 1))

                    # Apply same centering as main motif
                    mm_all_atom = mm_all_atom - centering_delta

                    # Apply same rotation
                    mm_all_atom = torch.einsum("bij,blnj->blni", random_rots, mm_all_atom)

                    # Apply atom mask
                    mm_all_atom = mm_all_atom * mm_atom_mask.unsqueeze(-1)

                    # Apply same translations
                    if dx is not None and dx != "":
                        mm_all_atom[..., 0] = mm_all_atom[..., 0] + dx
                    if dy is not None and dy != "":
                        mm_all_atom[..., 1] = mm_all_atom[..., 1] + dy
                    if dz is not None and dz != "":
                        mm_all_atom[..., 2] = mm_all_atom[..., 2] + dz

                    mm_entry['all_atom'] = mm_all_atom

                print(f"Using multi-motif conditioning with {len(multi_motif_data)} motifs.")

        # ------------------------------------------------------------------
        # NEGATIVE MOTIF(S) (negative design / repulsive guidance)
        # Each negative motif is loaded, contig-placed and rigid-transformed
        # EXACTLY like the positive motif, so the only thing differing between
        # the positive and negative conditioning tensors is motif geometry.
        # Multiple negatives are supported (band-pass selectivity: push away from
        # both larger and smaller variants at once); the per-step cost is
        # (1 + n_negatives) denoiser forwards.
        # ------------------------------------------------------------------
        neg_motif_coords_list: list = []          # transformed coords per negative
        neg_motif_weights: list[float] = []       # relative weight w_i per negative
        negative_guidance_weight = float(negative_guidance_weight or 0.0)
        negative_motif_entries: list[tuple[str, float]] = []
        if negative_motifs is not None:
            if isinstance(negative_motifs, (str, Path)):
                negative_motif_entries = [(str(negative_motifs), 1.0)]
            else:
                for entry in negative_motifs:
                    if isinstance(entry, (str, Path)):
                        negative_motif_entries.append((str(entry), 1.0))
                    else:
                        negative_motif_entries.append((str(entry[0]), float(entry[1])))
        if (
            cc.enabled
            and negative_motif_entries
            and negative_guidance_weight != 0.0
        ):
            if getattr(self.struct_model, "is_unindexed", False):
                raise NotImplementedError(
                    "negative_motifs is only supported for indexed "
                    "(struct_crop_cond) conditioning; this model is unindexed."
                )
            if negative_motif_all_atom_stage1 is not None:
                # stage2 / Caliby re-denoise: reuse the transformed coords from
                # the main pass (centering_delta / random_rots are unavailable here).
                cached = negative_motif_all_atom_stage1
                if torch.is_tensor(cached):
                    cached = [cached]
                if len(cached) != len(negative_motif_entries):
                    raise ValueError(
                        f"negative_motif_all_atom_stage1 has {len(cached)} entries but "
                        f"{len(negative_motif_entries)} negative motifs were requested."
                    )
                neg_motif_coords_list = [c.clone().to(self.device) for c in cached]
                neg_motif_weights = [w for _, w in negative_motif_entries]
            elif motif_all_atom_stage1 is not None:
                raise ValueError(
                    "Negative-design guidance is enabled and the positive motif was "
                    "supplied via motif_all_atom_stage1, but negative_motif_all_atom_stage1 "
                    "was not. The negative motifs cannot be re-derived because the "
                    "centering/rotation of the main pass is not available here. Pass the "
                    "cached negative coords (aux['negative_motif_all_atom'])."
                )
            else:
                _neg_strategy = (
                    cc.crop_conditional_guidance.strategy
                    if cc.crop_conditional_guidance is not None
                    else ""
                ) or ""
                for neg_path, neg_w in negative_motif_entries:
                    neg_motif_feats, _ = load_feats_from_pdb(
                        neg_path,
                        include_pos_feats=True,
                        inject_hoh_as_bb_o=("water-bbO" in _neg_strategy),
                    )
                    neg_chain_id_mapping = neg_motif_feats.pop("chain_id_mapping")
                    # Hard guard: identical residues/atoms, coords may differ.
                    assert_identical_motif_atom_sets(
                        motif_feats,
                        neg_motif_feats,
                        str(motif_file_path),
                        str(neg_path),
                    )
                    if motif_placements_full is not None:
                        neg_all_motif_feats = place_motif_feats_on_contigs(
                            neg_motif_feats, neg_chain_id_mapping, motif_placements_full
                        )
                        neg_coords = torch.stack(
                            [
                                mf["atom_positions"].to(self.device)
                                for mf in neg_all_motif_feats
                            ]
                        )
                    else:
                        neg_coords = torch.tile(
                            neg_motif_feats["atom_positions"]
                            .to(self.device)
                            .unsqueeze(0),
                            dims=(batch_size, 1, 1, 1),
                        )
                    # Apply the SAME rigid transform as the positive motif.
                    neg_coords = neg_coords - centering_delta
                    neg_coords = torch.einsum(
                        "bij,blnj->blni", random_rots, neg_coords
                    )
                    neg_coords = neg_coords * motif_atom_mask.unsqueeze(-1).to(
                        neg_coords
                    )
                    if dx is not None and dx != "":
                        neg_coords[..., 0] = neg_coords[..., 0] + dx
                    if dy is not None and dy != "":
                        neg_coords[..., 1] = neg_coords[..., 1] + dy
                    if dz is not None and dz != "":
                        neg_coords[..., 2] = neg_coords[..., 2] + dz
                    neg_motif_coords_list.append(neg_coords)
                    neg_motif_weights.append(neg_w)
            print(
                f"Using {len(neg_motif_coords_list)} NEGATIVE motif(s) "
                f"{[p for p, _ in negative_motif_entries]} with relative weights "
                f"{neg_motif_weights} and global negative_guidance_weight="
                f"{negative_guidance_weight}."
            )

        def ode_step(
            sigma_in,
            sigma_next,
            xt_in,
            x0_pred,
            guidance_in=None,
            curr_step=0,
            stage2=False,
        ):

            mask = (sigma_in > 0).float()
            score = (xt_in - x0_pred) / unsqueeze_trailing_dims(
                sigma_in.clamp(min=1e-6), xt_in
            )
            score = score * unsqueeze_trailing_dims(mask, score)

            # reconstruction guidance
            recon_on = curr_step >= (
                cc.reconstruction_guidance.start * n_steps
            ) and curr_step < (cc.reconstruction_guidance.end * n_steps)
            if (
                cc.enabled
                and cc.reconstruction_guidance.enabled
                and recon_on
                and guidance_in is not None
            ):
                guidance, guidance_mask = guidance_in
                guidance = guidance * guidance_mask.unsqueeze(-1)
                guidance_scale = get_time_dependent_scale(
                    cc.reconstruction_guidance.schedule,
                    cc.reconstruction_guidance.max_scale,
                    curr_step,
                    n_steps,
                    stage2=stage2,
                )
                score = score + guidance * guidance_scale

            step = (
                score
                * step_scale
                * unsqueeze_trailing_dims(sigma_next - sigma_in, score)
            )
            new_xt = xt_in + step
            return new_xt

        def sample_aatype(logprobs):
            # Top-p truncation
            probs = F.softmax(logprobs.clone(), dim=-1)
            sorted_prob, sorted_idxs = torch.sort(probs, descending=True)
            cumsum_prob = torch.cumsum(sorted_prob, dim=-1)
            sorted_indices_to_remove = cumsum_prob > top_p
            sorted_indices_to_remove[..., 0] = 0
            sorted_prob[sorted_indices_to_remove] = 0
            orig_probs = torch.scatter(
                torch.zeros_like(sorted_prob),
                dim=-1,
                index=sorted_idxs,
                src=sorted_prob,
            )

            # Apply temperature and disallowed AAs and sample
            assert temperature >= 0.0
            scaled_logits = orig_probs.clamp(min=1e-9).log() / (temperature + 1e-4)
            if disallow_aas:
                unwanted_mask = torch.zeros(scaled_logits.shape[-1]).to(scaled_logits)
                unwanted_mask[disallow_aas] = 1
                scaled_logits -= unwanted_mask * 1e3
            orig_probs = F.softmax(scaled_logits, dim=-1)
            categorical = torch.distributions.Categorical(probs=orig_probs)
            samp_aatype = categorical.sample()
            return samp_aatype

        def design_with_fullmpnn(
            batched_coords, seq_mask, motif_aatype=None, motif_idx=None
        ):
            seq_lens = seq_mask.sum(-1).long()
            if motif_aatype is not None:
                designed_seqs = []
                for i, c in enumerate(batched_coords):
                    _input_aatype = (
                        torch.ones(seq_lens[i], dtype=torch.long)
                        * residue_constants.restype_order["G"]
                    )
                    _fixed_pos_mask = torch.zeros_like(
                        _input_aatype
                    )  #  0 for positions to redesign, 1 for positions to keep fixed
                    _input_aatype[motif_idx[i]] = motif_aatype[i]
                    _fixed_pos_mask[motif_idx[i]] = 1

                    designed_seqs.append(
                        design_sequence(
                            c[: seq_lens[i]],
                            model=fullmpnn_model,
                            chain_index=chain_index[i, : seq_lens[i]].cpu(),
                            input_aatype=_input_aatype,
                            fixed_pos_mask=_fixed_pos_mask,
                        )[0]
                    )
            else:
                designed_seqs = [
                    design_sequence(
                        c[: seq_lens[i]],
                        model=fullmpnn_model,
                        chain_index=chain_index[i, : seq_lens[i]].cpu(),
                    )[0]
                    for i, c in enumerate(batched_coords)
                ]
            designed_aatypes, _ = batched_seq_to_aatype_and_mask(
                designed_seqs, max_len=seq_mask.shape[-1]
            )
            return designed_aatypes

        # Initialize masks/features
        if use_fullmpnn or use_fullmpnn_for_final:
            fullmpnn_model = protein_mpnn.get_mpnn_model(
                PROTEINMPNN_WEIGHTS, device=self.device
            )

        # Initialize noise schedule/parameters
        s_t_min = s_t_min * self.sigma_data
        s_t_max = s_t_max * self.sigma_data

        if noise_schedule is None:
            noise_schedule = self.sampling_noise_schedule_default

        sigma = sigma_float = noise_schedule(1)

        timesteps = torch.linspace(1, 0, n_steps + 1)

        crop_cond_coords = None

        coords_shape = seq_mask.shape + (self.n_atoms, 3)
        if xt_start is not None:
            print(f"Using supplied xt to start diffusion")
            xt = xt_start
        elif partial_diffusion is not None and pd.enabled:
            pd_step = n_steps - pd.n_steps
            pd_timestep = timesteps[pd_step]
            if not isinstance(pd.pdb_file_path, list):
                pd.pdb_file_path = [pd.pdb_file_path] * batch_size

            pd_motif_aatype, pd_motif_idx, pd_coords = [], [], []
            for pd_idx, pd_fp in enumerate(pd.pdb_file_path):
                pd_feats, _ = load_feats_from_pdb(pd_fp, include_pos_feats=True)
                pd_motif_aatype.append(
                    make_fixed_size_1d(
                        pd_feats["aatype"].flatten().clone().detach(),
                        fixed_size=seq_mask.shape[-1],
                    )[0]
                )
                pd_motif_idx.append(torch.arange(pd_feats["aatype"].shape[0]))

                if motif_all_atom_stage1 is not None:
                    pd_atom_positions = pd_feats["atom_positions"].to(self.device)
                else:
                    pd_atom_positions = pd_feats[
                        "atom_positions"
                    ] - torch.mean(
                        pd_feats["atom_positions"][:, 1:2, :], dim=-3, keepdim=True
                    )
                    pd_atom_positions = pd_atom_positions.to(self.device)
                    if cc.enabled:
                        pd_atom_positions = torch.einsum(
                            "ij,lnj->lni", random_rots[pd_idx], pd_atom_positions
                        )
                
                pd_coords.append(
                    make_fixed_size_1d(
                        pd_atom_positions,
                        fixed_size=seq_mask.shape[-1],
                    )[0]
                )
            pd_motif_aatype = (
                torch.stack(pd_motif_aatype).long().to(seq_mask.device)
            )
            pd_coords = torch.stack(pd_coords).to(self.device)

            pd_noise_level = torch.full(
                (seq_mask.shape[0],), noise_schedule(pd_timestep), device=self.device
            )
            if "repack" in pd and pd.repack:
                for pi, pd_aa in enumerate(pd.seq):
                    pd_motif_aatype[:, pi] = residue_constants.restype_order[pd_aa]
                pd_atom_mask = atom37_mask_from_aatype(pd_motif_aatype, seq_mask)
                bb_seq = (seq_mask * residue_constants.restype_order["G"]).long()
                bb_atom_mask = atom37_mask_from_aatype(bb_seq, seq_mask)
                xt = diffusion.noise_coords(
                    pd_coords,
                    pd_noise_level,
                    atom_mask=bb_atom_mask,
                    dummy_fill_mode=dummy_fill_mode,
                )
            else:
                pd_atom_mask = atom37_mask_from_aatype(pd_motif_aatype, seq_mask)
                xt = diffusion.noise_coords(
                    pd_coords,
                    pd_noise_level,
                    atom_mask=pd_atom_mask,
                    dummy_fill_mode=dummy_fill_mode,
                )
            print(
                f"Partial diffusion, going back to step {pd_step}, T={(pd_step / n_steps):.2f}"
            )

            if latent_interpolation:
                print(f"Using supplied xt_0 and xt_1 for latent interpolation")
                xt_0 = torch.load(xt_0_path)
                xt_1 = torch.load(xt_1_path)
                for ii, interp_i in enumerate(range(batch_size)):
                    alpha = interp_i / (batch_size - 1)
                    interp_xt = xt_0 * (1 - alpha) + xt_1 * alpha
                    xt[ii] = interp_xt
        else:
            xt = torch.randn(*coords_shape).to(self.device)
            xt = xt * sigma

        xt = xt * unsqueeze_trailing_dims(seq_mask, xt)

        # Seqhat and mask used to choose sidechains for euler step (b, n)
        if not pd.enabled:
            if jump_steps or uniform_steps:
                if gt_aatype is None:
                    fake_logits = repeat(seq_mask, "b n -> b n t", t=self.n_tokens)
                    s_hat = (sample_aatype(fake_logits) * seq_mask).long()
                else:
                    s_hat = gt_aatype
            else:
                s_hat = (seq_mask * 7).long()
        else:
            if jump_steps or uniform_steps:
                if isinstance(pd_motif_aatype, list):
                    pd_motif_aatype = torch.stack(pd_motif_aatype)
                s_hat = pd_motif_aatype.clone()
            else:
                s_hat = (seq_mask * 7).long()

        # Initialize superposition for all-atom sampling
        if jump_steps:
            b, n = seq_mask.shape[:2]

            # Latest predicted x0 for sidechain superpositions
            atom73_state_0 = torch.zeros(b, n, 73, 3).to(xt)

            # Current state xt for sidechain superpositions (denoised to different levels)
            atom73_state_t = torch.randn(b, n, 73, 3).to(xt) * sigma

            # Noise level of xt
            sigma73_last = torch.ones(b, n, 73).to(xt) * sigma

            mask37 = atom37_mask_from_aatype(s_hat, seq_mask).bool()
            mask73 = atom73_mask_from_aatype(s_hat, seq_mask).bool()

        begin_mpnn_step = int(n_steps * skip_mpnn_proportion)

        # Prepare to run sampling trajectory
        sigma = torch.full((seq_mask.shape[0],), sigma, device=self.device)
        x0 = None
        x_self_cond = None
        s_logprobs = None
        s_self_cond = None
        if tqdm_pbar is None:
            tqdm_pbar = lambda x: x
        torch.set_grad_enabled(False)

        # t_traj is the denoising trajectory; *0_traj is the evolution of predicted clean data
        # s0 are aatype probs of shape (b n t); s_hat are discrete aatype of shape (b n)
        x0 = None

        # Unindexed conditioning prep: motif residues are appended along seq dim
        # in Protpardelle.forward rather than placed at indexed scaffold positions.
        is_unindexed = getattr(self.struct_model, "is_unindexed", False)
        motif_clean_coords_unindexed = None
        motif_condmask_unindexed = None
        if cc.enabled and is_unindexed:
            M_cap = self.config.model.max_motif_size
            K = motif_all_atom.shape[1]
            assert K <= M_cap, (
                f"Motif has {K} residues but max_motif_size is {M_cap}. "
                "Increase max_motif_size or trim the motif."
            )
            # Pad motif coords to M_cap along sequence dim.
            motif_clean_coords_unindexed = torch.zeros(
                batch_size, M_cap, motif_all_atom.shape[2], 3,
                device=motif_all_atom.device, dtype=motif_all_atom.dtype,
            )
            motif_clean_coords_unindexed[:, :K] = motif_all_atom

            # Build condmask following the configured crop strategy.
            motif_condmask_unindexed = torch.zeros(
                batch_size, M_cap, motif_all_atom.shape[2],
                device=motif_all_atom.device, dtype=motif_all_atom.dtype,
            )
            strategy = cc.crop_conditional_guidance.strategy
            is_acidic_tip = "acidic-tip" in strategy
            # Parse "+<aa3>-full" suffixes for residues that get full-atom conditioning.
            full_aa3_set = set()
            for token in strategy.split("+"):
                token = token.strip().lower()
                if token.endswith("-full"):
                    aa_code = token[: -len("-full")].upper()
                    if aa_code in residue_constants.restype_3to1:
                        full_aa3_set.add(aa_code)
            # motif_aa3 is flat when motif_placements_full is None; list-per-batch otherwise.
            bb_o_idx = residue_constants.atom_order["O"]
            for bi in range(batch_size):
                aa3_list = motif_aa3[bi] if (isinstance(motif_aa3, list) and len(motif_aa3) > 0 and isinstance(motif_aa3[0], list)) else motif_aa3
                for mi, aa3 in enumerate(aa3_list[:K]):
                    if aa3 == "HOH":
                        # Water placeholder: only the backbone O (atom index 4)
                        # is anchored; everything else is unconstrained so the
                        # scaffold can adopt any residue identity here.
                        motif_condmask_unindexed[bi, mi, bb_o_idx] = 1.0
                        continue
                    if is_acidic_tip:
                        if aa3 in ("ASP", "GLU"):
                            tip_atoms = residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS[aa3]
                            atom_indices = [residue_constants.atom_order.get(at) for at in tip_atoms]
                            for ai in atom_indices:
                                motif_condmask_unindexed[bi, mi, ai] = 1.0
                        elif aa3 in full_aa3_set:
                            # Full residue (all atoms including backbone) — preserve all
                            motif_condmask_unindexed[bi, mi, :] = 1.0
                    elif strategy == "sidechain-tip" and aa3 in residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS:
                        tip_atoms = residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS[aa3]
                        atom_indices = [residue_constants.atom_order.get(at) for at in tip_atoms]
                        for ai in atom_indices:
                            motif_condmask_unindexed[bi, mi, ai] = 1.0
                    else:
                        # Default (incl. "backbone-sidechain"): derive from non-zero
                        # coords in the source motif PDB. This means atoms missing in
                        # the input PDB (e.g. AME specific_atoms files where only
                        # certain atoms per residue are listed) get condmask=0, so
                        # the model isn't told "this atom is at origin".
                        motif_condmask_unindexed[bi, mi] = (
                            motif_clean_coords_unindexed[bi, mi].abs().sum(-1) > 0
                        ).to(motif_condmask_unindexed.dtype)
            # Zero backbone atoms globally ONLY for pure tip strategies. When full_aa3_set
            # is non-empty, those residues' backbone was explicitly set above and must
            # not be stripped. HOH water residues need their backbone O preserved.
            if strategy == "acidic-tip" or strategy == "sidechain-tip":
                motif_condmask_unindexed[:, :, (0, 1, 2, 4)] = 0
            elif is_acidic_tip and full_aa3_set:
                # Zero backbone for ASP/GLU / other non-full residues; keep backbone for
                # full residues by restoring their backbone mask explicitly. Water
                # placeholders restore their single bb-O slot the same way.
                bb_idx = (0, 1, 2, 4)
                motif_condmask_unindexed[:, :, bb_idx] = 0
                for bi in range(batch_size):
                    aa3_list = motif_aa3[bi] if (isinstance(motif_aa3, list) and len(motif_aa3) > 0 and isinstance(motif_aa3[0], list)) else motif_aa3
                    for mi, aa3 in enumerate(aa3_list[:K]):
                        if aa3 in full_aa3_set:
                            for ai in bb_idx:
                                motif_condmask_unindexed[bi, mi, ai] = 1.0
                        elif aa3 == "HOH":
                            motif_condmask_unindexed[bi, mi, bb_o_idx] = 1.0
            # Zero out non-conditioned atom coords so padded/unconditioned positions don't leak.
            motif_clean_coords_unindexed = (
                motif_clean_coords_unindexed * motif_condmask_unindexed.unsqueeze(-1)
            )

        if cc.enabled and not is_unindexed:
            if motif_idx_stage1 is None:
                if len(motif_residx) != batch_size:
                    motif_residx = [
                        motif_residx for _ in range(seq_mask.shape[0])
                    ]  # batchify
                motif_idx_contigs = [
                    group_consecutive_idx(mr) for mr in motif_residx
                ]  # this only defines the groups, not the actual indices (could be arbitrary from PDB index)
                motif_idx_contigs = [
                    contig_to_idx(mic) for mic in motif_idx_contigs
                ]  # convert to grouped indices, 0-indexed
                motif_idx = [
                    [mi for sublist in mic for mi in sublist]
                    for mic in motif_idx_contigs
                ]  # flattened motif_idx, to use in actual indexing

                if cc.enabled and (
                    cc.discontiguous_motif_assignment.enabled
                    and cc.discontiguous_motif_assignment.strategy == "fixed"
                ):
                    assert len(cc.discontiguous_motif_assignment.fixed_motif_pos) > 0
                    m_idx = cc.discontiguous_motif_assignment.fixed_motif_pos
                    if isinstance(m_idx[0], int):
                        motif_idx = [m_idx for _ in range(batch_size)]
                    else:
                        motif_idx = m_idx
            else:
                motif_idx = motif_idx_stage1

            to_motif_size = lambda x: x * torch.ones(batch_size, motif_size).to(
                self.device
            )
        elif cc.enabled and is_unindexed:
            # For unindexed, motif is not placed at scaffold indices. Preserve the
            # real motif aatypes so end-of-sampling bipartite matching can pin the
            # matched scaffold positions to ASP/GLU etc. Use empty motif_idx during
            # the loop so intermediate `for bi, _ in enumerate(motif_idx)` loops are
            # no-ops.
            motif_idx = [[] for _ in range(batch_size)]
            if isinstance(motif_aatype, torch.Tensor):
                # Flat tensor [K] -> list-per-batch, tiled
                motif_aatype_unindexed = [
                    motif_aatype.clone().to(torch.long) for _ in range(batch_size)
                ]
            else:
                motif_aatype_unindexed = motif_aatype
            # Parallel per-batch bool list flagging HOH placeholder rows so the
            # end-of-sample MPNN-fix step can skip them (waters constrain only the
            # backbone O atom; the residue identity at the matched position is
            # designed freely).
            if isinstance(motif_aa3, list) and len(motif_aa3) > 0 and isinstance(motif_aa3[0], list):
                motif_is_water_unindexed = [
                    torch.tensor([a == "HOH" for a in aa3_list], dtype=torch.bool)
                    for aa3_list in motif_aa3
                ]
            else:
                _flat_water = [a == "HOH" for a in motif_aa3] if motif_aa3 else []
                motif_is_water_unindexed = [
                    torch.tensor(_flat_water, dtype=torch.bool) for _ in range(batch_size)
                ]
            motif_aatype = [torch.tensor([], dtype=torch.long) for _ in range(batch_size)]

        # `*_for_fix` are the AA-pinning lists handed to fill_motif_seq /
        # design_with_fullmpnn. They start as a 1:1 alias of motif_idx /
        # motif_aatype and get HOH-row-filtered below for the indexed path; the
        # unindexed final-step block (post-bipartite match) reassigns them with
        # its own filtered match output. Spatial conditioning still uses the
        # unfiltered motif_idx so water backbone-O constraints reach the model.
        motif_idx_for_fix = motif_idx
        motif_aatype_for_fix = motif_aatype
        if cc.enabled and not is_unindexed and motif_aa3 is not None:
            _is_water_indexed = []
            if isinstance(motif_aa3, list) and len(motif_aa3) > 0 and isinstance(motif_aa3[0], list):
                _is_water_indexed = [
                    [a == "HOH" for a in aa3_list] for aa3_list in motif_aa3
                ]
            else:
                _flat = [a == "HOH" for a in motif_aa3] if motif_aa3 else []
                _is_water_indexed = [_flat for _ in range(batch_size)]
            if any(any(iw) for iw in _is_water_indexed):
                motif_idx_for_fix = []
                motif_aatype_for_fix = []
                for bi in range(batch_size):
                    iw = _is_water_indexed[bi]
                    src_idx = motif_idx[bi] if bi < len(motif_idx) else []
                    src_aat = motif_aatype[bi] if bi < len(motif_aatype) else torch.tensor([], dtype=torch.long)
                    keep = [k for k, w in enumerate(iw) if not w]
                    motif_idx_for_fix.append([src_idx[k] for k in keep])
                    if isinstance(src_aat, torch.Tensor):
                        motif_aatype_for_fix.append(
                            src_aat[torch.tensor(keep, dtype=torch.long)]
                            if keep else torch.tensor([], dtype=src_aat.dtype)
                        )
                    else:
                        motif_aatype_for_fix.append([src_aat[k] for k in keep])

        all_crop_cond_coords = None
        multistate_motif_coords = []   # per-state full-atom motif coords (multi-state branch)
        if (
            cc.crop_conditional_guidance.enabled
            and cc.crop_conditional_guidance.start == 0.0
            and not is_unindexed
        ):
            if multi_motif_data is not None:
                # Multi-motif: create crop_cond_coords for each motif
                all_crop_cond_coords = []
                for mm_entry in multi_motif_data:
                    mm_all_atom = mm_entry['all_atom']

                    # Keep each state's MOTIF-ONLY placed coords [B, n_motif, 37, 3]
                    # (same layout as motif_all_atom) so multi-state sampling can pin
                    # that state's chromophore in the branch re-denoise.
                    multistate_motif_coords.append(mm_all_atom.detach().clone())

                    ccc = torch.zeros_like(xt).to(xt.device)
                    for bi in range(batch_size):
                        ccc[bi, motif_idx[bi]] = mm_all_atom[bi]

                    ccc = apply_crop_cond_strategy(
                        ccc,
                        motif_idx,
                        motif_aa3,
                        cc.crop_conditional_guidance.strategy,
                    )
                    ccc_mask = (ccc.abs().sum(-1) > 0).float()
                    all_crop_cond_coords.append((ccc, ccc_mask, mm_entry['weight']))

                # Set crop_cond_coords to first motif's for any code that expects it
                crop_cond_coords = all_crop_cond_coords[0][0]
            else:
                # Single motif (existing logic)
                crop_cond_coords = torch.zeros_like(xt).to(xt.device)
                # fill in with motif coords at the current motif_idx
                for bi, _ in enumerate(motif_idx):
                    crop_cond_coords[bi, motif_idx[bi]] = motif_all_atom[bi]

                crop_cond_coords = apply_crop_cond_strategy(
                    crop_cond_coords,
                    motif_idx,
                    motif_aa3,
                    cc.crop_conditional_guidance.strategy,
                )

        # Construct crop_cond_mask for condmask channel (derive from non-zero coords)
        crop_cond_mask = None
        if (
            "conditioning_style" in self.config.model
            and "condmask" in self.config.model.conditioning_style
        ):
            if all_crop_cond_coords is not None:
                crop_cond_mask = all_crop_cond_coords[0][1]
            elif crop_cond_coords is not None:
                crop_cond_mask = (crop_cond_coords.abs().sum(-1) > 0).float()

        # Negative-motif crop-conditional tensor. Same motif_idx, same
        # apply_crop_cond_strategy, same condmask -- only coords differ. We reuse the
        # POSITIVE condmask verbatim (rather than re-deriving it from non-zero
        # negative coords) so that a coordinate that lands exactly on the origin in
        # one of the two motifs cannot flip a mask bit and turn "atom moved" into
        # "atom appeared". The atom sets are asserted identical, so the correct
        # condmask is by construction the positive one.
        negative_crop_conds: list[tuple] = []
        if (
            neg_motif_coords_list
            and cc.crop_conditional_guidance.enabled
            and cc.crop_conditional_guidance.start == 0.0
            and not is_unindexed
        ):
            for neg_coords, neg_w in zip(neg_motif_coords_list, neg_motif_weights):
                nccc = torch.zeros_like(xt).to(xt.device)
                for bi, _ in enumerate(motif_idx):
                    nccc[bi, motif_idx[bi]] = neg_coords[bi]
                nccc = apply_crop_cond_strategy(
                    nccc,
                    motif_idx,
                    motif_aa3,
                    cc.crop_conditional_guidance.strategy,
                )
                negative_crop_conds.append((nccc, crop_cond_mask, neg_w))

        crop_cond_seq_oh = None
        if (
            "minimpnn_seqcond" in cc.crop_conditional_guidance
            and cc.crop_conditional_guidance.minimpnn_seqcond
        ):
            crop_cond_seq_oh = torch.zeros(batch_size, seq_length, self.n_tokens).to(
                xt.device
            )
            # fill in with motif aatype at the current motif_idx
            for bi in range(batch_size):
                crop_cond_seq_oh[bi, motif_idx[bi]] = motif_aatype[bi]

        # Place motif hotspot mask into full hotspot mask
        hotspot_mask = torch.zeros_like(residue_index)
        if hotspots is not None:
            for bi in range(batch_size):
                for ii, mi in enumerate(motif_idx[bi]):
                    hotspot_mask[bi, mi] = motif_hotspot_mask[bi, ii]

        if os.environ.get("PROTPARDELLE_DEBUG_COND"):
            logger.info(
                "[cond-debug] conditioning_style=%s hotspots=%r "
                "motif_hotspot_mask.sum=%s hotspot_mask.sum=%s per_sample=%s",
                self.config.model.conditioning_style,
                hotspots,
                int(locals()["motif_hotspot_mask"].sum())
                if locals().get("motif_hotspot_mask") is not None
                else None,
                int(hotspot_mask.sum()),
                hotspot_mask.sum(-1).tolist(),
            )

        # Build motif SSE conditioning (cc143/cc144 dssp channel). For unindexed,
        # the model extends inputs to seq_dim N+M inside Protpardelle.forward, so
        # motif_sse_cond must be sized [B, N+M, 3]. For indexed (concat+condmask),
        # it's [B, N, 3] with one-hot at scaffold motif positions.
        #
        # `motif_sse_label` accepts:
        #   - None: no SSE conditioning (all zero channels)
        #   - keyword: "helix"/"strand"/"loop"/"any" — broadcast to every motif residue
        #   - single letter "H"/"E"/"L"/"X" — same broadcast
        #   - per-residue string of length K (motif size), letters drawn from
        #     {H, E, L, X} where X means "no info" (zero vector for that residue)
        motif_sse_cond = None
        char_to_idx = {"H": 0, "E": 1, "L": 2, "X": -1}
        keyword_to_letter = {
            "helix": "H", "alpha": "H", "h": "H",
            "strand": "E", "beta": "E", "sheet": "E", "e": "E",
            "loop": "L", "coil": "L", "l": "L",
            "any": "X", "none": "X", "x": "X", "": "X",
        }
        if (
            motif_sse_label is not None
            and "dssp" in self.config.model.conditioning_style
        ):
            if is_unindexed:
                K = motif_all_atom.shape[1] if motif_all_atom is not None else 0
            else:
                K = len(motif_idx[0]) if len(motif_idx) > 0 else 0
            raw = str(motif_sse_label).strip()
            lo = raw.lower()
            if lo in keyword_to_letter:
                letters = keyword_to_letter[lo] * K
            elif all(ch.upper() in char_to_idx for ch in raw):
                if len(raw) != K:
                    raise ValueError(
                        f"motif_sse_label string {raw!r} has length {len(raw)} but "
                        f"the motif has {K} residues; must match exactly"
                    )
                letters = raw.upper()
            else:
                raise ValueError(
                    f"motif_sse_label must be one of helix/strand/loop/any or a "
                    f"per-residue string of letters in {{H,E,L,X}}, got {motif_sse_label!r}"
                )
            if is_unindexed:
                M_cap = self.config.model.max_motif_size
                motif_sse_cond = torch.zeros(
                    batch_size, seq_length + M_cap, 3, device=xt.device, dtype=xt.dtype
                )
                # One-hot only on the real motif residues (positions [N..N+K)),
                # not on padding which is gated separately by motif_seq_mask.
                for r, ch in enumerate(letters):
                    idx = char_to_idx[ch]
                    if idx >= 0:
                        motif_sse_cond[:, seq_length + r, idx] = 1.0
            else:
                motif_sse_cond = torch.zeros(
                    batch_size, seq_length, 3, device=xt.device, dtype=xt.dtype
                )
                # Place motif SSE one-hot at scaffold motif positions.
                for bi in range(batch_size):
                    for r, mi in enumerate(motif_idx[bi]):
                        if r >= len(letters):
                            break
                        idx = char_to_idx[letters[r]]
                        if idx >= 0:
                            motif_sse_cond[bi, mi, idx] = 1.0

                # Optional flanking SSE conditioning (cc147): for each motif residue,
                # also set its left/right neighbors' SSE channel to the same letter
                # (motif-position-local letter), respecting chain boundaries and the
                # valid sequence mask. Does not overwrite existing motif positions.
                if motif_sse_flank is not None and int(motif_sse_flank) > 0:
                    flank = int(motif_sse_flank)
                    if chain_index is None:
                        ci_t = torch.zeros(
                            batch_size, seq_length, device=xt.device, dtype=torch.long
                        )
                    else:
                        ci_t = chain_index
                    for bi in range(batch_size):
                        for r, mi in enumerate(motif_idx[bi]):
                            if r >= len(letters):
                                break
                            idx = char_to_idx[letters[r]]
                            if idx < 0:
                                continue
                            mi_int = int(mi)
                            chain_mi = int(ci_t[bi, mi_int].item())
                            for j in range(1, flank + 1):
                                mj = mi_int - j
                                if mj < 0:
                                    break
                                if int(ci_t[bi, mj].item()) != chain_mi:
                                    break
                                if float(seq_mask[bi, mj].item()) < 0.5:
                                    break
                                if motif_sse_cond[bi, mj].abs().sum() > 0:
                                    continue
                                motif_sse_cond[bi, mj, idx] = 1.0
                            for j in range(1, flank + 1):
                                mj = mi_int + j
                                if mj >= seq_length:
                                    break
                                if int(ci_t[bi, mj].item()) != chain_mi:
                                    break
                                if float(seq_mask[bi, mj].item()) < 0.5:
                                    break
                                if motif_sse_cond[bi, mj].abs().sum() > 0:
                                    continue
                                motif_sse_cond[bi, mj, idx] = 1.0

        # Binder global SSE-proportion conditioning (cc145). Provided as a 3-tuple
        # (h_frac, e_frac, l_frac) and broadcast to every binder (non-motif) residue,
        # zero at motif positions. Quantized to 0.1 to mirror training.
        binder_sse_prop_cond = None
        if (
            binder_sse_prop is not None
            and "binderprop" in self.config.model.conditioning_style
        ):
            if len(binder_sse_prop) != 3:
                raise ValueError(
                    f"binder_sse_prop must be a length-3 sequence (h, e, l); got {binder_sse_prop!r}"
                )
            prop = torch.tensor(
                [round(float(v) * 10.0) / 10.0 for v in binder_sse_prop],
                device=xt.device, dtype=xt.dtype,
            )  # [3]
            # Build per-residue mask: 1 at binder (non-motif) positions, 0 at motif.
            binder_res_mask = seq_mask.float().clone()
            for bi in range(batch_size):
                for mi in motif_idx[bi]:
                    binder_res_mask[bi, mi] = 0.0
            binder_sse_prop_cond = (
                prop.view(1, 1, 3) * binder_res_mask.unsqueeze(-1)
            )  # [B, N, 3]
            if os.environ.get("PROTPARDELLE_DEBUG_COND"):
                logger.info(
                    "[cond-debug] binder_sse_prop=%r quantized=%s n_binder_res=%s "
                    "cond.abs().sum=%.1f",
                    binder_sse_prop,
                    prop.tolist(),
                    binder_res_mask.sum(-1).tolist(),
                    float(binder_sse_prop_cond.abs().sum()),
                )

        # Per-residue motif SASA burial-class conditioning (cc91-sasa). Mirrors
        # motif_sse_cond but with classes buried/partial/exposed (channels 0/1/2).
        # `motif_sasa_label` accepts:
        #   - None: no SASA conditioning (all zero channels)
        #   - keyword "buried"/"partial"/"exposed"/"any" — broadcast to every motif residue
        #   - single letter "B"/"P"/"E"/"X" — same broadcast
        #   - per-residue string of length K (motif size) over {B, P, E, X}, X = no info
        motif_sasa_cond = None
        sasa_char_to_idx = {"B": 0, "P": 1, "E": 2, "X": -1}
        sasa_keyword_to_letter = {
            "buried": "B", "core": "B", "b": "B",
            "partial": "P", "partially_buried": "P", "intermediate": "P", "p": "P",
            "exposed": "E", "surface": "E", "solvent": "E", "e": "E",
            "any": "X", "none": "X", "x": "X", "": "X",
        }
        if (
            motif_sasa_label is not None
            and "sasa" in self.config.model.conditioning_style
        ):
            if is_unindexed:
                K = motif_all_atom.shape[1] if motif_all_atom is not None else 0
            else:
                K = len(motif_idx[0]) if len(motif_idx) > 0 else 0
            raw = str(motif_sasa_label).strip()
            lo = raw.lower()
            if lo in sasa_keyword_to_letter:
                letters = sasa_keyword_to_letter[lo] * K
            elif all(ch.upper() in sasa_char_to_idx for ch in raw):
                if len(raw) != K:
                    raise ValueError(
                        f"motif_sasa_label string {raw!r} has length {len(raw)} but "
                        f"the motif has {K} residues; must match exactly"
                    )
                letters = raw.upper()
            else:
                raise ValueError(
                    f"motif_sasa_label must be one of buried/partial/exposed/any or a "
                    f"per-residue string of letters in {{B,P,E,X}}, got {motif_sasa_label!r}"
                )
            if is_unindexed:
                M_cap = self.config.model.max_motif_size
                motif_sasa_cond = torch.zeros(
                    batch_size, seq_length + M_cap, 3, device=xt.device, dtype=xt.dtype
                )
                for r, ch in enumerate(letters):
                    idx = sasa_char_to_idx[ch]
                    if idx >= 0:
                        motif_sasa_cond[:, seq_length + r, idx] = 1.0
            else:
                motif_sasa_cond = torch.zeros(
                    batch_size, seq_length, 3, device=xt.device, dtype=xt.dtype
                )
                for bi in range(batch_size):
                    for r, mi in enumerate(motif_idx[bi]):
                        if r >= len(letters):
                            break
                        idx = sasa_char_to_idx[letters[r]]
                        if idx >= 0:
                            motif_sasa_cond[bi, mi, idx] = 1.0

        # Per-atom motif hydrogen-bond conditioning (cc91-sasa-hbond). Marks
        # specific motif ATOMS (not whole residues) as participating in a
        # side-chain hydrogen bond, matching the per-atom [B, N, 37] `hbond`
        # channel the model was trained with. `motif_hbond_label` is a single
        # string or a list of "<resspec>:<atom_name>" entries, where <resspec>
        # uses the motif PDB's original chain+residue numbering (e.g. "A66") and
        # <atom_name> is an atom37 name (e.g. "OH" for a Tyr tip hydroxyl).
        # Residues are resolved to scaffold positions exactly like hotspots
        # (parse on the source motif feats, then map through motif_idx). Indexed
        # path only (unindexed h-bond conditioning was never trained).
        motif_hbond_cond = None
        if (
            motif_hbond_label is not None
            and "hbond" in self.config.model.conditioning_style
        ):
            if is_unindexed:
                raise ValueError(
                    "motif_hbond_label is only supported for indexed sampling; "
                    "the hbond channel was not trained for unindexed motifs."
                )
            specs = (
                [motif_hbond_label]
                if isinstance(motif_hbond_label, str)
                else list(motif_hbond_label)
            )
            n_atoms = motif_all_atom.shape[-2]
            motif_hbond_cond = torch.zeros(
                batch_size, seq_length, n_atoms, device=xt.device, dtype=xt.dtype
            )
            for spec in specs:
                if spec is None:
                    continue
                if ":" not in str(spec):
                    raise ValueError(
                        f"motif_hbond_label entry {spec!r} must be "
                        f"'<chain><resnum>:<atom_name>', e.g. 'A66:OH'"
                    )
                resspec, atom_name = str(spec).rsplit(":", 1)
                atom_name = atom_name.strip()
                if atom_name not in residue_constants.atom_order:
                    raise ValueError(
                        f"motif_hbond_label atom {atom_name!r} is not a valid atom37 "
                        f"name (in spec {spec!r})"
                    )
                atom_idx = residue_constants.atom_order[atom_name]
                rows = parse_fixed_pos_str(
                    resspec.strip(),
                    chain_id_mapping,
                    motif_feats["residue_index_orig"],
                    motif_feats["chain_index"],
                )
                if not rows:
                    logger.warning(
                        "motif_hbond_label spec %r matched no motif residue; skipping.",
                        spec,
                    )
                for bi in range(batch_size):
                    for row in rows:
                        if row < len(motif_idx[bi]):
                            motif_hbond_cond[bi, motif_idx[bi][row], atom_idx] = 1.0

        xt_traj, x0_traj, st_traj, s0_traj = [], [], [], []
        xt_raw_traj = []   # unscaled xt (physical coords) for trajectory visualisation

        residue_index_orig = residue_index.clone()
        chain_id_mapping = None
        if partial_diffusion is not None and pd.enabled:
            sigma = sigma_float = noise_schedule(timesteps[pd_step])
            timesteps = timesteps[pd_step:]

        # Sampling trajectory
        pbar = tqdm(total=len(timesteps[1:]), desc="Sampling backbones")

        for i, t in enumerate(iter(timesteps[1:])):

            run_mpnn = sidechain_mode and i > begin_mpnn_step

            # Set up noise levels
            sigma_next = sigma_next_float = noise_schedule(t)

            if i == n_steps - 1:
                sigma_next *= 0
                sigma_next_float *= 0
            gamma = (
                s_churn / n_steps
                if (sigma_next >= s_t_min and sigma_next <= s_t_max)
                else 0.0
            )
            sigma = torch.full((seq_mask.shape[0],), sigma_float, device=self.device)
            sigma_next = torch.full(
                (seq_mask.shape[0],), sigma_next_float, device=self.device
            )

            bb_seq = (seq_mask * residue_constants.restype_order["G"]).long()
            bb_atom_mask = atom37_mask_from_aatype(bb_seq, seq_mask)

            if sidechain_mode and jump_steps:
                # Fill in noise for masked positions since xt is initialized to zeros at each step
                zero_atom_mask = atom37_mask_from_aatype(s_hat, seq_mask)
                dummy_fill_mask = 1 - zero_atom_mask.unsqueeze(-1)

                if dummy_fill_mode == "CA":
                    if x0 is not None:
                        dummy_fill_noise = (
                            torch.randn_like(xt) * unsqueeze_trailing_dims(sigma, xt)
                            + x0[:, :, 1:2, :]
                        )
                    else:
                        dummy_fill_noise = (
                            torch.randn_like(xt) * unsqueeze_trailing_dims(sigma, xt)
                            + xt[:, :, 1:2, :]
                        )
                else:
                    dummy_fill_noise = torch.randn_like(xt) * unsqueeze_trailing_dims(
                        sigma, xt
                    )

                xt = xt * zero_atom_mask.unsqueeze(-1)
                xt = xt + dummy_fill_noise * dummy_fill_mask
                atom_mask = zero_atom_mask

                if self.config.model.task == "ai-allatom-hybrid":
                    xt = xt * bb_atom_mask.unsqueeze(-1)
                    atom_mask = bb_atom_mask

            if self.config.model.task == "ai-allatom" and uniform_steps:
                if pd.enabled:
                    s_hat = pd_motif_aatype.long()
                    ai_atom_mask = atom37_mask_from_aatype(
                        pd_motif_aatype.long(), seq_mask
                    )
                elif s_logprobs is not None and cc.enabled and gt_aatype is None:
                    s_hat = sample_aatype(s_logprobs)
                    s_hat = fill_motif_seq(s_hat, motif_idx_for_fix, motif_aatype_for_fix)
                    ai_atom_mask = atom37_mask_from_aatype(s_hat, seq_mask)
                elif gt_aatype is not None:
                    ai_atom_mask = atom37_mask_from_aatype(gt_aatype, seq_mask)
                xt = xt * ai_atom_mask.unsqueeze(-1)
                atom_mask = ai_atom_mask
            elif self.config.model.task == "ai-allatom-nomask" and uniform_steps:
                if pd.enabled:
                    s_hat = pd_motif_aatype.long()
                    zero_atom_mask = atom37_mask_from_aatype(
                        pd_motif_aatype.long(), seq_mask
                    )
                elif s_logprobs is not None and cc.enabled and gt_aatype is None:
                    s_hat = sample_aatype(s_logprobs)
                    s_hat = fill_motif_seq(s_hat, motif_idx_for_fix, motif_aatype_for_fix)
                    zero_atom_mask = atom37_mask_from_aatype(s_hat, seq_mask)
                elif gt_aatype is not None:
                    zero_atom_mask = atom37_mask_from_aatype(gt_aatype, seq_mask)
                else:
                    zero_atom_mask = atom37_mask_from_aatype(s_hat, seq_mask)

                if x0 is not None:
                    dummy_fill_mask = 1 - zero_atom_mask.unsqueeze(-1)
                    if dummy_fill_mode == "zero":
                        dummy_fill_noise = torch.randn_like(
                            xt
                        ) * unsqueeze_trailing_dims(sigma, xt)
                    else:
                        dummy_fill_noise = (
                            torch.randn_like(xt) * unsqueeze_trailing_dims(sigma, xt)
                            + x0[:, :, 1:2, :]
                        )
                    xt = xt * zero_atom_mask.unsqueeze(-1)
                    xt = xt + dummy_fill_noise * dummy_fill_mask
                atom_mask = zero_atom_mask
            elif self.config.model.task == "ai-allatom-hybrid" and uniform_steps:
                if i < (0.5 * n_steps):
                    xt = xt * bb_atom_mask.unsqueeze(-1)
                    atom_mask = bb_atom_mask
                else:
                    if pd.enabled:
                        s_hat = pd_motif_aatype.long()
                        zero_atom_mask = atom37_mask_from_aatype(
                            pd_motif_aatype.long(), seq_mask
                        )
                    elif s_logprobs is not None and cc.enabled and gt_aatype is None:
                        s_hat = sample_aatype(s_logprobs)
                        s_hat = fill_motif_seq(s_hat, motif_idx_for_fix, motif_aatype_for_fix)
                        zero_atom_mask = atom37_mask_from_aatype(s_hat, seq_mask)
                    elif gt_aatype is not None:
                        zero_atom_mask = atom37_mask_from_aatype(gt_aatype, seq_mask)

                    if x0 is not None:
                        dummy_fill_mask = 1 - zero_atom_mask.unsqueeze(-1)
                        if dummy_fill_mode == "zero":
                            dummy_fill_noise = torch.randn_like(
                                xt
                            ) * unsqueeze_trailing_dims(sigma, xt)
                        else:
                            dummy_fill_noise = (
                                torch.randn_like(xt)
                                * unsqueeze_trailing_dims(sigma, xt)
                                + x0[:, :, 1:2, :]
                            )
                        xt = xt * zero_atom_mask.unsqueeze(-1)
                        xt = xt + dummy_fill_noise * dummy_fill_mask
                    atom_mask = zero_atom_mask
            elif self.config.model.task == "backbone":
                xt = xt * bb_atom_mask.unsqueeze(-1)
                atom_mask = bb_atom_mask

            # Structure denoising step
            if sigma_float > 0:
                for k in range(cc.num_recurrence_steps):
                    # Enable grad for reconstruction guidance
                    if cc.reconstruction_guidance.enabled:
                        torch.set_grad_enabled(True)
                        xt.requires_grad = True

                    if gamma > 0:
                        if k > 0:  # self-recurrence from Universal Guidance paper
                            sigma_hat = sigma
                            sigma_delta = torch.sqrt(
                                sigma**2 - sigma_next**2
                            )  # self-recurrence xt has slightly less noise
                        else:
                            sigma_hat = sigma + gamma * sigma
                            sigma_delta = torch.sqrt(sigma_hat**2 - sigma**2)

                        noisier_x = xt + unsqueeze_trailing_dims(
                            sigma_delta, xt
                        ) * noise_scale * torch.randn_like(xt).to(xt)

                        xt_hat = noisier_x * unsqueeze_trailing_dims(
                            seq_mask, noisier_x
                        )

                        if self.config.model.task == "ai-allatom" and uniform_steps:
                            if pd.enabled:
                                s_hat = pd_motif_aatype.long()
                                ai_atom_mask = atom37_mask_from_aatype(
                                    pd_motif_aatype.long(), seq_mask
                                )
                            elif (
                                s_logprobs is not None
                                and cc.enabled
                                and gt_aatype is None
                            ):
                                s_hat = sample_aatype(s_logprobs)
                                s_hat = fill_motif_seq(s_hat, motif_idx_for_fix, motif_aatype_for_fix)
                                ai_atom_mask = atom37_mask_from_aatype(s_hat, seq_mask)
                            elif gt_aatype is not None:
                                ai_atom_mask = atom37_mask_from_aatype(
                                    gt_aatype, seq_mask
                                )
                            xt_hat = xt_hat * ai_atom_mask.unsqueeze(-1)
                            atom_mask = ai_atom_mask
                        elif (
                            self.config.model.task == "ai-allatom-nomask"
                            and uniform_steps
                        ):
                            if pd.enabled:
                                s_hat = pd_motif_aatype.long()
                                zero_atom_mask = atom37_mask_from_aatype(
                                    pd_motif_aatype.long(), seq_mask
                                )
                            elif (
                                s_logprobs is not None
                                and cc.enabled
                                and gt_aatype is None
                            ):
                                s_hat = sample_aatype(s_logprobs)
                                s_hat = fill_motif_seq(s_hat, motif_idx_for_fix, motif_aatype_for_fix)

                                zero_atom_mask = atom37_mask_from_aatype(
                                    s_hat, seq_mask
                                )
                            elif gt_aatype is not None:
                                zero_atom_mask = atom37_mask_from_aatype(
                                    gt_aatype, seq_mask
                                )

                            if x0 is not None:
                                dummy_fill_mask = 1 - zero_atom_mask.unsqueeze(-1)
                                if dummy_fill_mode == "zero":
                                    dummy_fill_noise = torch.randn_like(
                                        xt_hat
                                    ) * unsqueeze_trailing_dims(sigma_hat, xt_hat)
                                else:
                                    dummy_fill_noise = (
                                        torch.randn_like(xt_hat)
                                        * unsqueeze_trailing_dims(sigma_hat, xt_hat)
                                        + x0[:, :, 1:2, :]
                                    )
                                xt_hat = xt_hat * zero_atom_mask.unsqueeze(-1)
                                xt_hat = xt_hat + dummy_fill_noise * dummy_fill_mask
                            atom_mask = zero_atom_mask
                        elif (
                            self.config.model.task == "ai-allatom-hybrid"
                            and uniform_steps
                        ):
                            if i < (0.5 * n_steps):
                                xt_hat = xt_hat * bb_atom_mask.unsqueeze(-1)
                                atom_mask = bb_atom_mask
                            else:
                                if pd.enabled:
                                    s_hat = pd_motif_aatype.long()
                                    zero_atom_mask = atom37_mask_from_aatype(
                                        pd_motif_aatype.long(), seq_mask
                                    )
                                elif (
                                    s_logprobs is not None
                                    and cc.enabled
                                    and gt_aatype is None
                                ):
                                    s_hat = sample_aatype(s_logprobs)
                                    s_hat = fill_motif_seq(
                                        s_hat, motif_idx_for_fix, motif_aatype_for_fix
                                    )

                                    zero_atom_mask = atom37_mask_from_aatype(
                                        s_hat, seq_mask
                                    )
                                elif gt_aatype is not None:
                                    zero_atom_mask = atom37_mask_from_aatype(
                                        gt_aatype, seq_mask
                                    )

                                if x0 is not None:
                                    dummy_fill_mask = 1 - zero_atom_mask.unsqueeze(-1)
                                    if dummy_fill_mode == "zero":
                                        dummy_fill_noise = torch.randn_like(
                                            xt_hat
                                        ) * unsqueeze_trailing_dims(sigma_hat, xt_hat)
                                    else:
                                        dummy_fill_noise = (
                                            torch.randn_like(xt_hat)
                                            * unsqueeze_trailing_dims(sigma_hat, xt_hat)
                                            + x0[:, :, 1:2, :]
                                        )
                                    xt_hat = xt_hat * zero_atom_mask.unsqueeze(-1)
                                    xt_hat = xt_hat + dummy_fill_noise * dummy_fill_mask
                                atom_mask = zero_atom_mask
                        elif self.config.model.task == "backbone":
                            xt_hat = xt_hat * bb_atom_mask.unsqueeze(-1)
                            atom_mask = bb_atom_mask

                        xt_rep = xt_hat.clone()
                        if cc.enabled:
                            # Replacement guidance
                            if (
                                cc.replacement_guidance.enabled
                                and i >= (cc.replacement_guidance.start * n_steps)
                                and i < (cc.replacement_guidance.end * n_steps)
                            ):
                                for bi, _ in enumerate(motif_idx):
                                    for raw_mi, mi in enumerate(motif_idx[bi]):
                                        replacement_idx = torch.nonzero(
                                            atom_mask[bi, mi]
                                        ).flatten()
                                        if tip_atom_conditioning:
                                            aatype_int = motif_aatype[bi][raw_mi]
                                            motif_aatype_str = (
                                                residue_constants.restype_1to3[
                                                    residue_constants.order_restype[
                                                        aatype_int.item()
                                                    ]
                                                ]
                                            )
                                            tip_atomtypes = residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS[
                                                motif_aatype_str
                                            ]
                                            replacement_idx = [
                                                residue_constants.atom_order.get(atype)
                                                for atype in tip_atomtypes
                                            ]
                                        xt_rep[bi, mi, replacement_idx] = (
                                            motif_all_atom[bi, raw_mi, replacement_idx]
                                        )

                        if all_crop_cond_coords is not None:
                            # Multi-motif weighted averaging
                            x0_accum = None
                            x_self_cond_accum = None

                            for ccc, ccc_mask, weight in all_crop_cond_coords:
                                x0_m, s_logprobs_m, x_self_cond_m, s_self_cond_m = self._forward_guided(guidance_model, autoguidance_weight,
                                    negative_crop_conds=negative_crop_conds,
                                    negative_guidance_weight=negative_guidance_weight,
                                    noisy_coords=xt_rep,
                                    noise_level=sigma_hat,
                                    seq_mask=seq_mask,
                                    residue_index=residue_index,
                                    chain_index=chain_index,
                                    hotspot_mask=hotspot_mask,
                                    struct_self_cond=(
                                        x_self_cond
                                        if self.config.train.self_cond_train_prob > 0.5 and not latent_interpolation
                                        else None
                                    ),
                                    struct_crop_cond=ccc,
                                    crop_cond_mask=ccc_mask,
                                    sse_cond=sse_cond,
                                    adj_cond=adj_cond,
                                    motif_sse_cond=motif_sse_cond,
                                    binder_sse_prop_cond=binder_sse_prop_cond,
                                    motif_sasa_cond=motif_sasa_cond,
                                    motif_hbond_cond=motif_hbond_cond,
                                    seq_self_cond=(
                                        s_self_cond
                                        if self.config.model.mpnn_model.use_self_conditioning
                                        else None
                                    ),
                                    seq_crop_cond=crop_cond_seq_oh,
                                    run_mpnn_model=run_mpnn,
                                )

                                if x0_accum is None:
                                    x0_accum = weight * x0_m
                                    x_self_cond_accum = weight * x_self_cond_m
                                    s_logprobs = s_logprobs_m
                                    s_self_cond = s_self_cond_m
                                else:
                                    x0_accum = x0_accum + weight * x0_m
                                    x_self_cond_accum = x_self_cond_accum + weight * x_self_cond_m

                            x0 = x0_accum
                            x_self_cond = x_self_cond_accum
                        else:
                            # Single motif (existing logic)
                            x0, s_logprobs, x_self_cond, s_self_cond = self._forward_guided(guidance_model, autoguidance_weight,
                                negative_crop_conds=negative_crop_conds,
                                negative_guidance_weight=negative_guidance_weight,
                                noisy_coords=xt_rep,
                                noise_level=sigma_hat,
                                seq_mask=seq_mask,
                                residue_index=residue_index,
                                chain_index=chain_index,
                                hotspot_mask=hotspot_mask,
                                struct_self_cond=(
                                    x_self_cond
                                    if self.config.train.self_cond_train_prob > 0.5 and not latent_interpolation
                                    else None
                                ),
                                struct_crop_cond=crop_cond_coords,
                                crop_cond_mask=crop_cond_mask,
                                sse_cond=sse_cond,
                                adj_cond=adj_cond,
                                motif_sse_cond=motif_sse_cond,
                                binder_sse_prop_cond=binder_sse_prop_cond,
                                motif_sasa_cond=motif_sasa_cond,
                                motif_hbond_cond=motif_hbond_cond,
                                seq_self_cond=(
                                    s_self_cond
                                    if self.config.model.mpnn_model.use_self_conditioning
                                    else None
                                ),
                                seq_crop_cond=crop_cond_seq_oh,
                                motif_clean_coords=motif_clean_coords_unindexed,
                                motif_condmask=motif_condmask_unindexed,
                                run_mpnn_model=run_mpnn,
                            )

                    else:
                        if k > 0:
                            sigma_delta = torch.sqrt(sigma**2 - sigma_next**2)
                        else:
                            sigma_delta = torch.sqrt(
                                sigma_next**2 - sigma_next**2
                            )  # don't add additional noise for the first pass

                        noisier_x = xt + unsqueeze_trailing_dims(
                            sigma_delta, xt
                        ) * noise_scale * torch.randn_like(xt).to(xt)

                        xt_hat = noisier_x * unsqueeze_trailing_dims(
                            seq_mask, noisier_x
                        )

                        if self.config.model.task == "ai-allatom" and uniform_steps:
                            if pd.enabled:
                                s_hat = pd_motif_aatype.long()
                                ai_atom_mask = atom37_mask_from_aatype(
                                    pd_motif_aatype.long(), seq_mask
                                )
                            elif (
                                s_logprobs is not None
                                and cc.enabled
                                and gt_aatype is None
                            ):
                                s_hat = sample_aatype(s_logprobs)
                                s_hat = fill_motif_seq(s_hat, motif_idx_for_fix, motif_aatype_for_fix)
                                ai_atom_mask = atom37_mask_from_aatype(s_hat, seq_mask)
                            elif gt_aatype is not None:
                                ai_atom_mask = atom37_mask_from_aatype(
                                    gt_aatype, seq_mask
                                )
                            xt_hat = xt_hat * ai_atom_mask.unsqueeze(-1)
                            atom_mask = ai_atom_mask
                        elif (
                            self.config.model.task == "ai-allatom-nomask"
                            and uniform_steps
                        ):
                            if pd.enabled:
                                s_hat = pd_motif_aatype.long()
                                zero_atom_mask = atom37_mask_from_aatype(
                                    pd_motif_aatype.long(), seq_mask
                                )
                            elif (
                                s_logprobs is not None
                                and cc.enabled
                                and gt_aatype is None
                            ):
                                s_hat = sample_aatype(s_logprobs)
                                s_hat = fill_motif_seq(s_hat, motif_idx_for_fix, motif_aatype_for_fix)
                                zero_atom_mask = atom37_mask_from_aatype(
                                    s_hat, seq_mask
                                )
                            elif gt_aatype is not None:
                                zero_atom_mask = atom37_mask_from_aatype(
                                    gt_aatype, seq_mask
                                )

                            if x0 is not None:
                                dummy_fill_mask = 1 - zero_atom_mask.unsqueeze(-1)

                                if dummy_fill_mode == "zero":
                                    dummy_fill_noise = torch.randn_like(
                                        xt
                                    ) * unsqueeze_trailing_dims(sigma, xt)
                                else:
                                    dummy_fill_noise = (
                                        torch.randn_like(xt)
                                        * unsqueeze_trailing_dims(sigma, xt)
                                        + x0[:, :, 1:2, :]
                                    )
                                xt_hat = xt_hat * zero_atom_mask.unsqueeze(-1)
                                xt_hat = xt_hat + dummy_fill_noise * dummy_fill_mask
                            atom_mask = zero_atom_mask
                        elif (
                            self.config.model.task == "ai-allatom-hybrid"
                            and uniform_steps
                        ):
                            if i < (0.5 * n_steps):
                                xt_hat = xt_hat * bb_atom_mask.unsqueeze(-1)
                                atom_mask = bb_atom_mask
                            else:
                                if pd.enabled:
                                    s_hat = pd_motif_aatype.long()
                                    zero_atom_mask = atom37_mask_from_aatype(
                                        pd_motif_aatype.long(), seq_mask
                                    )
                                elif (
                                    s_logprobs is not None
                                    and cc.enabled
                                    and gt_aatype is None
                                ):
                                    s_hat = sample_aatype(s_logprobs)
                                    s_hat = fill_motif_seq(
                                        s_hat, motif_idx_for_fix, motif_aatype_for_fix
                                    )
                                    zero_atom_mask = atom37_mask_from_aatype(
                                        s_hat, seq_mask
                                    )
                                elif gt_aatype is not None:
                                    zero_atom_mask = atom37_mask_from_aatype(
                                        gt_aatype, seq_mask
                                    )

                                if x0 is not None:
                                    dummy_fill_mask = 1 - zero_atom_mask.unsqueeze(-1)

                                    if dummy_fill_mode == "zero":
                                        dummy_fill_noise = torch.randn_like(
                                            xt
                                        ) * unsqueeze_trailing_dims(sigma, xt)
                                    else:
                                        dummy_fill_noise = (
                                            torch.randn_like(xt)
                                            * unsqueeze_trailing_dims(sigma, xt)
                                            + x0[:, :, 1:2, :]
                                        )
                                    xt_hat = xt_hat * zero_atom_mask.unsqueeze(-1)
                                    xt_hat = xt_hat + dummy_fill_noise * dummy_fill_mask
                                atom_mask = zero_atom_mask
                        elif self.config.model.task == "backbone":
                            xt_hat = xt_hat * bb_atom_mask.unsqueeze(-1)
                            atom_mask = bb_atom_mask

                        xt_rep = xt_hat.clone()
                        if cc.enabled:
                            # Replacement guidance
                            if (
                                cc.replacement_guidance.enabled
                                and i >= (cc.replacement_guidance.start * n_steps)
                                and i < (cc.replacement_guidance.end * n_steps)
                            ):
                                for bi, _ in enumerate(motif_idx):
                                    for raw_mi, mi in enumerate(motif_idx[bi]):
                                        replacement_idx = torch.nonzero(
                                            atom_mask[bi, mi]
                                        ).flatten()
                                        if tip_atom_conditioning:
                                            aatype_int = motif_aatype[bi][raw_mi]
                                            motif_aatype_str = (
                                                residue_constants.restype_1to3[
                                                    residue_constants.order_restype[
                                                        aatype_int.item()
                                                    ]
                                                ]
                                            )
                                            tip_atomtypes = residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS[
                                                motif_aatype_str
                                            ]
                                            replacement_idx = [
                                                residue_constants.atom_order.get(atype)
                                                for atype in tip_atomtypes
                                            ]
                                        xt_rep[bi, mi, replacement_idx] = (
                                            motif_all_atom[bi, raw_mi, replacement_idx]
                                        )

                        if all_crop_cond_coords is not None:
                            # Multi-motif weighted averaging
                            x0_accum = None
                            x_self_cond_accum = None

                            for ccc, ccc_mask, weight in all_crop_cond_coords:
                                x0_m, s_logprobs_m, x_self_cond_m, s_self_cond_m = self._forward_guided(guidance_model, autoguidance_weight,
                                    negative_crop_conds=negative_crop_conds,
                                    negative_guidance_weight=negative_guidance_weight,
                                    noisy_coords=xt_rep,
                                    noise_level=sigma,
                                    seq_mask=seq_mask,
                                    residue_index=residue_index,
                                    chain_index=chain_index,
                                    hotspot_mask=hotspot_mask,
                                    struct_self_cond=(
                                        x_self_cond
                                        if self.config.train.self_cond_train_prob > 0.5 and not latent_interpolation
                                        else None
                                    ),
                                    struct_crop_cond=ccc,
                                    crop_cond_mask=ccc_mask,
                                    sse_cond=sse_cond,
                                    adj_cond=adj_cond,
                                    motif_sse_cond=motif_sse_cond,
                                    binder_sse_prop_cond=binder_sse_prop_cond,
                                    motif_sasa_cond=motif_sasa_cond,
                                    motif_hbond_cond=motif_hbond_cond,
                                    seq_self_cond=(
                                        s_self_cond
                                        if self.config.model.mpnn_model.use_self_conditioning
                                        else None
                                    ),
                                    seq_crop_cond=crop_cond_seq_oh,
                                    run_mpnn_model=run_mpnn,
                                )

                                if x0_accum is None:
                                    x0_accum = weight * x0_m
                                    x_self_cond_accum = weight * x_self_cond_m
                                    s_logprobs = s_logprobs_m
                                    s_self_cond = s_self_cond_m
                                else:
                                    x0_accum = x0_accum + weight * x0_m
                                    x_self_cond_accum = x_self_cond_accum + weight * x_self_cond_m

                            x0 = x0_accum
                            x_self_cond = x_self_cond_accum
                        else:
                            # Single motif (existing logic)
                            x0, s_logprobs, x_self_cond, s_self_cond = self._forward_guided(guidance_model, autoguidance_weight,
                                negative_crop_conds=negative_crop_conds,
                                negative_guidance_weight=negative_guidance_weight,
                                noisy_coords=xt_rep,
                                noise_level=sigma,
                                seq_mask=seq_mask,
                                residue_index=residue_index,
                                chain_index=chain_index,
                                hotspot_mask=hotspot_mask,
                                struct_self_cond=(
                                    x_self_cond
                                    if self.config.train.self_cond_train_prob > 0.5 and not latent_interpolation
                                    else None
                                ),
                                struct_crop_cond=crop_cond_coords,
                                crop_cond_mask=crop_cond_mask,
                                sse_cond=sse_cond,
                                adj_cond=adj_cond,
                                motif_sse_cond=motif_sse_cond,
                                binder_sse_prop_cond=binder_sse_prop_cond,
                                motif_sasa_cond=motif_sasa_cond,
                                motif_hbond_cond=motif_hbond_cond,
                                seq_self_cond=(
                                    s_self_cond
                                    if self.config.model.mpnn_model.use_self_conditioning
                                    else None
                                ),
                                seq_crop_cond=crop_cond_seq_oh,
                                motif_clean_coords=motif_clean_coords_unindexed,
                                motif_condmask=motif_condmask_unindexed,
                                run_mpnn_model=run_mpnn,
                            )

                    if jump_steps or uniform_steps:
                        guidance_mask37 = atom37_mask_from_aatype(
                            s_hat, seq_mask
                        ).bool()
                    else:
                        bb_s = (seq_mask * 7).long()
                        guidance_mask37 = atom37_mask_from_aatype(bb_s, seq_mask).bool()

                    guidance_in = None
                    if cc.reconstruction_guidance.enabled:
                        loss_mask37 = guidance_mask37
                        if tip_atom_conditioning:
                            for bi, _ in enumerate(motif_idx):
                                for raw_mi, mi in enumerate(motif_idx[bi]):
                                    aatype_int = motif_aatype[bi][raw_mi]
                                    motif_aatype_str = residue_constants.restype_1to3[
                                        residue_constants.order_restype[
                                            aatype_int.item()
                                        ]
                                    ]
                                    tip_atomtypes = residue_constants.RFDIFFUSION_BENCHMARK_TIP_ATOMS[
                                        motif_aatype_str
                                    ]
                                    tip_idx = [
                                        residue_constants.atom_order.get(atype)
                                        for atype in tip_atomtypes
                                    ]
                                    nontip_idx = tuple(
                                        [
                                            ni
                                            for ni in range(guidance_mask37.shape[-1])
                                            if ni not in tip_idx
                                        ]
                                    )
                                    loss_mask37[bi, mi, nontip_idx] = 0

                        loss = torch.sum(
                            motif_loss(x0, motif_idx, motif_all_atom, loss_mask37)
                        )
                        loss = loss_weights.motif * loss

                        guidance = torch.autograd.grad(loss, xt_hat)[0]
                        guidance_in = (guidance, guidance_mask37.float())
                        torch.set_grad_enabled(False)

                    if jump_steps or uniform_steps:
                        # Determine sequence resampling probability
                        if anneal_seq_resampling_rate is not None:
                            step_time = 1 - (i - begin_mpnn_step) / max(
                                1, n_steps - begin_mpnn_step
                            )
                            if anneal_seq_resampling_rate == "linear":
                                resampling_rate = step_time
                            elif anneal_seq_resampling_rate == "cosine":
                                k = 2
                                resampling_rate = (
                                    1 + np.cos(2 * np.pi * (step_time - 0.5))
                                ) / k
                            resample_this_step = np.random.uniform() < resampling_rate

                        # Unindexed: at the final step, match rotated motif residues to
                        # scaffold positions via bipartite matching so we can fix residue
                        # types at those positions when designing with ProteinMPNN.
                        if (
                            is_unindexed
                            and i == n_steps - 1
                            and use_fullmpnn_for_final
                            and cc.enabled
                        ):
                            motif_idx, motif_aatype, motif_is_water_match = (
                                match_motif_to_scaffold_unindexed(
                                    x0,
                                    motif_all_atom,
                                    motif_atom_mask,
                                    motif_condmask_unindexed,
                                    motif_aatype_unindexed,
                                    seq_mask,
                                    motif_is_water_unindexed=motif_is_water_unindexed,
                                )
                            )
                            # Build a parallel filtered version that drops HOH
                            # placeholder rows so MPNN / fill_motif_seq don't pin
                            # a residue identity at water-anchor positions —
                            # those positions should be designed freely. The
                            # full lists (with waters) are preserved on the
                            # `*_full` variables so samp_aux can still record
                            # them for downstream analysis.
                            motif_idx_for_fix = []
                            motif_aatype_for_fix = []
                            for bi in range(len(motif_idx)):
                                iw = motif_is_water_match[bi]
                                if iw.numel() == 0 or not iw.any():
                                    motif_idx_for_fix.append(list(motif_idx[bi]))
                                    motif_aatype_for_fix.append(motif_aatype[bi].clone())
                                    continue
                                keep = ~iw
                                motif_idx_for_fix.append(
                                    [motif_idx[bi][j] for j in range(len(motif_idx[bi])) if bool(keep[j].item())]
                                )
                                motif_aatype_for_fix.append(motif_aatype[bi][keep])
                        # else: motif_idx_for_fix / motif_aatype_for_fix retain
                        # whatever was set at the start of sample() — for the
                        # indexed path they're the HOH-filtered versions; for
                        # the unindexed loop iterations before the final step,
                        # they alias the empty per-batch lists.

                        # Resample sequence or design with full ProteinMPNN
                        if gt_aatype is None and not pd.enabled:
                            if i == n_steps - 1 and use_fullmpnn_for_final:
                                s_hat = design_with_fullmpnn(
                                    x0,
                                    seq_mask,
                                    motif_aatype=motif_aatype_for_fix,
                                    motif_idx=motif_idx_for_fix,
                                ).to(x0.device)
                            elif (
                                anneal_seq_resampling_rate is None or resample_this_step
                            ):
                                if run_mpnn and use_fullmpnn:
                                    s_hat = design_with_fullmpnn(
                                        x0,
                                        seq_mask,
                                        motif_aatype=motif_aatype_for_fix,
                                        motif_idx=motif_idx_for_fix,
                                    ).to(x0.device)
                                else:
                                    s_hat = sample_aatype(s_logprobs)
                            if motif_aatype_for_fix is not None:
                                s_hat = fill_motif_seq(s_hat, motif_idx_for_fix, motif_aatype_for_fix)

                    if jump_steps:
                        # Write x0 into atom73_state_0 for atoms corresponding to old seqhat
                        atom73_state_0[mask73] = x0[
                            mask37
                        ]  # mask73 and mask37 have the same number of 1 bits

                        mask37 = atom37_mask_from_aatype(s_hat, seq_mask).bool()
                        mask73 = atom73_mask_from_aatype(s_hat, seq_mask).bool()

                        # Determine prev noise levels for atoms corresponding to new sequence
                        if gamma > 0:
                            step_sigma_prev_hat = (
                                torch.ones(*xt.shape[:-1]).to(xt)
                                * sigma_hat[..., None, None]
                            )
                            step_sigma_prev_hat[mask37] = (
                                sigma73_last[mask73] + gamma * sigma73_last[mask73]
                            )  # b, n, 37
                        else:
                            step_sigma_prev = (
                                torch.ones(*xt.shape[:-1]).to(xt)
                                * sigma[..., None, None]
                            )
                            step_sigma_prev[mask37] = sigma73_last[mask73]  # b, n, 37
                        step_sigma_next = sigma_next[..., None, None]  # b, 1, 1

                        # Denoising step on atoms corresponding to new sequence
                        b, n = mask37.shape[:2]
                        step_xt = torch.zeros(b, n, 37, 3).to(xt)
                        step_x0 = torch.zeros(b, n, 37, 3).to(xt)

                        step_xt[mask37] = atom73_state_t[mask73]
                        step_x0[mask37] = atom73_state_0[mask73]

                        if gamma > 0:
                            step_xt = ode_step(
                                step_sigma_prev_hat,
                                step_sigma_next,
                                step_xt,
                                step_x0,
                                guidance_in=guidance_in,
                                curr_step=i,
                                stage2=stage2,
                            )
                        else:
                            step_xt = ode_step(
                                step_sigma_prev,
                                step_sigma_next,
                                step_xt,
                                step_x0,
                                guidance_in=guidance_in,
                                curr_step=i,
                                stage2=stage2,
                            )
                        xt = step_xt

                        # Write new xt into atom73_state_t for atoms corresponding to new seqhat and update sigma_last
                        atom73_state_t[mask73] = step_xt[mask37]
                        sigma73_last[mask73] = step_sigma_next[0].item()

                    else:
                        if gamma > 0:
                            xt = ode_step(
                                sigma_hat,
                                sigma_next,
                                xt_hat,
                                x0,
                                guidance_in=guidance_in,
                                curr_step=i,
                            )
                        else:
                            xt = ode_step(
                                sigma,
                                sigma_next,
                                xt_hat,
                                x0,
                                guidance_in=guidance_in,
                                curr_step=i,
                            )

                    torch.cuda.empty_cache()

            else:
                xt = x0

            sigma = sigma_next
            sigma_float = sigma_next_float

            # Logging
            xt_scale = self.sigma_data / unsqueeze_trailing_dims(
                torch.sqrt(sigma_next**2 + self.sigma_data**2), xt
            )
            scaled_xt = xt * xt_scale
            xt_traj.append(scaled_xt.cpu())
            xt_raw_traj.append(xt.cpu())
            x0_traj.append(x0.cpu())
            st_traj.append(s_hat.cpu())
            s0_traj.append(s_logprobs.cpu())

            pbar.update(1)
        pbar.close()

        if return_last:
            return xt, s_hat, seq_mask
        elif return_aux:
            atom_mask = atom37_mask_from_aatype(s_hat, seq_mask)
            return {
                "x": xt,
                "s": s_hat,
                "seq_mask": seq_mask,
                "atom_mask": atom_mask,
                "xt_traj": xt_traj,
                "xt_raw_traj": xt_raw_traj,
                "x0_traj": x0_traj,
                "st_traj": st_traj,
                "s0_traj": s0_traj,
                "motif_idx": motif_idx,
                "motif_aatype": motif_aatype,
                "motif_all_atom": motif_all_atom,
                "negative_motif_all_atom": (neg_motif_coords_list or None),
                "multistate_motif_coords": (multistate_motif_coords or None),
                "motif_atom_mask": motif_atom_mask,
                "motif_aa3": motif_aa3,
                "motif_random_rots": random_rots,
                "motif_centering_delta": centering_delta,
                "residue_index": residue_index_orig,
                "chain_index": chain_index,
            }
        else:
            return xt_traj, x0_traj, st_traj, s0_traj, seq_mask


def load_model(
    config_path: StrPath, checkpoint_path: StrPath, device: Device = None
) -> Protpardelle:
    """Load a Protpardelle model from a configuration file and a checkpoint."""
    if device is None:
        device = get_default_device()
    assert isinstance(device, torch.device)  # for mypy
    config = load_config(config_path)

    checkpoint_path = norm_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if isinstance(checkpoint, dict):
        # training_state format: dict with "model_state_dict"
        state_dict = checkpoint["model_state_dict"]
    else:
        # full model object saved via torch.save(model)
        state_dict = checkpoint.state_dict()

    model = Protpardelle(config, device=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model


def load_model_phema(
    config_path: StrPath,
    checkpoint_dir: StrPath,
    epoch: int,
    target_std: float,
    device: Device = None,
) -> "Protpardelle":
    """Load a model with post-hoc EMA reconstruction.

    Finds all EMA snapshot files for the given epoch (e.g., epoch10_ema-0.050.pth,
    epoch10_ema-0.100.pth), solves for coefficients to reconstruct the desired
    target_std, and returns a model with the reconstructed weights.

    Args:
        config_path: Path to model config YAML.
        checkpoint_dir: Directory containing checkpoint files.
        epoch: Epoch number to reconstruct from.
        target_std: Desired EMA relative standard deviation.
        device: Device to load model on.

    Returns:
        Protpardelle model with reconstructed EMA weights.
    """
    import re
    from protpardelle.core.phema import solve_posthoc_coefficients

    if device is None:
        device = get_default_device()
    if isinstance(device, str):
        device = torch.device(device)
    config = load_config(config_path)

    checkpoint_dir = norm_path(checkpoint_dir)

    # Find all EMA snapshot files for this epoch
    ema_files = []
    for f in sorted(checkpoint_dir.iterdir()):
        m = re.match(rf"epoch{epoch}_ema-(\d+\.\d+)\.pth$", f.name)
        if m:
            ema_files.append((float(m.group(1)), f))

    if len(ema_files) == 0:
        raise FileNotFoundError(
            f"No EMA snapshots found for epoch {epoch} in {checkpoint_dir}"
        )

    in_stds = [std for std, _ in ema_files]
    # For single-epoch reconstruction, all snapshots share the same nimg.
    # Use epoch as a proxy (the exact nimg value cancels out when all inputs
    # come from the same training step).
    in_nimg = [float(epoch)] * len(ema_files)
    out_nimg = float(epoch)

    coefs = solve_posthoc_coefficients(
        np.array(in_nimg), np.array(in_stds),
        np.array([out_nimg]), np.array([target_std]),
    )  # [n_inputs, 1]

    # Load and combine EMA state dicts
    combined_state_dict = None
    for i, (std, fpath) in enumerate(ema_files):
        sd = torch.load(fpath, map_location=device, weights_only=False)
        if combined_state_dict is None:
            combined_state_dict = {k: v.float() * coefs[i, 0] for k, v in sd.items()}
        else:
            for k in combined_state_dict:
                combined_state_dict[k] += sd[k].float() * coefs[i, 0]

    logger.info(
        "Reconstructed EMA (std=%.3f) from %d snapshots at epoch %d: stds=%s, coefs=%s",
        target_std, len(ema_files), epoch, in_stds, coefs[:, 0].tolist(),
    )

    model = Protpardelle(config, device=device)
    model.load_state_dict(combined_state_dict, strict=False)
    model.to(device)
    model.eval()

    return model
