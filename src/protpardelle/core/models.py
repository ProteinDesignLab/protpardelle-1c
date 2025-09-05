"""Top-level model definitions.

Typically these are initialized with config rather than arguments.

Authors: Alex Chu, Jinho Kim, Richard Shuai, Tianyu Lu, Zhaoyang Li
"""

import argparse
import copy
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig
from torch.types import Device
from torchtyping import TensorType
from tqdm.auto import tqdm

from protpardelle.common import residue_constants
from protpardelle.core import diffusion, modules
from protpardelle.data.atom import atom37_mask_from_aatype, atom73_mask_from_aatype
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
    # remove heteroatoms from motif_aatype
    motif_aatype = [
        [ma for ma in mab if ma in residue_constants.restype_3to1]
        for mab in motif_aatype
    ]
    crop_cond_coords = coords.clone()
    if strategy is None:
        strategy = "backbone"
    if "backbone" in strategy and "sidechain" not in strategy:
        crop_cond_coords[:, :, 3, :] = 0
        crop_cond_coords[:, :, 5:, :] = 0
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
        sse_cond: TensorType["b n", int] | None = None,
        adj_cond: TensorType["b n n", int] | None = None,
        return_emb: bool = False,
    ):

        noise_level = noise_level.clamp(min=1e-6)

        # Prep inputs and time conditioning
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

        if self.config.model.crop_conditional:
            if (
                "conditioning_style" in self.config.model
                and "concat" in self.config.model.conditioning_style
            ):
                if struct_crop_cond is None:
                    struct_crop_cond = torch.zeros_like(noisy_coords)
                else:
                    struct_crop_cond = struct_crop_cond / self.sigma_data
                emb = torch.cat([emb, struct_self_cond, struct_crop_cond], dim=-1)

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

        emb, hidden = self.net(
            emb,
            noise_cond,
            motif_cond=None,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            return_emb=return_emb,
            pair_bias=adj_cond,
        )

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
        ckpt_dict = torch.load(ckpt_path, weights_only=False, map_location=self.device)
        model_state_dict = ckpt_dict["model_state_dict"]

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
        sse_cond: TensorType["b n", int] | None = None,
        adj_cond: TensorType["b n n", int] | None = None,
        seq_self_cond: TensorType["b n t", float] | None = None,  # logprobs
        seq_crop_cond: TensorType["b n", int] | None = None,  # motif aatypes
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
            seq_self_cond: mpnn-predicted sequence logprobs from the previous step.
            run_struct_model: flag to optionally not run structure denoiser.
            run_mpnn_model: flag to optionally not run MiniMPNN.
        """

        # Coordinate denoiser
        if run_struct_model:
            denoised_coords, _ = self.struct_model(
                noisy_coords,
                noise_level,
                seq_mask,
                residue_index=residue_index,
                chain_index=chain_index,
                hotspot_mask=hotspot_mask,
                struct_self_cond=struct_self_cond,
                struct_crop_cond=struct_crop_cond,
                sse_cond=sse_cond,
                adj_cond=adj_cond,
                return_emb=True,
            )
        else:
            denoised_coords = noisy_coords

        # MiniMPNN
        if self.use_mpnn_model and run_mpnn_model:
            assert isinstance(self.mpnn_model, MiniMPNN)
            aatype_logprobs = self.mpnn_model(
                denoised_coords.detach(),
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

        struct_self_cond_out = denoised_coords.detach() / self.sigma_data

        seq_self_cond_out = aatype_logprobs.detach()

        return denoised_coords, aatype_logprobs, struct_self_cond_out, seq_self_cond_out

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
        dx: float | None = None,
        dy: float | None = None,
        dz: float | None = None,
        dummy_fill_mode: Literal["zero", "CA"] = "zero",
        xt_start: TensorType["b n a x", float] | None = None,
        partial_diffusion: DictConfig | None = None,
        conditional_cfg: DictConfig | None = None,
        motif_placements_full: list[str] | None = None,
        motif_all_atom_stage1: TensorType["b n a x", float] | None = None,
        motif_idx_stage1: list[list[int]] | None = None,
        stage2: bool = False,
        tip_atom_conditioning: bool = False,
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
        """

        cc = apply_dotdict_recursively(conditional_cfg)  # shorthand
        pd = apply_dotdict_recursively(partial_diffusion)

        if sse_cond is not None and adj_cond is not None:
            sse_cond = sse_cond.to(self.device)
            adj_cond = adj_cond.to(self.device)

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
        if cc.enabled:
            motif_feats, hetero_obj = load_feats_from_pdb(
                motif_file_path, include_pos_feats=True
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
            motif_aa3 = [
                residue_constants.restype_1to3[residue_constants.restypes[aa_idx]]
                for aa_idx in motif_aatype
            ]

            motif_all_atom = motif_feats["atom_positions"].to(
                self.device
            )  # [num_res, 37, 3]

        if cc.enabled:
            if motif_placements_full is not None:

                all_motif_feats = []

                for mp_chains in motif_placements_full:
                    chain_motif_feats = defaultdict(list)
                    prev_motif_segments = 0
                    for mp in mp_chains.split(";/;"):
                        curr_motif_feats = copy.deepcopy(motif_feats)
                        motif_chain_resids = [
                            s for s in mp.split("/") if s[0].isalpha()
                        ]
                        flat_raw_idx = parse_fixed_pos_str(
                            ",".join(motif_chain_resids),
                            chain_id_mapping,
                            motif_feats["residue_index_orig"],
                            motif_feats["chain_index"],
                        )

                        if motif_chain_resids:
                            for k, v in curr_motif_feats.items():
                                chain_motif_feats[k].append(v[flat_raw_idx])
                            prev_motif_segments += len(motif_chain_resids)
                    for k, v in chain_motif_feats.items():
                        chain_motif_feats[k] = torch.cat(v, dim=0)
                    all_motif_feats.append(chain_motif_feats)

                motif_residx = [
                    mf["residue_index"].numpy().astype(int) for mf in all_motif_feats
                ]
                motif_aatype = [mf["aatype"] for mf in all_motif_feats]
                motif_aa3 = []
                for ma in motif_aatype:
                    motif_aa3.append(
                        [
                            residue_constants.restype_1to3[
                                residue_constants.restypes[aa_idx]
                            ]
                            for aa_idx in ma
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

            if motif_all_atom_stage1 is not None:
                motif_all_atom = motif_all_atom_stage1.clone()
            else:
                # center the motif on CA coords
                motif_all_atom = motif_all_atom - torch.mean(
                    motif_all_atom[..., 1:2, :], dim=-3, keepdim=True
                )

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
            if isinstance(pd.pdb_file_path, list):
                pd_motif_aatype, pd_motif_idx, pd_coords = [], [], []
                for pd_fp in pd.pdb_file_path:
                    pd_feats, pd_hetero_obj = load_feats_from_pdb(pd_fp, include_pos_feats=True)
                    pd_motif_aatype.append(
                        make_fixed_size_1d(
                            pd_feats["aatype"].flatten().clone().detach(),
                            fixed_size=seq_mask.shape[-1],
                        )[0]
                    )
                    pd_motif_idx.append(torch.arange(pd_feats["aatype"].shape[0]))
                    pd_feats["atom_positions"] = pd_feats[
                        "atom_positions"
                    ] - torch.mean(
                        pd_feats["atom_positions"][:, 1:2, :], dim=-3, keepdim=True
                    )
                    pd_coords.append(
                        make_fixed_size_1d(
                            pd_feats["atom_positions"],
                            fixed_size=seq_mask.shape[-1],
                        )[0]
                    )
                pd_motif_aatype = (
                    torch.stack(pd_motif_aatype).long().to(seq_mask.device)
                )
                pd_coords = torch.stack(pd_coords).to(self.device)
            else:
                pd_feats, pd_hetero_obj = load_feats_from_pdb(pd.pdb_file_path, include_pos_feats=True)
                pd_motif_aatype = pd_feats["aatype"].clone().detach()
                pd_motif_idx = torch.arange(pd_motif_aatype.shape[0])
                pd_motif_aatype = (
                    torch.stack([pd_motif_aatype for _ in range(batch_size)])
                    .long()
                    .to(seq_mask.device)
                )
                pd_motif_idx = [pd_motif_idx for _ in range(batch_size)]
                pd_feats["atom_positions"] = pd_feats["atom_positions"] - torch.mean(
                    pd_feats["atom_positions"][:, 1:2, :], dim=-3, keepdim=True
                )
                pd_coords = torch.tile(
                    pd_feats["atom_positions"], (batch_size, 1, 1, 1)
                ).to(self.device)

            pd_noise_level = torch.full(
                (seq_mask.shape[0],), noise_schedule(pd_timestep), device=self.device
            )
            if "repack" in pd and pd.repack:
                for pi, pd_aa in enumerate(pd.seq):
                    pd_motif_aatype[:, pi] = residue_constants.restype_order[pd_aa]
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
                f"Using {pd.pdb_file_path} for partial diffusion, going back to step {pd_step}"
            )
        else:
            xt = torch.randn(*coords_shape).to(self.device)

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

        if cc.enabled:
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

        if (
            cc.crop_conditional_guidance.enabled
            and cc.crop_conditional_guidance.start == 0.0
        ):
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

        xt_traj, x0_traj, st_traj, s0_traj = [], [], [], []

        residue_index_orig = residue_index.clone()
        chain_id_mapping = None
        if partial_diffusion is not None and pd.enabled:
            sigma = sigma_float = noise_schedule(timesteps[pd_step])
            timesteps = timesteps[pd_step:]

            # update residue index based on partial diffusion input PDB (chain breaks + multiple chains)
            residue_index = torch.tile(pd_feats["residue_index"][None], (batch_size, 1)).to(residue_index)
            residue_index_orig = torch.tile(pd_feats["residue_index_orig"][None], (batch_size, 1)).to(residue_index)
            chain_index = torch.tile(pd_feats["chain_index"][None], (batch_size, 1)).to(chain_index)
            chain_id_mapping = pd_feats["chain_id_mapping"]

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
                    s_hat = fill_motif_seq(s_hat, motif_idx, motif_aatype)
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
                    s_hat = fill_motif_seq(s_hat, motif_idx, motif_aatype)
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
                        s_hat = fill_motif_seq(s_hat, motif_idx, motif_aatype)
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
                                s_hat = fill_motif_seq(s_hat, motif_idx, motif_aatype)
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
                                s_hat = fill_motif_seq(s_hat, motif_idx, motif_aatype)

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
                                        s_hat, motif_idx, motif_aatype
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

                        x0, s_logprobs, x_self_cond, s_self_cond = self.forward(
                            noisy_coords=xt_rep,
                            noise_level=sigma_hat,
                            seq_mask=seq_mask,
                            residue_index=residue_index,
                            chain_index=chain_index,
                            hotspot_mask=hotspot_mask,
                            struct_self_cond=(
                                x_self_cond
                                if self.config.train.self_cond_train_prob > 0.5
                                else None
                            ),
                            struct_crop_cond=crop_cond_coords,
                            sse_cond=sse_cond,
                            adj_cond=adj_cond,
                            seq_self_cond=(
                                s_self_cond
                                if self.config.model.mpnn_model.use_self_conditioning
                                else None
                            ),
                            seq_crop_cond=crop_cond_seq_oh,
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
                                s_hat = fill_motif_seq(s_hat, motif_idx, motif_aatype)
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
                                s_hat = fill_motif_seq(s_hat, motif_idx, motif_aatype)
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
                                        s_hat, motif_idx, motif_aatype
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

                        x0, s_logprobs, x_self_cond, s_self_cond = self.forward(
                            noisy_coords=xt_rep,
                            noise_level=sigma,
                            seq_mask=seq_mask,
                            residue_index=residue_index,
                            chain_index=chain_index,
                            hotspot_mask=hotspot_mask,
                            struct_self_cond=(
                                x_self_cond
                                if self.config.train.self_cond_train_prob > 0.5
                                else None
                            ),
                            struct_crop_cond=crop_cond_coords,
                            sse_cond=sse_cond,
                            adj_cond=adj_cond,
                            seq_self_cond=(
                                s_self_cond
                                if self.config.model.mpnn_model.use_self_conditioning
                                else None
                            ),
                            seq_crop_cond=crop_cond_seq_oh,
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

                        # Resample sequence or design with full ProteinMPNN
                        if gt_aatype is None and not pd.enabled:
                            if i == n_steps - 1 and use_fullmpnn_for_final:
                                s_hat = design_with_fullmpnn(
                                    x0,
                                    seq_mask,
                                    motif_aatype=motif_aatype,
                                    motif_idx=motif_idx,
                                ).to(x0.device)
                            elif (
                                anneal_seq_resampling_rate is None or resample_this_step
                            ):
                                if run_mpnn and use_fullmpnn:
                                    s_hat = design_with_fullmpnn(
                                        x0,
                                        seq_mask,
                                        motif_aatype=motif_aatype,
                                        motif_idx=motif_idx,
                                    ).to(x0.device)
                                else:
                                    s_hat = sample_aatype(s_logprobs)
                            if motif_aatype is not None:
                                s_hat = fill_motif_seq(s_hat, motif_idx, motif_aatype)

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
                "x0_traj": x0_traj,
                "st_traj": st_traj,
                "s0_traj": s0_traj,
                "motif_idx": motif_idx,
                "motif_aatype": motif_aatype,
                "motif_all_atom": motif_all_atom,
                "motif_atom_mask": motif_atom_mask,
                "motif_aa3": motif_aa3,
                "residue_index": residue_index_orig,
                "chain_index": chain_index,
                "chain_id_mapping": chain_id_mapping,
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
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )["model_state_dict"]

    model = Protpardelle(config, device=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model
